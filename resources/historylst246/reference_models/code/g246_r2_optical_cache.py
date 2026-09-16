#!/usr/bin/env python3
"""Resumable target-free canonical optical cache for G246 R2.

The cache is deliberately separate from the published texture sidecars.  It
stores one uncompressed uint16 ``640 x 640`` array for each public
fit/validation scene and B2--B7 band.  A network prefetch can therefore finish
once, after which texture feature construction is strictly local and may be
repeated without reopening a COG.

Only callers that already resolved :func:`g246_data.load_splits` with
``role="fit+validation"`` may supply scenes.  This module has no source-
manifest or locked-test argument, never persists a SAS token, and never stores
thermal/target arrays.
"""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Iterator, Mapping, Sequence

from affine import Affine
import numpy as np
import rasterio

try:
    from . import build_landsat30_texture_sidecars as texture
    from . import g246_data
except ImportError:
    import build_landsat30_texture_sidecars as texture
    import g246_data


CACHE_SCHEMA = "uhi-cdc-g246-r2-canonical-optical-cache-v1"
MANIFEST_NAME = "manifest.jsonl"
PROGRESS_NAME = "progress.json"
BAND_NAMES = tuple(texture.BAND_NAMES)
EXPECTED_SHAPE = (640, 640)
EXPECTED_DTYPE = np.dtype(np.uint16)
CACHE_MAX_BYTES = 40 * 1024**3
COMBINED_MAX_BYTES = 40 * 1024**3
MAX_ATTEMPTS = 8
MAX_RETRY_BACKOFF_SECONDS = 30.0
DEFAULT_REMOTE_WORKERS = 8
MAX_REMOTE_WORKERS = 16

# Conservative transport defaults already exercised by the diverse-source
# builder.  Only absent environment keys are passed to rasterio.Env, so an
# operator setting is never overwritten.
_TRANSPORT_DEFAULTS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif",
    "CPL_VSIL_CURL_CHUNK_SIZE": "65536",
    "CPL_VSIL_CURL_CACHE_SIZE": "33554432",
    "GDAL_INGESTED_BYTES_AT_OPEN": "65536",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_MAX_RETRY": "1",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_TCP_KEEPALIVE": "YES",
    "GDAL_HTTP_TCP_KEEPIDLE": "30",
    "GDAL_HTTP_TCP_KEEPINTVL": "30",
}
_CAP_COMMIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class CacheIndex:
    root: Path
    header: Mapping[str, Any]
    records: Mapping[tuple[str, str], Mapping[str, Any]]
    complete: bool

    def record_for(self, spec: Any, band: str) -> Mapping[str, Any]:
        key = (str(spec.entry.scene_id), str(band))
        try:
            return self.records[key]
        except KeyError as exc:
            raise ValueError(
                f"canonical optical cache lacks {key[0]}/{key[1]}"
            ) from exc


class CacheTransferStats:
    """Small target-free transfer receipt shared with the prefetch loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.remote_errors = 0
        self.retries = 0
        self.item_fetches = 0
        self.asset_opens = 0
        self.bands_written = 0

    def record_item_fetch(self) -> None:
        with self._lock:
            self.item_fetches += 1

    def record_asset_open(self) -> None:
        with self._lock:
            self.asset_opens += 1

    def record_band_written(self) -> None:
        with self._lock:
            self.bands_written += 1

    def record_remote_error(self, *, retrying: bool) -> None:
        with self._lock:
            self.remote_errors += 1
            self.retries += int(retrying)

    def document(self) -> dict[str, int]:
        with self._lock:
            return {
                "remote_errors": int(self.remote_errors),
                "retries": int(self.retries),
                "item_fetches": int(self.item_fetches),
                "asset_opens": int(self.asset_opens),
                "bands_written": int(self.bands_written),
            }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _safe_cache(path: str | os.PathLike[str]) -> Path:
    return g246_data.reject_forbidden_path(path).resolve()


def _scope_header(scope: Any) -> dict[str, Any]:
    splits = scope.splits
    return {
        "input_scope": "g246_data.load_splits:fit+validation",
        "receipt_sha256": splits.receipt_sha256,
        "campaign_id": splits.campaign_id,
        "source_manifest_sha256": splits.source_manifest_sha256,
        "fit_view_sha256": splits.fit_view_sha256,
        "validation_view_sha256": splits.validation_view_sha256,
        "fit_scene_count": len(scope.fit_entries),
        "validation_scene_count": len(scope.validation_entries),
        "allowed_scene_count": len(scope.entries),
        "selected_cities": list(scope.selected_cities),
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }


def _canonical_grid(spec: Any) -> dict[str, Any]:
    metadata = spec.metadata
    return {
        "crs": str(metadata["canonical_grid_crs"]),
        "transform": [float(value) for value in metadata["transform30"]],
        "shape": [640, 640],
        "resampling": "nearest",
        "destination_fill": 0,
    }


def _signature_sha(spec: Any, band: str) -> str:
    signatures = spec.metadata.get("asset_grid_signatures")
    if not isinstance(signatures, Mapping) or not isinstance(
        signatures.get(band), Mapping
    ):
        raise ValueError(
            f"scene lacks delivered signature: {spec.entry.scene_id}/{band}"
        )
    return canonical_sha256(signatures[band])


def band_name(spec: Any, band: str) -> str:
    if band not in BAND_NAMES:
        raise ValueError(f"unsupported optical cache band: {band!r}")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(spec.entry.city)).strip(
        "._-"
    ) or "city"
    identity = hashlib.sha256(str(spec.entry.scene_id).encode("utf-8")).hexdigest()[:12]
    return (
        f"{spec.entry.view_role}_{slug}_{int(spec.entry.year)}_"
        f"{identity}_{band}_canonical_u16.npy"
    )


def _expected_record(spec: Any, band: str, path: Path) -> dict[str, Any]:
    return {
        "kind": "band",
        "status": "complete",
        "view_role": spec.entry.view_role,
        "region": spec.entry.region,
        "city": spec.entry.city,
        "target_year": int(spec.entry.year),
        "scene_id": spec.entry.scene_id,
        "source_scene_sha256": spec.entry.sha256,
        "item_id": spec.entry.item_id,
        "datetime": spec.entry.datetime,
        "band": band,
        "canonical_grid_sha256": canonical_sha256(_canonical_grid(spec)),
        "delivered_grid_signature_sha256": _signature_sha(spec, band),
        "file": path.name,
        "shape": [640, 640],
        "dtype": "uint16",
        "locked_test_opened": False,
        "target_arrays_opened": False,
        "sas_persisted": False,
    }


def record_from_existing(path: Path, spec: Any, band: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.shape != EXPECTED_SHAPE or array.dtype != EXPECTED_DTYPE:
            return None
        record = _expected_record(spec, band, path)
        record.update({"sha256": sha256_file(path), "bytes": path.stat().st_size})
        return record
    except (OSError, ValueError, EOFError):
        return None


def _persistent_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Another cache worker may have atomically renamed its temporary
            # file between enumeration and stat.
            continue
    return total


def _enforce_cache_cap(root: Path) -> None:
    if _persistent_bytes(root) > CACHE_MAX_BYTES:
        raise RuntimeError("canonical optical cache exceeds the 40 GiB cap")


def enforce_combined_cap(cache_root: Path, output: Path) -> None:
    if cache_root == output or cache_root in output.parents or output in cache_root.parents:
        raise ValueError("raw cache and texture output must be separate non-nested paths")
    if _persistent_bytes(cache_root) + _persistent_bytes(output) > COMBINED_MAX_BYTES:
        raise RuntimeError("raw cache plus texture output exceeds the 40 GiB cap")


@contextmanager
def exclusive_cache_lock(root: Path) -> Iterator[Path]:
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.canonical-optical-cache.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another canonical optical cache writer holds {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "cache": str(root), "acquired": time.time()},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        temporary = Path(name)
        if temporary.exists():
            temporary.unlink()


def _write_manifest(
    path: Path, header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    payload = "\n".join(
        [json.dumps(dict(header), sort_keys=True)]
        + [json.dumps(dict(row), sort_keys=True) for row in records]
    ) + "\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        temporary = Path(name)
        if temporary.exists():
            temporary.unlink()


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("canonical optical cache manifest is empty")
    try:
        values = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError("canonical optical cache manifest is malformed") from exc
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("canonical optical cache manifest lines must be objects")
    return values[0], values[1:]


def _ordered_specs(specs: Sequence[Any]) -> tuple[Any, ...]:
    result = tuple(sorted(specs, key=lambda spec: str(spec.entry.scene_id)))
    identities = [str(spec.entry.scene_id) for spec in result]
    if len(identities) != len(set(identities)):
        raise ValueError("canonical optical cache received duplicate scenes")
    return result


def _validate_scope_specs(scope: Any, specs: Sequence[Any]) -> tuple[Any, ...]:
    ordered = _ordered_specs(specs)
    expected = {str(entry.scene_id) for entry in scope.entries}
    actual = {str(spec.entry.scene_id) for spec in ordered}
    if actual != expected:
        raise ValueError("canonical optical cache specs differ from guarded public scope")
    if any(spec.entry.view_role not in {"fit", "validation"} for spec in ordered):
        raise ValueError("canonical optical cache accepts only fit/validation scenes")
    return ordered


def _all_existing_records(
    root: Path, specs: Sequence[Any], *, remove_invalid: bool = False
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _ordered_specs(specs):
        for band in BAND_NAMES:
            path = root / band_name(spec, band)
            record = record_from_existing(path, spec, band)
            if record is not None:
                records.append(record)
            elif remove_invalid and path.exists():
                path.unlink()
    return records


def _header(
    scope: Any, records: Sequence[Mapping[str, Any]], *, status: str
) -> dict[str, Any]:
    scene_ids = {str(row["scene_id"]) for row in records}
    return {
        "kind": "dataset",
        "schema": CACHE_SCHEMA,
        "status": status,
        **_scope_header(scope),
        "cached_scene_count": len(scene_ids),
        "cached_band_count": len(records),
        "expected_band_count": len(scope.entries) * len(BAND_NAMES),
        "bands": list(BAND_NAMES),
        "array": {"shape": [640, 640], "dtype": "uint16"},
        "persistent_cap_bytes": CACHE_MAX_BYTES,
        "combined_cache_output_cap_bytes": COMBINED_MAX_BYTES,
        "signed_urls_persisted": False,
    }


def publish_manifest(root: Path, scope: Any, specs: Sequence[Any]) -> CacheIndex:
    specs = _validate_scope_specs(scope, specs)
    records = _all_existing_records(root, specs, remove_invalid=True)
    complete = len(records) == len(specs) * len(BAND_NAMES)
    status = "complete" if complete else "partial"
    header = _header(scope, records, status=status)
    _write_manifest(root / MANIFEST_NAME, header, records)
    progress = root / PROGRESS_NAME
    if progress.exists():
        progress.unlink()
    _enforce_cache_cap(root)
    return CacheIndex(
        root=root,
        header=header,
        records={(str(row["scene_id"]), str(row["band"])): row for row in records},
        complete=complete,
    )


def verify_cache(
    cache: str | os.PathLike[str],
    scope: Any,
    specs: Sequence[Any],
    *,
    required_specs: Sequence[Any] = (),
    require_complete: bool = False,
) -> CacheIndex:
    """Verify a complete or canonical partial cache without any network call."""

    root = _safe_cache(cache)
    specs = _validate_scope_specs(scope, specs)
    header, rows = _read_manifest(root / MANIFEST_NAME)
    if header.get("kind") != "dataset" or header.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported canonical optical cache manifest")
    if header.get("status") not in {"partial", "complete"}:
        raise ValueError("canonical optical cache status is invalid")
    for key, value in _scope_header(scope).items():
        if header.get(key) != value:
            raise ValueError(f"canonical optical cache {key} mismatch")
    allowed = {str(spec.entry.scene_id): spec for spec in specs}
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_files = {MANIFEST_NAME}
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        scene_id, band = str(row.get("scene_id", "")), str(row.get("band", ""))
        key = (scene_id, band)
        if key in by_key or scene_id not in allowed or band not in BAND_NAMES:
            raise ValueError(f"unexpected/duplicate canonical cache record: {key}")
        file_name = row.get("file")
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError("unsafe canonical optical cache filename")
        spec = allowed[scene_id]
        expected_name = band_name(spec, band)
        if file_name != expected_name:
            raise ValueError(f"canonical optical cache filename mismatch: {key}")
        canonical = record_from_existing(root / file_name, spec, band)
        if canonical is None or canonical != row:
            raise ValueError(f"canonical optical cache content mismatch: {key}")
        by_key[key] = row
        canonical_rows.append(canonical)
        expected_files.add(file_name)
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if any(not path.is_file() for path in root.iterdir()):
        raise ValueError("canonical optical cache must contain only regular files")
    if actual_files != expected_files:
        raise ValueError(
            "canonical optical cache file set mismatch; "
            f"extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )
    required = _ordered_specs(required_specs)
    missing = [
        (spec.entry.scene_id, band)
        for spec in required
        for band in BAND_NAMES
        if (str(spec.entry.scene_id), band) not in by_key
    ]
    if missing:
        raise ValueError(f"canonical optical cache lacks required bands: {missing[:5]}")
    is_complete = len(by_key) == len(allowed) * len(BAND_NAMES)
    expected_status = "complete" if is_complete else "partial"
    canonical_header = _header(scope, canonical_rows, status=expected_status)
    if header != canonical_header:
        raise ValueError("canonical optical cache header is not canonical")
    if require_complete and not is_complete:
        raise ValueError("canonical optical cache is partial")
    _enforce_cache_cap(root)
    return CacheIndex(root, header, by_key, is_complete)


def _write_band(root: Path, spec: Any, band: str, array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    if values.shape != EXPECTED_SHAPE or values.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"canonical optical band must be uint16 [640,640]: "
            f"{spec.entry.scene_id}/{band}"
        )
    path = root / band_name(spec, band)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-", suffix=".npy", dir=root
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(name)
        # Do not keep a mmap handle across os.replace.  WSL/DrvFS may retain a
        # delete-pending alias for the renamed file when the source is mapped.
        check = np.load(temporary, allow_pickle=False)
        if check.shape != EXPECTED_SHAPE or check.dtype != EXPECTED_DTYPE:
            raise ValueError("atomic canonical cache write failed validation")
        del check
        with _CAP_COMMIT_LOCK:
            _enforce_cache_cap(root)  # old target + every current temporary
            os.replace(temporary, path)
            _enforce_cache_cap(root)
    finally:
        temporary = Path(name)
        if temporary.exists():
            temporary.unlink()
    # The temporary payload was shape/dtype checked immediately before the
    # atomic replace.  Build the receipt from the committed bytes without a
    # second mmap open; full resume/manifest verification still reopens every
    # file and checks the same contract independently.
    record = _expected_record(spec, band, path)
    record.update({"sha256": sha256_file(path), "bytes": path.stat().st_size})
    return record


def load_scene(index: CacheIndex, spec: Any) -> np.ndarray:
    """Load one verified six-band scene using local files only."""

    arrays: list[np.ndarray] = []
    for band in BAND_NAMES:
        row = index.record_for(spec, band)
        path = index.root / str(row["file"])
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.shape != EXPECTED_SHAPE or array.dtype != EXPECTED_DTYPE:
            raise ValueError(f"verified cache band changed after verification: {path.name}")
        arrays.append(np.asarray(array))
    result = np.stack(arrays)
    if result.shape != (6, 640, 640) or result.dtype != EXPECTED_DTYPE:
        raise ValueError("canonical optical cache scene stack is invalid")
    return result


def _transport_env() -> dict[str, str]:
    return {
        key: value for key, value in _TRANSPORT_DEFAULTS.items() if key not in os.environ
    }


def _same_signature(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("crs") == expected.get("crs")
        and actual.get("shape") == expected.get("shape")
        and actual.get("dtype") == expected.get("dtype")
        and actual.get("nodata") == expected.get("nodata")
        and Affine(*actual["transform"]).almost_equals(
            Affine(*expected["transform"]), precision=1e-10
        )
    )


def _window_order(spec: Any) -> tuple[float, float, str]:
    window = spec.metadata.get("source_window30", ())
    if isinstance(window, Sequence) and len(window) >= 2:
        try:
            return float(window[1]), float(window[0]), str(spec.entry.scene_id)
        except (TypeError, ValueError):
            pass
    transform = spec.metadata.get("transform30", ())
    if isinstance(transform, Sequence) and len(transform) >= 6:
        return float(transform[5]), float(transform[2]), str(spec.entry.scene_id)
    return math.inf, math.inf, str(spec.entry.scene_id)


def _token_snapshot(tokens: Any) -> tuple[str, int | None]:
    snapshot = getattr(tokens, "snapshot", None)
    if callable(snapshot):
        token, generation = snapshot()
        return str(token), int(generation)
    return str(tokens.get()), None


def _refresh(tokens: Any, token: str, generation: int | None) -> tuple[str, int | None]:
    refresh_if_stale = getattr(tokens, "refresh_if_stale", None)
    if callable(refresh_if_stale):
        refreshed = str(refresh_if_stale(token, generation))
        return _token_snapshot(tokens) if generation is not None else (refreshed, None)
    return str(tokens.refresh()), None


def _validate_item(item: Mapping[str, Any], specs: Sequence[Any]) -> None:
    if not specs:
        raise ValueError("empty Landsat item group")
    item_id = str(specs[0].entry.item_id)
    timestamp = str(specs[0].entry.datetime)
    if item.get("id") != item_id or item.get("collection") != texture.builder.LANDSAT_COLLECTION:
        raise ValueError(f"exact Landsat item mismatch: {item_id}")
    if item.get("properties", {}).get("datetime") != timestamp:
        raise ValueError(f"exact Landsat datetime mismatch: {item_id}")
    if any(
        str(spec.entry.item_id) != item_id or str(spec.entry.datetime) != timestamp
        for spec in specs
    ):
        raise ValueError(f"inconsistent scene group for Landsat item: {item_id}")
    missing = set(BAND_NAMES).difference(item.get("assets", {}))
    if missing:
        raise ValueError(f"exact Landsat item lacks bands: {sorted(missing)}")


def _prefetch_item(
    root: Path,
    specs: Sequence[Any],
    *,
    tokens: Any,
    fetch_exact_item_func: Any,
    max_attempts: int,
    retry_backoff_seconds: float,
    stats: CacheTransferStats,
) -> None:
    ordered = tuple(sorted(specs, key=_window_order))
    missing_by_band = {
        band: [
            spec
            for spec in ordered
            if record_from_existing(root / band_name(spec, band), spec, band) is None
        ]
        for band in BAND_NAMES
    }
    if not any(missing_by_band.values()):
        return
    item = fetch_exact_item_func(ordered[0])
    stats.record_item_fetch()
    _validate_item(item, ordered)
    token, generation = _token_snapshot(tokens)
    for band in BAND_NAMES:
        pending = missing_by_band[band]
        if not pending:
            continue
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                href = texture.builder.signed_href(item, band, token)
                # The signed href is intentionally confined to this stack frame.
                with rasterio.Env(**_transport_env()):
                    with rasterio.open(href) as dataset:
                        stats.record_asset_open()
                        actual = texture.builder.grid_signature(dataset)
                        for spec in pending:
                            expected = spec.metadata["asset_grid_signatures"][band]
                            if not _same_signature(actual, expected):
                                raise ValueError(
                                    "delivered-grid signature changed: "
                                    f"{spec.entry.scene_id}/{band}"
                                )
                        for spec in tuple(pending):
                            grid = {
                                "crs": str(spec.metadata["canonical_grid_crs"]),
                                "transform": Affine(*spec.metadata["transform30"]),
                                "shape": EXPECTED_SHAPE,
                            }
                            array = texture.builder.reproject_asset_to_canonical(
                                dataset, grid, fill_value=0
                            )
                            _write_band(root, spec, band, array)
                            stats.record_band_written()
                            pending.remove(spec)
                last_error = None
                break
            except (rasterio.errors.RasterioIOError, texture.builder.RemoteAssetReadError) as exc:
                last_error = exc
                retrying = attempt + 1 < max_attempts
                stats.record_remote_error(retrying=retrying)
                if not retrying:
                    break
                delay = min(
                    retry_backoff_seconds * (2.0**attempt),
                    MAX_RETRY_BACKOFF_SECONDS,
                )
                if delay > 0:
                    time.sleep(delay)
                token, generation = _refresh(tokens, token, generation)
                # Files committed before the error stay valid; retry only the
                # still-pending scene windows for this band.
            if not pending:
                last_error = None
                break
        if last_error is not None:
            raise last_error


def populate_cache(
    cache: str | os.PathLike[str],
    scope: Any,
    all_specs: Sequence[Any],
    requested_specs: Sequence[Any],
    *,
    tokens: Any,
    fetch_exact_item_func: Any,
    max_attempts: int = 4,
    retry_backoff_seconds: float = 1.0,
    remote_workers: int = DEFAULT_REMOTE_WORKERS,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Populate requested public scenes in deterministic item/band/window order."""

    if not 1 <= int(max_attempts) <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be in [1,{MAX_ATTEMPTS}]")
    if not 0 <= float(retry_backoff_seconds) <= MAX_RETRY_BACKOFF_SECONDS:
        raise ValueError("retry_backoff_seconds is outside the supported range")
    if not 1 <= int(remote_workers) <= MAX_REMOTE_WORKERS:
        raise ValueError(f"remote_workers must be in [1,{MAX_REMOTE_WORKERS}]")
    root = _safe_cache(cache)
    root.mkdir(parents=True, exist_ok=True)
    all_specs = _validate_scope_specs(scope, all_specs)
    allowed = {str(spec.entry.scene_id): spec for spec in all_specs}
    requested = _ordered_specs(requested_specs)
    if any(str(spec.entry.scene_id) not in allowed for spec in requested):
        raise ValueError("cache request includes a scene outside fit+validation")
    if output is not None:
        enforce_combined_cap(root, Path(output).resolve())
    stats = CacheTransferStats()
    started = time.monotonic()
    with exclusive_cache_lock(root):
        # Atomic-write leftovers are never authoritative; committed filenames
        # and content validation are the sole resume source.
        for path in root.glob(".*.tmp-*"):
            if path.is_file():
                path.unlink()
        groups: dict[str, list[Any]] = {}
        for spec in requested:
            if any(
                record_from_existing(root / band_name(spec, band), spec, band) is None
                for band in BAND_NAMES
            ):
                groups.setdefault(str(spec.entry.item_id), []).append(spec)
        _write_json(
            root / PROGRESS_NAME,
            {
                "kind": "dataset-progress",
                "schema": CACHE_SCHEMA,
                "status": "in_progress",
                **_scope_header(scope),
                "requested_scene_count": len(requested),
                "remaining_item_count": len(groups),
                "locked_test_opened": False,
                "target_arrays_opened": False,
            },
        )
        token_manager = tokens() if groups and callable(tokens) else tokens
        item_ids = tuple(sorted(groups))
        executor = ThreadPoolExecutor(max_workers=int(remote_workers))
        iterator = iter(item_ids)
        pending: dict[Future[None], str] = {}
        exhausted = False

        def fill() -> None:
            nonlocal exhausted
            while not exhausted and len(pending) < int(remote_workers) * 2:
                try:
                    item_id = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(
                    _prefetch_item,
                    root,
                    groups[item_id],
                    tokens=token_manager,
                    fetch_exact_item_func=fetch_exact_item_func,
                    max_attempts=int(max_attempts),
                    retry_backoff_seconds=float(retry_backoff_seconds),
                    stats=stats,
                )
                pending[future] = item_id

        completed_items = 0
        fill()
        try:
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                # Sorting simultaneous completions keeps progress documents
                # deterministic; request order inside each item is already
                # fixed as band -> spatially ordered windows.
                for future in sorted(completed, key=lambda value: pending[value]):
                    pending.pop(future)
                    future.result()
                    completed_items += 1
                    fill()
                    _write_json(
                        root / PROGRESS_NAME,
                        {
                            "kind": "dataset-progress",
                            "schema": CACHE_SCHEMA,
                            "status": "in_progress",
                            **_scope_header(scope),
                            "requested_scene_count": len(requested),
                            "remote_workers": int(remote_workers),
                            "items_complete_this_run": completed_items,
                            "remaining_item_count": len(groups) - completed_items,
                            "transfer": stats.document(),
                            "locked_test_opened": False,
                            "target_arrays_opened": False,
                        },
                    )
                    if output is not None:
                        enforce_combined_cap(root, Path(output).resolve())
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        index = publish_manifest(root, scope, all_specs)
        # Partial manifests are valid; only the requested subset is mandatory.
        index = verify_cache(
            root, scope, all_specs, required_specs=requested, require_complete=False
        )
    return {
        "status": "complete" if index.complete else "partial",
        "requested_scenes": len(requested),
        "cached_scenes": int(index.header["cached_scene_count"]),
        "cached_bands": int(index.header["cached_band_count"]),
        "bytes": _persistent_bytes(root),
        "wall_seconds": time.monotonic() - started,
        "cache": str(root),
        "transfer": stats.document(),
        "remote_workers": int(remote_workers),
        "network_access": bool(stats.document()["item_fetches"]),
        "locked_test_opened": False,
        "target_arrays_opened": False,
        "signed_urls_persisted": False,
    }
