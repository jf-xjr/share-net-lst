#!/usr/bin/env python3
"""Build compact target-free 30 m texture sidecars for G246 R2.

The input authority is deliberately narrow: :func:`g246_data.load_splits`
materialises the public fit and validation views and this module consumes only
the returned ``G246Scene`` objects.  There is no source-manifest argument and
the locked-test descriptor is never resolved here.

Each v2 sidecar stores 30 optical texture summaries, 30 m valid coverage, and
strictly target-free B2--B7 means on the 120 m grid as compressed float16.
Exact Landsat item lookup, delivered-grid checks, and deployable optical QA
masking are reused from the frozen Landsat texture builder.  Thermal/target QA
bits never enter either optical array.  Direct mode persists no raw COG.  The
optional two-stage mode may persist only canonical target-free B2--B7 crops in
a separate bounded cache; it never persists a full COG, thermal/target array,
or signed URL.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import warnings

import numpy as np
import rasterio

try:
    from . import build_landsat30_texture_sidecars as texture
    from . import g246_data
    from . import g246_r2_optical_cache as optical_cache
except ImportError:
    import build_landsat30_texture_sidecars as texture
    import g246_data
    import g246_r2_optical_cache as optical_cache


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WORKSPACE / "data/g246_r2_texture_v2"
DEFAULT_RECEIPT = g246_data.DEFAULT_SPLIT_RECEIPT

EXPECTED_FIT_SCENES = 603
EXPECTED_VALIDATION_SCENES = 45
EXPECTED_SCENES = EXPECTED_FIT_SCENES + EXPECTED_VALIDATION_SCENES
MAX_BYTES = 20 * 1024**3
DEFAULT_WORKERS = 8
MAX_WORKERS = 16
DEFAULT_MAX_ATTEMPTS = 4
MAX_ATTEMPTS = 8
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
MAX_RETRY_BACKOFF_SECONDS = 30.0
DEFAULT_INFLIGHT_PER_WORKER = 2

# Keep one implementation of the scientific feature contract.  Assigning the
# functions (rather than wrapping them) also makes accidental drift visible in
# tests and code review.
aggregate_texture = texture.aggregate_texture
optical_valid_mask = texture.optical_valid_mask
fetch_exact_item = texture.fetch_exact_item
read_optical = texture.read_optical
CHANNEL_NAMES = texture.CHANNEL_NAMES
FEATURE_CHANNELS = texture.FEATURE_CHANNELS
TOTAL_CHANNELS = texture.TOTAL_CHANNELS
BAND_NAMES = texture.BAND_NAMES
SR_SCALE = texture.SR_SCALE
SR_OFFSET = texture.SR_OFFSET
OPTICAL_MEAN_NAMES = tuple(f"{band}_mean120" for band in BAND_NAMES)
NORMALIZED_CHANNEL_NAMES = (*CHANNEL_NAMES[:FEATURE_CHANNELS], *OPTICAL_MEAN_NAMES)
STORED_CHANNELS = TOTAL_CHANNELS + len(OPTICAL_MEAN_NAMES)

MANIFEST_SCHEMA = "uhi-cdc-g246-r2-texture-sidecars-v2"
SCENE_SCHEMA = "uhi-cdc-g246-r2-texture-scene-v2"
NORMALIZATION_SCHEMA = "uhi-cdc-g246-r2-texture-normalization-v2"
MANIFEST_NAME = "manifest.jsonl"
PROGRESS_NAME = "progress.jsonl"
NORMALIZATION_NAME = "fit_region_balanced_normalization.json"
LEGACY_MANIFEST_SCHEMA = "uhi-cdc-g246-r2-texture-sidecars-v1"


class TokenManager:
    """Thread-safe SAS token cache with single-flight stale-token refresh.

    A burst of COG failures used to make every worker refresh the token in
    series.  ``refresh_if_stale`` refreshes only when the caller still holds
    the current token; workers arriving after the first refresh reuse it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = texture.builder.sas_token(texture.builder.LANDSAT_COLLECTION)
        self._generation = 0
        self._refresh_count = 0

    def get(self) -> str:
        with self._lock:
            return self._token

    def refresh(self) -> str:
        with self._lock:
            self._token = texture.builder.sas_token(texture.builder.LANDSAT_COLLECTION)
            self._generation += 1
            self._refresh_count += 1
            return self._token

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self._token, self._generation

    def refresh_if_stale(
        self,
        stale_token: str,
        stale_generation: int | None = None,
    ) -> str:
        with self._lock:
            if (
                (stale_generation is not None and self._generation != stale_generation)
                or (stale_generation is None and self._token != stale_token)
            ):
                return self._token
            self._token = texture.builder.sas_token(texture.builder.LANDSAT_COLLECTION)
            self._generation += 1
            self._refresh_count += 1
            return self._token

    @property
    def refresh_count(self) -> int:
        with self._lock:
            return self._refresh_count


class TransferStats:
    """Small thread-safe transfer receipt; it never contains source payloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._remote_errors = 0
        self._retries = 0
        self._max_in_flight = 0

    def record_remote_error(self, *, retrying: bool) -> None:
        with self._lock:
            self._remote_errors += 1
            self._retries += int(retrying)

    def observe_in_flight(self, count: int) -> None:
        with self._lock:
            self._max_in_flight = max(self._max_in_flight, int(count))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "remote_errors": self._remote_errors,
                "retries": self._retries,
                "max_in_flight": self._max_in_flight,
            }


@dataclass(frozen=True)
class SceneSpec:
    entry: g246_data.G246Scene
    metadata: Mapping[str, Any]
    qa_reason30: np.ndarray


@dataclass(frozen=True)
class InputScope:
    splits: g246_data.G246Splits
    entries: tuple[g246_data.G246Scene, ...]
    selected_cities: tuple[str, ...]
    full: bool

    @property
    def fit_entries(self) -> tuple[g246_data.G246Scene, ...]:
        return tuple(entry for entry in self.entries if entry.view_role == "fit")

    @property
    def validation_entries(self) -> tuple[g246_data.G246Scene, ...]:
        return tuple(entry for entry in self.entries if entry.view_role == "validation")


def sha256_file(path: Path) -> str:
    return texture.sha256_file(path)


def _safe_output(path: str | os.PathLike[str]) -> Path:
    return g246_data.reject_forbidden_path(path).resolve()


@contextmanager
def _exclusive_build_lock(output: Path) -> Iterator[Path]:
    """Prevent two future builders from writing one output concurrently.

    The advisory lock lives beside, rather than inside, the dataset so it is
    never included in the sidecar manifest or persistent-byte accounting.  A
    process exit releases the kernel lock even though the small lock file is
    intentionally retained to avoid inode races between successive builders.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.builder.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another texture builder holds the output lock: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "output": str(output),
                    "acquired_unix_seconds": time.time(),
                },
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


@contextmanager
def _exclusive_scene_publish_lock(output: Path, sidecar: Path) -> Iterator[Path]:
    """Serialize publication of one scene across primary and assist processes.

    The reverse assistant intentionally does not take the dataset-wide builder
    lock.  Both processes may therefore finish computing the same frontier
    scene.  A per-scene publish lock prevents two successive ``os.replace``
    calls from making a previously returned manifest record stale.  Lock files
    live beside the output and are retained so unlink/recreate inode races are
    impossible; they are not dataset artifacts or part of byte accounting.
    """

    lock_root = output.parent / f".{output.name}.scene-publish-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(sidecar.name.encode("utf-8")).hexdigest()
    lock_path = lock_root / f"{identity}.lock"
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _validate_public_entries(
    fit: Sequence[g246_data.G246Scene],
    validation: Sequence[g246_data.G246Scene],
) -> None:
    entries = tuple(fit) + tuple(validation)
    if not entries:
        raise ValueError("G246 fit+validation scope is empty")
    if any(entry.view_role != "fit" for entry in fit):
        raise ValueError("fit view returned a non-fit scene")
    if any(entry.view_role != "validation" for entry in validation):
        raise ValueError("validation view returned a non-validation scene")
    identities = [entry.scene_id for entry in entries]
    if len(identities) != len(set(identities)):
        raise ValueError("G246 fit+validation contains duplicate scene identities")
    # Nearby city windows may legitimately come from the same delivered item;
    # city/year/item (``scene_id``), not item_id alone, is the sidecar identity.
    # ``g246_data.load_splits`` has already established that fit and validation
    # do not share an item.
    if set(entry.region for entry in fit) != set(g246_data.MACRO_REGIONS):
        raise ValueError("G246 fit scope must contain US, China, and Europe")


def resolve_input_scope(
    receipt: str | os.PathLike[str] = DEFAULT_RECEIPT,
    cities: Sequence[str] | None = None,
) -> InputScope:
    """Resolve only the entries returned by the guarded G246 split loader."""

    # This is the sole input resolution call in this module.  In particular,
    # do not parse ``splits.source_manifest`` or any receipt descriptor here.
    splits = g246_data.load_splits(receipt, role="fit+validation")
    _validate_public_entries(splits.fit, splits.validation)
    all_entries = tuple(splits.fit) + tuple(splits.validation)

    selected = tuple(dict.fromkeys(str(city).strip() for city in (cities or ()) if str(city).strip()))
    if cities is not None and len(selected) != len(cities):
        raise ValueError("--cities must contain non-empty unique city names")
    if selected:
        wanted = set(selected)
        entries = tuple(entry for entry in all_entries if entry.city in wanted)
        missing = wanted.difference(entry.city for entry in entries)
        if missing:
            raise ValueError(f"requested smoke cities are outside G246 fit+validation: {sorted(missing)}")
        return InputScope(splits, entries, selected, False)

    counts = (len(splits.fit), len(splits.validation), len(all_entries))
    expected = (EXPECTED_FIT_SCENES, EXPECTED_VALIDATION_SCENES, EXPECTED_SCENES)
    if counts != expected:
        raise ValueError(
            "full G246 R2 texture build requires exactly "
            f"{expected[0]} fit + {expected[1]} validation = {expected[2]} scenes; got "
            f"{counts[0]} + {counts[1]} = {counts[2]}"
        )
    return InputScope(splits, all_entries, (), True)


def sidecar_name(entry: g246_data.G246Scene) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", entry.city).strip("._-") or "city"
    identity = hashlib.sha256(entry.scene_id.encode("utf-8")).hexdigest()[:12]
    return f"{entry.view_role}_{slug}_{int(entry.year)}_{identity}_optical_v2.npz"


def aggregate_optical_mean(optical30: np.ndarray, valid30: np.ndarray) -> np.ndarray:
    """Aggregate fresh B2--B7 reflectance with optical-only QA support.

    ``valid30`` must already be produced by :func:`optical_valid_mask`, whose
    reject bits are exactly 0--5 and 9.  Bits 6--8 encode thermal/target state
    and are intentionally absent.  A 120 m cell with no optical support is
    stored as zero and identified by ``texture31[-1] == 0``.
    """

    optical = np.asarray(optical30, dtype=np.float32)
    valid = np.asarray(valid30, dtype=bool)
    if optical.shape != (6, 640, 640) or valid.shape != (640, 640):
        raise ValueError("optical mean aggregation requires [6,640,640] and [640,640]")
    if not np.all(np.isfinite(optical)):
        raise ValueError("fresh B2--B7 reflectance must be finite")
    blocks = optical.reshape(6, 160, 4, 160, 4)
    valid_blocks = valid.reshape(160, 4, 160, 4)
    counts = valid_blocks.sum((1, 3), dtype=np.uint8)
    totals = np.where(valid_blocks[None], blocks, 0.0).sum((2, 4), dtype=np.float64)
    means = np.divide(
        totals,
        counts[None],
        out=np.zeros((6, 160, 160), dtype=np.float64),
        where=counts[None] > 0,
    ).astype(np.float16)
    if means.shape != (6, 160, 160) or not np.all(np.isfinite(means)):
        raise ValueError("invalid target-free B2--B7 120 m means")
    if np.any(means[:, counts == 0] != 0):
        raise ValueError("unsupported optical means must remain zero")
    return means


def load_scene_spec(entry: g246_data.G246Scene) -> SceneSpec:
    path = g246_data.reject_forbidden_path(entry.file)
    payload = path.read_bytes()
    if g246_data.sha256_bytes(payload) != entry.sha256:
        raise ValueError(f"G246 scene SHA-256 mismatch: {entry.scene_id}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "qa_reason30" not in archive.files or "metadata" not in archive.files:
            raise ValueError(f"G246 scene lacks qa_reason30 or metadata: {entry.scene_id}")
        reason = np.asarray(archive["qa_reason30"], dtype=np.uint16)
        try:
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"G246 scene metadata is malformed: {entry.scene_id}") from exc
    if reason.shape != (640, 640):
        raise ValueError(f"G246 qa_reason30 geometry mismatch: {entry.scene_id}")
    expected = {
        "city": entry.city,
        "year": int(entry.year),
        "item_id": entry.item_id,
        "datetime": entry.datetime,
        "shape30": [640, 640],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"G246 scene metadata {key} mismatch: {entry.scene_id}")
    crs = metadata.get("canonical_grid_crs")
    if not isinstance(crs, str) or not crs or metadata.get("crs") != crs:
        raise ValueError(f"G246 canonical CRS is invalid: {entry.scene_id}")
    transform = np.asarray(metadata.get("transform30"), dtype=np.float64)
    if transform.shape != (9,) or not np.all(np.isfinite(transform)):
        raise ValueError(f"G246 transform30 is invalid: {entry.scene_id}")
    signatures = metadata.get("asset_grid_signatures")
    if not isinstance(signatures, Mapping) or any(
        not isinstance(signatures.get(band), Mapping) for band in BAND_NAMES
    ):
        raise ValueError(f"G246 optical grid signatures are incomplete: {entry.scene_id}")
    return SceneSpec(entry, metadata, reason)


def _metadata_document(
    spec: SceneSpec,
    item: Mapping[str, Any],
    features: np.ndarray,
    optical_means: np.ndarray,
    *,
    cache_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = np.asarray(features[-1], dtype=np.float32)
    document = {
        "schema": SCENE_SCHEMA,
        "view_role": spec.entry.view_role,
        "region": spec.entry.region,
        "input_role": spec.entry.input_role,
        "city": spec.entry.city,
        "target_year": int(spec.entry.year),
        "scene_id": spec.entry.scene_id,
        "source_scene_sha256": spec.entry.sha256,
        "item_id": item["id"],
        "datetime": item["properties"]["datetime"],
        "canonical_grid_crs": spec.metadata["canonical_grid_crs"],
        "transform30": spec.metadata["transform30"],
        "shape30": [640, 640],
        "shape120": [160, 160],
        "texture_channels": list(CHANNEL_NAMES),
        "optical_mean_channels": list(OPTICAL_MEAN_NAMES),
        "arrays": {
            "texture31": {"shape": [TOTAL_CHANNELS, 160, 160], "dtype": "float16"},
            "optical_mean6": {"shape": [6, 160, 160], "dtype": "float16"},
        },
        "dtype": "float16",
        "coverage_mean": float(coverage.mean()),
        "coverage_nonzero_cells": int(np.count_nonzero(coverage > 0)),
        "optical_mask_source": (
            "stored qa_reason30 bits 0,1,2,3,4,5,9; "
            "thermal/target bits 6,7,8 ignored"
        ),
        "optical_mean_definition": (
            "fresh exact-item B2-B7 30m reflectance, optical-QA-supported arithmetic "
            "mean in each aligned 4x4 cell; zero when valid_coverage30=0"
        ),
        "stored_optical_used": False,
        "stac_access": (
            "offline canonical optical cache; no STAC access during materialization"
            if cache_provenance is not None
            else "exact item metadata only; no page persisted"
        ),
        "asset_access": (
            "six local canonical uint16 cache bands; no COG access during materialization"
            if cache_provenance is not None
            else "six optical COG range windows only; no raw asset persisted"
        ),
        "raw_cache_used": cache_provenance is not None,
        "locked_test_opened": False,
    }
    if cache_provenance is not None:
        document["raw_cache"] = dict(cache_provenance)
    return document


def write_sidecar(
    path: Path,
    features: np.ndarray,
    optical_means: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            texture31=np.asarray(features, dtype=np.float16),
            optical_mean6=np.asarray(optical_means, dtype=np.float16),
            channel_names=np.asarray(CHANNEL_NAMES),
            optical_channel_names=np.asarray(OPTICAL_MEAN_NAMES),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def manifest_record(
    path: Path,
    spec: SceneSpec,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "scene",
        "status": "complete",
        "view_role": spec.entry.view_role,
        "region": spec.entry.region,
        "city": spec.entry.city,
        "target_year": int(spec.entry.year),
        "scene_id": spec.entry.scene_id,
        "source_scene_sha256": spec.entry.sha256,
        "item_id": metadata["item_id"],
        "datetime": metadata["datetime"],
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": [TOTAL_CHANNELS, 160, 160],
        "optical_mean_shape": [6, 160, 160],
        "arrays": ["texture31", "optical_mean6"],
        "dtype": "float16",
        "coverage_mean": float(metadata["coverage_mean"]),
        "locked_test_opened": False,
    }


def record_from_existing(path: Path, spec: SceneSpec) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "texture31", "optical_mean6", "channel_names",
                "optical_channel_names", "metadata",
            }:
                return None
            features = np.asarray(archive["texture31"])
            optical_means = np.asarray(archive["optical_mean6"])
            names = tuple(np.asarray(archive["channel_names"]).astype(str).tolist())
            optical_names = tuple(
                np.asarray(archive["optical_channel_names"]).astype(str).tolist()
            )
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
        coverage = np.asarray(features[-1], dtype=np.float32)
        coverage_mean = float(coverage.mean())
        coverage_nonzero = int(np.count_nonzero(coverage > 0))
        expected = {
            "schema": SCENE_SCHEMA,
            "view_role": spec.entry.view_role,
            "region": spec.entry.region,
            "city": spec.entry.city,
            "target_year": int(spec.entry.year),
            "scene_id": spec.entry.scene_id,
            "source_scene_sha256": spec.entry.sha256,
            "item_id": spec.entry.item_id,
            "datetime": spec.entry.datetime,
            "canonical_grid_crs": spec.metadata["canonical_grid_crs"],
            "transform30": spec.metadata["transform30"],
            "shape30": [640, 640],
            "shape120": [160, 160],
            "dtype": "float16",
            "coverage_nonzero_cells": coverage_nonzero,
            "stored_optical_used": False,
            "locked_test_opened": False,
        }
        if (
            features.shape != (TOTAL_CHANNELS, 160, 160)
            or optical_means.shape != (6, 160, 160)
            or features.dtype != np.float16
            or optical_means.dtype != np.float16
            or names != CHANNEL_NAMES
            or optical_names != OPTICAL_MEAN_NAMES
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(optical_means))
            or np.any(coverage < 0)
            or np.any(coverage > 1)
            or np.any(features[:-1, coverage == 0] != 0)
            or np.any(optical_means[:, coverage == 0] != 0)
            or any(metadata.get(key) != value for key, value in expected.items())
            or metadata.get("texture_channels") != list(CHANNEL_NAMES)
            or metadata.get("optical_mean_channels") != list(OPTICAL_MEAN_NAMES)
            or not math.isclose(
                float(metadata.get("coverage_mean", math.nan)),
                coverage_mean,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            return None
        return manifest_record(path, spec, metadata)
    except (KeyError, ValueError, OSError, json.JSONDecodeError, IndexError):
        return None


def _publish_sidecar(
    output: Path,
    spec: SceneSpec,
    features: np.ndarray,
    optical_means: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish once, or atomically adopt a concurrent writer's valid result."""

    path = output / sidecar_name(spec.entry)
    with _exclusive_scene_publish_lock(output, path):
        existing = record_from_existing(path, spec)
        if existing is not None:
            return existing
        write_sidecar(path, features, optical_means, metadata)
        return manifest_record(path, spec, metadata)


def _token_snapshot(tokens: Any) -> tuple[str, int | None]:
    snapshot = getattr(tokens, "snapshot", None)
    if callable(snapshot):
        token, generation = snapshot()
        return str(token), int(generation)
    return str(tokens.get()), None


def _refresh_if_stale(
    tokens: Any,
    stale_token: str,
    stale_generation: int | None,
) -> tuple[str, int | None]:
    refresh_if_stale = getattr(tokens, "refresh_if_stale", None)
    if callable(refresh_if_stale):
        token = refresh_if_stale(stale_token, stale_generation)
        return _token_snapshot(tokens) if stale_generation is not None else (str(token), None)
    # Compatibility for small test doubles and old external wrappers.  The
    # production TokenManager always takes the single-flight path above.
    return str(tokens.refresh()), None


def _build_sidecar_from_raw(
    spec: SceneSpec,
    output: Path,
    raw: np.ndarray,
    *,
    item: Mapping[str, Any] | None = None,
    cache_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialise one texture sidecar from an in-memory canonical raw stack.

    This is the common numerical path for direct COG access and strict local
    cache replay.  Keeping the QA mask, scaling and aggregators here makes the
    two modes bitwise comparable.
    """

    raw_array = np.asarray(raw)
    if raw_array.shape != (6, 640, 640) or raw_array.dtype != np.uint16:
        raise ValueError(
            f"canonical optical raw stack is not uint16 [6,640,640]: "
            f"{spec.entry.scene_id}"
        )
    valid = optical_valid_mask(spec.qa_reason30, raw_array)
    optical = raw_array.astype(np.float32) * SR_SCALE + SR_OFFSET
    features = aggregate_texture(optical, valid)
    optical_means = aggregate_optical_mean(optical, valid)
    expected_coverage = valid.reshape(160, 4, 160, 4).mean((1, 3), dtype=np.float32)
    if not np.array_equal(features[-1].astype(np.float32), expected_coverage):
        raise ValueError("texture and optical-mean QA coverage disagree")
    identity = item or {
        "id": spec.entry.item_id,
        "properties": {"datetime": spec.entry.datetime},
    }
    metadata = _metadata_document(
        spec,
        identity,
        features,
        optical_means,
        cache_provenance=cache_provenance,
    )
    return _publish_sidecar(output, spec, features, optical_means, metadata)


def build_one(
    spec: SceneSpec,
    output: Path,
    tokens: TokenManager,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    stats: TransferStats | None = None,
) -> dict[str, Any]:
    path = output / sidecar_name(spec.entry)
    existing = record_from_existing(path, spec)
    if existing is not None:
        return existing
    if not 1 <= int(max_attempts) <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be in [1,{MAX_ATTEMPTS}]")
    if not 0.0 <= float(retry_backoff_seconds) <= MAX_RETRY_BACKOFF_SECONDS:
        raise ValueError(
            "retry_backoff_seconds must be in "
            f"[0,{MAX_RETRY_BACKOFF_SECONDS:g}]"
        )
    item = fetch_exact_item(spec)
    last_error: Exception | None = None
    raw: np.ndarray | None = None
    token, token_generation = _token_snapshot(tokens)
    for attempt in range(int(max_attempts)):
        try:
            raw = read_optical(spec, item, token)
            break
        except (rasterio.errors.RasterioIOError, texture.builder.RemoteAssetReadError) as error:
            last_error = error
            retrying = attempt + 1 < int(max_attempts)
            if stats is not None:
                stats.record_remote_error(retrying=retrying)
            if not retrying:
                break
            delay = min(
                float(retry_backoff_seconds) * (2.0 ** attempt),
                MAX_RETRY_BACKOFF_SECONDS,
            )
            if delay > 0:
                time.sleep(delay)
            token, token_generation = _refresh_if_stale(
                tokens,
                token,
                token_generation,
            )
    if raw is None:
        assert last_error is not None
        raise last_error
    return _build_sidecar_from_raw(spec, output, raw, item=item)


def build_one_cached(
    spec: SceneSpec,
    output: Path,
    index: optical_cache.CacheIndex,
    *,
    cache_manifest_sha256: str,
) -> dict[str, Any]:
    """Build one sidecar through the verified, strictly local cache path."""

    path = output / sidecar_name(spec.entry)
    existing = record_from_existing(path, spec)
    if existing is not None:
        return existing
    rows = [index.record_for(spec, band) for band in BAND_NAMES]
    provenance = {
        "schema": optical_cache.CACHE_SCHEMA,
        "manifest_sha256_at_materialization": cache_manifest_sha256,
        "cache_status": str(index.header["status"]),
        "band_sha256": {
            str(row["band"]): str(row["sha256"]) for row in rows
        },
        "network_access_during_materialization": False,
        "signed_urls_persisted": False,
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }
    raw = optical_cache.load_scene(index, spec)
    return _build_sidecar_from_raw(
        spec, output, raw, cache_provenance=provenance
    )


def _bounded_results(
    executor: ThreadPoolExecutor,
    specs: Sequence[SceneSpec],
    submit: Callable[[SceneSpec], Future[dict[str, Any]]],
    *,
    max_in_flight: int,
    stats: TransferStats | None = None,
) -> Iterator[tuple[SceneSpec, dict[str, Any]]]:
    """Yield completed scenes while keeping only a bounded future queue.

    Limiting queued work makes Ctrl-C/restart responsive: at most
    ``max_in_flight`` scenes have been handed to the executor, while completed
    sidecars remain independently atomic and discoverable on the next scan.
    """

    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    iterator = iter(specs)
    pending: dict[Future[dict[str, Any]], SceneSpec] = {}
    exhausted = False

    def fill() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < max_in_flight:
            try:
                spec = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[submit(spec)] = spec
        if stats is not None:
            stats.observe_in_flight(len(pending))

    fill()
    while pending:
        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            spec = pending.pop(future)
            record = future.result()
            # Refill before manifest hashing/compression in the consumer so
            # network workers do not idle on local bookkeeping.
            fill()
            yield spec, record


def _moment_component(accumulator: texture.WeightedMoments) -> dict[str, Any]:
    """Represent one coverage-weighted scene distribution by two moments."""

    document = accumulator.document()
    return {
        **document,
        "second_moment": accumulator.total2 / accumulator.weight,
    }


def _equal_mixture(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine component distributions with exactly equal mixture weights."""

    if not components:
        raise ValueError("normalization mixture has no components")
    count = len(components)
    mean = math.fsum(float(value["mean"]) for value in components) / count
    second = math.fsum(float(value["second_moment"]) for value in components) / count
    variance = max(second - mean * mean, 0.0)
    return {
        "mean": mean,
        "std": max(float(math.sqrt(variance)), 1e-8),
        "second_moment": second,
        "min": min(float(value["min"]) for value in components),
        "max": max(float(value["max"]) for value in components),
        # Diagnostics only: neither quantity below is used as a mixture weight.
        "weight_sum": math.fsum(float(value["weight_sum"]) for value in components),
        "valid_cell_count": sum(int(value["valid_cell_count"]) for value in components),
        "component_count": count,
    }


def _hierarchical_balanced_channel(
    moments: Mapping[
        str,
        Mapping[str, Sequence[tuple[str, texture.WeightedMoments]]],
    ],
) -> dict[str, Any]:
    """Mix scene -> city -> region distributions without coverage dominance."""

    regions = tuple(g246_data.MACRO_REGIONS)
    region_components: list[dict[str, Any]] = []
    region_documents: dict[str, dict[str, Any]] = {}
    for region in regions:
        cities = moments.get(region, {})
        if not cities:
            raise ValueError(f"normalization region has no fit cities: {region}")
        city_components: list[dict[str, Any]] = []
        city_documents: dict[str, dict[str, Any]] = {}
        for city in sorted(cities):
            scene_pairs = sorted(cities[city], key=lambda pair: pair[0])
            if not scene_pairs:
                raise ValueError(f"normalization city has no fit scenes: {city}")
            scene_components = [
                _moment_component(accumulator) for _, accumulator in scene_pairs
            ]
            scene_weight = 1.0 / len(scene_components)
            scene_documents = {
                scene_id: {**component, "normalization_weight": scene_weight}
                for (scene_id, _), component in zip(scene_pairs, scene_components)
            }
            city_component = _equal_mixture(scene_components)
            city_components.append(city_component)
            city_documents[city] = {
                **city_component,
                "normalization_weight": 1.0 / len(cities),
                "scene_count": len(scene_pairs),
                "scenes": scene_documents,
            }
        region_component = _equal_mixture(city_components)
        region_components.append(region_component)
        region_documents[region] = {
            **region_component,
            "normalization_weight": 1.0 / len(regions),
            "city_count": len(cities),
            "cities": city_documents,
        }
    global_component = _equal_mixture(region_components)
    return {**global_component, "regions": region_documents}


def build_normalization(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    receipt_sha256: str,
    fit_view_sha256: str,
) -> dict[str, Any]:
    """Compute fit-only scene/city/region balanced texture statistics.

    Coverage weights pixels only inside one scene.  Each scene then receives
    equal weight inside its city, each city equal weight inside its region, and
    US/China/Europe each one third globally.  Thus neither cloud-free area nor
    the number of valid cells can make one city dominate another.
    """

    fit_rows = [row for row in records if row.get("view_role") == "fit"]
    validation_rows = [row for row in records if row.get("view_role") == "validation"]
    regions = tuple(g246_data.MACRO_REGIONS)
    actual_regions = {str(row.get("region")) for row in fit_rows}
    if actual_regions != set(regions):
        raise ValueError("fit-only texture normalization requires all three regions")
    accumulators: dict[
        str,
        dict[str, list[tuple[str, dict[str, texture.WeightedMoments]]]],
    ] = {region: {} for region in regions}
    city_regions: dict[str, str] = {}
    scene_ids: set[str] = set()
    for row in fit_rows:
        region = str(row["region"])
        city = str(row.get("city", "")).strip()
        scene_id = str(row.get("scene_id", row.get("file", ""))).strip()
        if not city or not scene_id:
            raise ValueError("fit normalization row lacks city or scene identity")
        if scene_id in scene_ids:
            raise ValueError(f"duplicate fit normalization scene: {scene_id}")
        scene_ids.add(scene_id)
        previous_region = city_regions.setdefault(city, region)
        if previous_region != region:
            raise ValueError(f"fit city occurs in multiple regions: {city}")
        path = output / str(row["file"])
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["texture31"], dtype=np.float32)
            optical_means = np.asarray(archive["optical_mean6"], dtype=np.float32)
        if (
            values.shape != (TOTAL_CHANNELS, 160, 160)
            or optical_means.shape != (6, 160, 160)
        ):
            raise ValueError(f"normalization sidecar shape mismatch: {path.name}")
        coverage = values[-1]
        normalized_values = np.concatenate((values[:FEATURE_CHANNELS], optical_means))
        scene_moments = {
            name: texture.WeightedMoments()
            for name in NORMALIZED_CHANNEL_NAMES
        }
        for index, name in enumerate(NORMALIZED_CHANNEL_NAMES):
            scene_moments[name].update(normalized_values[index], coverage)
        accumulators[region].setdefault(city, []).append((scene_id, scene_moments))
    channels = {
        name: _hierarchical_balanced_channel(
            {
                region: {
                    city: [
                        (scene_id, scene_moments[name])
                        for scene_id, scene_moments in scenes
                    ]
                    for city, scenes in accumulators[region].items()
                }
                for region in regions
            }
        )
        for name in NORMALIZED_CHANNEL_NAMES
    }
    return {
        "schema": NORMALIZATION_SCHEMA,
        "scope": {
            "view_role": "fit",
            "validation_included": False,
            "locked_test_opened": False,
        },
        "receipt_sha256": receipt_sha256,
        "fit_view_sha256": fit_view_sha256,
        "fit_city_count": len({str(row["city"]) for row in fit_rows}),
        "fit_scene_count": len(fit_rows),
        "validation_scene_count_excluded": len(validation_rows),
        "regions": list(regions),
        "region_scene_counts": {
            region: sum(str(row["region"]) == region for row in fit_rows)
            for region in regions
        },
        "region_city_counts": {
            region: len(accumulators[region]) for region in regions
        },
        "city_scene_counts": {
            city: len(scenes)
            for region in regions
            for city, scenes in sorted(accumulators[region].items())
        },
        "value_arrays": {
            "texture31": list(CHANNEL_NAMES[:FEATURE_CHANNELS]),
            "optical_mean6": list(OPTICAL_MEAN_NAMES),
        },
        "coverage_channel": "valid_coverage30",
        "missing_rule": "coverage=0 => feature values remain exactly 0 after normalization",
        "weighting": (
            "valid_coverage30 weights cells only within each scene; scenes are equal "
            "within city; cities are equal within region; US, China, and Europe each "
            "have global weight 1/3"
        ),
        "channels": channels,
        "normalized_channel_count": len(NORMALIZED_CHANNEL_NAMES),
        "excluded": ["valid_coverage30"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_manifest(
    path: Path,
    header: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    lines = [json.dumps(header, sort_keys=True)]
    lines.extend(json.dumps(row, sort_keys=True) for row in records)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ordered_records(
    records: Sequence[Mapping[str, Any]],
    entries: Sequence[g246_data.G246Scene],
) -> list[dict[str, Any]]:
    order = {entry.scene_id: index for index, entry in enumerate(entries)}
    if len(order) != len(entries):
        raise ValueError("cannot order duplicate G246 scenes")
    result = [dict(row) for row in records]
    result.sort(key=lambda row: order[str(row["scene_id"])])
    return result


def _scope_header(scope: InputScope) -> dict[str, Any]:
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
        "selected_cities": list(scope.selected_cities),
        "locked_test_opened": False,
    }


def _persistent_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def _enforce_cap(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    include_inflight_directory: bool = True,
) -> None:
    sidecar_bytes = sum(int(row["bytes"]) for row in records)
    # During concurrent writes, hidden atomic temporary files may disappear
    # between directory enumeration and stat.  Completed record bytes are the
    # race-free online gate; the whole directory is measured after workers end.
    directory_exceeds = include_inflight_directory and _persistent_bytes(output) > MAX_BYTES
    if sidecar_bytes > MAX_BYTES or directory_exceeds:
        raise RuntimeError("G246 R2 texture output exceeds the 20 GiB persistent cap")


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("texture manifest is empty")
    try:
        values = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError("texture manifest JSONL is malformed") from exc
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("texture manifest lines must be JSON objects")
    return values[0], values[1:]


def _reject_legacy_output_mix(output: Path) -> None:
    """Keep irrecoverable v1 sidecars separate from the target-free v2 set."""

    manifest = output / MANIFEST_NAME
    if manifest.is_file():
        header, _ = _read_manifest(manifest)
        if header.get("schema") == LEGACY_MANIFEST_SCHEMA:
            raise ValueError(
                "v1 texture sidecars lack fresh B2-B7 means; use the v2 output directory"
            )
    legacy = sorted(path.name for path in output.glob("*_texture31.npz"))
    if legacy:
        raise ValueError(
            "legacy texture31-only sidecars cannot be upgraded offline; "
            "use a separate v2 output directory"
        )


def _complete_header(
    scope: InputScope,
    records: Sequence[Mapping[str, Any]],
    normalization_sha256: str,
) -> dict[str, Any]:
    return {
        "kind": "dataset",
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "scene_count": len(records),
        **_scope_header(scope),
        "normalization_file": NORMALIZATION_NAME,
        "normalization_sha256": normalization_sha256,
        "normalization_scope": "fit-only equal-scene/equal-city/equal-region",
        "stored_channel_count": STORED_CHANNELS,
        "arrays": {
            "texture31": {
                "shape": [TOTAL_CHANNELS, 160, 160],
                "dtype": "float16",
                "channels": list(CHANNEL_NAMES),
            },
            "optical_mean6": {
                "shape": [6, 160, 160],
                "dtype": "float16",
                "channels": list(OPTICAL_MEAN_NAMES),
                "qa_scope": "optical bits 0-5,9 only; thermal/target bits 6-8 ignored",
            },
        },
        "sidecar_bytes": sum(int(row["bytes"]) for row in records),
        "persistent_cap_bytes": MAX_BYTES,
    }


def verify_dataset(
    output: str | os.PathLike[str] = DEFAULT_OUTPUT,
    receipt: str | os.PathLike[str] = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Perform a complete local-only integrity check of the published dataset."""

    output_path = _safe_output(output)
    scope = resolve_input_scope(receipt)
    manifest_path = output_path / MANIFEST_NAME
    header, rows = _read_manifest(manifest_path)
    if header.get("kind") != "dataset" or header.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported G246 R2 texture manifest")
    if header.get("status") != "complete" or header.get("locked_test_opened") is not False:
        raise ValueError("G246 R2 texture manifest is not safely complete")
    if header.get("scene_count") != EXPECTED_SCENES or len(rows) != EXPECTED_SCENES:
        raise ValueError("G246 R2 texture manifest scene count mismatch")
    for key, value in _scope_header(scope).items():
        if header.get(key) != value:
            raise ValueError(f"G246 R2 texture manifest {key} mismatch")

    entries = tuple(scope.entries)
    entry_by_id = {entry.scene_id: entry for entry in entries}
    if len(entry_by_id) != len(entries):
        raise ValueError("duplicate allowed G246 identities")
    row_by_id: dict[str, Mapping[str, Any]] = {}
    expected_files: set[str] = {MANIFEST_NAME, NORMALIZATION_NAME}
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if scene_id in row_by_id or scene_id not in entry_by_id:
            raise ValueError(f"unexpected or duplicate sidecar record: {scene_id}")
        row_by_id[scene_id] = row
        file_name = row.get("file")
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError(f"unsafe sidecar filename: {file_name!r}")
        expected_files.add(file_name)
        spec = load_scene_spec(entry_by_id[scene_id])
        canonical = record_from_existing(output_path / file_name, spec)
        if canonical is None or canonical != row:
            raise ValueError(f"sidecar content/manifest mismatch: {scene_id}")
        canonical_rows.append(canonical)
    if set(row_by_id) != set(entry_by_id):
        raise ValueError("G246 R2 texture manifest omits allowed scenes")

    actual_files = {path.name for path in output_path.iterdir() if path.is_file()}
    if any(not path.is_file() for path in output_path.iterdir()):
        raise ValueError("texture output must not contain subdirectories or special files")
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise ValueError(f"texture output file set mismatch; extra={extra}, missing={missing}")
    normalization_path = output_path / NORMALIZATION_NAME
    if sha256_file(normalization_path) != header.get("normalization_sha256"):
        raise ValueError("fit normalization SHA-256 mismatch")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    recomputed = build_normalization(
        output_path,
        canonical_rows,
        receipt_sha256=scope.splits.receipt_sha256,
        fit_view_sha256=scope.splits.fit_view_sha256,
    )
    if normalization != recomputed:
        raise ValueError("fit-only region-balanced normalization does not recompute")
    expected_header = _complete_header(
        scope, _ordered_records(canonical_rows, entries), sha256_file(normalization_path)
    )
    if header != expected_header:
        raise ValueError("G246 R2 texture dataset header is not canonical")
    _enforce_cap(output_path, canonical_rows)
    return {
        "status": "verified",
        "scenes": len(canonical_rows),
        "bytes": _persistent_bytes(output_path),
        "output": str(output_path),
        "network_access": False,
        "locked_test_opened": False,
    }


def _build_locked(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    warnings.filterwarnings(
        "ignore", message="All-NaN slice encountered", category=RuntimeWarning
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    workers = int(args.workers)
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    max_attempts = int(getattr(args, "max_attempts", DEFAULT_MAX_ATTEMPTS))
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be in [1,{MAX_ATTEMPTS}]")
    retry_backoff_seconds = float(
        getattr(args, "retry_backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS)
    )
    if not 0.0 <= retry_backoff_seconds <= MAX_RETRY_BACKOFF_SECONDS:
        raise ValueError(
            "retry_backoff_seconds must be in "
            f"[0,{MAX_RETRY_BACKOFF_SECONDS:g}]"
        )
    requested_in_flight = int(getattr(args, "max_in_flight", 0))
    max_in_flight = (
        workers * DEFAULT_INFLIGHT_PER_WORKER
        if requested_in_flight == 0
        else requested_in_flight
    )
    if not workers <= max_in_flight <= MAX_WORKERS * DEFAULT_INFLIGHT_PER_WORKER:
        raise ValueError(
            "max_in_flight must be 0 (auto) or in "
            f"[{workers},{MAX_WORKERS * DEFAULT_INFLIGHT_PER_WORKER}]"
        )
    _reject_legacy_output_mix(output)
    scope = resolve_input_scope(args.receipt, args.cities)
    entries = tuple(scope.entries)
    specs = [load_scene_spec(entry) for entry in entries]
    started = time.monotonic()

    records: list[dict[str, Any]] = []
    missing: list[SceneSpec] = []
    for spec in specs:
        record = record_from_existing(output / sidecar_name(spec.entry), spec)
        if record is None:
            missing.append(spec)
        else:
            records.append(record)
    progress_header = {
        "kind": "dataset-progress",
        "schema": MANIFEST_SCHEMA,
        "status": "in_progress",
        "scene_count_complete": len(records),
        "scene_count_expected": len(entries),
        "workers": workers,
        "max_in_flight": max_in_flight,
        "max_attempts": max_attempts,
        "retry_backoff_seconds": retry_backoff_seconds,
        **_scope_header(scope),
    }
    _write_manifest(
        output / PROGRESS_NAME,
        progress_header,
        _ordered_records(records, entries),
    )

    # Do not request a SAS token when a complete local checkpoint is resumed.
    stats = TransferStats()
    initial_complete = len(records)
    if missing:
        tokens = TokenManager()
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            def submit(spec: SceneSpec) -> Future[dict[str, Any]]:
                return executor.submit(
                    build_one,
                    spec,
                    output,
                    tokens,
                    max_attempts=max_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    stats=stats,
                )

            for spec, record in _bounded_results(
                executor,
                missing,
                submit,
                max_in_flight=max_in_flight,
                stats=stats,
            ):
                records.append(record)
                ordered = _ordered_records(records, entries)
                _enforce_cap(output, ordered, include_inflight_directory=False)
                progress_header["scene_count_complete"] = len(ordered)
                progress_header["transfer"] = stats.snapshot()
                _write_manifest(output / PROGRESS_NAME, progress_header, ordered)
                completed_now = len(ordered) - initial_complete
                elapsed = max(time.monotonic() - started, 1e-9)
                scenes_per_hour = completed_now * 3600.0 / elapsed
                remaining = len(entries) - len(ordered)
                eta_hours = remaining / scenes_per_hour if scenes_per_hour > 0 else math.inf
                print(
                    f"complete {len(ordered)}/{len(entries)} "
                    f"{spec.entry.scene_id} {record['bytes']} bytes "
                    f"rate={scenes_per_hour:.1f}/h eta={eta_hours:.2f}h "
                    f"retries={stats.snapshot()['retries']}",
                    flush=True,
                )
        except BaseException:
            # A failure/Ctrl-C must not start the queued second wave.  Running
            # scene writes still finish atomically and are discovered by the
            # next content/SHA scan even if their progress row was not emitted.
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    records = _ordered_records(records, entries)
    if len(records) != len(entries):
        raise RuntimeError("not every selected G246 scene produced a texture sidecar")

    if scope.full:
        normalization = build_normalization(
            output,
            records,
            receipt_sha256=scope.splits.receipt_sha256,
            fit_view_sha256=scope.splits.fit_view_sha256,
        )
        _write_json(output / NORMALIZATION_NAME, normalization)
        normalization_sha = sha256_file(output / NORMALIZATION_NAME)
        header = _complete_header(scope, records, normalization_sha)
        status = "complete"
    else:
        header = {
            "kind": "dataset",
            "schema": MANIFEST_SCHEMA,
            "status": "smoke_complete",
            "scene_count": len(records),
            **_scope_header(scope),
            "normalization_file": None,
            "normalization_sha256": None,
            "stored_channel_count": STORED_CHANNELS,
            "arrays": {
                "texture31": {
                    "shape": [TOTAL_CHANNELS, 160, 160],
                    "dtype": "float16",
                    "channels": list(CHANNEL_NAMES),
                },
                "optical_mean6": {
                    "shape": [6, 160, 160],
                    "dtype": "float16",
                    "channels": list(OPTICAL_MEAN_NAMES),
                    "qa_scope": "optical bits 0-5,9 only; thermal/target bits 6-8 ignored",
                },
            },
            "sidecar_bytes": sum(int(row["bytes"]) for row in records),
            "persistent_cap_bytes": MAX_BYTES,
        }
        status = "smoke_complete"
    _write_manifest(output / MANIFEST_NAME, header, records)
    progress = output / PROGRESS_NAME
    if progress.exists():
        progress.unlink()
    _enforce_cap(output, records)
    result = {
        "status": status,
        "scenes": len(records),
        "resumed_scenes": len(entries) - len(missing),
        "downloaded_scenes": len(missing),
        "bytes": _persistent_bytes(output),
        "wall_seconds": time.monotonic() - started,
        "output": str(output),
        "workers": workers,
        "max_in_flight": max_in_flight,
        "transfer": stats.snapshot(),
        "locked_test_opened": False,
    }
    if missing:
        result["token_refreshes"] = int(getattr(tokens, "refresh_count", 0))
    if scope.full:
        verification = verify_dataset(output, args.receipt)
        result["offline_verification"] = verification["status"]
    return result


def _scope_specs(args: argparse.Namespace) -> tuple[InputScope, list[SceneSpec]]:
    scope = resolve_input_scope(args.receipt, args.cities)
    return scope, [load_scene_spec(entry) for entry in scope.entries]


def _existing_output_records(
    output: Path, specs: Sequence[SceneSpec]
) -> tuple[list[dict[str, Any]], list[SceneSpec]]:
    records: list[dict[str, Any]] = []
    missing: list[SceneSpec] = []
    for spec in specs:
        record = record_from_existing(output / sidecar_name(spec.entry), spec)
        if record is None:
            missing.append(spec)
        else:
            records.append(record)
    return records, missing


def _cache_path(args: argparse.Namespace) -> Path:
    value = getattr(args, "raw_cache", None)
    if value is None:
        raise ValueError("canonical cache mode requires --raw-cache")
    cache = optical_cache._safe_cache(value)
    output = _safe_output(args.output)
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("--raw-cache and --output must be separate non-nested paths")
    return cache


def prefetch_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Populate all requested raw bands without creating texture output."""

    cache = _cache_path(args)
    scope, specs = _scope_specs(args)
    output = _safe_output(args.output)
    _, missing = _existing_output_records(output, specs)
    requested = specs if bool(args.cache_all) else missing
    return optical_cache.populate_cache(
        cache,
        scope,
        specs,
        requested,
        # Passing the class keeps SAS acquisition lazy: a fully cached resume
        # never constructs TokenManager and therefore performs no token call.
        tokens=TokenManager,
        fetch_exact_item_func=fetch_exact_item,
        max_attempts=int(args.max_attempts),
        retry_backoff_seconds=float(args.retry_backoff_seconds),
        remote_workers=int(args.remote_workers),
        output=output if output.exists() else None,
    )


def verify_raw_cache(args: argparse.Namespace) -> dict[str, Any]:
    cache = _cache_path(args)
    scope, specs = _scope_specs(args)
    index = optical_cache.verify_cache(
        cache,
        scope,
        specs,
        require_complete=bool(args.cache_all),
    )
    return {
        "status": "verified_complete" if index.complete else "verified_partial",
        "cached_scenes": int(index.header["cached_scene_count"]),
        "cached_bands": len(index.records),
        "bytes": optical_cache._persistent_bytes(cache),
        "cache": str(cache),
        "network_access": False,
        "locked_test_opened": False,
        "target_arrays_opened": False,
        "signed_urls_persisted": False,
    }


def _build_cached_locked(
    args: argparse.Namespace, output: Path, cache: Path
) -> dict[str, Any]:
    """Complete missing sidecars through an online-prefetch/offline-compute barrier."""

    warnings.filterwarnings(
        "ignore", message="All-NaN slice encountered", category=RuntimeWarning
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    local_workers = int(args.local_workers)
    if not 1 <= local_workers <= 4:
        raise ValueError("local_workers must be in [1,4]")
    _reject_legacy_output_mix(output)
    scope, specs = _scope_specs(args)
    records, missing = _existing_output_records(output, specs)
    cache_result: Mapping[str, Any] | None = None
    if not bool(args.offline_cache):
        requested = specs if bool(args.cache_all) else missing
        cache_result = optical_cache.populate_cache(
            cache,
            scope,
            specs,
            requested,
            tokens=TokenManager,
            fetch_exact_item_func=fetch_exact_item,
            max_attempts=int(args.max_attempts),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
            remote_workers=int(args.remote_workers),
            output=output,
        )
    # This is the strict phase barrier.  Every raw file needed for an output
    # write is SHA/identity checked before the first aggregate starts.
    index = optical_cache.verify_cache(
        cache,
        scope,
        specs,
        required_specs=missing,
        require_complete=bool(args.cache_all),
    )
    optical_cache.enforce_combined_cap(cache, output)
    cache_manifest_sha = optical_cache.sha256_file(cache / optical_cache.MANIFEST_NAME)
    started = time.monotonic()
    progress_header = {
        "kind": "dataset-progress",
        "schema": MANIFEST_SCHEMA,
        "status": "in_progress",
        "scene_count_complete": len(records),
        "scene_count_expected": len(specs),
        "workers": local_workers,
        "materialization": "strict-offline-canonical-optical-cache",
        "raw_cache_manifest_sha256": cache_manifest_sha,
        **_scope_header(scope),
    }
    _write_manifest(
        output / PROGRESS_NAME,
        progress_header,
        _ordered_records(records, scope.entries),
    )

    if missing:
        executor = ThreadPoolExecutor(max_workers=local_workers)
        try:
            def submit(spec: SceneSpec) -> Future[dict[str, Any]]:
                return executor.submit(
                    build_one_cached,
                    spec,
                    output,
                    index,
                    cache_manifest_sha256=cache_manifest_sha,
                )

            for spec, record in _bounded_results(
                executor,
                missing,
                submit,
                max_in_flight=local_workers,
            ):
                records.append(record)
                ordered = _ordered_records(records, scope.entries)
                _enforce_cap(output, ordered, include_inflight_directory=False)
                optical_cache.enforce_combined_cap(cache, output)
                progress_header["scene_count_complete"] = len(ordered)
                _write_manifest(output / PROGRESS_NAME, progress_header, ordered)
                print(
                    f"offline complete {len(ordered)}/{len(specs)} "
                    f"{spec.entry.scene_id} {record['bytes']} bytes",
                    flush=True,
                )
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    records = _ordered_records(records, scope.entries)
    if len(records) != len(specs):
        raise RuntimeError("not every selected scene produced a texture sidecar")

    if scope.full:
        normalization = build_normalization(
            output,
            records,
            receipt_sha256=scope.splits.receipt_sha256,
            fit_view_sha256=scope.splits.fit_view_sha256,
        )
        _write_json(output / NORMALIZATION_NAME, normalization)
        normalization_sha = sha256_file(output / NORMALIZATION_NAME)
        header = _complete_header(scope, records, normalization_sha)
        status = "complete"
    else:
        header = {
            "kind": "dataset",
            "schema": MANIFEST_SCHEMA,
            "status": "smoke_complete",
            "scene_count": len(records),
            **_scope_header(scope),
            "normalization_file": None,
            "normalization_sha256": None,
            "stored_channel_count": STORED_CHANNELS,
            "arrays": {
                "texture31": {
                    "shape": [TOTAL_CHANNELS, 160, 160],
                    "dtype": "float16",
                    "channels": list(CHANNEL_NAMES),
                },
                "optical_mean6": {
                    "shape": [6, 160, 160],
                    "dtype": "float16",
                    "channels": list(OPTICAL_MEAN_NAMES),
                    "qa_scope": "optical bits 0-5,9 only; thermal/target bits 6-8 ignored",
                },
            },
            "sidecar_bytes": sum(int(row["bytes"]) for row in records),
            "persistent_cap_bytes": MAX_BYTES,
        }
        status = "smoke_complete"
    _write_manifest(output / MANIFEST_NAME, header, records)
    progress = output / PROGRESS_NAME
    if progress.exists():
        progress.unlink()
    _enforce_cap(output, records)
    optical_cache.enforce_combined_cap(cache, output)
    result: dict[str, Any] = {
        "status": status,
        "scenes": len(records),
        "resumed_scenes": len(specs) - len(missing),
        "materialized_scenes": len(missing),
        "bytes": _persistent_bytes(output),
        "wall_seconds": time.monotonic() - started,
        "output": str(output),
        "local_workers": local_workers,
        "raw_cache": str(cache),
        "raw_cache_status": str(index.header["status"]),
        "raw_cache_manifest_sha256": cache_manifest_sha,
        "network_access_during_materialization": False,
        "locked_test_opened": False,
    }
    if cache_result is not None:
        result["prefetch"] = dict(cache_result)
    if scope.full:
        verification = verify_dataset(output, args.receipt)
        result["offline_verification"] = verification["status"]
    return result


def build_cached(args: argparse.Namespace) -> dict[str, Any]:
    output = _safe_output(args.output)
    cache = _cache_path(args)
    output.mkdir(parents=True, exist_ok=True)
    optical_cache.enforce_combined_cap(cache, output)
    with _exclusive_build_lock(output):
        return _build_cached_locked(args, output, cache)


def assist_cached_reverse(args: argparse.Namespace) -> dict[str, Any]:
    """Safely assist an active offline build from the opposite end.

    The primary builder owns the dataset progress/manifest lock.  This helper
    deliberately writes *only* independently atomic scene sidecars, in reverse
    order, and never touches progress, normalization, or the final manifest.
    Primary and assistant share a per-scene publication lock: if their work
    fronts meet after both have computed a scene, exactly one publishes and the
    other validates/adopts that immutable result.  When the primary iterator
    later reaches an already assisted scene, :func:`build_one_cached` adopts it
    without recomputation.
    """

    warnings.filterwarnings(
        "ignore", message="All-NaN slice encountered", category=RuntimeWarning
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    output = _safe_output(args.output)
    cache = _cache_path(args)
    output.mkdir(parents=True, exist_ok=True)
    _reject_legacy_output_mix(output)
    scope, specs = _scope_specs(args)
    _, missing = _existing_output_records(output, specs)
    index = optical_cache.verify_cache(
        cache, scope, specs, required_specs=missing, require_complete=False,
    )
    cache_manifest_sha = optical_cache.sha256_file(
        cache / optical_cache.MANIFEST_NAME
    )
    started = time.monotonic()
    assisted = 0
    for spec in reversed(missing):
        existed_before = record_from_existing(
            output / sidecar_name(spec.entry), spec
        ) is not None
        record = build_one_cached(
            spec,
            output,
            index,
            cache_manifest_sha256=cache_manifest_sha,
        )
        if not existed_before:
            assisted += 1
        _enforce_cap(output, (), include_inflight_directory=True)
        optical_cache.enforce_combined_cap(cache, output)
        print(
            f"assist reverse {assisted}/{len(missing)} "
            f"{spec.entry.scene_id} {record['bytes']} bytes",
            flush=True,
        )
    return {
        "status": "assist_complete",
        "direction": "reverse",
        "missing_at_start": len(missing),
        "sidecars_materialized": assisted,
        "wall_seconds": time.monotonic() - started,
        "output": str(output),
        "raw_cache": str(cache),
        "network_access": False,
        "progress_or_manifest_written": False,
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Build under an output-specific inter-process exclusion lock."""

    output = _safe_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_build_lock(output):
        return _build_locked(args, output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"concurrent scene downloads (default {DEFAULT_WORKERS}, maximum {MAX_WORKERS})",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="bounded submitted-scene queue; 0 uses two futures per worker",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"remote COG attempts per scene (default {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="initial exponential COG retry delay",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        help="optional allowed-city subset for a network smoke build",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--verify-only",
        action="store_true",
        help="perform the complete local integrity check without any network call",
    )
    modes.add_argument(
        "--cache-only",
        action="store_true",
        help="prefetch missing canonical raw bands and do not build sidecars",
    )
    modes.add_argument(
        "--offline-cache",
        action="store_true",
        help="strictly materialize missing sidecars from an existing local cache",
    )
    modes.add_argument(
        "--verify-cache-only",
        action="store_true",
        help="verify a partial/complete raw cache without any network call",
    )
    modes.add_argument(
        "--assist-reverse",
        action="store_true",
        help=("assist an active offline build from the reverse end; writes only "
              "atomic sidecars and leaves progress/manifest ownership to the primary"),
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        help="independent canonical uint16 B2-B7 cache directory",
    )
    parser.add_argument(
        "--cache-all",
        action="store_true",
        help="require/prefetch all allowed scenes instead of only missing output scenes",
    )
    parser.add_argument(
        "--local-workers",
        type=int,
        default=1,
        help="offline texture materialization workers (default 1, maximum 4)",
    )
    parser.add_argument(
        "--remote-workers",
        type=int,
        default=optical_cache.DEFAULT_REMOTE_WORKERS,
        help=(
            "bounded item-level cache prefetch workers "
            f"(default {optical_cache.DEFAULT_REMOTE_WORKERS}, "
            f"maximum {optical_cache.MAX_REMOTE_WORKERS})"
        ),
    )
    args = parser.parse_args(argv)
    if args.verify_only and args.cities:
        parser.error("--verify-only checks the complete 648-scene dataset; omit --cities")
    cache_requested = any(
        (
            args.raw_cache is not None,
            args.cache_only,
            args.offline_cache,
            args.verify_cache_only,
            args.assist_reverse,
            args.cache_all,
        )
    )
    if cache_requested and args.raw_cache is None:
        parser.error("cache modes require --raw-cache")
    if args.verify_only and args.raw_cache is not None:
        parser.error("--verify-only is the direct sidecar verifier; omit --raw-cache")
    if not 1 <= int(args.local_workers) <= 4:
        parser.error("--local-workers must be in [1,4]")
    if not 1 <= int(args.remote_workers) <= optical_cache.MAX_REMOTE_WORKERS:
        parser.error(
            f"--remote-workers must be in [1,{optical_cache.MAX_REMOTE_WORKERS}]"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        result = verify_dataset(args.output, args.receipt)
    elif args.verify_cache_only:
        result = verify_raw_cache(args)
    elif args.cache_only:
        result = prefetch_cache(args)
    elif args.assist_reverse:
        result = assist_cached_reverse(args)
    elif args.raw_cache is not None:
        result = build_cached(args)
    else:
        result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
