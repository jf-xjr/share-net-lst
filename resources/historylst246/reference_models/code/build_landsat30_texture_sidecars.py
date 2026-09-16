#!/usr/bin/env python3
"""Build target-free contemporaneous Landsat 30 m texture sidecars.

Only the six optical COG windows are range-read.  QA and thermal assets are
not reopened: the deployable optical mask is reconstructed from the stored
``qa_reason30`` bits while explicitly ignoring target-derived reason bits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote
import warnings

from affine import Affine
import numpy as np
import rasterio

try:
    from . import acquire_effective120_v2 as builder
    from . import train_expanded_unet as expanded
except ImportError:
    import acquire_effective120_v2 as builder
    import train_expanded_unet as expanded


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WORKSPACE / "data/landsat30_texture_expanded_v1"
STAC_ITEM = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
    "landsat-c2-l2/items/{item_id}"
)
SOURCE_SHA256 = expanded.FROZEN_SOURCE_SHA256
VALIDATION_SHA256 = expanded.FROZEN_VALIDATION_SHA256
MAX_BYTES = 10 * 1024**3
MAX_WORKERS = 4
SR_SCALE = np.float32(2.75e-5)
SR_OFFSET = np.float32(-0.2)

BAND_NAMES = tuple(builder.OPTICAL_KEYS)
INDEX_NAMES = ("ndvi", "ndbi", "mndwi")
CHANNEL_NAMES = tuple(
    [f"{band}_{stat}" for band in BAND_NAMES for stat in ("std", "q25", "q75")]
    + [f"{index}_{stat}" for index in INDEX_NAMES for stat in ("mean", "std", "q25", "q75")]
    + ["valid_coverage30"]
)
FEATURE_CHANNELS = 30
TOTAL_CHANNELS = 31
OPTICAL_REJECT_BITS = (0, 1, 2, 3, 4, 5, 9)
TARGET_ONLY_BITS = (6, 7, 8)
OPTICAL_REJECT_MASK = sum(1 << bit for bit in OPTICAL_REJECT_BITS)


@dataclass(frozen=True)
class SceneSpec:
    entry: expanded.SceneEntry
    metadata: Mapping[str, Any]
    qa_reason30: np.ndarray


@dataclass(frozen=True)
class InputScope:
    """Resolved sidecar inputs without weakening the frozen default gate."""

    source_entries: tuple[expanded.SceneEntry, ...]
    validation_entries: tuple[expanded.SceneEntry, ...]
    source_manifest_sha256: str
    validation_manifest_sha256: str | None
    source_city_count: int
    validation_city_count: int
    source_only: bool

    @property
    def source_scene_count(self) -> int:
        return len(self.source_entries)

    @property
    def validation_scene_count(self) -> int:
        return len(self.validation_entries)

    @property
    def scene_count(self) -> int:
        return self.source_scene_count + self.validation_scene_count


class WeightedMoments:
    def __init__(self) -> None:
        self.weight = 0.0
        self.total = 0.0
        self.total2 = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.cells = 0

    def update(self, values: np.ndarray, coverage: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        w = np.asarray(coverage, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            return
        xv, wv = x[valid], w[valid]
        self.weight += float(wv.sum())
        self.total += float(np.dot(xv, wv))
        self.total2 += float(np.dot(xv * xv, wv))
        self.minimum = min(self.minimum, float(xv.min()))
        self.maximum = max(self.maximum, float(xv.max()))
        self.cells += int(valid.sum())

    def document(self) -> dict[str, Any]:
        if self.weight <= 0:
            raise ValueError("texture normalization channel has no source support")
        mean = self.total / self.weight
        variance = max(self.total2 / self.weight - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": max(float(np.sqrt(variance)), 1e-8),
            "min": self.minimum,
            "max": self.maximum,
            "weight_sum": self.weight,
            "valid_cell_count": self.cells,
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_city_years(
    entries: Sequence[expanded.SceneEntry],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    identities: set[tuple[str, int]] = set()
    for entry in entries:
        identity = (entry.city, int(entry.year))
        if identity in identities:
            raise ValueError(
                f"source-only manifest repeats city/year identity: "
                f"{entry.city} {entry.year}"
            )
        identities.add(identity)
        grouped.setdefault(entry.city, []).append(int(entry.year))
    return {city: sorted(years) for city, years in sorted(grouped.items())}


def resolve_input_scope(
    source_manifest: Path,
    validation_manifest: Path | None,
    *,
    source_only: bool,
) -> InputScope:
    """Resolve frozen source+validation or an arbitrary complete source manifest.

    The default path deliberately delegates to the existing frozen verifier.
    In source-only mode the validation argument is never inspected or opened.
    """
    if not source_only:
        if validation_manifest is None:
            raise ValueError("validation manifest is required outside source-only mode")
        verified = expanded._verified_inputs(source_manifest, validation_manifest)
        return InputScope(
            source_entries=tuple(verified.source_entries),
            validation_entries=tuple(verified.validation_entries),
            source_manifest_sha256=verified.source_manifest_sha256,
            validation_manifest_sha256=verified.validation_manifest_sha256,
            source_city_count=len(verified.source_city_years),
            validation_city_count=len(verified.validation_city_years),
            source_only=False,
        )

    source_path = expanded.reject_sealed(source_manifest).resolve()
    manifest_bytes = source_path.read_bytes()
    source_sha256 = sha256_bytes(manifest_bytes)
    payload = json.loads(manifest_bytes)
    if not isinstance(payload, dict):
        raise ValueError("source-only manifest root must be a JSON object")
    if (
        payload.get("build_complete") is not True
        or payload.get("split") not in (None, "source")
    ):
        raise ValueError("source-only manifest must be a complete source split")
    sealed_flags = ("sealed_test_unlocked", "sealed_test_opened", "sealed_access")
    if any(payload.get(name) not in (None, False) for name in sealed_flags):
        raise ValueError("source-only manifest must keep every sealed-access flag false")
    selected = payload.get("selected_cities")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(city, str) or not city.strip() for city in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("source-only manifest requires non-empty unique selected_cities")
    entries = tuple(
        expanded.read_manifest(
            source_path,
            "source",
            expected_manifest_sha256=source_sha256,
            expected_cities=selected,
        )
    )
    city_years = _entry_city_years(entries)
    if "scene_count" in payload and payload["scene_count"] != len(entries):
        raise ValueError("source-only manifest scene_count differs from its scene list")
    if "city_count" in payload and payload["city_count"] != len(city_years):
        raise ValueError("source-only manifest city_count differs from its scene list")
    if "city_years" in payload:
        declared = payload["city_years"]
        if not isinstance(declared, Mapping):
            raise ValueError("source-only manifest city_years must be an object")
        try:
            normalized = {
                str(city): sorted(int(year) for year in years)
                for city, years in declared.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("source-only manifest city_years is invalid") from exc
        if normalized != city_years:
            raise ValueError("source-only manifest city_years differs from its scenes")
    return InputScope(
        source_entries=entries,
        validation_entries=(),
        source_manifest_sha256=source_sha256,
        validation_manifest_sha256=None,
        source_city_count=len(city_years),
        validation_city_count=0,
        source_only=True,
    )


def sidecar_name(entry: expanded.SceneEntry) -> str:
    return f"{entry.city}_{int(entry.year)}_landsat30_texture.npz"


def optical_valid_mask(qa_reason30: np.ndarray, optical_raw: np.ndarray | None = None) -> np.ndarray:
    reason = np.asarray(qa_reason30)
    if reason.shape != (640, 640) or not np.issubdtype(reason.dtype, np.integer):
        raise ValueError("qa_reason30 must be an integer 640x640 array")
    valid = (reason.astype(np.uint16) & np.uint16(OPTICAL_REJECT_MASK)) == 0
    if optical_raw is not None:
        raw = np.asarray(optical_raw)
        if raw.shape != (6, 640, 640):
            raise ValueError("fresh optical stack must have shape [6,640,640]")
        nonzero = np.all(raw != 0, axis=0)
        stored_zero = (reason.astype(np.uint16) & np.uint16(1 << 9)) != 0
        if not np.array_equal(stored_zero, ~nonzero):
            mismatch = int(np.count_nonzero(stored_zero != ~nonzero))
            raise ValueError(f"fresh optical-zero mask differs from stored bit 9 at {mismatch} pixels")
        valid &= nonzero
    return valid


def _blocks(array: np.ndarray) -> np.ndarray:
    leading = array.shape[:-2]
    if array.shape[-2:] != (640, 640):
        raise ValueError("30 m arrays must be 640x640")
    return array.reshape(*leading, 160, 4, 160, 4).swapaxes(-3, -2).reshape(*leading, 160, 160, 16)


def _normalized_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = first + second
    result = np.zeros_like(first, dtype=np.float32)
    np.divide(first - second, denominator, out=result, where=np.abs(denominator) > 1e-6)
    return np.clip(result, -1.0, 1.0)


def aggregate_texture(optical30: np.ndarray, valid30: np.ndarray) -> np.ndarray:
    optical = np.asarray(optical30, dtype=np.float32)
    valid = np.asarray(valid30, dtype=bool)
    if optical.shape != (6, 640, 640) or valid.shape != (640, 640):
        raise ValueError("texture aggregation requires [6,640,640] optical and [640,640] mask")
    indices = np.stack(
        (
            _normalized_difference(optical[3], optical[2]),
            _normalized_difference(optical[4], optical[3]),
            _normalized_difference(optical[1], optical[4]),
        )
    )
    values = np.concatenate((optical, indices), axis=0)
    blocked = _blocks(values)
    blocked_valid = _blocks(valid)[None]
    masked = np.where(blocked_valid, blocked, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(masked, axis=-1)
        q25 = np.nanquantile(masked, 0.25, axis=-1)
        q75 = np.nanquantile(masked, 0.75, axis=-1)
        mean = np.nanmean(masked, axis=-1)
    channels: list[np.ndarray] = []
    for index in range(6):
        channels.extend((std[index], q25[index], q75[index]))
    for index in range(6, 9):
        channels.extend((mean[index], std[index], q25[index], q75[index]))
    coverage = _blocks(valid).sum(axis=-1, dtype=np.uint8).astype(np.float32) / np.float32(16.0)
    result = np.stack((*channels, coverage), axis=0)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    if result.shape != (TOTAL_CHANNELS, 160, 160) or not np.all(np.isfinite(result)):
        raise ValueError("invalid 31-channel texture aggregation")
    if np.any(result[-1] < 0) or np.any(result[-1] > 1):
        raise ValueError("texture coverage is outside [0,1]")
    return result.astype(np.float16)


def load_scene_spec(entry: expanded.SceneEntry) -> SceneSpec:
    payload = entry.file.read_bytes()
    if entry.sha256 is None or sha256_bytes(payload) != entry.sha256:
        raise ValueError(f"source/validation scene SHA mismatch: {entry.scene_id}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "qa_reason30" not in archive.files or "metadata" not in archive.files:
            raise ValueError(f"scene lacks stored QA reason or metadata: {entry.scene_id}")
        reason = np.asarray(archive["qa_reason30"], dtype=np.uint16)
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
    expected = {
        "city": entry.city,
        "year": int(entry.year),
        "split": entry.role,
        "shape30": [640, 640],
        "canonical_grid_crs": str(metadata.get("crs")),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"scene metadata {key} mismatch: {entry.scene_id}")
    if not isinstance(metadata.get("item_id"), str) or not isinstance(metadata.get("datetime"), str):
        raise ValueError(f"scene lacks Landsat item identity: {entry.scene_id}")
    transform = np.asarray(metadata.get("transform30"), dtype=np.float64)
    if transform.shape != (9,) or not np.all(np.isfinite(transform)):
        raise ValueError(f"scene transform30 is invalid: {entry.scene_id}")
    return SceneSpec(entry, metadata, reason)


def fetch_exact_item(spec: SceneSpec) -> dict[str, Any]:
    item_id = str(spec.metadata["item_id"])
    item = builder.request_json(STAC_ITEM.format(item_id=quote(item_id, safe="")))
    if item.get("id") != item_id or item.get("collection") != builder.LANDSAT_COLLECTION:
        raise ValueError(f"STAC exact-item identity mismatch: {spec.entry.scene_id}")
    if item.get("properties", {}).get("datetime") != spec.metadata["datetime"]:
        raise ValueError(f"STAC datetime mismatch: {spec.entry.scene_id}")
    missing = set(BAND_NAMES).difference(item.get("assets", {}))
    if missing:
        raise ValueError(f"STAC item lacks optical assets {sorted(missing)}")
    return item


def _same_signature(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("crs") == expected.get("crs")
        and actual.get("shape") == expected.get("shape")
        and actual.get("dtype") == expected.get("dtype")
        and actual.get("nodata") == expected.get("nodata")
        and Affine(*actual["transform"]).almost_equals(Affine(*expected["transform"]), precision=1e-10)
    )


def read_optical(spec: SceneSpec, item: Mapping[str, Any], token: str) -> np.ndarray:
    metadata = spec.metadata
    grid = {
        "crs": str(metadata["canonical_grid_crs"]),
        "transform": Affine(*metadata["transform30"]),
        "shape": (640, 640),
    }
    stored_signatures = metadata.get("asset_grid_signatures", {})
    arrays: list[np.ndarray] = []
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**env):
        for key in BAND_NAMES:
            href = builder.signed_href(item, key, token)
            with rasterio.open(href) as dataset:
                actual = builder.grid_signature(dataset)
                expected = stored_signatures.get(key)
                if not isinstance(expected, Mapping) or not _same_signature(actual, expected):
                    raise ValueError(f"delivered-grid signature changed: {spec.entry.scene_id}/{key}")
                arrays.append(builder.reproject_asset_to_canonical(dataset, grid, fill_value=0))
    return np.stack(arrays)


def _metadata_document(spec: SceneSpec, item: Mapping[str, Any], features: np.ndarray) -> dict[str, Any]:
    coverage = features[-1].astype(np.float32)
    observed = coverage > 0
    return {
        "schema": "landsat30-texture-scene-v1",
        "split": spec.entry.role,
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
        "channels": list(CHANNEL_NAMES),
        "dtype": "float16",
        "coverage_mean": float(coverage.mean()),
        "coverage_nonzero_cells": int(observed.sum()),
        "optical_mask_source": "stored qa_reason30 bits 0,1,2,3,4,5,9; target-only bits 6,7,8 ignored",
        "stac_access": "exact item metadata only; no page persisted",
        "asset_access": "six optical COG range windows only; no raw asset persisted",
        "sealed_access": False,
    }


def write_sidecar(path: Path, features: np.ndarray, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            texture31=np.asarray(features, dtype=np.float16),
            channel_names=np.asarray(CHANNEL_NAMES),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def record_from_existing(path: Path, spec: SceneSpec) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["texture31"])
            names = tuple(np.asarray(archive["channel_names"]).astype(str).tolist())
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
        if (
            features.shape != (TOTAL_CHANNELS, 160, 160)
            or features.dtype != np.float16
            or names != CHANNEL_NAMES
            or metadata.get("source_scene_sha256") != spec.entry.sha256
            or metadata.get("item_id") != spec.metadata["item_id"]
            or metadata.get("datetime") != spec.metadata["datetime"]
            or metadata.get("sealed_access") is not False
        ):
            return None
        return manifest_record(path, spec, metadata)
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def manifest_record(path: Path, spec: SceneSpec, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "scene",
        "status": "complete",
        "split": spec.entry.role,
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
        "dtype": "float16",
        "coverage_mean": float(metadata["coverage_mean"]),
        "sealed_access": False,
    }


class TokenManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = builder.sas_token(builder.LANDSAT_COLLECTION)

    def get(self) -> str:
        with self._lock:
            return self._token

    def refresh(self) -> str:
        with self._lock:
            self._token = builder.sas_token(builder.LANDSAT_COLLECTION)
            return self._token


def build_one(spec: SceneSpec, output: Path, tokens: TokenManager) -> dict[str, Any]:
    path = output / sidecar_name(spec.entry)
    existing = record_from_existing(path, spec)
    if existing is not None:
        return existing
    item = fetch_exact_item(spec)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = read_optical(spec, item, tokens.get() if attempt == 0 else tokens.refresh())
            break
        except (rasterio.errors.RasterioIOError, builder.RemoteAssetReadError) as error:
            last_error = error
    else:
        assert last_error is not None
        raise last_error
    valid = optical_valid_mask(spec.qa_reason30, raw)
    optical = raw.astype(np.float32) * SR_SCALE + SR_OFFSET
    features = aggregate_texture(optical, valid)
    metadata = _metadata_document(spec, item, features)
    write_sidecar(path, features, metadata)
    return manifest_record(path, spec, metadata)


def build_normalization(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    source_manifest_sha256: str = SOURCE_SHA256,
    source_city_count: int | None = None,
) -> dict[str, Any]:
    accumulators = {name: WeightedMoments() for name in CHANNEL_NAMES[:FEATURE_CHANNELS]}
    source_rows = [row for row in records if row["split"] == "source"]
    actual_source_cities = {str(row["city"]) for row in source_rows}
    if source_city_count is None:
        source_city_count = len(actual_source_cities)
    elif source_city_count != len(actual_source_cities):
        raise ValueError("source city count differs from normalization records")
    for row in source_rows:
        with np.load(output / str(row["file"]), allow_pickle=False) as archive:
            values = np.asarray(archive["texture31"], dtype=np.float32)
        coverage = values[-1]
        for index, name in enumerate(CHANNEL_NAMES[:FEATURE_CHANNELS]):
            accumulators[name].update(values[index], coverage)
    return {
        "schema": "landsat30-texture-source-normalization-v1",
        "scope": {"split": "source", "validation_included": False, "sealed_access": False},
        "source_manifest_sha256": source_manifest_sha256,
        "source_city_count": source_city_count,
        "source_scene_count": len(source_rows),
        "value_array": "texture31",
        "coverage_channel": "valid_coverage30",
        "missing_rule": "coverage=0 => feature values stored as 0; normalized loader must retain 0",
        "weighting": "120m valid-coverage weighted over source scenes",
        "channels": {name: accumulators[name].document() for name in CHANNEL_NAMES[:FEATURE_CHANNELS]},
        "excluded": ["valid_coverage30"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_manifest(path: Path, header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(header, sort_keys=True), *(json.dumps(row, sort_keys=True) for row in records)]
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    # NumPy emits these for the intentionally missing 120 m cells.  Their
    # values are deterministically replaced by zero below; suppressing the
    # warnings keeps multi-worker progress output readable.
    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    if not 1 <= int(args.workers) <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    output = expanded.reject_sealed(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scope = resolve_input_scope(
        args.source_manifest,
        getattr(args, "validation_manifest", expanded.DEFAULT_VALIDATION),
        source_only=bool(getattr(args, "source_only", False)),
    )
    entries = list(scope.source_entries) + list(scope.validation_entries)
    if args.cities:
        wanted = set(args.cities)
        entries = [entry for entry in entries if entry.city in wanted]
        missing = wanted.difference(entry.city for entry in entries)
        if missing:
            raise ValueError(f"requested smoke cities not in expanded data: {sorted(missing)}")
    specs = [load_scene_spec(entry) for entry in entries]
    tokens = TokenManager()
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        pending = {executor.submit(build_one, spec, output, tokens): spec for spec in specs}
        for future in as_completed(pending):
            spec = pending[future]
            record = future.result()
            records.append(record)
            # Sum completed final records rather than globbing the output
            # directory: another worker may atomically rename a hidden
            # temporary .npz between glob() and stat().
            total = sum(int(row["bytes"]) for row in records)
            if total > MAX_BYTES:
                raise RuntimeError("texture sidecars exceed the 10 GiB persistent cap")
            print(f"complete {spec.entry.scene_id} {record['bytes']} bytes", flush=True)
    order = {entry.scene_id: index for index, entry in enumerate(entries)}
    records.sort(key=lambda row: order[str(row["scene_id"])])
    full = not args.cities and (scope.source_only or len(entries) == 162)
    if full:
        normalization = build_normalization(
            output,
            records,
            source_manifest_sha256=scope.source_manifest_sha256,
            source_city_count=scope.source_city_count,
        )
        _write_json(output / "source_feature_normalization.json", normalization)
        normalization_sha = sha256_file(output / "source_feature_normalization.json")
        header = {
            "kind": "dataset",
            "schema": "landsat30-texture-sidecars-v1",
            "status": "complete",
            "scene_count": scope.scene_count,
            "source_scene_count": scope.source_scene_count,
            "validation_scene_count": scope.validation_scene_count,
            "source_manifest_sha256": scope.source_manifest_sha256,
            "validation_manifest_sha256": scope.validation_manifest_sha256,
            "source_normalization_sha256": normalization_sha,
            "channel_count": TOTAL_CHANNELS,
            "channels": list(CHANNEL_NAMES),
            "persistent_cap_bytes": MAX_BYTES,
            "persistent_bytes": sum(int(row["bytes"]) for row in records),
            "sealed_access": False,
        }
        if scope.source_only:
            header["input_scope"] = "source_only"
        _write_manifest(output / "manifest.jsonl", header, records)
    return {
        "status": "complete" if full else "smoke_complete",
        "scenes": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "wall_seconds": time.monotonic() - started,
        "output": str(output),
        "sealed_access": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=expanded.DEFAULT_SOURCE)
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument("--source-only", action="store_true")
    input_mode.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--cities", nargs="*")
    args = parser.parse_args(argv)
    if not args.source_only and args.validation_manifest is None:
        args.validation_manifest = expanded.DEFAULT_VALIDATION
    return args


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
