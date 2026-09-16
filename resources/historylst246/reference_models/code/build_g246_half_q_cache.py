#!/usr/bin/env python3
"""Build the Fit-only r6a/r9 fixed-half Q cache for G246 U1-Lite.

The cache core is intentionally dependency-injected: callers must supply a
predictor source that explicitly proves it did not open supervision arrays.
The formal CLI uses the lower-level target-free R2 encoders and local Stage-C
sidecars; it never calls ``R2TemporalDataset._sample`` or
``evaluation_batch``, both of which materialise query supervision today.

Only ``q_half = 0.5 * (q_r6a + q_r9)`` is persisted.  The array, row index,
and manifest are built in a private sibling directory and published by one
atomic directory rename.  Any contract or numerical failure removes the
staging directory and leaves no accepted cache.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator, Mapping, Protocol, Sequence
import uuid

import numpy as np
import torch
from torch import Tensor


WORKSPACE = Path(__file__).resolve().parents[1]
CODE_ROOT = WORKSPACE / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import g246_data  # noqa: E402
import g246_q_bands  # noqa: E402
import g246_r2_data  # noqa: E402
import g246_r2_multisource  # noqa: E402
import train_g246_r2  # noqa: E402


CACHE_SCHEMA = "uhi-cdc-g246-half-q-cache-v1"
INDEX_SCHEMA = "uhi-cdc-g246-half-q-scene-index-v1"
PREDICTOR_SCHEMA = "uhi-cdc-g246-fit-fine52-predictor-only-v1"
VALIDATION_PREDICTOR_SCHEMA = (
    "uhi-cdc-g246-validation-fine52-predictor-only-v1"
)
ARRAY_NAME = "q_half_f32.npy"
INDEX_NAME = "scene_index.json"
MANIFEST_NAME = "manifest.json"

REGISTERED_FIT_SCENE_COUNT = 603
REGISTERED_FIT_CITY_COUNT = 201
REGISTERED_VALIDATION_SCENE_COUNT = 45
REGISTERED_VALIDATION_CITY_COUNT = 15
REGISTERED_GRID_SHAPE = (160, 160)
REGISTERED_FIT_VIEW_SHA256 = (
    "7252a2922cf70d07b418ffdbf308c3ebaf0969afff2521217387ad7019c35a54"
)
REGISTERED_RECEIPT_SHA256 = (
    "06bdd26e267bae1007060aa81d5dfe73e3b478565df7bda9a0964681f99ebd20"
)
REGISTERED_SOURCE_MANIFEST_SHA256 = (
    "801a9814eb74ab5d805d0e40cef67eaada31bf4b2fdc8c0ac7f4253446b93889"
)
REGISTERED_NORMALIZATION_SHA256 = (
    "452d909e209b9e022067a5f0b9240f4ac7f684e51f9be04b6501beb8631de36c"
)
REGISTERED_FINE52_BINDING_SHA256 = (
    "4806aa9931134f5f04f48d204f4e171a8b4ba840be8dbea788fb48dae078fee9"
)

DEFAULT_SHALLOW_CHECKPOINT = (
    WORKSPACE
    / "artifacts/g246_r2/runs/"
    "r6a_calibratedq_s_context15_surgery_fine52_public_seed20260818/best.pt"
)
DEFAULT_DEEP_CHECKPOINT = (
    WORKSPACE
    / "artifacts/g246_r2/runs/"
    "r9_ipmrq_s_context15_fine52_public_seed20260818/best.pt"
)
DEFAULT_NORMALIZATION = DEFAULT_SHALLOW_CHECKPOINT.parent / "normalization.json"
DEFAULT_TEXTURE_MANIFEST = WORKSPACE / "data/g246_r2_texture_v2/manifest.jsonl"
DEFAULT_WEATHER_MANIFEST = (
    WORKSPACE / "artifacts/g246_r2_weather_sidecars_v1/manifest.json"
)

FORBIDDEN_SAMPLE_KEYS = frozenset(
    {
        "target",
        "target_k",
        "valid",
        "eligible",
        "label",
        "labels",
        "q90",
        "hotspot_target",
    }
)


class CacheContractError(ValueError):
    """A cache input or publication contract failed closed."""


class PredictorOnlySource(Protocol):
    predictor_only: bool
    target_arrays_opened: bool
    locked_test_opened: bool
    scene_ids: Sequence[str]

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        ...

    def provenance_record(self) -> Mapping[str, Any]:
        ...


class TeacherPair(Protocol):
    locked_test_opened: bool

    def predict_q(self, sample: Mapping[str, Any]) -> tuple[Any, Any]:
        ...

    def provenance_record(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class CacheBuildResult:
    output_dir: Path
    array_sha256: str
    scene_index_sha256: str
    scene_order_sha256: str
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "array_sha256": self.array_sha256,
            "scene_index_sha256": self.scene_index_sha256,
            "scene_order_sha256": self.scene_order_sha256,
            "row_count": self.row_count,
            "locked_test_opened": False,
            "target_arrays_opened": False,
        }


def _hex_digest(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CacheContractError(f"{label} must be a lowercase SHA-256")
    return digest


def _reject_forbidden_path(path: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(path).replace("\\", "/").casefold()
    if "sealed" in raw or "locked_test" in raw or "locked-test" in raw:
        raise CacheContractError(f"{label} points at a sealed/locked-test path")
    return Path(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise CacheContractError(f"value is not JSON-safe: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _entry_scene_id(entry: Any) -> str:
    value = str(getattr(entry, "scene_id", "")).strip()
    if not value:
        raise CacheContractError("Fit scene has an empty scene_id")
    return value


def guarded_fit_scene_index(
    entries: Sequence[Any],
    *,
    expected_scene_count: int = REGISTERED_FIT_SCENE_COUNT,
    expected_city_count: int | None = REGISTERED_FIT_CITY_COUNT,
) -> tuple[Any, ...]:
    """Return a stable Fit-only order, checking roles before every path access."""

    if expected_scene_count <= 0:
        raise CacheContractError("expected scene count must be positive")
    rows = tuple(entries)
    # This pass is deliberately first.  A forged dev/test entry must fail
    # before its path property can trigger any filesystem operation.
    for entry in rows:
        if getattr(entry, "view_role", None) != "fit":
            raise CacheContractError(
                f"half-Q cache accepts only role='fit', got "
                f"{getattr(entry, 'view_role', None)!r}"
            )
    if len(rows) != expected_scene_count:
        raise CacheContractError(
            f"Fit scene count differs: {len(rows)} != {expected_scene_count}"
        )

    scene_ids = [_entry_scene_id(entry) for entry in rows]
    if len(set(scene_ids)) != len(scene_ids):
        raise CacheContractError("Fit scene index contains duplicate scene IDs")
    cities = [str(getattr(entry, "city", "")).strip() for entry in rows]
    if any(not city for city in cities):
        raise CacheContractError("Fit scene index contains an empty city")
    if expected_city_count is not None and len(set(cities)) != expected_city_count:
        raise CacheContractError(
            f"Fit city count differs: {len(set(cities))} != {expected_city_count}"
        )
    if expected_city_count is not None:
        city_counts = {city: cities.count(city) for city in set(cities)}
        if any(count != 3 for count in city_counts.values()):
            raise CacheContractError("registered Fit index requires three scenes per city")

    for entry in rows:
        _hex_digest(getattr(entry, "sha256", None), f"source {entry.scene_id}")
        _reject_forbidden_path(getattr(entry, "file"), f"source {entry.scene_id}")
    return tuple(sorted(rows, key=_entry_scene_id))


def _guarded_validation_scene_index(
    entries: Sequence[Any],
    *,
    expected_scene_count: int = REGISTERED_VALIDATION_SCENE_COUNT,
    expected_city_count: int | None = REGISTERED_VALIDATION_CITY_COUNT,
) -> tuple[Any, ...]:
    """Return a stable predictor-only validation order without weakening cache guards."""

    if expected_scene_count <= 0:
        raise CacheContractError("expected validation scene count must be positive")
    rows = tuple(entries)
    # As for the Fit cache guard, role is checked before any path is touched.
    for entry in rows:
        if getattr(entry, "view_role", None) != "validation":
            raise CacheContractError(
                "validation predictor accepts only role='validation', got "
                f"{getattr(entry, 'view_role', None)!r}"
            )
    if len(rows) != expected_scene_count:
        raise CacheContractError(
            "Validation scene count differs: "
            f"{len(rows)} != {expected_scene_count}"
        )

    scene_ids = [str(getattr(entry, "scene_id", "")).strip() for entry in rows]
    if any(not scene_id for scene_id in scene_ids):
        raise CacheContractError("Validation scene has an empty scene_id")
    if len(set(scene_ids)) != len(scene_ids):
        raise CacheContractError("Validation scene index contains duplicate scene IDs")
    cities = [str(getattr(entry, "city", "")).strip() for entry in rows]
    if any(not city for city in cities):
        raise CacheContractError("Validation scene index contains an empty city")
    if expected_city_count is not None and len(set(cities)) != expected_city_count:
        raise CacheContractError(
            "Validation city count differs: "
            f"{len(set(cities))} != {expected_city_count}"
        )
    if expected_city_count is not None:
        city_counts = {city: cities.count(city) for city in set(cities)}
        if any(count != 3 for count in city_counts.values()):
            raise CacheContractError(
                "registered Validation index requires three scenes per city"
            )

    for entry in rows:
        _hex_digest(getattr(entry, "sha256", None), f"source {entry.scene_id}")
        _reject_forbidden_path(getattr(entry, "file"), f"source {entry.scene_id}")
    return tuple(sorted(rows, key=lambda entry: str(entry.scene_id)))


def _validate_source_binding(
    source_binding: Mapping[str, Any], *, expected_fit_view_sha256: str,
) -> dict[str, Any]:
    if not isinstance(source_binding, Mapping):
        raise CacheContractError("source binding must be a mapping")
    result = dict(source_binding)
    for key, label in (
        ("receipt_sha256", "split receipt"),
        ("source_manifest_sha256", "source manifest"),
        ("fit_view_sha256", "Fit view"),
        ("normalization_sha256", "normalization"),
        ("fine52_binding_sha256", "Fine52 binding"),
    ):
        result[key] = _hex_digest(result.get(key), f"{label} SHA-256")
    expected = _hex_digest(expected_fit_view_sha256, "expected Fit view SHA-256")
    if result["fit_view_sha256"] != expected:
        raise CacheContractError("source binding is not the expected Fit view")
    if result.get("locked_test_opened") is not False:
        raise CacheContractError("source binding does not keep locked test closed")
    if result.get("target_arrays_opened") is not False:
        raise CacheContractError("source binding opened or ambiguously handled targets")
    if result.get("campaign_id") is not None:
        result["campaign_id"] = _hex_digest(result["campaign_id"], "campaign ID")
    return result


def _validate_predictor_source(
    predictor_source: PredictorOnlySource, expected_scene_ids: Sequence[str],
) -> dict[str, Any]:
    if getattr(predictor_source, "predictor_only", None) is not True:
        raise CacheContractError("predictor source lacks predictor_only=True")
    if getattr(predictor_source, "target_arrays_opened", None) is not False:
        raise CacheContractError(
            "predictor source must prove target_arrays_opened=False"
        )
    if getattr(predictor_source, "locked_test_opened", None) is not False:
        raise CacheContractError("predictor source did not keep locked test closed")
    if getattr(predictor_source, "view_role", "fit") != "fit":
        raise CacheContractError("half-Q cache accepts only a Fit predictor source")
    declared = tuple(str(value) for value in getattr(predictor_source, "scene_ids", ()))
    if declared != tuple(expected_scene_ids):
        raise CacheContractError("predictor source scene order differs from Fit index")
    provenance = predictor_source.provenance_record()
    if not isinstance(provenance, Mapping):
        raise CacheContractError("predictor provenance must be a mapping")
    if provenance.get("target_arrays_opened") is not False \
            or provenance.get("locked_test_opened") is not False:
        raise CacheContractError("predictor provenance is not target/locked-test safe")
    if provenance.get("view_role", "fit") != "fit":
        raise CacheContractError("half-Q cache predictor provenance is not Fit-only")
    return dict(provenance)


def _validate_teacher_pair(teacher_pair: TeacherPair) -> dict[str, Any]:
    if getattr(teacher_pair, "locked_test_opened", None) is not False:
        raise CacheContractError("teacher pair did not keep locked test closed")
    provenance = teacher_pair.provenance_record()
    if not isinstance(provenance, Mapping) or provenance.get("locked_test_opened") is not False:
        raise CacheContractError("teacher provenance is not locked-test closed")
    for label in ("shallow", "deep"):
        source = provenance.get(label)
        if not isinstance(source, Mapping) or source.get("locked_test_opened") is not False:
            raise CacheContractError(f"{label} teacher provenance is unsafe")
        _hex_digest(source.get("checkpoint_sha256"), f"{label} checkpoint SHA-256")
        _hex_digest(
            source.get("selected_tensor_state_sha256"),
            f"{label} selected tensor SHA-256",
        )
    return dict(provenance)


def _sample_spatial(
    sample: Mapping[str, Any], key: str, grid_shape: tuple[int, int],
) -> np.ndarray:
    if key not in sample:
        raise CacheContractError(f"predictor sample lacks {key}")
    value = sample[key]
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    # Accepted source layouts are C,H,W; B,C,H,W; or B,T,C,H,W, with
    # singleton B/T/C because cache construction is deliberately sequential.
    while array.ndim > 3:
        if array.shape[0] != 1:
            raise CacheContractError(f"{key} must be a sequential singleton sample")
        array = array[0]
    if array.shape != (1, *grid_shape):
        raise CacheContractError(
            f"{key} shape differs: {array.shape} != {(1, *grid_shape)}"
        )
    return array


def _normalise_q(value: Any, grid_shape: tuple[int, int], label: str) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 3:
        if array.shape[0] != 1:
            raise CacheContractError(f"{label} must have singleton batch dimensions")
        array = array[0]
    if array.shape != (1, *grid_shape):
        raise CacheContractError(
            f"{label} shape differs: {array.shape} != {(1, *grid_shape)}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise CacheContractError(f"{label} must be floating point")
    result = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise CacheContractError(f"{label} contains NaN or infinity")
    return result


def _q_checks(
    q: np.ndarray,
    support: np.ndarray,
    coarse_valid: np.ndarray,
    *,
    label: str,
    closure_tolerance_k: float,
    unsupported_tolerance_k: float,
) -> dict[str, float]:
    support_bool = np.asarray(support) != 0
    unsupported = float(
        np.max(np.abs(q[~support_bool]), initial=0.0)
    )
    if unsupported > unsupported_tolerance_k:
        raise CacheContractError(
            f"{label} unsupported magnitude {unsupported:.8g} K exceeds "
            f"{unsupported_tolerance_k:.8g} K"
        )
    field_t = torch.from_numpy(np.ascontiguousarray(q[None]))
    support_t = torch.from_numpy(np.ascontiguousarray(support_bool[None]))
    valid_t = torch.from_numpy(np.ascontiguousarray(coarse_valid[None]))
    bands = g246_q_bands.orthogonal_q_bands(field_t, support_t, valid_t)
    parent_closure = float(bands.p40.abs().max().item())
    reconstruction = float((bands.q_total - field_t).abs().max().item())
    if parent_closure > closure_tolerance_k:
        raise CacheContractError(
            f"{label} valid-parent Q closure {parent_closure:.8g} K exceeds "
            f"{closure_tolerance_k:.8g} K"
        )
    if reconstruction > closure_tolerance_k:
        raise CacheContractError(
            f"{label} QM+QH reconstruction {reconstruction:.8g} K exceeds "
            f"{closure_tolerance_k:.8g} K"
        )
    return {
        "max_abs_unsupported_k": unsupported,
        "max_abs_valid_parent_mean_k": parent_closure,
        "max_abs_qm_qh_reconstruction_error_k": reconstruction,
    }


def _validate_sample(
    sample: Mapping[str, Any], scene_id: str, grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, Any]:
    if not isinstance(sample, Mapping):
        raise CacheContractError("predictor iterator must yield mappings")
    forbidden = FORBIDDEN_SAMPLE_KEYS.intersection(sample)
    if forbidden:
        raise CacheContractError(
            f"predictor sample exposes supervision keys: {sorted(forbidden)}"
        )
    if sample.get("target_arrays_opened", False) is not False \
            or sample.get("locked_test_opened", False) is not False:
        raise CacheContractError("predictor sample is not target/locked-test safe")
    if str(sample.get("scene_id", "")) != scene_id:
        raise CacheContractError("predictor iterator scene order differs from Fit index")
    support = _sample_spatial(sample, "support", grid_shape)
    if not np.all(np.isfinite(support)) or np.any((support != 0) & (support != 1)):
        raise CacheContractError(f"{scene_id} support must be finite and binary")
    coarse_shape = (grid_shape[0] // 4, grid_shape[1] // 4)
    coarse = _sample_spatial(sample, "coarse_k", coarse_shape)
    if np.any(np.isinf(coarse)):
        raise CacheContractError(f"{scene_id} coarse predictor contains infinity")
    coarse_valid = np.isfinite(coarse)
    counts = support.astype(bool).reshape(
        1, coarse_shape[0], 4, coarse_shape[1], 4
    ).sum((2, 4))
    if np.any(coarse_valid & (counts <= 0)):
        raise CacheContractError(f"{scene_id} valid parent has no fine support")
    signature = _json_safe(sample.get("grid_signature"))
    if signature is None:
        raise CacheContractError(f"{scene_id} lacks a grid signature")
    return support, coarse_valid, signature


def build_half_q_cache(
    *,
    entries: Sequence[Any],
    source_binding: Mapping[str, Any],
    predictor_source: PredictorOnlySource,
    teacher_pair: TeacherPair,
    output_dir: str | os.PathLike[str],
    expected_scene_count: int = REGISTERED_FIT_SCENE_COUNT,
    expected_city_count: int | None = REGISTERED_FIT_CITY_COUNT,
    expected_fit_view_sha256: str = REGISTERED_FIT_VIEW_SHA256,
    grid_shape: tuple[int, int] = REGISTERED_GRID_SHAPE,
    closure_tolerance_k: float = 5.0e-5,
    unsupported_tolerance_k: float = 0.0,
    code_hashes: Mapping[str, str] | None = None,
) -> CacheBuildResult:
    """Build and atomically publish one Fit-only fixed-half Q cache."""

    if (
        len(grid_shape) != 2
        or min(grid_shape) <= 0
        or grid_shape[0] % 4
        or grid_shape[1] % 4
    ):
        raise CacheContractError("cache grid must be positive and divisible by four")
    if closure_tolerance_k < 0 or unsupported_tolerance_k < 0:
        raise CacheContractError("Q tolerances must be nonnegative")

    ordered = guarded_fit_scene_index(
        entries,
        expected_scene_count=expected_scene_count,
        expected_city_count=expected_city_count,
    )
    scene_ids = tuple(_entry_scene_id(entry) for entry in ordered)
    binding = _validate_source_binding(
        source_binding, expected_fit_view_sha256=expected_fit_view_sha256
    )
    predictor_provenance = _validate_predictor_source(predictor_source, scene_ids)
    if predictor_provenance.get("binding_sha256") != binding["fine52_binding_sha256"]:
        raise CacheContractError("predictor source differs from registered Fine52 binding")
    teacher_provenance = _validate_teacher_pair(teacher_pair)

    hashes = {
        "builder": _sha256_file(Path(__file__).resolve()),
    }
    for key, value in (code_hashes or {}).items():
        hashes[str(key)] = _hex_digest(value, f"code hash {key}")

    output = _reject_forbidden_path(output_dir, "cache output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"cache output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if staging.exists():  # pragma: no cover - UUID collision guard
        raise FileExistsError(f"cache staging path already exists: {staging}")
    staging.mkdir(mode=0o700)

    array_path = staging / ARRAY_NAME
    memmap: np.memmap | None = None
    iterator: Iterator[Mapping[str, Any]] | None = None
    try:
        memmap = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.dtype("<f4"),
            shape=(len(ordered), 1, *grid_shape),
            fortran_order=False,
        )
        iterator = iter(predictor_source)
        row_records: list[dict[str, Any]] = []
        data_digest = hashlib.sha256()
        maxima = {
            "max_abs_unsupported_k": 0.0,
            "max_abs_valid_parent_mean_k": 0.0,
            "max_abs_qm_qh_reconstruction_error_k": 0.0,
        }
        for row_index, entry in enumerate(ordered):
            try:
                sample = next(iterator)
            except StopIteration as exc:
                raise CacheContractError("predictor iterator ended before Fit index") from exc
            scene_id = _entry_scene_id(entry)
            support, coarse_valid, grid_signature = _validate_sample(
                sample, scene_id, grid_shape
            )
            q_shallow_raw, q_deep_raw = teacher_pair.predict_q(sample)
            q_shallow = _normalise_q(q_shallow_raw, grid_shape, "shallow Q")
            q_deep = _normalise_q(q_deep_raw, grid_shape, "deep Q")
            q_half = np.multiply(
                np.add(q_shallow, q_deep, dtype=np.float32),
                np.float32(0.5),
                dtype=np.float32,
            )
            checks = {
                "shallow": _q_checks(
                    q_shallow, support, coarse_valid,
                    label=f"{scene_id} shallow Q",
                    closure_tolerance_k=closure_tolerance_k,
                    unsupported_tolerance_k=unsupported_tolerance_k,
                ),
                "deep": _q_checks(
                    q_deep, support, coarse_valid,
                    label=f"{scene_id} deep Q",
                    closure_tolerance_k=closure_tolerance_k,
                    unsupported_tolerance_k=unsupported_tolerance_k,
                ),
                "half": _q_checks(
                    q_half, support, coarse_valid,
                    label=f"{scene_id} half Q",
                    closure_tolerance_k=closure_tolerance_k,
                    unsupported_tolerance_k=unsupported_tolerance_k,
                ),
            }
            for record in checks.values():
                for key in maxima:
                    maxima[key] = max(maxima[key], float(record[key]))
            little = np.ascontiguousarray(q_half, dtype=np.dtype("<f4"))
            memmap[row_index] = little
            row_payload = little.tobytes(order="C")
            row_sha = hashlib.sha256(row_payload).hexdigest()
            data_digest.update(row_payload)
            row_records.append(
                {
                    "row": row_index,
                    "scene_id": scene_id,
                    "city": str(entry.city),
                    "region": str(getattr(entry, "region", "")),
                    "year": int(getattr(entry, "year")),
                    "item_id": str(getattr(entry, "item_id", "")),
                    "source_scene_sha256": _hex_digest(
                        entry.sha256, f"source {scene_id} SHA-256"
                    ),
                    "grid_signature": grid_signature,
                    "q_half_row_sha256": row_sha,
                    "checks": checks,
                    "view_role": "fit",
                    "locked_test_opened": False,
                    "target_arrays_opened": False,
                }
            )

        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise CacheContractError("predictor iterator contains rows outside Fit index")

        memmap.flush()
        mmap_object = getattr(memmap, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()
        memmap = None
        with array_path.open("rb") as handle:
            os.fsync(handle.fileno())

        identity_rows = [
            {
                "scene_id": _entry_scene_id(entry),
                "source_scene_sha256": str(entry.sha256),
            }
            for entry in ordered
        ]
        scene_order_sha = _canonical_sha256(identity_rows)
        index_document = {
            "schema_version": INDEX_SCHEMA,
            "array_file": ARRAY_NAME,
            "array_shape": [len(ordered), 1, *grid_shape],
            "array_dtype": "float32_little_endian",
            "scene_order_sha256": scene_order_sha,
            "fit_view_sha256": binding["fit_view_sha256"],
            "row_count": len(row_records),
            "rows": row_records,
            "locked_test_opened": False,
            "target_arrays_opened": False,
        }
        index_sha = _write_json(staging / INDEX_NAME, index_document)
        array_sha = _sha256_file(array_path)
        array_size = array_path.stat().st_size
        expected_data_bytes = len(ordered) * int(np.prod((1, *grid_shape))) * 4
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "U1-Lite output-distillation initialization only",
            "cached_quantity": "q_half=0.5*(q_r6a+q_r9)",
            "cached_arrays": ["q_half"],
            "array": {
                "file": ARRAY_NAME,
                "format": "npy_v1_or_later_c_order",
                "dtype": "float32_little_endian",
                "shape": [len(ordered), 1, *grid_shape],
                "data_nbytes": expected_data_bytes,
                "file_nbytes": array_size,
                "file_sha256": array_sha,
                "concatenated_row_data_sha256": data_digest.hexdigest(),
            },
            "scene_index": {
                "file": INDEX_NAME,
                "sha256": index_sha,
                "scene_order_sha256": scene_order_sha,
                "row_count": len(row_records),
            },
            "source_binding": {
                **binding,
                "fit_scene_count": len(ordered),
                "fit_city_count": len({entry.city for entry in ordered}),
                "scene_order_sha256": scene_order_sha,
            },
            "predictor_source": predictor_provenance,
            "teachers": teacher_provenance,
            "checks": {
                "closure_tolerance_k": closure_tolerance_k,
                "unsupported_tolerance_k": unsupported_tolerance_k,
                **maxima,
                "source_outputs_checked_before_averaging": True,
                "qm_qh_recomputed_not_cached": True,
            },
            "code_sha256": hashes,
            "publication": "private_sibling_directory_then_atomic_rename",
            "augmentation": "canonical_full_scene_no_crop_no_d4_no_dropout",
            "target_arrays_opened": False,
            "locked_test_descriptor_resolved": False,
            "locked_test_opened": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return CacheBuildResult(
            output.resolve(), array_sha, index_sha, scene_order_sha, len(ordered)
        )
    except BaseException:
        if memmap is not None:
            try:
                memmap.flush()
                mmap_object = getattr(memmap, "_mmap", None)
                if mmap_object is not None:
                    mmap_object.close()
            except Exception:
                pass
        if iterator is not None and hasattr(iterator, "close"):
            try:
                iterator.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if staging.exists():
            shutil.rmtree(staging)
        raise


class _TextureBackedOpticalAdapter:
    """Expose target-free optical means while sharing the texture NPZ read."""

    def __init__(self, texture_adapter: g246_r2_multisource.Texture30Adapter) -> None:
        self.texture_adapter = texture_adapter
        self.manifest_sha256 = texture_adapter.manifest_sha256
        self._last_scene_id: str | None = None
        self._last_record: dict[str, np.ndarray] | None = None

    def load(self, entry: g246_data.G246Scene) -> dict[str, np.ndarray]:
        record = self.texture_adapter.load(entry)
        coverage = np.asarray(record["coverage"], dtype=np.float32)
        self._last_scene_id = entry.scene_id
        self._last_record = record
        return {
            "optical": np.asarray(record["optical_mean6"], dtype=np.float32),
            "optical_coverage120": coverage,
            "optical_valid120": coverage > 0,
        }

    def take_texture(self, entry: g246_data.G246Scene) -> np.ndarray:
        if self._last_scene_id != entry.scene_id or self._last_record is None:
            raise CacheContractError("texture/optical predictor reads lost scene alignment")
        result = np.asarray(self._last_record["texture30"], dtype=np.float32)
        self._last_scene_id = None
        self._last_record = None
        return result


class Fine52PredictorOnlySource:
    """Sequential public-view Fine52/Context19 source with no supervision path.

    The default is the registered Fit603/201-city source used by the cache
    builder.  Validation is an explicit predictor-only evaluation view; it
    remains ineligible for ``build_half_q_cache`` and always reuses the Fit
    normalization scope.
    """

    predictor_only = True
    target_arrays_opened = False
    locked_test_opened = False

    def __init__(
        self,
        entries: Sequence[g246_data.G246Scene],
        normalization: g246_r2_data.R2Normalization,
        *,
        texture_manifest: str | os.PathLike[str],
        weather_manifest: str | os.PathLike[str],
        expected_binding_sha256: str = REGISTERED_FINE52_BINDING_SHA256,
        view_role: str = "fit",
        expected_scene_count: int | None = None,
        expected_city_count: int | None = None,
    ) -> None:
        if view_role not in {"fit", "validation"}:
            raise CacheContractError(
                "predictor view_role must be 'fit' or 'validation'"
            )
        # Omitting counts selects the registered public-view contract.  Tests
        # and deliberately scoped callers can inject a smaller scene count;
        # in that case ``expected_city_count=None`` intentionally disables the
        # city-count/three-scenes-per-city check, as the cache-core tests do.
        registered_counts = expected_scene_count is None
        if expected_scene_count is None:
            expected_scene_count = (
                REGISTERED_FIT_SCENE_COUNT
                if view_role == "fit"
                else REGISTERED_VALIDATION_SCENE_COUNT
            )
        if registered_counts and expected_city_count is None:
            expected_city_count = (
                REGISTERED_FIT_CITY_COUNT
                if view_role == "fit"
                else REGISTERED_VALIDATION_CITY_COUNT
            )
        if view_role == "fit":
            self.entries = guarded_fit_scene_index(
                entries,
                expected_scene_count=expected_scene_count,
                expected_city_count=expected_city_count,
            )
        else:
            self.entries = _guarded_validation_scene_index(
                entries,
                expected_scene_count=expected_scene_count,
                expected_city_count=expected_city_count,
            )
        self.view_role = view_role
        self.scene_ids = tuple(entry.scene_id for entry in self.entries)
        normalization_fit_scene_ids = tuple(sorted(map(
            str, getattr(normalization, "fit_scene_ids", ()),
        )))
        if not normalization_fit_scene_ids \
                or len(set(normalization_fit_scene_ids)) \
                != len(normalization_fit_scene_ids):
            raise CacheContractError("normalization lacks a unique Fit scene scope")
        if view_role == "fit":
            if normalization_fit_scene_ids != tuple(sorted(self.scene_ids)):
                raise CacheContractError(
                    "normalization scene scope differs from Fit iterator"
                )
        elif set(normalization_fit_scene_ids).intersection(self.scene_ids):
            raise CacheContractError(
                "validation predictor overlaps normalization Fit scenes"
            )
        self.normalization = normalization
        self.texture_adapter = g246_r2_multisource.Texture30Adapter(
            texture_manifest,
            self.entries,
            expected_manifest_sha256=normalization.optical_sidecar_manifest_sha256,
            normalization_scene_ids=normalization.fit_scene_ids,
            cache_size=max(1, min(len(self.entries), 128)),
        )
        self.optical_adapter = _TextureBackedOpticalAdapter(self.texture_adapter)
        self.weather_adapter = g246_r2_multisource.Weather7Adapter(
            weather_manifest,
            self.entries,
            expected_fit_view_sha256=self.texture_adapter.fit_view_sha256,
            expected_validation_view_sha256=(
                self.texture_adapter.validation_view_sha256
            ),
            normalization_scene_ids=normalization.fit_scene_ids,
            cache_size=max(1, min(len(self.entries), 768)),
        )
        if (
            self.weather_adapter.receipt_sha256
            != self.texture_adapter.receipt_sha256
            or self.weather_adapter.source_manifest_sha256
            != self.texture_adapter.source_manifest_sha256
            or self.weather_adapter.campaign_id != self.texture_adapter.campaign_id
            or self.weather_adapter.normalization_scope_sha256
            != self.texture_adapter.normalization_scope_sha256
        ):
            raise CacheContractError("texture/weather public bindings differ")
        binding = {
            "schema": g246_r2_multisource.MULTISOURCE_SCHEMA,
            "fine_channels": g246_r2_multisource.MULTISOURCE_FINE_CHANNELS,
            "context_dim": g246_r2_multisource.MULTISOURCE_CONTEXT_DIM,
            "core_fine_channels_preserved": g246_r2_data.FINE_CHANNELS,
            "core_context_dim_preserved": g246_r2_data.CONTEXT_DIM,
            "texture_manifest_sha256": self.texture_adapter.manifest_sha256,
            "texture_normalization_sha256": self.texture_adapter.normalization_sha256,
            "texture_active_normalization_sha256": (
                self.texture_adapter.active_normalization_sha256
            ),
            "weather_manifest_sha256": self.weather_adapter.manifest_sha256,
            "weather_normalization_sha256": self.weather_adapter.normalization_sha256,
            "weather_active_normalization_sha256": (
                self.weather_adapter.active_normalization_sha256
            ),
            "normalization_scope": self.texture_adapter.normalization_scope,
            "normalization_scene_count": len(
                self.texture_adapter.normalization_scene_ids
            ),
            "normalization_scope_sha256": (
                self.texture_adapter.normalization_scope_sha256
            ),
            "fit_view_sha256": self.texture_adapter.fit_view_sha256,
            "validation_view_sha256": self.texture_adapter.validation_view_sha256,
            "receipt_sha256": self.texture_adapter.receipt_sha256,
            "source_manifest_sha256": self.texture_adapter.source_manifest_sha256,
            "campaign_id": self.texture_adapter.campaign_id,
            "training_io": "local_sidecars_only_no_network",
            "locked_test_opened": False,
            "target_arrays_opened_by_multisource_adapter": False,
            "optical_texture_augmentation": "shared_affine_and_dropout_v1",
            "texture_index_transport": "local_delta_jacobian_v1",
        }
        binding["binding_sha256"] = _canonical_sha256(binding)
        expected = _hex_digest(expected_binding_sha256, "expected Fine52 binding")
        if binding["binding_sha256"] != expected:
            raise CacheContractError("assembled predictor binding differs from Fine52 source")
        self._provenance = {
            "schema_version": (
                PREDICTOR_SCHEMA
                if view_role == "fit"
                else VALIDATION_PREDICTOR_SCHEMA
            ),
            **binding,
            "view_role": view_role,
            "view_scene_count": len(self.entries),
            "view_city_count": len({entry.city for entry in self.entries}),
            "normalization_view_role": "fit",
            "iterator": (
                "encode_r2_predictors_plus_texture30_weather7_no_dataset_sample"
            ),
            "canonical_full160": True,
            "query_only": True,
            "augment": False,
            "target_arrays_opened": False,
            "locked_test_opened": False,
        }

    def provenance_record(self) -> Mapping[str, Any]:
        return dict(self._provenance)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for entry in self.entries:
            core = g246_r2_data.encode_r2_predictors(
                entry, self.normalization, self.optical_adapter
            )
            texture = self.optical_adapter.take_texture(entry)
            weather = self.weather_adapter.load(entry)
            fine = np.concatenate((core["fine"], texture), axis=0).astype(
                np.float32, copy=False
            )
            context = np.concatenate((core["context"], weather), axis=0).astype(
                np.float32, copy=False
            )
            if fine.shape != (52, 160, 160) or context.shape != (19,):
                raise CacheContractError(f"{entry.scene_id} Fine52 geometry drifted")
            if not np.all(np.isfinite(fine)) or not np.all(np.isfinite(context)):
                raise CacheContractError(f"{entry.scene_id} predictor is not finite")
            yield {
                "scene_id": entry.scene_id,
                "fine": fine[None, None],
                "coarse_k": np.asarray(core["coarse_k"], np.float32)[None, None],
                "support": np.asarray(core["support"], np.float32)[None, None],
                "context": context[None, None],
                "temporal_available": np.ones((1, 1), dtype=bool),
                "query_index": np.zeros((1,), dtype=np.int64),
                "grid_signature": core["grid_signature"],
                "view_role": self.view_role,
                "target_arrays_opened": False,
                "locked_test_opened": False,
            }


class RegisteredR6aR9TeacherPair:
    """Exact registered r6a raw / r9 EMA Q inference pair."""

    locked_test_opened = False

    def __init__(
        self,
        shallow_checkpoint: str | os.PathLike[str],
        deep_checkpoint: str | os.PathLike[str],
        *,
        source_binding: Mapping[str, Any],
        device: torch.device,
        amp: bool,
    ) -> None:
        shallow_state, shallow_config, shallow_record = (
            train_g246_r2._parent_router_source_state(
                Path(shallow_checkpoint),
                label="shallow",
                expected_file_sha256=(
                    train_g246_r2.PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256
                ),
                expected_model="calibrated_q",
                expected_update=2000,
                expected_state_key="raw_model_state_dict",
                expected_state_sha256=(
                    train_g246_r2.PARENT_ROUTER_SHALLOW_STATE_SHA256
                ),
            )
        )
        deep_state, deep_config, deep_record = train_g246_r2._parent_router_source_state(
            Path(deep_checkpoint),
            label="deep",
            expected_file_sha256=train_g246_r2.PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
            expected_model="ipmr_q",
            expected_update=8000,
            expected_state_key="ema_model_state_dict",
            expected_state_sha256=train_g246_r2.PARENT_ROUTER_DEEP_STATE_SHA256,
        )
        for config, label in ((shallow_config, "shallow"), (deep_config, "deep")):
            multisource = config.get("multisource_provenance")
            expected = {
                "fit_view_sha256": source_binding["fit_view_sha256"],
                "receipt_sha256": source_binding["receipt_sha256"],
                "normalization_sha256": source_binding["normalization_sha256"],
                "fine_channels": 52,
                "context_dim": 19,
                "input_mode": "multisource",
                "temporal_mode": "single",
                "locked_test_opened": False,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise CacheContractError(f"{label} teacher data contract differs")
            if config.get("data_scope") != "public_validation" \
                    or config.get("provisional_optical_authorized") is not False:
                raise CacheContractError(f"{label} teacher scientific scope differs")
            if not isinstance(multisource, Mapping) \
                    or multisource.get("binding_sha256") \
                    != source_binding["fine52_binding_sha256"] \
                    or multisource.get("source_manifest_sha256") \
                    != source_binding["source_manifest_sha256"] \
                    or multisource.get("locked_test_opened") is not False \
                    or multisource.get("target_arrays_opened_by_multisource_adapter") \
                    is not False:
                raise CacheContractError(f"{label} Fine52 provenance differs")

        shallow_no_geo = shallow_config.get("calibrated_no_geo_core")
        deep_context = deep_config.get("ipmr_context_contract")
        if not isinstance(shallow_no_geo, Mapping) \
                or shallow_no_geo.get("enabled") is not True \
                or shallow_no_geo.get("removed_context_indices") != [5, 6, 7, 8]:
            raise CacheContractError("shallow teacher is not registered Context15")
        if not isinstance(deep_context, Mapping) \
                or deep_context.get("coordinates_used") is not False \
                or deep_context.get("physically_removed_indices") != [5, 6, 7, 8]:
            raise CacheContractError("deep teacher is not registered Context15")

        self.shallow = train_g246_r2.build_model(
            "calibrated_q",
            width=48,
            calibrated_activation_checkpointing=False,
            calibrated_no_geo_core=True,
            fine_channels=52,
            context_dim=19,
        )
        self.deep = train_g246_r2.build_model(
            "ipmr_q",
            width=48,
            ipmr_activation_checkpointing=False,
            fine_channels=52,
            context_dim=19,
        )
        self.shallow.load_state_dict(shallow_state, strict=True)
        self.deep.load_state_dict(deep_state, strict=True)
        self.device = device
        self.amp_requested = bool(amp)
        self.amp_enabled = bool(amp and device.type == "cuda")
        for model in (self.shallow, self.deep):
            model.requires_grad_(False)
            model.eval()
            model.to(device)
        self._provenance = {
            "schema_version": "uhi-cdc-g246-r6a-r9-half-q-teachers-v1",
            "shallow": dict(shallow_record),
            "deep": dict(deep_record),
            "formula": "0.5*(r6a_raw_u2000_q+r9_ema_u8000_q)",
            "device_type": device.type,
            "amp_requested": self.amp_requested,
            "amp_enabled": self.amp_enabled,
            "cuda_autocast_dtype": "float16" if self.amp_enabled else None,
            "cache_cast": "float32",
            "target_arrays_opened": False,
            "locked_test_opened": False,
        }

    def provenance_record(self) -> Mapping[str, Any]:
        return dict(self._provenance)

    def _tensor(self, sample: Mapping[str, Any], key: str, dtype: torch.dtype) -> Tensor:
        value = sample.get(key)
        if isinstance(value, Tensor):
            return value.to(device=self.device, dtype=dtype, non_blocking=True)
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def predict_q(self, sample: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        fine = self._tensor(sample, "fine", torch.float32)
        coarse = self._tensor(sample, "coarse_k", torch.float32)
        support = self._tensor(sample, "support", torch.float32)
        context = self._tensor(sample, "context", torch.float32)
        available = self._tensor(sample, "temporal_available", torch.bool)
        query = self._tensor(sample, "query_index", torch.long)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp_enabled
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            shallow = self.shallow.forward_components(
                fine, coarse, support, context, available, query
            ).q_k
            deep = self.deep.forward_components(
                fine, coarse, support, context, available, query
            ).q_k
        return shallow.float(), deep.float()


def _load_registered_normalization(
    path: Path,
    *,
    expected_sha256: str,
    expected_fit_view_sha256: str,
    expected_scene_ids: Sequence[str],
) -> g246_r2_data.R2Normalization:
    candidate = _reject_forbidden_path(path, "normalization")
    if candidate.is_symlink() or not candidate.is_file():
        raise CacheContractError("normalization must be a regular public file")
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _hex_digest(
        expected_sha256, "expected normalization SHA-256"
    ):
        raise CacheContractError("normalization bytes differ from registered source")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheContractError("normalization JSON is malformed") from exc
    if not isinstance(document, Mapping) or document.get("locked_test_opened") is not False:
        raise CacheContractError("normalization is not locked-test closed")
    normalization = g246_r2_data.R2Normalization.from_dict(document)
    if normalization.fit_view_sha256 != expected_fit_view_sha256:
        raise CacheContractError("normalization Fit view differs")
    if tuple(sorted(normalization.fit_scene_ids)) != tuple(sorted(expected_scene_ids)):
        raise CacheContractError("normalization Fit scene scope differs")
    return normalization


def _registered_code_hashes() -> dict[str, str]:
    paths = {
        "split_loader": CODE_ROOT / "g246_data.py",
        "r2_predictor_encoder": CODE_ROOT / "g246_r2_data.py",
        "multisource_adapter": CODE_ROOT / "g246_r2_multisource.py",
        "q_bands": CODE_ROOT / "g246_q_bands.py",
        "support_projection": CODE_ROOT / "ocnir.py",
        "shallow_model": CODE_ROOT / "g246_calibrated_continuous_q.py",
        "deep_model": CODE_ROOT / "g246_ipmr_q.py",
        "teacher_loader": CODE_ROOT / "train_g246_r2.py",
    }
    return {label: _sha256_file(path) for label, path in paths.items()}


def build_registered_cache(args: argparse.Namespace) -> CacheBuildResult:
    splits = g246_data.load_splits(args.split_receipt, role="fit+validation")
    if splits.receipt_sha256 != REGISTERED_RECEIPT_SHA256:
        raise CacheContractError("split receipt differs from registered G246 campaign")
    if splits.source_manifest_sha256 != REGISTERED_SOURCE_MANIFEST_SHA256:
        raise CacheContractError("source manifest differs from registered G246 campaign")
    if splits.fit_view_sha256 != REGISTERED_FIT_VIEW_SHA256:
        raise CacheContractError("Fit view differs from registered G246 campaign")
    ordered = guarded_fit_scene_index(splits.fit)
    normalization = _load_registered_normalization(
        args.normalization,
        expected_sha256=REGISTERED_NORMALIZATION_SHA256,
        expected_fit_view_sha256=splits.fit_view_sha256,
        expected_scene_ids=[entry.scene_id for entry in ordered],
    )
    predictor_source = Fine52PredictorOnlySource(
        ordered,
        normalization,
        texture_manifest=args.texture_manifest,
        weather_manifest=args.weather_manifest,
        expected_binding_sha256=REGISTERED_FINE52_BINDING_SHA256,
        view_role="fit",
    )
    source_binding = {
        "receipt_sha256": splits.receipt_sha256,
        "source_manifest_sha256": splits.source_manifest_sha256,
        "fit_view_sha256": splits.fit_view_sha256,
        "normalization_sha256": REGISTERED_NORMALIZATION_SHA256,
        "fine52_binding_sha256": REGISTERED_FINE52_BINDING_SHA256,
        "campaign_id": splits.campaign_id,
        "target_arrays_opened": False,
        "locked_test_opened": False,
    }
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CacheContractError("CUDA was requested but is unavailable")
    teachers = RegisteredR6aR9TeacherPair(
        args.shallow_checkpoint,
        args.deep_checkpoint,
        source_binding=source_binding,
        device=device,
        amp=args.amp,
    )
    return build_half_q_cache(
        entries=ordered,
        source_binding=source_binding,
        predictor_source=predictor_source,
        teacher_pair=teachers,
        output_dir=args.output,
        code_hashes=_registered_code_hashes(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the atomic Fit603 r6a/r9 fixed-half Q cache"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-receipt", type=Path, default=g246_data.DEFAULT_SPLIT_RECEIPT
    )
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument(
        "--texture-manifest", type=Path, default=DEFAULT_TEXTURE_MANIFEST
    )
    parser.add_argument(
        "--weather-manifest", type=Path, default=DEFAULT_WEATHER_MANIFEST
    )
    parser.add_argument(
        "--shallow-checkpoint", type=Path, default=DEFAULT_SHALLOW_CHECKPOINT
    )
    parser.add_argument(
        "--deep-checkpoint", type=Path, default=DEFAULT_DEEP_CHECKPOINT
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_registered_cache(args)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
