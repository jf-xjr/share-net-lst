#!/usr/bin/env python3
"""Frozen 50-city residual U-Net control for expanded development data.

The runner is intentionally narrow: every scientific choice is read from and
checked against ``expanded_model_plan_v1.json``.  It never accepts a sealed
path, fits/loads normalization from source data only, and a successful run
publishes exactly ``history.json``, ``normalization.json``, and ``best.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


WORKSPACE = Path(__file__).resolve().parents[1]
FROZEN_PLAN = WORKSPACE / "research/protocol/expanded_model_plan_v1.json"
FROZEN_SOURCE_GEOMETRY_AMENDMENT = (
    WORKSPACE / "research/protocol/expanded_unet_source_geometry_amendment_v1.json"
)
DEFAULT_SOURCE = WORKSPACE / "data/v2/source_expanded_final_v1/manifest.json"
DEFAULT_VALIDATION = (
    WORKSPACE / "data/v2/validation_primary_v2_1_quota4_2b0a25b5/manifest.json"
)
DEFAULT_OUTPUT = WORKSPACE / "artifacts/expanded_v1/residual_unet_control"

FROZEN_PLAN_SHA256 = "5c44aad7945689b8ef41197748e5e0432175c4f750740db96950f0b4329e928b"
FROZEN_SOURCE_GEOMETRY_AMENDMENT_SHA256 = (
    "3a10bf193ad49dd93591d2114a6400cf82234e60fc635288c2303a429b22e334"
)
FROZEN_SOURCE_SHA256 = "3d699445259612ddf6828be7c617ca3edeb1e9aaf9668690dc13cee869bf0d51"
FROZEN_VALIDATION_SHA256 = "cf56514eb8f5e4bdeaf973ec4c0e8a04302be8cbfcc957d1266373b125c4b521"

EXPECTED_SOURCE_CITIES = 50
EXPECTED_SOURCE_SCENES = 150
EXPECTED_VALIDATION_CITIES = 4
EXPECTED_VALIDATION_SCENES = 12
YEAR_MIN = 2021
YEAR_MAX = 2025

INPUT_CHANNELS = 8
WIDTH = 32
BATCH_SIZE = 8
PATCH_SIZE = 64
MINIMUM_PATCH_MASK_FRACTION = 0.10
MAXIMUM_UPDATES = 5_000
VALIDATION_INTERVAL = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 20260818
OUTPUT_FILES = frozenset({"history.json", "normalization.json", "best.pt"})


try:  # Support both package imports and ``python code/train_expanded_unet.py``.
    from . import models as models_module
    from . import v2_data as data_module
    from .models import ResidualUNet, trainable_parameter_count
    from .v2_data import (
        SceneEntry,
        V2Normalization,
        V2PatchDataset,
        decode_residual,
        encode_scene,
        load_scene,
        read_manifest,
        source_validation_overlap,
    )
except ImportError:
    import models as models_module
    import v2_data as data_module
    from models import ResidualUNet, trainable_parameter_count
    from v2_data import (
        SceneEntry,
        V2Normalization,
        V2PatchDataset,
        decode_residual,
        encode_scene,
        load_scene,
        read_manifest,
        source_validation_overlap,
    )

def _average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary unweighted AP equivalent to sklearn for the fixed metric contract."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0 or not np.any(labels):
        raise ValueError("average precision requires aligned scores and positives")
    order = np.argsort(scores, kind="mergesort")[::-1]
    labels, scores = labels[order], scores[order]
    threshold_indices = np.r_[np.flatnonzero(np.diff(scores)), labels.size - 1]
    true_positives = np.cumsum(labels, dtype=np.float64)[threshold_indices]
    precision = true_positives / (threshold_indices + 1.0)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _repository_metrics() -> tuple[Any, Any, Any]:
    """Import the repository metrics with a narrow sklearn-AP compatibility shim."""
    try:
        module = importlib.import_module(
            f"{__package__}.metrics" if __package__ else "metrics"
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "sklearn.metrics"}:
            raise
        sklearn = types.ModuleType("sklearn")
        sklearn_metrics = types.ModuleType("sklearn.metrics")
        sklearn.__spec__ = ModuleSpec("sklearn", loader=None, is_package=True)
        sklearn.__path__ = []
        sklearn_metrics.__spec__ = ModuleSpec("sklearn.metrics", loader=None)
        sklearn_metrics.__package__ = "sklearn"
        sklearn_metrics.average_precision_score = _average_precision_score
        sklearn.metrics = sklearn_metrics
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = sklearn_metrics
        sys.modules.pop(f"{__package__}.metrics" if __package__ else "metrics", None)
        module = importlib.import_module(
            f"{__package__}.metrics" if __package__ else "metrics"
        )
    return module, module.evaluate_field, module.macro_city_aggregate


# Some lean GPU environments omit scikit-learn.  Only average-precision is
# needed from it, so retain the repository metric implementation and supply a
# tested NumPy equivalent rather than changing the metric contract.
try:
    metrics_module, evaluate_field, macro_city_aggregate = _repository_metrics()
except ModuleNotFoundError:
    metrics_module = evaluate_field = macro_city_aggregate = None


@dataclass(frozen=True)
class VerifiedInputs:
    plan_sha256: str
    source_manifest_sha256: str
    validation_manifest_sha256: str
    source_entries: tuple[SceneEntry, ...]
    validation_entries: tuple[SceneEntry, ...]
    source_city_years: dict[str, list[int]]
    validation_city_years: dict[str, list[int]]
    source_payload: dict[str, Any]
    source_geometry_amendment_sha256: str

    def public_record(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "source_cities": len(self.source_city_years),
            "source_scenes": len(self.source_entries),
            "validation_cities": len(self.validation_city_years),
            "validation_scenes": len(self.validation_entries),
            "source_geometry_amendment_sha256": self.source_geometry_amendment_sha256,
            "minimum_patch_mask_fraction": MINIMUM_PATCH_MASK_FRACTION,
            "source_city_years": self.source_city_years,
            "validation_city_years": self.validation_city_years,
            "year_rule": "exactly three distinct manifest-declared years per city in 2021-2025",
            "sealed_test_opened": False,
        }


def reject_sealed(path: str | os.PathLike[str]) -> Path:
    """Reject every sealed-looking path before any filesystem operation."""
    candidate = Path(path)
    normalized = os.fspath(path).replace("\\", "/").casefold()
    if any("sealed" in part for part in normalized.split("/")):
        raise ValueError(f"sealed paths are forbidden: {candidate}")
    return candidate


def sha256_file(path: str | os.PathLike[str]) -> str:
    candidate = reject_sealed(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    candidate = reject_sealed(path)
    payload = json.loads(candidate.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {candidate}")
    return payload


def _assert_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != "uhi-cdc-expanded-model-plan-v1":
        raise ValueError("unsupported expanded model plan schema")
    if plan.get("status") != "frozen_before_any_50_city_model_result":
        raise ValueError("expanded model plan is not frozen before results")
    if plan.get("sealed_test_opened") is not False:
        raise ValueError("expanded model plan must keep sealed_test_opened=false")
    expected_control = {
        "input": "interpolated low-resolution LST, interpolation weight, six optical bands",
        "model": "residual U-Net, width 32",
        "loss": "masked Huber",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "patch_size": PATCH_SIZE,
        "augmentation": "deterministic D4",
        "maximum_updates": MAXIMUM_UPDATES,
        "validation_interval_updates": VALIDATION_INTERVAL,
        "seed": SEED,
    }
    control = plan.get("baseline_runs", {}).get("residual_unet_control")
    if control != expected_control:
        raise ValueError("residual_unet_control differs from the frozen implementation contract")
    fixed = plan.get("fixed_evaluation", {})
    if (
        fixed.get("field") != ["RMSE_K", "MAE_K", "R2", "Pearson"]
        or fixed.get("hotspot_primary")
        != "per-scene truth top-decile (q90), fixed before prediction"
        or fixed.get("aggregation") != "scene metrics -> city mean -> equal-weight city macro"
        or fixed.get("same_output_rule")
        != "one deterministic kelvin field supplies both field and hotspot scores"
    ):
        raise ValueError("fixed expanded evaluation contract has changed")
    selection = plan.get("checkpoint_selection", {})
    if (
        selection.get("field_eligible")
        != "RMSE <= run minimum + max(0.10 K, 0.02*minimum) and R2 >= run maximum - 0.01"
        or selection.get("score") != "0.5*AUPRC_q90 + 0.5*equal_area_IoU_q90"
        or selection.get("tie_breaks") != ["lower RMSE", "earlier update or epoch"]
    ):
        raise ValueError("expanded checkpoint-selection contract has changed")


def _verify_source_geometry_amendment() -> str:
    amendment_sha = sha256_file(FROZEN_SOURCE_GEOMETRY_AMENDMENT)
    if amendment_sha != FROZEN_SOURCE_GEOMETRY_AMENDMENT_SHA256:
        raise ValueError("expanded U-Net source-geometry amendment SHA-256 mismatch")
    amendment = _read_json(FROZEN_SOURCE_GEOMETRY_AMENDMENT)
    expected = {
        "schema": "uhi-cdc-expanded-unet-source-geometry-amendment-v1",
        "parent_plan_sha256": FROZEN_PLAN_SHA256,
        "source_manifest_sha256": FROZEN_SOURCE_SHA256,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_SIZE,
        "source_scenes_checked": EXPECTED_SOURCE_SCENES,
        "frozen_minimum_patch_mask_fraction": MINIMUM_PATCH_MASK_FRACTION,
        "optimizer_steps_before_amendment": 0,
        "validation_metrics_consulted": False,
        "sealed_test_opened": False,
    }
    mismatches = {
        key: {"actual": amendment.get(key), "expected": value}
        for key, value in expected.items()
        if amendment.get(key) != value
    }
    if mismatches:
        raise ValueError(f"expanded U-Net source-geometry amendment differs: {mismatches}")
    return amendment_sha


def _reject_manifest_redirects(payload: Mapping[str, Any]) -> None:
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("manifest has no scenes list")
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping) or "file" not in scene:
            raise ValueError(f"malformed manifest scene {index}")
        reject_sealed(str(scene["file"]))


def _three_declared_years(entries: Sequence[SceneEntry], label: str) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for entry in entries:
        grouped.setdefault(entry.city, []).append(int(entry.year))
    invalid: dict[str, list[int]] = {}
    result: dict[str, list[int]] = {}
    for city, years in grouped.items():
        ordered = sorted(years)
        if (
            len(ordered) != 3
            or len(set(ordered)) != 3
            or any(year < YEAR_MIN or year > YEAR_MAX for year in ordered)
        ):
            invalid[city] = ordered
        result[city] = ordered
    if invalid:
        raise ValueError(
            f"{label} cities must have exactly three distinct manifest-declared "
            f"years in {YEAR_MIN}-{YEAR_MAX}: {invalid}"
        )
    return dict(sorted(result.items()))


def _verified_inputs(
    source_manifest: str | os.PathLike[str] = DEFAULT_SOURCE,
    validation_manifest: str | os.PathLike[str] = DEFAULT_VALIDATION,
    plan_path: str | os.PathLike[str] = FROZEN_PLAN,
) -> VerifiedInputs:
    source_path = reject_sealed(source_manifest)
    validation_path = reject_sealed(validation_manifest)
    plan_candidate = reject_sealed(plan_path)

    plan_sha = sha256_file(plan_candidate)
    if plan_sha != FROZEN_PLAN_SHA256:
        raise ValueError("expanded model plan SHA-256 mismatch")
    plan = _read_json(plan_candidate)
    _assert_plan(plan)
    amendment_sha = _verify_source_geometry_amendment()

    planned_source = plan.get("datasets", {}).get("source", {})
    planned_validation = plan.get("datasets", {}).get("validation", {})
    if (
        planned_source.get("sha256") != FROZEN_SOURCE_SHA256
        or planned_validation.get("sha256") != FROZEN_VALIDATION_SHA256
        or planned_source.get("cities") != EXPECTED_SOURCE_CITIES
        or planned_source.get("scenes") != EXPECTED_SOURCE_SCENES
        or planned_validation.get("cities") != EXPECTED_VALIDATION_CITIES
        or planned_validation.get("scenes") != EXPECTED_VALIDATION_SCENES
    ):
        raise ValueError("dataset hashes/counts differ from the code-pinned expanded plan")

    source_sha = sha256_file(source_path)
    validation_sha = sha256_file(validation_path)
    if source_sha != FROZEN_SOURCE_SHA256 or source_sha != planned_source.get("sha256"):
        raise ValueError("source manifest SHA-256 differs from the frozen plan")
    if (
        validation_sha != FROZEN_VALIDATION_SHA256
        or validation_sha != planned_validation.get("sha256")
    ):
        raise ValueError("validation manifest SHA-256 differs from the frozen plan")

    source_payload = _read_json(source_path)
    validation_payload = _read_json(validation_path)
    _reject_manifest_redirects(source_payload)
    _reject_manifest_redirects(validation_payload)
    if (
        source_payload.get("schema_version") != planned_source.get("schema")
        or source_payload.get("artifact_kind") != "expanded_source_training_root"
        or source_payload.get("build_complete") is not True
        or source_payload.get("split") != "source"
        or source_payload.get("sealed_test_unlocked") is not False
    ):
        raise ValueError("expanded source manifest is not a complete non-sealed training root")
    if (
        validation_payload.get("build_complete") is not True
        or validation_payload.get("split") != "validation"
        or validation_payload.get("sealed_test_unlocked") is not False
    ):
        raise ValueError("validation manifest is not a complete non-sealed validation split")

    source_selected = source_payload.get("selected_cities")
    validation_selected = validation_payload.get("selected_cities")
    if not isinstance(source_selected, list) or not isinstance(validation_selected, list):
        raise ValueError("source/validation selected city lists are required")
    source_entries = tuple(
        read_manifest(
            source_path,
            "source",
            expected_manifest_sha256=source_sha,
            expected_cities=source_selected,
        )
    )
    validation_entries = tuple(
        read_manifest(
            validation_path,
            "validation",
            expected_manifest_sha256=validation_sha,
            expected_cities=validation_selected,
        )
    )
    source_years = _three_declared_years(source_entries, "source")
    validation_years = _three_declared_years(validation_entries, "validation")
    if len(source_entries) != EXPECTED_SOURCE_SCENES or len(source_years) != EXPECTED_SOURCE_CITIES:
        raise ValueError("expanded U-Net requires exactly 50 source cities / 150 scenes")
    if (
        len(validation_entries) != EXPECTED_VALIDATION_SCENES
        or len(validation_years) != EXPECTED_VALIDATION_CITIES
    ):
        raise ValueError("expanded U-Net requires exactly 4 validation cities / 12 scenes")
    if source_payload.get("scene_count") != len(source_entries):
        raise ValueError("source manifest scene_count does not match its scene list")
    if source_payload.get("city_count") != len(source_years):
        raise ValueError("source manifest city_count does not match its scene list")
    declared_years = source_payload.get("city_years")
    if not isinstance(declared_years, Mapping) or {
        str(city): sorted(int(year) for year in years)
        for city, years in declared_years.items()
    } != source_years:
        raise ValueError("source city_years differs from the manifest scene declarations")

    frozen_validation = source_payload.get("fixed_validation_manifest")
    if (
        not isinstance(frozen_validation, Mapping)
        or frozen_validation.get("sha256") != validation_sha
        or frozen_validation.get("scene_count") != len(validation_entries)
        or set(map(str, frozen_validation.get("cities", []))) != set(validation_years)
    ):
        raise ValueError("expanded source does not bind the supplied fixed validation manifest")
    reject_sealed(str(frozen_validation.get("path", "")))

    if set(source_years).intersection(validation_years):
        raise ValueError("source and validation cities overlap")
    if source_validation_overlap(source_entries, validation_entries):
        raise ValueError("source and validation scene files overlap")
    source_scene_hashes = {entry.sha256 for entry in source_entries}
    validation_scene_hashes = {entry.sha256 for entry in validation_entries}
    if None in source_scene_hashes or None in validation_scene_hashes:
        raise ValueError("scientific manifests must hash every scene")
    if source_scene_hashes.intersection(validation_scene_hashes):
        raise ValueError("source and validation scene content hashes overlap")

    return VerifiedInputs(
        plan_sha,
        source_sha,
        validation_sha,
        source_entries,
        validation_entries,
        source_years,
        validation_years,
        source_payload,
        amendment_sha,
    )


def validate_training_contract(
    source_manifest: str | os.PathLike[str] = DEFAULT_SOURCE,
    validation_manifest: str | os.PathLike[str] = DEFAULT_VALIDATION,
    plan_path: str | os.PathLike[str] = FROZEN_PLAN,
) -> dict[str, Any]:
    """Validate frozen hashes, counts, disjointness, and dynamic three-year groups."""
    return _verified_inputs(source_manifest, validation_manifest, plan_path).public_record()


def load_source_normalization(
    source_manifest: str | os.PathLike[str],
    *,
    expected_source_scenes: int = EXPECTED_SOURCE_SCENES,
) -> tuple[V2Normalization, dict[str, Any]]:
    """Load the source-pinned normalization without reading validation data."""
    source_path = reject_sealed(source_manifest)
    source = _read_json(source_path)
    descriptor = source.get("normalization")
    if not isinstance(descriptor, Mapping):
        raise ValueError("source manifest has no normalization descriptor")
    if (
        descriptor.get("scope") != "source_only_valid_and_eligible_pixels"
        or descriptor.get("implementation") != "v2_data.fit_normalization"
    ):
        raise ValueError("normalization is not the frozen source-only v2_data fit")
    relative = Path(str(descriptor.get("file", "")))
    if relative.is_absolute() or relative.name != str(relative) or not relative.name:
        raise ValueError("source normalization must be one relative filename")
    normalization_path = reject_sealed(source_path.parent / relative)
    normalization_sha = sha256_file(normalization_path)
    if normalization_sha != descriptor.get("sha256"):
        raise ValueError("source normalization SHA-256 differs from its manifest")
    raw = _read_json(normalization_path)
    if raw.get("fit_scope") != descriptor.get("scope"):
        raise ValueError("source normalization fit scope mismatch")
    normalization = V2Normalization.from_dict(raw)
    if (
        normalization.n_source_scenes != expected_source_scenes
        or raw.get("source_scene_count") != expected_source_scenes
        or raw.get("source_city_count") != EXPECTED_SOURCE_CITIES
    ):
        raise ValueError("source normalization provenance counts are invalid")
    public = dict(raw)
    public.update(
        {
            "schema": "uhi-cdc-expanded-source-normalization-v1",
            "source_manifest_sha256": sha256_file(source_path),
            "input_normalization_sha256": normalization_sha,
            "validation_included": False,
        }
    )
    return normalization, public


def seed_everything(seed: int = SEED) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_residual_unet() -> ResidualUNet:
    return ResidualUNet(in_channels=INPUT_CHANNELS, width=WIDTH)


def masked_huber(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target, and mask must have identical NCHW shape")
    selected = mask.bool()
    if not torch.any(selected):
        raise ValueError("masked Huber batch contains no valid & eligible pixels")
    return torch.nn.functional.smooth_l1_loss(
        prediction[selected], target[selected], beta=1.0, reduction="mean"
    )


def _amp_context(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if enabled and device.type == "cuda"
        else nullcontext()
    )


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _tile_origins(length: int, tile_size: int = PATCH_SIZE, stride: int = 48) -> list[int]:
    if length < tile_size or not 0 < stride <= tile_size:
        raise ValueError("invalid validation tile geometry")
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def predict_normalized_residual(
    model: nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
    amp_enabled: bool = True,
) -> np.ndarray:
    if inputs.ndim != 3 or inputs.shape[0] != INPUT_CHANNELS:
        raise ValueError("validation input must have shape [8,H,W]")
    height, width = inputs.shape[-2:]
    coordinates = [
        (row, column)
        for row in _tile_origins(height)
        for column in _tile_origins(width)
    ]
    total = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.uint16)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), batch_size):
            group = coordinates[start : start + batch_size]
            tiles = np.stack(
                [
                    inputs[:, row : row + PATCH_SIZE, column : column + PATCH_SIZE]
                    for row, column in group
                ]
            )
            tensor = torch.from_numpy(tiles).to(device=device, dtype=torch.float32)
            with _amp_context(device, amp_enabled):
                prediction = model(tensor)
            for values, (row, column) in zip(prediction[:, 0].float().cpu().numpy(), group):
                total[row : row + PATCH_SIZE, column : column + PATCH_SIZE] += values
                counts[row : row + PATCH_SIZE, column : column + PATCH_SIZE] += 1
    if np.any(counts == 0):
        raise RuntimeError("validation tiling left pixels uncovered")
    return (total / counts).astype(np.float32)


def evaluate_validation(
    model: nn.Module,
    entries: Sequence[SceneEntry],
    normalization: V2Normalization,
    device: torch.device,
    *,
    amp_enabled: bool = True,
) -> dict[str, Any]:
    if evaluate_field is None or macro_city_aggregate is None:
        raise RuntimeError("scientific validation requires the repository metrics dependencies")
    if not entries or any(entry.role != "validation" for entry in entries):
        raise ValueError("validation evaluation requires only validation-role scenes")
    records: list[dict[str, Any]] = []
    metric_keys: list[str] | None = None
    for entry in entries:
        scene = load_scene(entry)
        inputs, _, mask = encode_scene(scene, normalization)
        predicted_normalized = predict_normalized_residual(
            model, inputs, device, amp_enabled=amp_enabled
        )
        predicted_lst = scene["lowres_lst"] + decode_residual(
            predicted_normalized, normalization
        )
        values = evaluate_field(
            scene["target_lst"], predicted_lst, mask=mask, score=predicted_lst
        )
        if metric_keys is None:
            metric_keys = sorted(key for key in values if not key.startswith("n_"))
        records.append(
            {"city": entry.city, "year": entry.year, "scene_id": entry.scene_id, **values}
        )
    aggregate = macro_city_aggregate(records, metric_keys=metric_keys)
    return {"scenes": records, **aggregate}


def select_field_hotspot_checkpoint(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Apply field eligibility, then joint q90 hotspot score and frozen ties."""
    if not records:
        raise ValueError("cannot select a checkpoint without validation records")
    rows: list[dict[str, float | int]] = []
    for record in records:
        macro = record.get("validation", {}).get("macro", {})
        try:
            row: dict[str, float | int] = {
                "step": int(record["step"]),
                "rmse": float(macro["rmse"]),
                "r2": float(macro["r2"]),
                "auprc_q90": float(macro["auprc_q90"]),
                "top_iou_q90": float(macro["top_iou_q90"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("validation record lacks a frozen selection metric") from exc
        if not all(math.isfinite(float(value)) for key, value in row.items() if key != "step"):
            raise ValueError("validation selection metrics must be finite")
        row["hotspot_score"] = 0.5 * (
            float(row["auprc_q90"]) + float(row["top_iou_q90"])
        )
        rows.append(row)
    minimum_rmse = min(float(row["rmse"]) for row in rows)
    maximum_r2 = max(float(row["r2"]) for row in rows)
    tolerance = max(0.10, 0.02 * minimum_rmse)
    eligible = [
        row
        for row in rows
        if float(row["rmse"]) <= minimum_rmse + tolerance
        and float(row["r2"]) >= maximum_r2 - 0.01
    ]
    selected = min(
        eligible,
        key=lambda row: (-float(row["hotspot_score"]), float(row["rmse"]), int(row["step"])),
    )
    return {
        **selected,
        "minimum_rmse": minimum_rmse,
        "maximum_r2": maximum_r2,
        "rmse_tolerance": tolerance,
        "eligible_checkpoint_count": len(eligible),
    }


def runtime_identity(device: torch.device | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device is not None:
        result["device"] = str(device)
        if device.type == "cuda" and torch.cuda.is_available():
            result["cuda_device_name"] = torch.cuda.get_device_name(device)
            result["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def code_identity() -> dict[str, str]:
    files = {
        "trainer": Path(__file__).resolve(),
        "v2_data": Path(data_module.__file__).resolve(),
        "models": Path(models_module.__file__).resolve(),
        "metrics": (
            Path(metrics_module.__file__).resolve()
            if metrics_module is not None
            else WORKSPACE / "code/metrics.py"
        ),
    }
    return {f"{name}_sha256": sha256_file(path) for name, path in files.items()}


def architecture_hash(parameters: int, code_hashes: Mapping[str, str]) -> str:
    record = {
        "class": "code.models.ResidualUNet",
        "in_channels": INPUT_CHANNELS,
        "width": WIDTH,
        "parameters": int(parameters),
        "models_sha256": code_hashes["models_sha256"],
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_finite_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_output_contract(output: str | os.PathLike[str]) -> None:
    root = reject_sealed(output)
    if not root.is_dir():
        raise ValueError("completed run output is not a directory")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    subdirectories = [path.name for path in root.iterdir() if path.is_dir()]
    if actual != OUTPUT_FILES or subdirectories:
        raise ValueError(
            f"expanded U-Net output must contain only {sorted(OUTPUT_FILES)}; "
            f"files={sorted(actual)}, directories={sorted(subdirectories)}"
        )


def cuda_single_step_smoke(device: str | torch.device = "cuda") -> dict[str, Any]:
    """One exact-shape AMP update; intended for the Windows CUDA runtime gate."""
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the expanded U-Net smoke requires an available CUDA device")
    seed_everything(SEED)
    model = build_residual_unet().to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = _grad_scaler(True)
    generator = torch.Generator(device=selected_device)
    generator.manual_seed(SEED)
    inputs = torch.randn(
        BATCH_SIZE,
        INPUT_CHANNELS,
        PATCH_SIZE,
        PATCH_SIZE,
        device=selected_device,
        generator=generator,
    )
    target = torch.randn(
        BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE, device=selected_device, generator=generator
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(selected_device, True):
        prediction = model(inputs)
        loss = masked_huber(prediction, target, mask)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(selected_device)
    result = {
        "ok": bool(math.isfinite(float(loss.detach().cpu()))),
        "loss": float(loss.detach().cpu()),
        "input_shape": list(inputs.shape),
        "output_shape": list(prediction.shape),
        "model": "ResidualUNet",
        "width": WIDTH,
        "optimizer": "AdamW",
        "batch_size": BATCH_SIZE,
        "patch_size": PATCH_SIZE,
        "loss_name": "masked Huber",
        "amp_enabled": True,
        "seed": SEED,
        "runtime": runtime_identity(selected_device),
        "code_hashes": code_identity(),
        "sealed_test_opened": False,
    }
    if not result["ok"]:
        raise RuntimeError("CUDA AMP single-step smoke produced a non-finite loss")
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = reject_sealed(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing training output: {output}")
    if not 1 <= int(args.updates) <= MAXIMUM_UPDATES:
        raise ValueError(f"updates must be in [1,{MAXIMUM_UPDATES}]")
    # Freeze scientific identity before constructing a device/model/optimizer.
    verified = _verified_inputs(args.source_manifest, args.validation_manifest, args.plan)
    normalization, normalization_record = load_source_normalization(args.source_manifest)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("expanded residual U-Net training requires CUDA AMP")

    seed_everything(SEED)
    dataset = V2PatchDataset(
        verified.source_entries,
        normalization,
        stride=PATCH_SIZE,
        min_mask_fraction=MINIMUM_PATCH_MASK_FRACTION,
        augment=True,
        seed=SEED,
    )
    sampler = dataset.make_sampler(int(args.updates) * BATCH_SIZE, SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=int(args.workers),
        pin_memory=True,
        drop_last=True,
    )
    model = build_residual_unet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = _grad_scaler(True)
    hashes = code_identity()
    runtime = runtime_identity(device)
    parameters = trainable_parameter_count(model)
    model_hash = architecture_hash(parameters, hashes)

    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    rolling_loss = 0.0
    rolling_count = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="expanded_unet_", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        checkpoints = temporary_root / "checkpoints"
        checkpoints.mkdir()
        model.train()
        for step, batch in enumerate(loader, start=1):
            if step > int(args.updates):
                break
            inputs = batch["input"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            mask = batch["mask"].to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, True):
                prediction = model(inputs)
                loss = masked_huber(prediction, target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            rolling_loss += float(loss.detach().cpu())
            rolling_count += 1

            validate_now = step % VALIDATION_INTERVAL == 0 or step == int(args.updates)
            if validate_now:
                validation = evaluate_validation(
                    model,
                    verified.validation_entries,
                    normalization,
                    device,
                    amp_enabled=True,
                )
                record = {
                    "step": step,
                    "train_loss": rolling_loss / rolling_count,
                    "validation": validation,
                }
                records.append(record)
                torch.save(model.state_dict(), checkpoints / f"step_{step:07d}.pt")
                rolling_loss = 0.0
                rolling_count = 0
                model.train()

        if not records or int(records[-1]["step"]) != int(args.updates):
            raise RuntimeError("training ended before the requested update budget")
        torch.cuda.synchronize(device)
        wall_seconds = time.monotonic() - started
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        selection = select_field_hotspot_checkpoint(records)
        selected_path = checkpoints / f"step_{int(selection['step']):07d}.pt"
        try:
            selected_state = torch.load(selected_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("this PyTorch build lacks safe weights-only loading") from exc

        stage = temporary_root / "final"
        stage.mkdir()
        _write_json(stage / "normalization.json", normalization_record)
        normalization_output_sha = sha256_file(stage / "normalization.json")
        checkpoint = {
            "schema": "uhi-cdc-expanded-residual-unet-checkpoint-v1",
            "model": selected_state,
            "step": int(selection["step"]),
            "model_hash": model_hash,
            "model_spec": {
                "class": "ResidualUNet",
                "in_channels": INPUT_CHANNELS,
                "width": WIDTH,
                "parameters": parameters,
            },
            "selection": selection,
            "normalization_sha256": normalization_output_sha,
            "contract": verified.public_record(),
            "runtime": runtime,
            "code_hashes": hashes,
            "sealed_test_opened": False,
        }
        torch.save(checkpoint, stage / "best.pt")
        best_sha = sha256_file(stage / "best.pt")
        history = {
            "schema": "uhi-cdc-expanded-residual-unet-history-v1",
            "status": "complete",
            "run_mode": (
                "frozen_full_budget" if int(args.updates) == MAXIMUM_UPDATES else "bounded_smoke_budget"
            ),
            "contract": verified.public_record(),
            "configuration": {
                "input_order": [
                    "interpolated_lowres_lst",
                    "interpolation_weight120",
                    "optical_0",
                    "optical_1",
                    "optical_2",
                    "optical_3",
                    "optical_4",
                    "optical_5",
                ],
                "target": "normalized signed target_lst-minus-lowres_lst residual",
                "mask": "valid & eligible",
                "model": "ResidualUNet",
                "width": WIDTH,
                "loss": "masked Huber beta=1.0",
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "patch_size": PATCH_SIZE,
                "minimum_patch_mask_fraction": MINIMUM_PATCH_MASK_FRACTION,
                "augmentation": "deterministic D4",
                "maximum_updates": MAXIMUM_UPDATES,
                "executed_updates": int(args.updates),
                "validation_interval_updates": VALIDATION_INTERVAL,
                "seed": SEED,
                "cuda_amp": True,
                "normalization_fit": "source_only_valid_and_eligible_pixels",
            },
            "model_hash": model_hash,
            "parameters": parameters,
            "records": records,
            "checkpoint_selection": selection,
            "resource_accounting": {
                "optimizer_updates": int(args.updates),
                "processed_patches": int(args.updates) * BATCH_SIZE,
                "training_wall_seconds": wall_seconds,
                "peak_gpu_memory_bytes": peak_memory,
            },
            "runtime": runtime,
            "code_hashes": hashes,
            "artifact_hashes": {
                "normalization.json": normalization_output_sha,
                "best.pt": best_sha,
            },
            "output_allowlist": sorted(OUTPUT_FILES),
            "sealed_test_opened": False,
        }
        _write_json(stage / "history.json", history)
        assert_output_contract(stage)
        if output.exists():
            raise FileExistsError(f"training output appeared during the run: {output}")
        os.replace(stage, output)

    assert_output_contract(output)
    return {
        "status": "complete",
        "output": str(output),
        "selected_step": int(selection["step"]),
        "optimizer_updates": int(args.updates),
        "best_sha256": best_sha,
        "sealed_test_opened": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=FROZEN_PLAN)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--updates", type=int, default=MAXIMUM_UPDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--cuda-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        contract = validate_training_contract(
            args.source_manifest, args.validation_manifest, args.plan
        )
        _, normalization = load_source_normalization(args.source_manifest)
        print(json.dumps({"status": "GO", "contract": contract, "normalization": normalization}, indent=2))
        return 0
    if args.cuda_smoke:
        contract = validate_training_contract(
            args.source_manifest, args.validation_manifest, args.plan
        )
        load_source_normalization(args.source_manifest)
        result = cuda_single_step_smoke(args.device)
        print(json.dumps({"status": "GO", "contract": contract, "smoke": result}, indent=2))
        return 0
    result = train(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
