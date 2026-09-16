#!/usr/bin/env python3
"""Independent CPU verifier for the public G246 eight-hour experiment.

Import ``score_fields`` and ``projection_diagnostics`` in a training runner.
The command-line receipt additionally verifies saved float64 fields and a
loadable deployment state. No locked-test path is accepted or resolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REGIONS = ("us", "china", "europe")
METRICS = ("rmse_k", "mae_k", "true_hotspot_mae_q90_k", "auprc_q90", "iou_q90")
ROOT = Path(__file__).resolve().parents[1]


def load_deployment(source: Any) -> dict[str, Any]:
    """Weights-only load with the sole historical TorchVersion string exception."""
    import torch
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("deployment must be a dictionary")

    def plain(value: Any) -> Any:
        if isinstance(value, TorchVersion):
            return str(value)
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        if isinstance(value, tuple):
            return tuple(plain(item) for item in value)
        return value

    if "config" in payload:
        payload["config"] = plain(payload["config"])
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [N,H,W] or [N,1,H,W]")
    return array


def _binary(value: Any, name: str) -> np.ndarray:
    array = _fields(value, name)
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be finite and binary")
    return array.astype(bool, copy=False)


def _scene_metrics(prediction: np.ndarray, target: np.ndarray,
                   mask: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=np.float64)[mask]
    truth = np.asarray(target, dtype=np.float64)[mask]
    if not pred.size or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("each formal scene must contain finite predictions and targets")
    error = pred - truth
    count = max(1, math.ceil(0.10 * pred.size))
    target_order = np.argsort(-truth, kind="stable")
    prediction_order = np.argsort(-pred, kind="stable")
    target_hot = np.zeros(pred.size, dtype=bool)
    predicted_hot = np.zeros(pred.size, dtype=bool)
    target_hot[target_order[:count]] = True
    predicted_hot[prediction_order[:count]] = True
    ranked_hot = target_hot[prediction_order]
    precision = np.cumsum(ranked_hot) / np.arange(1, pred.size + 1)
    intersection = np.count_nonzero(target_hot & predicted_hot)
    union = np.count_nonzero(target_hot | predicted_hot)
    return {
        "rmse_k": float(np.sqrt(np.mean(error * error))),
        "mae_k": float(np.mean(np.abs(error))),
        "true_hotspot_mae_q90_k": float(np.mean(np.abs(error[target_hot]))),
        "auprc_q90": float(np.sum(precision[ranked_hot]) / count),
        "iou_q90": float(intersection / union),
        "n_pixels": int(pred.size),
    }


def _means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in METRICS}


def score_fields(predictions: Any, target: Any, formal: Any,
                 records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the unchanged scene -> city -> region -> equal-region metric."""
    predictions = _fields(predictions, "predictions")
    target = _fields(target, "target")
    formal = _binary(formal, "formal")
    if predictions.shape != target.shape or formal.shape != target.shape:
        raise ValueError("prediction, target and formal geometry differs")
    if len(records) != len(predictions):
        raise ValueError("scene metadata and field counts differ")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("saved predictions contain nonfinite values, including outside formal mask")
    per_scene: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        scene_id, city, region = (str(record[key]) for key in ("scene_id", "city", "region"))
        if not scene_id or not city or region not in REGIONS or scene_id in per_scene:
            raise ValueError("scene identity is empty, duplicated, or has an unknown region")
        row = {"city": city, "region": region,
               **_scene_metrics(predictions[index], target[index], formal[index])}
        per_scene[scene_id] = row
        grouped.setdefault((region, city), []).append(row)
    per_city = {
        f"{region}/{city}": {"city": city, "region": region,
                              "scene_count": len(rows), **_means(rows)}
        for (region, city), rows in sorted(grouped.items())
    }
    per_region = {}
    for region in REGIONS:
        rows = [row for row in per_city.values() if row["region"] == region]
        if not rows:
            raise ValueError(f"formal view has no cities in {region}")
        per_region[region] = {"city_count": len(rows), **_means(rows)}
    return {
        "aggregation": "scene_to_city_to_region_equal_region",
        "equal_region": _means(list(per_region.values())),
        "per_region": per_region, "per_city": per_city, "per_scene": per_scene,
        "scene_count": len(per_scene), "city_count": len(per_city),
        "formal_pixel_count": int(formal.sum()),
    }


def projection_diagnostics(predictions: Any, coarse: Any, support: Any,
                           tolerance_k: float = 1e-10) -> dict[str, Any]:
    """Recompute actual-support O-parent means with float64 arithmetic."""
    pred = _fields(predictions, "predictions").astype(np.float64, copy=False)
    coarse = _fields(coarse, "coarse").astype(np.float64, copy=False)
    support = _binary(support, "support")
    n, height, width = pred.shape
    if support.shape != pred.shape or height % 4 or width % 4:
        raise ValueError("support/fine geometry is incompatible with the 4x operator")
    if coarse.shape != (n, height // 4, width // 4):
        raise ValueError("coarse/fine geometry differs from 4x")
    if not np.all(np.isfinite(pred)) or np.any(np.isinf(coarse)):
        raise ValueError("prediction must be finite and missing coarse must use NaN")
    shape = (n, height // 4, 4, width // 4, 4)
    counts = support.reshape(shape).sum(axis=(2, 4))
    sums = np.where(support, pred, 0.0).reshape(shape).sum(axis=(2, 4), dtype=np.float64)
    observed = np.isfinite(coarse)
    if np.any(observed & (counts == 0)):
        raise ValueError("observed coarse parent has zero physical support")
    means = sums / np.maximum(counts, 1)
    error = np.zeros_like(coarse)
    error[observed] = means[observed] - coarse[observed]
    strata = {"full_O": observed & (counts == 16),
              "partial_O": observed & (counts > 0) & (counts < 16),
              "singleton_O": observed & (counts == 1),
              "all_O": observed,
              "U": ~observed & (counts > 0), "Z": counts == 0}
    report: dict[str, Any] = {}
    for name, mask in strata.items():
        item: dict[str, Any] = {"parent_count": int(mask.sum())}
        if name.endswith("O"):
            values = error[mask]
            item["max_abs_closure_k"] = float(np.max(np.abs(values))) if values.size else 0.0
            item["rms_closure_k"] = float(np.sqrt(np.mean(values * values))) if values.size else 0.0
        report[name] = item
    report["tolerance_k"] = float(tolerance_k)
    report["closure_pass"] = bool(report["all_O"]["max_abs_closure_k"] <= tolerance_k)
    report["unsupported_max_abs_k"] = float(np.max(np.abs(pred[~support]))) if np.any(~support) else 0.0
    return report


def _records(metadata: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    if metadata.get("role") == "validation" and isinstance(metadata.get("scenes"), list):
        return metadata["scenes"]
    split = metadata.get("validation")
    if isinstance(split, list):
        return split
    if isinstance(split, Mapping):
        for key in ("records", "scenes"):
            if isinstance(split.get(key), list):
                return split[key]
    for key in ("records", "scenes", "splits"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            value = value.get("validation")
            if isinstance(value, list):
                return value
            if isinstance(value, Mapping):
                for nested in ("records", "scenes"):
                    if isinstance(value.get(nested), list):
                        return value[nested]
    raise ValueError("metadata does not contain an ordered validation scene list")


def check_hourly_cache(root: Path, expected_sha256: str, base_manifest_sha256: str,
                       expected_scene_ids: Sequence[str],
                       base_metadata_sha256: str) -> dict[str, Any]:
    """Bind all public hourly validation tokens before any label scoring."""
    from build_g246_8h_hourly_cache import HourlyCache

    root = root.resolve() if root.is_absolute() else (ROOT / root).resolve()
    if "locked" in str(root).lower():
        raise ValueError("hourly cache must use a public path")
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("hourly manifest differs from the deployment binding")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("base_cache_manifest", {}).get("sha256") != base_manifest_sha256:
        raise ValueError("hourly and base caches are bound to different manifests")
    if set(manifest.get("roles", {})) != {"fit", "validation"}:
        raise ValueError("hourly cache requires the public fit/validation roles")
    normalization = manifest.get("normalization", {})
    if (normalization.get("reproduced_from_raw_fit_rows") is not True
            or normalization.get("source_document", {}).get("fit_scene_count") != 603
            or manifest.get("identity_metadata_not_model_inputs") is not True):
        raise ValueError("hourly cache must bind reproduced Fit603 normalization and predictor-only inputs")
    cache = HourlyCache(root, "validation", expected_scene_ids=expected_scene_ids, verify_hash=True)
    metadata = cache.metadata
    if (metadata.get("role") != "validation" or metadata.get("locked_test_opened") is not False
            or metadata.get("target_arrays_opened") is not False
            or metadata.get("identity_metadata_not_model_inputs") is not True
            or metadata.get("base_metadata_sha256") != base_metadata_sha256):
        raise ValueError("hourly validation metadata has an invalid public scene binding")
    records = metadata["scenes"]
    if any(row.get("index") != index or row.get("view_role") != "validation"
           for index, row in enumerate(records)):
        raise ValueError("hourly validation scene order or role is invalid")
    detail = manifest["roles"]["validation"]
    expected = detail["tokens"]
    tokens_path = root / "validation" / "tokens.npy"
    if (detail.get("scene_count") != 45 or list(cache.array.shape) != expected.get("shape")
            or str(cache.array.dtype) != expected.get("dtype")
            or tokens_path.stat().st_size != expected.get("bytes")
            or not np.all(np.isfinite(cache.array))):
        raise ValueError("hourly validation tokens differ from their manifest or contain nonfinite values")
    return {"root": str(root), "manifest_sha256": expected_sha256,
            "validation_tokens_sha256": expected["sha256"],
            "validation_metadata_sha256": detail["metadata_sha256"],
            "base_metadata_sha256": base_metadata_sha256,
            "normalization_sha256": normalization["sha256"],
            "normalization_fit_scene_count": 603,
            "scene_count": 45, "shape": list(cache.array.shape),
            "dtype": str(cache.array.dtype), "finite": True,
            "identity_metadata_not_model_inputs": True, "locked_test_opened": False}


def check_emissivity_cache(root: Path, expected_sha256: str, base_manifest_sha256: str,
                           expected_scene_ids: Sequence[str],
                           base_metadata_sha256: str) -> dict[str, Any]:
    """Verify the approved joint-masked four-channel ancillary input cache."""
    from build_g246_8h_emissivity_cache import CHANNELS, SCHEMA

    root = root.resolve() if root.is_absolute() else (ROOT / root).resolve()
    if "locked" in str(root).lower():
        raise ValueError("emissivity cache must use a public path")
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("emissivity manifest differs from the deployment binding")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "complete"
            or manifest.get("locked_test_opened") is not False
            or manifest.get("target_arrays_opened") is not False
            or manifest.get("target_masks_opened") is not False
            or manifest.get("channel_names") != CHANNELS
            or manifest.get("base_manifest_sha256") != base_manifest_sha256
            or set(manifest.get("roles", {})) != {"fit", "validation"}
            or manifest.get("original_qa_water_fill_or_separate_nodata_masks_not_features") is not True):
        raise ValueError("emissivity cache has an invalid approved-input contract")
    audit_path = Path(manifest["admissibility_receipt"])
    audit_path = audit_path.resolve() if audit_path.is_absolute() else (ROOT / audit_path).resolve()
    if "locked" in str(audit_path).lower() or sha256_file(audit_path) != manifest["admissibility_receipt_sha256"]:
        raise ValueError("emissivity admissibility audit binding changed")
    audit = json.loads(audit_path.read_text())
    if (audit.get("pass") is not True or audit.get("joint_transform_invariance_pass") is not True
            or audit.get("postprocessing_invariance_pass") is not True
            or audit.get("locked_test_opened") is not False):
        raise ValueError("emissivity input audit did not pass")
    detail = manifest["roles"]["validation"]
    metadata_path = root / "validation" / "metadata.json"
    feature_path = root / "validation" / "features.npy"
    if (detail.get("scene_count") != 45 or detail.get("base_metadata_sha256") != base_metadata_sha256
            or sha256_file(metadata_path) != detail["metadata_sha256"]
            or sha256_file(feature_path) != detail["features_sha256"]):
        raise ValueError("emissivity validation files differ from the frozen manifest")
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("schema") != SCHEMA or metadata.get("role") != "validation"
            or [row["scene_id"] for row in metadata["scenes"]] != list(expected_scene_ids)):
        raise ValueError("emissivity validation scene order or schema changed")
    array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if (array.shape != (45, 4, 160, 160) or array.dtype != np.float32
            or list(array.shape) != detail["features_shape"] or str(array.dtype) != detail["features_dtype"]
            or feature_path.stat().st_size != detail["features_bytes"] or not np.all(np.isfinite(array))):
        raise ValueError("emissivity validation tensor differs or contains nonfinite values")
    coverage = array[:, 3]
    if np.any((coverage < 0) | (coverage > 1)) or np.any(coverage * 16 != np.rint(coverage * 16)):
        raise ValueError("emissivity coverage does not represent an actual 4x4 common domain")
    return {"root": str(root), "manifest_sha256": expected_sha256,
            "validation_features_sha256": detail["features_sha256"],
            "validation_metadata_sha256": detail["metadata_sha256"],
            "admissibility_receipt_sha256": manifest["admissibility_receipt_sha256"],
            "admissibility_pass": True, "channel_names": CHANNELS,
            "scene_count": 45, "shape": list(array.shape), "dtype": str(array.dtype),
            "finite": True, "locked_test_opened": False}


def check_optical_cache(root: Path, expected_sha256: str,
                        expected_scene_ids: Sequence[str]) -> dict[str, Any]:
    from build_g246_8h_optical_detail import OpticalDetailCache

    root = root.resolve() if root.is_absolute() else (ROOT / root).resolve()
    if "locked" in str(root).lower() or sha256_file(root / "manifest.json") != expected_sha256:
        raise ValueError("optical cache path or deployment manifest binding differs")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("locked_test_opened") is not False or manifest.get("target_arrays_opened") is not False:
        raise ValueError("optical cache has an invalid predictor-only contract")
    cache = OpticalDetailCache(root, "validation", expected_scene_ids=list(expected_scene_ids), verify_hash=True)
    if cache.array.shape != (45, 128, 160, 160):
        raise ValueError("optical validation cache must cover all 45 scenes")
    detail = manifest["roles"]["validation"]
    return {"root": str(root), "manifest_sha256": expected_sha256,
            "validation_array_sha256": detail["array_sha256"],
            "validation_metadata_sha256": detail["metadata_sha256"], "scene_count": 45}


def check_historical_cache(root: Path, expected_sha256: str, base_manifest_sha256: str,
                           expected_scene_ids: Sequence[str], base_metadata_sha256: str,
                           expected_records: Sequence[Mapping[str, Any]], *, family: str | None = None) -> dict[str, Any]:
    """Check frozen historical source identities and validation predictor fields."""
    from g246_8h_historical_contract import FAMILIES, SIX_FAMILIES, check_cache

    if family in ("historical_recent_innovation_emissivity_r6a_nine",
                  "historical_recent_refinement_emissivity_r6a_nine"):
        from g246_8h_historical_nine_contract import check_cache as check_nine_cache
        return check_nine_cache(root, expected_sha256, base_manifest_sha256,
                                expected_scene_ids, base_metadata_sha256, expected_records)
    if family in ("historical_recent_innovation_emissivity_r6a_seven",
                  "historical_recent_refinement_emissivity_r6a_seven"):
        from g246_8h_historical_seven_contract import check_cache as check_seven_cache
        return check_seven_cache(root, expected_sha256, base_manifest_sha256,
                                 expected_scene_ids, base_metadata_sha256, expected_records)
    if family in SIX_FAMILIES:
        from g246_8h_historical_six_contract import check_cache as check_six_cache
        return check_six_cache(root, expected_sha256, base_manifest_sha256,
                               expected_scene_ids, base_metadata_sha256, expected_records)
    if family is not None and family not in FAMILIES:
        raise ValueError("unregistered historical deployment family")

    return check_cache(root, expected_sha256, base_manifest_sha256,
                       expected_scene_ids, base_metadata_sha256, expected_records)


def _deployment(path: Path, manifest_sha256: str,
                records: Sequence[Mapping[str, Any]], metadata_sha256: str) -> dict[str, Any]:
    import torch
    from train_g246_8h import create_model

    payload = load_deployment(path)
    if not isinstance(payload, dict):
        raise ValueError("deploy.pt must contain a dictionary")
    state = payload.get("state_dict", payload.get("model_state_dict"))
    if not isinstance(state, dict) or not state:
        raise ValueError("deploy.pt requires state_dict or model_state_dict")
    if payload.get("locked_test_opened") is not False:
        raise ValueError("deployment must explicitly declare locked_test_opened=false")
    if payload.get("cache_manifest_sha256") != manifest_sha256:
        raise ValueError("deployment is bound to a different cache manifest")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item()
           for value in state.values()):
        raise ValueError("deployment state has nontensor or nonfinite entries")
    spec = payload.get("model_spec")
    if not isinstance(spec, dict) or spec.get("family") not in (
            "r6a", "r9", "multiscale", "temporal", "temporal_r6a", "resnet18",
            "optical_r6a", "optical_multiscale", "optical_native_r6a", "hourly_r6a", "emissivity_r6a",
            "optical_emissivity_r6a", "dino_r6a", "historical_r6a", "historical_innovation_r6a",
            "historical_innovation_emissivity_r6a", "historical_innovation_emissivity_r6a_six",
            "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine", "fixed_ensemble"):
        raise ValueError("unsupported deployment model_spec")
    hourly = None
    if spec["family"] == "hourly_r6a":
        config = payload["config"]
        hourly = check_hourly_cache(Path(config["hourly_root"]), config["hourly_manifest_sha256"],
                                    manifest_sha256, [str(row["scene_id"]) for row in records],
                                    metadata_sha256)
    emissivity = None
    if spec["family"] in ("emissivity_r6a", "optical_emissivity_r6a", "historical_innovation_emissivity_r6a",
                          "historical_innovation_emissivity_r6a_six", "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"):
        config = payload["config"]
        emissivity = check_emissivity_cache(Path(config["emissivity_root"]),
            config["emissivity_manifest_sha256"], manifest_sha256,
            [str(row["scene_id"]) for row in records], metadata_sha256)
    optical = None
    if spec["family"].startswith("optical_"):
        config = payload["config"]
        optical = check_optical_cache(Path(config["detail_root"]), config["optical_detail_manifest_sha256"],
                                      [str(row["scene_id"]) for row in records])
    historical = None
    if spec["family"] in ("historical_r6a", "historical_innovation_r6a", "historical_innovation_emissivity_r6a",
                          "historical_innovation_emissivity_r6a_six", "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"):
        config = payload["config"]
        historical = check_historical_cache(Path(config["historical_root"]),
            config["historical_manifest_sha256"], manifest_sha256,
            [str(row["scene_id"]) for row in records], metadata_sha256, records, family=spec["family"])
    from g246_8h_deployment_sources import prepare_dino_source
    source_dependency = prepare_dino_source(payload)
    if spec["family"] == "fixed_ensemble":
        from package_g246_8h_ensemble import build_ensemble
        model = build_ensemble(spec)
    else:
        model = create_model(spec["family"], int(spec["width"]))
    model.load_state_dict(state, strict=True)
    member_audit = None
    if spec["family"] == "fixed_ensemble":
        from package_g246_8h_ensemble import audit_ensemble
        member_audit = audit_ensemble(model, payload)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    actual_shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    state_elements = sum(value.numel() for value in state.values())
    named_shapes = payload.get("named_parameter_shapes")
    if named_shapes != actual_shapes:
        raise ValueError("named parameter shapes do not match the reconstructed model")
    if parameter_count >= 20_000_000:
        raise ValueError("deployment model is not strictly below 20M parameters")
    declared = payload.get("parameter_count")
    if declared is None or int(declared) != parameter_count:
        raise ValueError("declared parameter count differs from reconstructed model")
    return {"sha256": sha256_file(path), "tensor_state_elements": state_elements,
            "parameter_count": parameter_count, "model_spec": spec,
            "parameter_limit_strict": 20_000_000,
            "under_20m": True, "strict_load_pass": True,
            "declared_validation_rmse_k": payload.get("validation_rmse_k"),
            "state_finite": True, "hourly_cache": hourly, "emissivity_cache": emissivity,
            "optical_cache": optical, "historical_cache": historical, "member_parameter_audit": member_audit,
            "source_dependency": source_dependency}


def verify(cache_root: Path, run_root: Path, *, tolerance_k: float = 1e-10) -> dict[str, Any]:
    cache_root, run_root = cache_root.resolve(), run_root.resolve()
    if any("locked" in str(path).lower() for path in (cache_root, run_root)):
        raise ValueError("verifier accepts public cache/run paths only")
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("locked_test_opened") is not False:
        raise ValueError("cache manifest must be complete and explicitly locked-test closed")
    if set(manifest.get("roles", {})) != {"fit", "validation"}:
        raise ValueError("cache roles must be exactly fit and validation")
    manifest_sha = sha256_file(manifest_path)
    metadata_path = cache_root / "validation" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("locked_test_opened") is not False:
        raise ValueError("cache metadata must explicitly declare locked_test_opened=false")
    records = _records(metadata)
    if len(records) != 45:
        raise ValueError("the formal public validation view must contain all 45 scenes")
    detail = manifest["roles"]["validation"]
    if set(detail.get("arrays", {})) != {"fine", "coarse", "support", "context", "target", "formal", "valid"}:
        raise ValueError("validation manifest must bind all seven registered arrays")
    if sha256_file(metadata_path) != detail["metadata_sha256"]:
        raise ValueError("validation scene metadata differs from the cache manifest")
    if any(record.get("index") != index or record.get("view_role") != "validation"
           for index, record in enumerate(records)):
        raise ValueError("validation metadata order or role is invalid")
    prediction_path = run_root / "validation_predictions.npy"
    prediction_sha = sha256_file(prediction_path)
    predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if predictions.dtype != np.float64:
        raise ValueError("validation_predictions.npy must store float64 predictions")
    deployment = _deployment(run_root / "deploy.pt", manifest_sha, records, detail["metadata_sha256"])
    # Predictor fields and deployment hashes are bound before loading labels.
    folder = cache_root / "validation"
    values = {name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("target", "formal", "valid", "coarse", "support")}
    array_hashes = {}
    for name, expected in detail["arrays"].items():
        if name not in ("fine", "coarse", "support", "context", "target", "formal", "valid"):
            raise ValueError("unexpected validation array in cache manifest")
        path = folder / f"{name}.npy"
        array = values.get(name)
        if array is None:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        actual_hash = sha256_file(path)
        if (list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]
                or path.stat().st_size != expected["bytes"] or actual_hash != expected["sha256"]):
            raise ValueError(f"validation array differs from its manifest: {name}")
        array_hashes[name] = actual_hash
    formal, valid = _binary(values["formal"], "formal"), _binary(values["valid"], "valid")
    support = _binary(values["support"], "support")
    if np.any(formal & ~valid) or np.any(formal & ~support):
        raise ValueError("formal mask is not contained in valid and physical support")
    score = score_fields(predictions, values["target"], formal, records)
    if score["city_count"] != 15 or any(row["scene_count"] != 3 for row in score["per_city"].values()):
        raise ValueError("formal validation requires 15 cities with exactly three scenes each")
    if any(row["city_count"] != 5 for row in score["per_region"].values()):
        raise ValueError("formal validation requires five cities in each of three regions")
    projection = projection_diagnostics(predictions, values["coarse"], support, tolerance_k)
    raw_rmse = float(score["equal_region"]["rmse_k"])
    declared_rmse = deployment["declared_validation_rmse_k"]
    if declared_rmse is None or not math.isclose(raw_rmse, float(declared_rmse), rel_tol=0, abs_tol=1e-10):
        raise ValueError("independent RMSE differs from the saved deployment score")
    verified = bool(projection["closure_pass"])
    return {"schema": "g246-8h-independent-verification-v1",
            "cache_root": str(cache_root), "run_root": str(run_root),
            "cache_metadata_sha256": sha256_file(metadata_path),
            "cache_manifest_sha256": manifest_sha,
            "validation_predictions_sha256": prediction_sha,
            "validation_array_sha256": array_hashes,
            "deployment": deployment, "score": score, "projection": projection,
            "public_validation_is_adaptive_development": True,
            "deployment_forward_replayed": False,
            "locked_test_opened": False, "verification_pass": verified,
            "strict_rmse_threshold_k": 0.5,
            "raw_equal_region_rmse_k": raw_rmse,
            "rmse_below_0p5": bool(raw_rmse < 0.5),
            "goal_achieved_on_public_validation": bool(verified and raw_rmse < 0.5)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--closure-tolerance-k", type=float, default=1e-10)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    result = verify(args.cache_root, args.run_root, tolerance_k=args.closure_tolerance_k)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output = args.output or args.run_root / "verification.json"
    output.write_text(encoded)
    print(json.dumps({key: result[key] for key in (
        "verification_pass", "raw_equal_region_rmse_k", "rmse_below_0p5",
        "goal_achieved_on_public_validation")}, allow_nan=False))
    return 0 if result["verification_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
