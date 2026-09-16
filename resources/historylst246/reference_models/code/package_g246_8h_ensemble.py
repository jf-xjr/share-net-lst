#!/usr/bin/env python3
"""Freeze, cold-load and score fixed G246 network ensembles without label fitting.

All member tensors are embedded in deploy.pt. Source files are provenance only:
cold inference needs this deployment plus its registered predictor caches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

if __name__ == "__main__":
    # A long replay must not import a second, newly edited copy at final verify.
    sys.modules.setdefault("package_g246_8h_ensemble", sys.modules[__name__])

from g246_8h_augment import inverse_field, transform_batch
from g246_8h_deployment_sources import prepare_dino_source
from replay_g246_8h import _cache_check, _json, _optical_check, _path, _weight_sha256
from train_g246_8h import Cache, TemporalModel, create_model, forward_batch, initialize, repair_numpy
from verify_g246_8h_result import check_emissivity_cache, check_historical_cache, check_hourly_cache, load_deployment, score_fields, sha256_file, verify


SCHEMA = "g246-8h-ensemble-deploy-v1"
MEMBER_FAMILIES = {"r6a", "r9", "multiscale", "temporal", "temporal_r6a", "resnet18",
                   "optical_r6a", "optical_multiscale", "optical_native_r6a", "hourly_r6a", "emissivity_r6a",
                   "optical_emissivity_r6a", "dino_r6a", "historical_r6a", "historical_innovation_r6a",
                   "historical_innovation_emissivity_r6a", "historical_innovation_emissivity_r6a_six",
                   "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"}
PREDICTORS = ("fine", "coarse", "support", "context")


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _shapes(model: nn.Module) -> dict[str, list[int]]:
    return {name: list(parameter.shape) for name, parameter in model.named_parameters()}


def _member(spec: dict[str, Any]) -> nn.Module:
    if spec.get("family") not in MEMBER_FAMILIES:
        raise ValueError("member must be a registered single-network family")
    return create_model(spec["family"], int(spec["width"]))


class FixedEnsemble(nn.Module):
    """All trainable tensors count toward the joint budget; mixing adds none."""
    def __init__(self, spec: dict[str, Any]):
        super().__init__()
        entries = spec["members"]
        weights = np.asarray([entry["weight"] for entry in entries], np.float64)
        if (len(entries) < 2 or not np.all(np.isfinite(weights)) or np.any(weights <= 0)
                or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("at least two positive predeclared weights must sum to one")
        self.members = nn.ModuleList([_member(entry["model_spec"]) for entry in entries])
        self.register_buffer("fixed_weights", torch.as_tensor(weights.copy()))
        self.tta_d4 = bool(spec["tta_d4"])
        if _count(self) >= 20_000_000:
            raise ValueError("the sum of every member's parameters must be strictly below 20M")

    def forward(self, batches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        if len(batches) != len(self.members):
            raise ValueError("each member requires its registered predictor batch")
        result = None
        for index, (member, batch) in enumerate(zip(self.members, batches)):
            allowed = set(PREDICTORS) | {"detail", "hourly", "emissivity", "history"}
            if set(batch) - allowed or not set(PREDICTORS).issubset(batch):
                raise ValueError("ensemble forward accepts physical predictors only")
            total = None
            for code in range(8) if self.tta_d4 else (0,):
                transformed = transform_batch(batch, code) if self.tta_d4 else batch
                field = forward_batch(member, transformed)
                if self.tta_d4:
                    field = inverse_field(field, code)
                field = field.to(torch.float64)
                total = field if total is None else total + field
            contribution = total / (8 if self.tta_d4 else 1) * self.fixed_weights[index]
            result = contribution if result is None else result + contribution
        return result


def build_ensemble(spec: dict[str, Any]) -> FixedEnsemble:
    if spec.get("family") != "fixed_ensemble":
        raise ValueError("invalid ensemble family")
    return FixedEnsemble(spec)


def _cache_contract(config: dict[str, Any], spec: dict[str, Any], base_sha: str) -> dict[str, Any]:
    root = _path(config["cache"])
    manifest = _cache_check(root, base_sha)
    metadata = json.loads((root / "validation" / "metadata.json").read_text())
    scene_ids = [row["scene_id"] for row in metadata["scenes"]]
    report = {"base_root": str(root), "base_manifest_sha256": base_sha,
              "model_forward_keys": list(PREDICTORS), "detail": None, "hourly": None,
              "emissivity": None, "historical": None}
    if str(spec["family"]).startswith("optical_"):
        report["detail"] = _optical_check(_path(config["detail_root"]),
                                           config["optical_detail_manifest_sha256"])
        report["model_forward_keys"].append("detail")
    if spec["family"] == "hourly_r6a":
        report["hourly"] = check_hourly_cache(_path(config["hourly_root"]),
            config["hourly_manifest_sha256"], base_sha, scene_ids,
            manifest["roles"]["validation"]["metadata_sha256"])
        report["model_forward_keys"].append("hourly")
    if spec["family"] in ("emissivity_r6a", "optical_emissivity_r6a", "historical_innovation_emissivity_r6a",
                          "historical_innovation_emissivity_r6a_six", "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"):
        report["emissivity"] = check_emissivity_cache(_path(config["emissivity_root"]),
            config["emissivity_manifest_sha256"], base_sha, scene_ids,
            manifest["roles"]["validation"]["metadata_sha256"])
        report["model_forward_keys"].append("emissivity")
    if spec["family"] in ("historical_r6a", "historical_innovation_r6a", "historical_innovation_emissivity_r6a",
                          "historical_innovation_emissivity_r6a_six", "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"):
        report["historical"] = check_historical_cache(_path(config["historical_root"]),
            config["historical_manifest_sha256"], base_sha, scene_ids,
            manifest["roles"]["validation"]["metadata_sha256"], metadata["scenes"], family=spec["family"])
        report["model_forward_keys"].append("history")
    report["temporal_predictor_dates"] = 3 if spec["family"].startswith("temporal") else 1
    return report


def audit_ensemble(model: FixedEnsemble, payload: dict[str, Any], *, check_caches: bool = True) -> list[dict[str, Any]]:
    if payload.get("schema") != SCHEMA or payload.get("locked_test_opened") is not False:
        raise ValueError("invalid public ensemble deployment")
    entries = payload["model_spec"]["members"]
    if len(entries) != len(model.members):
        raise ValueError("member count changed")
    if not torch.equal(model.fixed_weights.cpu(), torch.tensor([e["weight"] for e in entries], dtype=torch.float64)):
        raise ValueError("stored mixing weights differ from their frozen specification")
    reports = []
    for member, entry in zip(model.members, entries):
        count, shapes = _count(member), _shapes(member)
        if count != entry["parameter_count"] or shapes != entry["named_parameter_shapes"]:
            raise ValueError("member parameter inventory changed")
        digest = _weight_sha256(member.state_dict())
        if digest != entry["weights_content_sha256"]:
            raise ValueError("embedded member tensors differ from their source snapshot")
        report = {"model_spec": entry["model_spec"], "weight": entry["weight"],
                  "parameter_count": count, "weights_content_sha256": digest,
                  "source_checkpoint_sha256": entry["source_checkpoint_sha256"]}
        if check_caches:
            report["predictor_contract"] = _cache_contract(entry["config"], entry["model_spec"],
                                                            payload["cache_manifest_sha256"])
        reports.append(report)
    if sum(row["parameter_count"] for row in reports) != _count(model):
        raise ValueError("ensemble contains uncounted parameters")
    return reports


def freeze(member_paths: list[Path], weights: list[float] | None, *, tta_d4: bool) -> dict[str, Any]:
    if len(member_paths) < 2:
        raise ValueError("at least two members are required")
    weights = weights if weights is not None else [1.0 / len(member_paths)] * len(member_paths)
    if len(weights) != len(member_paths):
        raise ValueError("one fixed weight is required per member")
    entries, states, cache_sha, cache_root = [], [], None, None
    dependencies = {}
    for path, weight in zip(member_paths, weights):
        path = _path(path)
        path = path / "deploy.pt" if path.is_dir() else path
        source_bytes = path.read_bytes()
        payload = load_deployment(io.BytesIO(source_bytes))
        if payload.get("schema") != "g246-8h-deploy-v1" or payload.get("locked_test_opened") is not False:
            raise ValueError("each source must be a public g246-8h-deploy-v1 checkpoint")
        if cache_sha is not None and payload["cache_manifest_sha256"] != cache_sha:
            raise ValueError("ensemble members must share the exact same base cache")
        cache_sha = payload["cache_manifest_sha256"]
        cache_root = str(_path(payload["config"]["cache"]))
        prepare_dino_source(payload)
        dependencies.update(payload.get("source_dependencies", {}))
        member = _member(payload["model_spec"])
        member.load_state_dict(payload["state_dict"], strict=True)
        if _count(member) != payload["parameter_count"] or _shapes(member) != payload["named_parameter_shapes"]:
            raise ValueError("source parameter declaration differs from the actual model")
        if any(not torch.isfinite(value).all().item() for value in member.state_dict().values()):
            raise ValueError("source has nonfinite model state")
        entries.append({"model_spec": payload["model_spec"], "weight": float(weight),
            "parameter_count": _count(member), "named_parameter_shapes": _shapes(member),
            "config": payload["config"], "source_path": str(path),
            "source_checkpoint_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "weights_content_sha256": _weight_sha256(member.state_dict()),
            "source_selected_update": payload.get("selected_update"),
            "source_selected_weights": payload.get("selected_weights"),
            "source_declared_validation_rmse_k": payload.get("validation_rmse_k")})
        states.append(member.state_dict())
    spec = {"family": "fixed_ensemble", "members": entries, "tta_d4": bool(tta_d4),
            "weight_policy": "predeclared_constants_not_fitted_to_validation_labels"}
    model = build_ensemble(spec)
    for member, state in zip(model.members, states):
        member.load_state_dict(state, strict=True)
    result = {"schema": SCHEMA, "model_spec": spec, "state_dict": model.state_dict(),
        "parameter_count": _count(model), "named_parameter_shapes": _shapes(model),
        "config": {"cache": cache_root, "amp": False}, "cache_manifest_sha256": cache_sha,
        "locked_test_opened": False, "validation_rmse_k": None,
        "ensemble_weights_fitted_to_validation": False, "tta_additional_parameters": 0,
        "source_dependencies": dependencies}
    audit_ensemble(model, result)
    return result


def export_legacy(checkpoint: Path, family: str, cache: Path, expected_sha: str) -> dict[str, Any]:
    checkpoint, cache = _path(checkpoint), _path(cache)
    if family not in ("r6a", "r9") or sha256_file(checkpoint) != expected_sha:
        raise ValueError("legacy export requires a registered family and exact checkpoint SHA")
    model = create_model(family, 48)
    provenance = initialize(model, checkpoint, "ema")
    return {"schema": "g246-8h-deploy-v1", "model_spec": {"family": family, "width": 48},
        "state_dict": model.state_dict(), "parameter_count": _count(model),
        "named_parameter_shapes": _shapes(model), "config": {"cache": str(cache), "amp": False},
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"), "locked_test_opened": False,
        "validation_rmse_k": None, "legacy_conversion": provenance,
        "selected_weights": "registered_original_ema", "conversion_did_not_train": True}


@torch.inference_mode()
def cold_run(checkpoint_path: Path, output_root: Path, *, batch_size: int = 2) -> dict[str, Any]:
    """CPU FP32 members, float64 accumulation; labels are used after saving fields."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    started = time.perf_counter()
    torch.set_num_threads(2)
    checkpoint_path, output_root = _path(checkpoint_path), _path(output_root)
    if output_root.exists():
        raise FileExistsError("cold replay requires a new output directory")
    source_bytes = checkpoint_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    payload = load_deployment(io.BytesIO(source_bytes))
    if payload.get("locked_test_opened") is not False:
        raise ValueError("deployment must be explicitly locked-test closed")
    ensemble = payload["model_spec"]["family"] == "fixed_ensemble"
    source_dependency = prepare_dino_source(payload)
    model = build_ensemble(payload["model_spec"]) if ensemble else _member(payload["model_spec"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if _count(model) != payload["parameter_count"] or _count(model) >= 20_000_000 or _shapes(model) != payload["named_parameter_shapes"]:
        raise ValueError("cold reconstructed parameter audit failed")
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("nonfinite deployment state")
    model.eval()
    state_sha = _weight_sha256(model.state_dict())
    entries = payload["model_spec"]["members"] if ensemble else [payload]
    member_reports = audit_ensemble(model, payload) if ensemble else []
    contracts = [_cache_contract(entry["config"], entry["model_spec"], payload["cache_manifest_sha256"])
                 for entry in entries]
    caches = []
    for entry, contract in zip(entries, contracts):
        cache = Cache(Path(contract["base_root"]), "validation", torch.device("cpu"), False,
            detail_root=Path(contract["detail"]["root"]) if contract["detail"] else None,
            hourly_root=Path(contract["hourly"]["root"]) if contract["hourly"] else None,
            emissivity_root=Path(contract["emissivity"]["root"]) if contract["emissivity"] else None,
            **({"historical_root": Path(contract["historical"]["root"])} if contract["historical"] else {}))
        cache.arrays = {name: cache.arrays[name] for name in PREDICTORS}
        caches.append(cache)
    if any(cache.records != caches[0].records for cache in caches[1:]):
        raise ValueError("member validation scene order differs")
    output_root.mkdir(parents=True, exist_ok=False)
    fields = np.empty((45, 1, 160, 160), np.float64)
    inference_started = time.perf_counter()
    member_models = model.members if ensemble else [model]
    for start in range(0, 45, batch_size):
        stop = min(start + batch_size, 45)
        batches = [(cache.temporal_batch if isinstance(member, TemporalModel) else cache.batch)(np.arange(start, stop))
                   for member, cache in zip(member_models, caches)]
        prediction = model(batches) if ensemble else forward_batch(model, batches[0])
        fields[start:stop] = repair_numpy(prediction.cpu().numpy().astype(np.float64),
            caches[0].arrays["coarse"][start:stop], caches[0].arrays["support"][start:stop])
        if start == 0 or stop == 45 or stop % 10 == 0:
            print(json.dumps({"event": "cold_ensemble_progress", "scenes": stop,
                "tta_views": 8 if ensemble and model.tta_d4 else 1,
                "seconds": time.perf_counter() - inference_started}), flush=True)
    inference_seconds = time.perf_counter() - inference_started
    if _weight_sha256(model.state_dict()) != state_sha or not np.all(np.isfinite(fields)):
        raise ValueError("state changed during inference or predictions are not finite")
    np.save(output_root / "validation_predictions.npy", fields, allow_pickle=False)
    prediction_sha = sha256_file(output_root / "validation_predictions.npy")
    # Only now are fine validation targets and formal evaluation masks loaded.
    base_root = Path(contracts[0]["base_root"])
    target = np.load(base_root / "validation" / "target.npy", mmap_mode="r", allow_pickle=False)
    formal = np.load(base_root / "validation" / "formal.npy", mmap_mode="r", allow_pickle=False)
    metrics = score_fields(fields, target, formal, caches[0].records)
    rmse = float(metrics["equal_region"]["rmse_k"])
    copied = dict(payload)
    copied.update(validation_rmse_k=rmse, cold_replay_parent_checkpoint_sha256=source_sha)
    torch.save(copied, output_root / "deploy.pt")
    _json(output_root / "best_metrics.json", metrics)
    verification = verify(base_root, output_root)
    verification["deployment_forward_replayed"] = True
    verification["cold_replay_parent_checkpoint_sha256"] = source_sha
    _json(output_root / "verification.json", verification)
    report = {"schema": "g246-8h-ensemble-cold-replay-v1", "source_checkpoint_sha256": source_sha,
        "deploy_sha256": sha256_file(output_root / "deploy.pt"), "weights_content_sha256": state_sha,
        "validation_predictions_sha256": prediction_sha, "parameter_count": _count(model),
        "member_parameter_audit": member_reports, "predictor_contracts": contracts,
        "source_dependency": source_dependency,
        "scene_count": 45, "device": "cpu", "cpu_threads": 2, "amp": False,
        "torch_version": str(torch.__version__), "member_inference_dtype": "float32",
        "accumulation_and_saved_dtype": "float64", "inference_seconds": inference_seconds,
        "tta_d4": bool(ensemble and model.tta_d4), "tta_additional_parameters": 0,
        "labels_used_only_after_all_predictions_saved": True,
        "ensemble_weights_fitted_to_validation": False, "locked_test_opened": False,
        "rmse_k": rmse, "verification_pass": verification["verification_pass"],
        "goal_achieved_on_public_validation": verification["goal_achieved_on_public_validation"],
        "elapsed_seconds": time.perf_counter() - started}
    _json(output_root / "replay.json", report)
    print(json.dumps({"event": "cold_ensemble_complete", "rmse_k": rmse,
                      "parameter_count": _count(model), "verification_pass": verification["verification_pass"]}), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("create")
    package.add_argument("--member", action="append", type=Path, required=True)
    package.add_argument("--weights", type=float, nargs="+")
    package.add_argument("--tta-d4", action="store_true")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--batch-size", type=int, default=2)
    replay = commands.add_parser("replay")
    replay.add_argument("--deploy", type=Path, required=True)
    replay.add_argument("--output-root", type=Path, required=True)
    replay.add_argument("--batch-size", type=int, default=2)
    legacy = commands.add_parser("export-legacy")
    legacy.add_argument("--checkpoint", type=Path, required=True)
    legacy.add_argument("--family", choices=("r6a", "r9"), required=True)
    legacy.add_argument("--expected-sha256", required=True)
    legacy.add_argument("--cache", type=Path, required=True)
    legacy.add_argument("--output-root", type=Path, required=True)
    legacy.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "replay":
        cold_run(args.deploy, args.output_root, batch_size=args.batch_size)
        return 0
    payload = (freeze(args.member, args.weights, tta_d4=args.tta_d4) if args.command == "create"
        else export_legacy(args.checkpoint, args.family, args.cache, args.expected_sha256))
    output = _path(args.output_root)
    if output.exists():
        raise FileExistsError("output must be a new directory")
    # Freeze complete tensors and all constants before fresh model construction.
    staging = output.with_name(output.name + ".frozen")
    staging.mkdir(parents=True, exist_ok=False)
    torch.save(payload, staging / "deploy.pt")
    del payload
    gc.collect()
    cold_run(staging / "deploy.pt", output, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
