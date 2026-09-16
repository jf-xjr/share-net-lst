#!/usr/bin/env python3
"""Cold-load a public G246 deployment and independently replay all 45 scenes.

The source run is read-only. A new output directory receives the float64
predictions, unchanged model weights with updated replay metadata, and receipts.
CPU uses float32 inference; CUDA defaults to the source run's AMP setting.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from g246_8h_augment import inverse_field, transform_batch
from g246_8h_deployment_sources import prepare_dino_source
from train_g246_8h import Cache, TemporalModel, create_model, forward_batch, repair_numpy
from verify_g246_8h_result import check_emissivity_cache, check_historical_cache, check_hourly_cache, load_deployment, projection_diagnostics, score_fields, sha256_file, verify


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_KEYS = ("fine", "coarse", "support", "context")


def _path(value: str | Path) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if "locked" in str(path).lower():
        raise ValueError("only public paths are supported")
    return path


def _json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _weight_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cache_check(root: Path, expected_sha: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_sha:
        raise ValueError("base cache manifest differs from the deployment binding")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("locked_test_opened") is not False:
        raise ValueError("base cache must be complete and explicitly locked-test closed")
    if set(manifest.get("roles", {})) != {"fit", "validation"}:
        raise ValueError("only the fit/validation cache contract is accepted")
    detail = manifest["roles"]["validation"]
    metadata_path = root / "validation" / "metadata.json"
    if sha256_file(metadata_path) != detail["metadata_sha256"]:
        raise ValueError("validation scene order differs from the deployment cache")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("locked_test_opened") is not False or metadata.get("role") != "validation":
        raise ValueError("invalid public validation metadata")
    rows = metadata["scenes"]
    if len(rows) != 45 or any(row["index"] != i or row["view_role"] != "validation"
                              for i, row in enumerate(rows)):
        raise ValueError("replay requires all 45 scenes in canonical order")
    # Validate inference inputs before running the model. Labels are checked by
    # the independent verifier only after predictions have been materialized.
    for name in PREDICTOR_KEYS:
        if sha256_file(root / "validation" / f"{name}.npy") != detail["arrays"][name]["sha256"]:
            raise ValueError(f"inference input has changed: {name}")
    return manifest


def _optical_check(root: Path, expected_sha: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_sha:
        raise ValueError("optical detail manifest differs from the deployment binding")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("locked_test_opened") is not False
            or manifest.get("target_arrays_opened") is not False):
        raise ValueError("optical detail cache has an invalid public predictor contract")
    detail = manifest["roles"]["validation"]
    for filename, expected in (("metadata.json", detail["metadata_sha256"]),
                               ("detail.npy", detail["array_sha256"])):
        if sha256_file(root / "validation" / filename) != expected:
            raise ValueError(f"optical detail validation {filename} changed")
    return {"root": str(root), "manifest_sha256": expected_sha,
            "validation_array_sha256": detail["array_sha256"]}


@torch.inference_mode()
def replay(run_root: Path, output_root: Path, *, device_name: str = "cpu",
           batch_size: int = 2, tta_d4: bool = False, amp: bool | None = None,
           threads: int = 2) -> dict[str, Any]:
    started = time.perf_counter()
    run_root, output_root = _path(run_root), _path(output_root)
    if run_root == output_root or run_root in output_root.parents or output_root in run_root.parents:
        raise ValueError("source and replay output directories must not overlap")
    if output_root.exists():
        raise FileExistsError("replay requires a new output directory")
    if batch_size < 1 or threads < 1 or device_name not in ("cpu", "cuda"):
        raise ValueError("invalid batch size, CPU thread count or device")
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay was requested but is unavailable")
    # Snapshot the source checkpoint bytes so concurrent atomic checkpoint
    # publication cannot mix metadata and model weights from different states.
    checkpoint_bytes = (run_root / "deploy.pt").read_bytes()
    parent_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    payload = load_deployment(io.BytesIO(checkpoint_bytes))
    if payload.get("locked_test_opened") is not False:
        raise ValueError("deployment must explicitly be locked-test closed")
    config, spec = payload["config"], payload["model_spec"]
    use_amp = bool(config.get("amp", True)) if amp is None else bool(amp)
    if device.type == "cpu":
        if amp is True:
            raise ValueError("CPU replay uses float32; --amp is CUDA-only")
        use_amp = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    cache_root = _path(config["cache"])
    base_manifest = _cache_check(cache_root, payload["cache_manifest_sha256"])
    detail_root, optical = None, None
    if str(spec["family"]).startswith("optical_"):
        detail_root = _path(config["detail_root"])
        optical = _optical_check(detail_root, config["optical_detail_manifest_sha256"])
    hourly_root, hourly = None, None
    if spec["family"] == "hourly_r6a":
        hourly_root = _path(config["hourly_root"])
        base_metadata = json.loads((cache_root / "validation" / "metadata.json").read_text())
        hourly = check_hourly_cache(hourly_root, config["hourly_manifest_sha256"],
                                    payload["cache_manifest_sha256"],
                                    [row["scene_id"] for row in base_metadata["scenes"]],
                                    base_manifest["roles"]["validation"]["metadata_sha256"])
    emissivity_root, emissivity = None, None
    if spec["family"] in ("emissivity_r6a", "optical_emissivity_r6a", "historical_innovation_emissivity_r6a",
                          "historical_innovation_emissivity_r6a_six", "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"):
        emissivity_root = _path(config["emissivity_root"])
        base_metadata = json.loads((cache_root / "validation" / "metadata.json").read_text())
        emissivity = check_emissivity_cache(emissivity_root, config["emissivity_manifest_sha256"],
            payload["cache_manifest_sha256"], [row["scene_id"] for row in base_metadata["scenes"]],
            base_manifest["roles"]["validation"]["metadata_sha256"])
    historical_root, historical = None, None
    if spec["family"] in ("historical_r6a", "historical_innovation_r6a", "historical_innovation_emissivity_r6a",
                          "historical_innovation_emissivity_r6a_six", "historical_attended_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a",
                          "historical_multiscale_innovation_emissivity_r6a_six",
                          "historical_recent_innovation_emissivity_r6a_seven",
                          "historical_recent_refinement_emissivity_r6a_seven",
                          "historical_recent_innovation_emissivity_r6a_nine",
                          "historical_recent_refinement_emissivity_r6a_nine"):
        historical_root = _path(config["historical_root"])
        base_metadata = json.loads((cache_root / "validation" / "metadata.json").read_text())
        historical = check_historical_cache(historical_root, config["historical_manifest_sha256"],
            payload["cache_manifest_sha256"], [row["scene_id"] for row in base_metadata["scenes"]],
            base_manifest["roles"]["validation"]["metadata_sha256"], base_metadata["scenes"], family=spec["family"])
    source_field_bytes = (run_root / "validation_predictions.npy").read_bytes()
    source_field_sha = hashlib.sha256(source_field_bytes).hexdigest()
    source_fields = np.load(io.BytesIO(source_field_bytes), allow_pickle=False)
    if (source_fields.shape != (45, 1, 160, 160) or source_fields.dtype != np.float64
            or not np.all(np.isfinite(source_fields))):
        raise ValueError("source fields must cover all 45 finite canonical scenes")
    source_dependency = prepare_dino_source(payload)
    model = create_model(spec["family"], int(spec["width"]))
    state = payload["state_dict"]
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    if parameter_count >= 20_000_000 or parameter_count != payload["parameter_count"]:
        raise ValueError("reconstructed model violates its strict parameter budget")
    if parameter_shapes != payload["named_parameter_shapes"]:
        raise ValueError("reconstructed model differs from its deployment parameter specification")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("deployment weights contain nonfinite values")
    weight_sha = _weight_sha256(state)
    model.to(device).eval()
    data = Cache(cache_root, "validation", device, False, detail_root=detail_root, hourly_root=hourly_root,
                 emissivity_root=emissivity_root, **({"historical_root": historical_root} if historical else {}))
    all_arrays = data.arrays
    # Batch materialization during inference receives no fine labels or masks.
    data.arrays = {name: all_arrays[name] for name in PREDICTOR_KEYS}
    output_root.mkdir(parents=True, exist_ok=False)
    fields = np.empty((45, 1, 160, 160), dtype=np.float64)
    inference_started = time.perf_counter()
    batch_times = []
    for start in range(0, 45, batch_size):
        batch_started = time.perf_counter()
        stop = min(start + batch_size, 45)
        batcher = data.temporal_batch if isinstance(model, TemporalModel) else data.batch
        batch = batcher(np.arange(start, stop))
        total = np.zeros((stop - start, 1, 160, 160), dtype=np.float64)
        codes = range(8) if tta_d4 else (0,)
        for code in codes:
            transformed = transform_batch(batch, code) if tta_d4 else batch
            context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with context:
                prediction = forward_batch(model, transformed)
            if tta_d4:
                prediction = inverse_field(prediction, code)
            total += prediction.float().cpu().numpy().astype(np.float64)
        total /= 8 if tta_d4 else 1
        fields[start:stop] = repair_numpy(total, all_arrays["coarse"][start:stop],
                                         all_arrays["support"][start:stop])
        batch_times.append(time.perf_counter() - batch_started)
        if start == 0 or stop == 45 or len(batch_times) % 5 == 0:
            print(json.dumps({"event": "cold_replay_progress", "scenes": stop,
                              "total_scenes": 45, "device": str(device),
                              "tta_views": 8 if tta_d4 else 1,
                              "elapsed_inference_seconds": time.perf_counter() - inference_started}), flush=True)
    inference_seconds = time.perf_counter() - inference_started
    if _weight_sha256(model.state_dict()) != weight_sha:
        raise ValueError("model state changed during inference; this is not a frozen-weight replay")
    data.arrays = all_arrays
    if not np.all(np.isfinite(fields)):
        raise ValueError("cold inference produced nonfinite values")
    np.save(output_root / "validation_predictions.npy", fields, allow_pickle=False)
    prediction_sha = sha256_file(output_root / "validation_predictions.npy")
    # Scoring begins only after every replay field has been written and hashed.
    metrics = score_fields(fields, all_arrays["target"], all_arrays["formal"], data.records)
    source_metrics = score_fields(source_fields, all_arrays["target"], all_arrays["formal"], data.records)
    source_rmse = float(source_metrics["equal_region"]["rmse_k"])
    if not math.isclose(source_rmse, float(payload["validation_rmse_k"]), rel_tol=0, abs_tol=1e-10):
        raise ValueError("source checkpoint and source fields are not a consistent saved pair")
    projection = projection_diagnostics(fields, all_arrays["coarse"], all_arrays["support"])
    if not projection["closure_pass"]:
        raise ValueError("cold replay failed exact observed-parent closure")
    difference = fields - source_fields
    formal = np.asarray(all_arrays["formal"], dtype=bool)
    formal_difference = difference[formal]
    rmse = float(metrics["equal_region"]["rmse_k"])
    report = {
        "schema": "g246-8h-cold-replay-v1", "source_run_root": str(run_root),
        "output_root": str(output_root), "parent_checkpoint_sha256": parent_sha,
        "weights_content_sha256": weight_sha, "parameter_count": parameter_count,
        "source_validation_predictions_sha256": source_field_sha,
        "replay_validation_predictions_sha256": prediction_sha,
        "cache_manifest_sha256": payload["cache_manifest_sha256"],
        "optical_detail": optical, "hourly_cache": hourly, "emissivity_cache": emissivity,
        "historical_cache": historical, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": use_amp, "inference_dtype": "float16_autocast_with_float32_output" if use_amp else "float32",
        "saved_prediction_dtype": "float64", "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda, "batch_size": batch_size,
        "cpu_threads": threads, "tta_d4": bool(tta_d4), "tta_views": 8 if tta_d4 else 1,
        "tta_aggregation": "inverse_transform_then_float64_mean_then_actual_support_repair",
        "source_rmse_k": source_rmse, "replay_rmse_k": rmse,
        "replay_minus_source_rmse_k": rmse - source_rmse,
        "field_difference": {"max_abs_all_k": float(np.max(np.abs(difference))),
                              "rms_all_k": float(np.sqrt(np.mean(difference * difference))),
                              "max_abs_formal_k": float(np.max(np.abs(formal_difference))),
                              "rms_formal_k": float(np.sqrt(np.mean(formal_difference * formal_difference)))},
        "inference_wall_seconds": inference_seconds, "first_batch_wall_seconds": batch_times[0],
        "model_forward_keys": list(PREDICTOR_KEYS) + (["detail"] if optical else []) + (["hourly"] if hourly else [])
                              + (["emissivity"] if emissivity else []) + (["history"] if historical else []),
        "query_target_only_for_scoring": True, "scene_count": 45,
        "locked_test_opened": False, "deployment_forward_replayed": True,
        "public_validation_is_adaptive_development": True,
        "source_sha256": {name: sha256_file(ROOT / "code" / name) for name in (
            "replay_g246_8h.py", "train_g246_8h.py", "g246_8h_augment.py", "verify_g246_8h_result.py",
            "g246_8h_network.py", "g246_8h_temporal.py", "g246_8h_optical_network.py",
            "g246_8h_resnet.py", "build_g246_8h_optical_detail.py",
            "g246_8h_hourly_network.py", "build_g246_8h_hourly_cache.py",
            "g246_8h_emissivity_network.py", "build_g246_8h_emissivity_cache.py",
            "g246_8h_dino.py", "g246_8h_deployment_sources.py", "g246_8h_historical_network.py",
            "g246_8h_historical_innovation_network.py",
            "g246_8h_historical_multisource.py", "g246_8h_historical_attended.py", "g246_8h_historical_multiscale.py",
            "g246_8h_historical_recent.py", "g246_8h_historical_recent_refinement.py", "g246_8h_recent_historical_features.py",
            "g246_8h_historical_seven_contract.py", "g246_8h_recent_source_contract.py",
            "g246_8h_historical_nine_contract.py", "g246_8h_recent_pair_source_contract.py",
            "g246_8h_historical_recent_multi.py", "g246_8h_historical_recent_refinement_multi.py",
            "build_g246_8h_historical_nine_cache.py",
            "build_g246_8h_historical_seven_cache.py", "build_g246_8h_acquisition_denylist.py",
            "g246_8h_historical_six_contract.py", "build_g246_8h_historical_six_cache.py",
            "g246_8h_historical_features.py", "g246_8h_historical_contract.py",
            "build_g246_8h_historical_cache.py") if (ROOT / "code" / name).exists()},
        "source_dependency": source_dependency,
    }
    copied = dict(payload)
    copied.update(validation_rmse_k=rmse, parent_checkpoint_sha256=parent_sha,
                  source_validation_rmse_k=source_rmse, replay=report,
                  named_parameter_shapes=parameter_shapes, parameter_count=parameter_count,
                  locked_test_opened=False)
    copied["config"] = {**config, "inference_tta_d4": bool(tta_d4), "replay_device": str(device),
                        "replay_amp": use_amp}
    temporary = output_root / "deploy.pt.partial"
    torch.save(copied, temporary)
    temporary.replace(output_root / "deploy.pt")
    _json(output_root / "best_metrics.json", metrics)
    verification = verify(cache_root, output_root)
    verification["deployment_forward_replayed"] = True
    verification["cold_replay_parent_checkpoint_sha256"] = parent_sha
    _json(output_root / "verification.json", verification)
    report["verification_pass"] = verification["verification_pass"]
    report["goal_achieved_on_public_validation"] = verification["goal_achieved_on_public_validation"]
    report["elapsed_wall_seconds"] = time.perf_counter() - started
    report["replay_deploy_sha256"] = sha256_file(output_root / "deploy.pt")
    _json(output_root / "replay.json", report)
    print(json.dumps({"event": "cold_replay_complete", "rmse_k": rmse,
                      "source_rmse_k": source_rmse, "verification_pass": verification["verification_pass"],
                      "goal_achieved_on_public_validation": verification["goal_achieved_on_public_validation"],
                      "inference_seconds": inference_seconds}), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tta-d4", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    replay(args.run_root, args.output_root, device_name=args.device, batch_size=args.batch_size,
           tta_d4=args.tta_d4, amp=args.amp, threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
