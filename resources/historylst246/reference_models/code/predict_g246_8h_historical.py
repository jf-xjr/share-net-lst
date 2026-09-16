#!/usr/bin/env python3
"""CPU inference from a deployment and already encoded physical inputs only.

This entry point neither constructs a dataset/cache nor follows paths stored
in deployment configuration. It does not encode raw satellite observations.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from g246_8h_augment import inverse_field, transform_batch
from package_g246_8h_ensemble import FixedEnsemble, audit_ensemble
from train_g246_8h import create_model, forward_batch, repair_numpy
from verify_g246_8h_result import load_deployment


INPUT_KEYS = ("fine", "coarse", "support", "context", "emissivity", "history")
SHARED_KEYS = INPUT_KEYS[:-1]
FAMILIES = {
    "historical_recent_innovation_emissivity_r6a_seven": 7,
    "historical_recent_refinement_emissivity_r6a_seven": 7,
    "historical_recent_innovation_emissivity_r6a_nine": 9,
    "historical_recent_refinement_emissivity_r6a_nine": 9,
}
PARAMETER_LIMIT = 20_000_000


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(label + " must be a positive integer")
    return value


def _member_spec(spec: Mapping[str, Any]) -> int:
    if not isinstance(spec, Mapping) or spec.get("family") not in FAMILIES:
        raise ValueError("only the registered seven/nine historical emissivity families are supported")
    _positive_int(spec.get("width"), "model width")
    slots = FAMILIES[spec["family"]]
    if "source_count" in spec and spec["source_count"] != slots:
        raise ValueError("deployment source_count conflicts with its registered family")
    if "history_shape" in spec and spec["history_shape"][1:3] != [slots, 9]:
        raise ValueError("deployment history_shape conflicts with its registered family")
    return slots


def _state(model: torch.nn.Module, payload: Mapping[str, Any]) -> None:
    state = payload.get("state_dict")
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("all deployment state keys must exactly match the registered network")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all().item()):
            raise ValueError("deployment state shape, dtype, device or finiteness differs: " + name)
    model.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    if not 0 < count < PARAMETER_LIMIT or count != payload.get("parameter_count"):
        raise ValueError("actual total parameter count differs or is not strictly below 20M")
    if shapes != payload.get("named_parameter_shapes"):
        raise ValueError("actual named parameter shapes differ from the deployment")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("registered learned parameters must execute in CPU FP32")


def _semantic_spec(model: torch.nn.Module, spec: Mapping[str, Any]) -> None:
    expected = model.model_config
    for key in ("schema_version", "class_name", "source_count", "history_shape", "history_date_slots",
                "history_channels", "backbone_family", "recent_history_slot", "recent_history_slots"):
        if key in expected and spec.get(key) != expected[key]:
            raise ValueError("deployment registered input/architecture metadata differs: " + key)


def load_inputs(path: str | Path) -> dict[str, np.ndarray]:
    """Load only an exact six-key physical NPZ; reject other keys before arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(INPUT_KEYS) or len(archive.files) != len(INPUT_KEYS):
            raise ValueError("NPZ must contain exactly fine/coarse/support/context/emissivity/history; target, masks, QA and geographic metadata are forbidden")
        return {name: np.array(archive[name], copy=True) for name in INPUT_KEYS}


def _array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("physical input tensors must already be on CPU")
        value = value.detach().numpy()
    if not isinstance(value, np.ndarray):
        raise TypeError(name + " must be a numpy array or CPU tensor")
    if name == "support":
        if value.dtype.kind not in "bufi" or not np.all((value == 0) | (value == 1)):
            raise ValueError("support must be a finite binary actual-support field")
    else:
        if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(name + " must retain float32/float64 encoded precision; Kelvin cannot use float16")
        if name == "coarse":
            if np.isinf(value).any():
                raise ValueError("coarse may use NaN for unobserved parents but must not contain infinity")
        elif not np.isfinite(value).all():
            raise ValueError(name + " must be finite")
    return value


def _inputs(batch: Mapping[str, Any], slots: int) -> dict[str, np.ndarray]:
    if not isinstance(batch, Mapping) or set(batch) != set(INPUT_KEYS):
        raise ValueError("only the exact six physical input keys are accepted; labels, target masks, QA and geographic fields are forbidden")
    arrays = {name: _array(batch[name], name) for name in INPUT_KEYS}
    if arrays["fine"].ndim != 4 or arrays["fine"].shape[0] < 1:
        raise ValueError("fine must be a nonempty [N,52,160,160] batch")
    n = arrays["fine"].shape[0]
    shapes = {"fine": (n, 52, 160, 160), "coarse": (n, 1, 40, 40),
        "support": (n, 1, 160, 160), "context": (n, 15),
        "emissivity": (n, 4, 160, 160), "history": (n, slots, 9, 160, 160)}
    for name in INPUT_KEYS:
        if arrays[name].shape != shapes[name]:
            raise ValueError(f"{name} must have shape {shapes[name]}, received {arrays[name].shape}")
    return arrays


class HistoricalPredictor:
    """Restore embedded weights and predict supplied physical batches.

    A single network accepts one mapping. A FixedEnsemble accepts one mapping
    per embedded member in its declared order. Ensemble members may have
    different seven/nine-slot histories, but share the same query predictors.
    """
    def __init__(self, deployment: str | Path, *, threads: int = 2):
        torch.set_num_threads(_positive_int(threads, "CPU threads"))
        contents = Path(deployment).read_bytes()
        payload = load_deployment(io.BytesIO(contents))
        if payload.get("locked_test_opened") is not False:
            raise ValueError("deployment must explicitly preserve the locked-test boundary")
        spec = payload.get("model_spec", {})
        self.is_ensemble = spec.get("family") == "fixed_ensemble"
        if self.is_ensemble:
            if payload.get("schema") != "g246-8h-ensemble-deploy-v1":
                raise ValueError("invalid ensemble deployment schema")
            entries = spec.get("members", [])
            self.source_counts = [_member_spec(entry["model_spec"]) for entry in entries]
            if not isinstance(spec.get("tta_d4"), bool):
                raise ValueError("ensemble model_spec must declare boolean tta_d4")
            self.tta_d4 = spec["tta_d4"]
            model = FixedEnsemble(dict(spec))
        else:
            if payload.get("schema") != "g246-8h-deploy-v1":
                raise ValueError("invalid single-network deployment schema")
            self.source_counts = [_member_spec(spec)]
            tta = payload.get("config", {}).get("inference_tta_d4", False)
            if not isinstance(tta, bool):
                raise ValueError("single-network inference_tta_d4 must be boolean when supplied")
            self.tta_d4 = tta
            model = create_model(spec["family"], int(spec["width"]))
        _state(model, payload)
        if self.is_ensemble:
            for member, entry in zip(model.members, entries):
                _semantic_spec(member, entry["model_spec"])
        else:
            _semantic_spec(model, spec)
        self.member_reports = audit_ensemble(model, payload, check_caches=False) if self.is_ensemble else []
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.deployment_sha256 = hashlib.sha256(contents).hexdigest()
        self.model_spec = spec
        self.report = {"deployment_sha256": self.deployment_sha256,
            "parameter_count": self.parameter_count, "member_count": len(self.source_counts),
            "history_slots_per_member": self.source_counts, "cpu_network_dtype": "float32",
            "accumulation_and_output_dtype": "float64", "tta_d4": self.tta_d4,
            "actual_support_repair": True, "cache_or_dataset_constructed": False,
            "deployment_config_paths_followed": False, "current_query_target_arrays_opened": False,
            "current_query_target_masks_opened": False, "current_query_qa_arrays_opened": False,
            "geographic_metadata_inputs": False, "locked_test_opened": False}

    @torch.inference_mode()
    def predict(self, member_inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                *, batch_size: int = 1) -> np.ndarray:
        _positive_int(batch_size, "batch_size")
        if isinstance(member_inputs, Mapping):
            member_inputs = [member_inputs]
        if not isinstance(member_inputs, Sequence) or len(member_inputs) != len(self.source_counts):
            raise ValueError("provide exactly one physical mapping per embedded member, in declared order")
        arrays = [_inputs(batch, slots) for batch, slots in zip(member_inputs, self.source_counts)]
        for other in arrays[1:]:
            for key in SHARED_KEYS:
                if not np.array_equal(other[key], arrays[0][key], equal_nan=True):
                    raise ValueError("ensemble members must share identical aligned query predictors: " + key)
        n = arrays[0]["fine"].shape[0]
        output = np.empty((n, 1, 160, 160), dtype=np.float64)
        self.model.eval()
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batches = [{name: torch.from_numpy(np.array(a[name][start:stop], copy=True)).to(
                dtype=torch.bool if name == "support" else torch.float32)
                for name in INPUT_KEYS} for a in arrays]
            # Explicitly disable inherited CPU autocast: learned forward is FP32.
            with torch.autocast("cpu", enabled=False):
                if self.is_ensemble:
                    accumulated = self.model(batches).cpu().numpy()
                else:
                    accumulated = np.zeros((stop-start, 1, 160, 160), dtype=np.float64)
                    for code in range(8) if self.tta_d4 else (0,):
                        transformed = transform_batch(batches[0], code) if self.tta_d4 else batches[0]
                        field = forward_batch(self.model, transformed)
                        if self.tta_d4:
                            field = inverse_field(field, code)
                        accumulated += field.float().cpu().numpy().astype(np.float64)
                    accumulated /= 8 if self.tta_d4 else 1
            output[start:stop] = repair_numpy(accumulated, arrays[0]["coarse"][start:stop],
                                               arrays[0]["support"][start:stop])
        if not np.isfinite(output).all():
            raise ValueError("model produced nonfinite predictions")
        return output


def predict(deployment: str | Path,
            member_inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
            *, batch_size: int = 1, threads: int = 2) -> np.ndarray:
    """One-call Python API; use HistoricalPredictor to reuse a loaded model."""
    return HistoricalPredictor(deployment, threads=threads).predict(member_inputs, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="one exact six-key physical NPZ per embedded member, in order")
    parser.add_argument("--output", type=Path, required=True, help="new float64 NPY output")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("output must use .npy")
    if args.output.exists():
        raise FileExistsError("output already exists; select a new output path")
    runner = HistoricalPredictor(args.deploy, threads=args.threads)
    fields = runner.predict([load_inputs(path) for path in args.inputs], batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.save(handle, fields, allow_pickle=False)
    print(json.dumps({**runner.report, "output": str(args.output.resolve()), "shape": list(fields.shape)}))


if __name__ == "__main__":
    main()
