#!/usr/bin/env python3
"""Train the upstream MoCoLSKNet baseline on the v2 CDC split.

This runner deliberately keeps all scene arrays in their native 160 m grid and
builds the seven-channel guidance tensor expected by MoCoLSKNet: six optical
bands plus the supplied interpolation weight.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data/v2/source_expanded_final_v1/manifest.json"
DEFAULT_VALIDATION = REPO_ROOT / "data/v2/validation_primary_v2_1_quota4_2b0a25b5/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/frontier_expanded_final_v1/mocolsk_x4"
DEFAULT_DPLUS = REPO_ROOT / "data/dplus_expanded_final_v1"
UPSTREAM_X4_CONFIG = REPO_ROOT / "baselines/GrokLST/configs/gisr/mocolsk/mocolsk_x4_4xb1-10k_groklst.py"
FROZEN_PLAN = REPO_ROOT / "research/protocol/expanded_model_plan_v1.json"
FROZEN_PLAN_SHA256 = "5c44aad7945689b8ef41197748e5e0432175c4f750740db96950f0b4329e928b"
FROZEN_SEED = 20260818
EXPECTED_SOURCE_CITIES = 50
EXPECTED_SOURCE_SCENES = 150
EXPECTED_VALIDATION_CITIES = 4
EXPECTED_VALIDATION_SCENES = 12
UPSTREAM_COMMIT = "956f76bbf565a78d3cd086abf479cb4609b9ac5c"
UPSTREAM_X4_CONFIG_SHA256 = "1034e5a2dfcf1ae9c8a0559015ccb167a2456a5c8f52b64c13fe9ea298540641"
UPSTREAM_MOCO_SHA256 = {
    "dynamic_mlp.py": "dc0aeab756f0d0219259ea73cdefcc5bcbf7703e2edcf538ad438179e161e087",
    "mocolsk.py": "583a316678fabdaf6385db4a670559653045ddf71b25cc521a59d6e70685c6b6",
    "mocolsk_net.py": "cfbfeb5469430614478694704992a4d67291b5ba52937c50f4efd501cdc0de54",
}


def reject_sealed(path: str | Path) -> Path:
    """Reject a sealed path before any filesystem operation is attempted."""
    candidate = Path(path)
    if any("sealed" in part.lower() for part in candidate.parts):
        raise ValueError(f"sealed paths are forbidden: {candidate}")
    return candidate


def safe_load_json(path: str | Path) -> dict[str, Any]:
    path = reject_sealed(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    path = reject_sealed(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_paths(manifest_path: str | Path) -> list[Path]:
    manifest_path = reject_sealed(manifest_path)
    manifest = safe_load_json(manifest_path)
    entries: Any = manifest.get("files", manifest.get("samples", manifest.get("scenes", [])))
    if isinstance(entries, dict):
        entries = list(entries.values())
    result: list[Path] = []
    for item in entries:
        raw = item if isinstance(item, str) else item.get("path", item.get("file", item.get("npz")))
        if raw is None:
            continue
        candidate = reject_sealed(raw)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        result.append(reject_sealed(candidate))
    if not result:
        # Support manifests that provide an explicit NPZ directory only.
        for candidate in sorted(manifest_path.parent.glob("*.npz")):
            result.append(reject_sealed(candidate))
    if not result:
        raise ValueError(f"no NPZ scenes listed by {manifest_path}")
    return result


def _nanmean_std(values: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sums = None
    sums2 = None
    counts = None
    for array in values:
        x = np.asarray(array, dtype=np.float64)
        finite = np.isfinite(x)
        current_sum = np.where(finite, x, 0).sum(axis=tuple(range(1, x.ndim)))
        current_sum2 = np.where(finite, x * x, 0).sum(axis=tuple(range(1, x.ndim)))
        current_count = finite.sum(axis=tuple(range(1, x.ndim)))
        if sums is None:
            sums, sums2, counts = current_sum, current_sum2, current_count
        else:
            sums += current_sum
            sums2 += current_sum2
            counts += current_count
    assert sums is not None and sums2 is not None and counts is not None
    means = sums / np.maximum(counts, 1)
    variances = np.maximum(sums2 / np.maximum(counts, 1) - means * means, 1e-12)
    return means.astype(np.float32), np.sqrt(variances).astype(np.float32)


@dataclass
class SourceStats:
    optical_mean: list[float]
    optical_std: list[float]
    thermal_mean: float
    thermal_std: float
    n_source_scenes: int = 0
    n_source_pixels: int = 0
    fit_scope: str = "source_only_valid_and_eligible_pixels"


def source_statistics(paths: list[Path]) -> SourceStats:
    optical_values: list[np.ndarray] = []
    thermal_values: list[np.ndarray] = []
    for path in paths:
        path = reject_sealed(path)
        with np.load(path) as scene:
            valid = np.asarray(scene["valid"], dtype=bool) & np.asarray(scene["eligible"], dtype=bool)
            optical_values.append(np.where(valid[None], np.asarray(scene["optical"], dtype=np.float32), np.nan))
            # Source-only thermal normalisation is based on the available native
            # target; lowres remains a fallback for layout variants.
            thermal = scene["target_lst"] if "target_lst" in scene else scene["lowres_lst"]
            thermal_values.append(np.where(valid[None], np.asarray(thermal, dtype=np.float32)[None], np.nan))
    opt_mean, opt_std = _nanmean_std(optical_values)
    therm_mean, therm_std = _nanmean_std(thermal_values)
    return SourceStats(
        opt_mean.tolist(), opt_std.tolist(), float(therm_mean[0]), float(therm_std[0]),
        n_source_scenes=len(paths),
        n_source_pixels=int(sum(np.isfinite(values).sum() for values in thermal_values)),
    )


def source_statistics_from_manifest(manifest_path: str | Path) -> SourceStats:
    """Load the finalized source-only normalization without touching validation."""
    manifest_path = reject_sealed(manifest_path)
    manifest = safe_load_json(manifest_path)
    descriptor = manifest.get("normalization")
    if not isinstance(descriptor, dict) or descriptor.get("scope") != "source_only_valid_and_eligible_pixels":
        raise ValueError("expanded source manifest lacks source-only normalization")
    relative = Path(str(descriptor.get("file", "")))
    if relative.is_absolute() or relative.name != str(relative):
        raise ValueError("source normalization must be one relative filename")
    normalization_path = reject_sealed(manifest_path.parent / relative)
    expected_sha = str(descriptor.get("sha256", ""))
    if sha256_file(normalization_path) != expected_sha:
        raise ValueError("source normalization SHA-256 differs from expanded manifest")
    values = safe_load_json(normalization_path)
    if values.get("fit_scope") != descriptor["scope"]:
        raise ValueError("source normalization fit scope mismatch")
    stats = SourceStats(
        optical_mean=[float(value) for value in values["optical_mean"]],
        optical_std=[float(value) for value in values["optical_std"]],
        # The published x4 model applies one LST transform to both LR and HR.
        thermal_mean=float(values["lowres_mean"]),
        thermal_std=float(values["lowres_std"]),
        n_source_scenes=int(values["n_source_scenes"]),
        n_source_pixels=int(values["n_source_pixels"]),
        fit_scope=str(values["fit_scope"]),
    )
    if len(stats.optical_mean) != 6 or len(stats.optical_std) != 6 or min(stats.optical_std) <= 0:
        raise ValueError("invalid six-channel source optical normalization")
    if stats.thermal_std <= 0 or stats.n_source_scenes != EXPECTED_SOURCE_SCENES:
        raise ValueError("invalid expanded source thermal normalization/provenance")
    return stats


def validate_training_contract(
    source_manifest: str | Path, validation_manifest: str | Path
) -> dict[str, Any]:
    """Freeze the expanded source and its manifest-declared validation split."""
    if sha256_file(FROZEN_PLAN) != FROZEN_PLAN_SHA256:
        raise ValueError("expanded model plan SHA-256 mismatch")
    plan = safe_load_json(FROZEN_PLAN)
    source_manifest = reject_sealed(source_manifest)
    validation_manifest = reject_sealed(validation_manifest)
    source_sha = sha256_file(source_manifest)
    validation_sha = sha256_file(validation_manifest)
    planned_source = plan["datasets"]["source"]
    planned_validation = plan["datasets"]["validation"]
    if source_sha != planned_source["sha256"] or validation_sha != planned_validation["sha256"]:
        raise ValueError("source/validation manifest SHA-256 differs from frozen plan")
    source = safe_load_json(source_manifest)
    if source.get("artifact_kind") != "expanded_source_training_root" or source.get("build_complete") is not True:
        raise ValueError("MoCoLSK training requires the complete expanded source root")
    if source.get("split") != "source" or source.get("sealed_test_unlocked") is not False:
        raise ValueError("expanded manifest must be non-sealed source data")
    source_scenes = source.get("scenes")
    if not isinstance(source_scenes, list):
        raise ValueError("expanded source manifest has no scenes list")
    source_cities = {str(row["city"]) for row in source_scenes}
    if len(source_scenes) != EXPECTED_SOURCE_SCENES or len(source_cities) != EXPECTED_SOURCE_CITIES:
        raise ValueError("MoCoLSK requires all 50 source cities / 150 scenes")
    for city in source_cities:
        years = [int(row["year"]) for row in source_scenes if str(row["city"]) == city]
        if len(years) != 3 or len(set(years)) != 3:
            raise ValueError(f"source city {city} does not have three distinct scenes")

    frozen = source.get("fixed_validation_manifest")
    if not isinstance(frozen, dict):
        raise ValueError("expanded source manifest does not freeze validation")
    if validation_sha != str(frozen.get("sha256", "")):
        raise ValueError("validation manifest differs from expanded source fixed validation")
    validation = safe_load_json(validation_manifest)
    validation_scenes = validation.get("scenes")
    if not isinstance(validation_scenes, list):
        raise ValueError("fixed validation manifest has no scenes list")
    validation_cities = {str(row["city"]) for row in validation_scenes}
    if (
        validation.get("split") != "validation"
        or len(validation_scenes) != EXPECTED_VALIDATION_SCENES
        or len(validation_cities) != EXPECTED_VALIDATION_CITIES
        or validation_cities != set(map(str, frozen.get("cities", [])))
    ):
        raise ValueError("MoCoLSK requires the fixed 4-city / 12-scene validation")
    if source_cities & validation_cities:
        raise ValueError("source and validation cities overlap")
    return {
        "plan_sha256": FROZEN_PLAN_SHA256,
        "source_manifest_sha256": source_sha,
        "validation_manifest_sha256": validation_sha,
        "source_cities": len(source_cities),
        "source_scenes": len(source_scenes),
        "validation_cities": len(validation_cities),
        "validation_scenes": len(validation_scenes),
    }


def validate_dplus_interface(
    root: str | Path, source_manifest_sha256: str, validation_manifest_sha256: str
) -> dict[str, Any]:
    """Verify that the available D+ sidecars align; exact MoCoLSK does not ingest them."""
    root = reject_sealed(root)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    dataset = next((row for row in records if row.get("kind") == "dataset"), None)
    normalization = next((row for row in records if row.get("kind") == "source_feature_normalization"), None)
    if not isinstance(dataset, dict) or not isinstance(normalization, dict):
        raise ValueError("D+ interface is incomplete")
    if (
        dataset.get("status") != "complete"
        or dataset.get("sealed_access") is not False
        or dataset.get("source_manifest_sha256") != source_manifest_sha256
        or dataset.get("validation_manifest_sha256") != validation_manifest_sha256
        or normalization.get("validation_included") is not False
    ):
        raise ValueError("D+ interface does not match expanded source/fixed validation")
    return {"root": str(root.resolve()), "compatible": True, "used_by_exact_mocolsk": False}


def manifest_metadata(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = reject_sealed(manifest_path)
    metadata: dict[str, dict[str, Any]] = {}
    for item in safe_load_json(manifest_path).get("scenes", []):
        if not isinstance(item, dict):
            continue
        raw = item.get("path", item.get("file", item.get("npz")))
        if raw is None:
            continue
        path = reject_sealed(raw)
        if not path.is_absolute():
            path = manifest_path.parent / path
        metadata[str(reject_sealed(path))] = {
            "city": str(item.get("city", path.stem.rsplit("_", 1)[0])),
            "year": int(item.get("year", 0)),
            "scene_id": str(item.get("item_id", path.stem)),
        }
    return metadata


class SceneDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: list[Path], stats: SourceStats, metadata: dict[str, dict[str, Any]] | None = None):
        self.paths = [reject_sealed(p) for p in paths]
        self.stats = stats
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = reject_sealed(self.paths[index])
        with np.load(path) as data:
            optical = np.asarray(data["optical"], dtype=np.float32)
            weight = np.asarray(data["interpolation_weight120"], dtype=np.float32)
            coarse = np.asarray(data["coarse_lst480"], dtype=np.float32)
            coarse_valid = np.asarray(data["coarse_valid480"], dtype=bool)
            target = np.asarray(data["target_lst"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            eligible = np.asarray(data["eligible"], dtype=bool)
        if optical.shape != (6, 160, 160) or coarse.shape != (40, 40):
            raise ValueError(f"{path}: exact x4 contract requires optical 6x160x160 and coarse 40x40")
        if coarse_valid.shape != coarse.shape:
            raise ValueError(f"{path}: coarse validity shape mismatch")
        for name, array in (("interpolation_weight120", weight), ("target_lst", target), ("valid", valid), ("eligible", eligible)):
            if array.shape != (160, 160):
                raise ValueError(f"{path}: {name} must have full-scene 160x160 shape")
        optical_mean = np.asarray(self.stats.optical_mean, dtype=np.float32)[:, None, None]
        optical_std = np.asarray(self.stats.optical_std, dtype=np.float32)[:, None, None]
        optical = (optical - optical_mean) / optical_std
        thermal_mean, thermal_std = self.stats.thermal_mean, self.stats.thermal_std
        coarse = ((coarse - thermal_mean) / thermal_std).astype(np.float32, copy=False)
        target = ((target - thermal_mean) / thermal_std).astype(np.float32, copy=False)
        coarse = np.nan_to_num(np.where(coarse_valid, coarse, np.nan), nan=0.0, posinf=0.0, neginf=0.0)
        guidance = np.concatenate([np.nan_to_num(optical), np.nan_to_num(weight)[None]], axis=0).astype(np.float32, copy=False)
        metadata = self.metadata.get(str(path), {"city": path.stem.rsplit("_", 1)[0], "year": 0, "scene_id": path.stem})
        return {
            "coarse": torch.from_numpy(coarse[None]),
            "guidance": torch.from_numpy(guidance),
            "target": torch.from_numpy(np.nan_to_num(target)[None]),
            "mask": torch.from_numpy(valid & eligible)[None],
            **metadata,
        }


def patch_dynamic_mlp_gradients(module: nn.Module) -> nn.Module:
    """Repair upstream MCWG's detached Parameter construction at runtime.

    Upstream turns the dynamic convolution weights into a new
    ``nn.Parameter(..., requires_grad=False)``.  That preserves its forward
    result but severs the dynamic MLP.  Replacing this with functional convs
    preserves B=1 output exactly and retains autograd connectivity.
    """
    for child in module.modules():
        # The actual upstream MCWG bridge. This is a copy of MoCoLSK.forward
        # with only ``nn.Parameter(data=weights, requires_grad=False)`` removed.
        if all(hasattr(child, name) for name in ("dynamic_mlp", "kernel_size", "conv_lst", "conv_spatial0", "conv1", "conv2", "conv")):
            def _forward(self: nn.Module, lst: torch.Tensor, gui: torch.Tensor) -> torch.Tensor:
                batch, channels, _, _ = lst.shape
                attn1 = self.conv_lst(lst)
                attn2 = self.conv_spatial0(attn1)
                attn1, attn2 = self.conv1(attn1), self.conv2(attn2)
                attn = torch.cat([attn1, attn2], dim=1)
                agg = torch.cat([torch.mean(attn, dim=1, keepdim=True), torch.max(attn, dim=1, keepdim=True)[0]], dim=1)
                lst_pools, gui_pools = [], []
                for pool_size in self.pool_sizes:
                    lst_pools.append(F.adaptive_avg_pool2d(lst, (pool_size, pool_size)).view(batch, channels, -1))
                    gui_pools.append(F.adaptive_avg_pool2d(gui, (pool_size, pool_size)).view(batch, channels, -1))
                lst_pooled = torch.cat(lst_pools, dim=2).permute(0, 2, 1)
                gui_pooled = torch.cat(gui_pools, dim=2).permute(0, 2, 1)
                # FusionModule's upstream forward averages over dimension zero,
                # which is the batch axis for these tensors.  Preserve its exact
                # B=1 behavior and obtain one kernel per sample for B>1 by
                # evaluating that small global pathway independently.
                generated = (
                    self.dynamic_mlp(lst_pooled, gui_pooled)
                    if batch == 1
                    else torch.cat([
                        self.dynamic_mlp(
                            lst_pooled[index:index + 1],
                            gui_pooled[index:index + 1],
                        )
                        for index in range(batch)
                    ], dim=0)
                )
                weights = torch.mean(generated, dim=1).reshape(
                    batch, 2, 2, self.kernel_size, self.kernel_size
                )
                if batch == 1:
                    agg = F.conv2d(agg, weights[0], stride=1, padding=self.kernel_size // 2, groups=1)
                else:
                    agg = torch.cat([F.conv2d(agg[i:i + 1], weights[i], stride=1, padding=self.kernel_size // 2, groups=1) for i in range(batch)], dim=0)
                sig = agg.sigmoid()
                return self.conv(attn1 * sig[:, 0].unsqueeze(1) + attn2 * sig[:, 1].unsqueeze(1)) * gui
            child.forward = _forward.__get__(child, child.__class__)
    return module


def import_model() -> type[nn.Module]:
    upstream = REPO_ROOT / "baselines/PGDM"
    reject_sealed(upstream)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from lib.moco.mocolsk_net import MoCoLSKNet
    return MoCoLSKNet


def build_model() -> nn.Module:
    if sha256_file(UPSTREAM_X4_CONFIG) != UPSTREAM_X4_CONFIG_SHA256:
        raise RuntimeError("official GrokLST x4 configuration bytes changed")
    moco_root = REPO_ROOT / "baselines/PGDM/lib/moco"
    if any(sha256_file(moco_root / name) != expected for name, expected in UPSTREAM_MOCO_SHA256.items()):
        raise RuntimeError("official MoCoLSK source bytes changed")
    cls = import_model()
    candidates = (
        {"in_channels": 1, "gui_channels": 7, "scale": 4, "num_feats": 32, "n_resblocks": 4, "num_stages": 4},
    )
    for kwargs in candidates:
        try:
            return patch_dynamic_mlp_gradients(cls(**kwargs))
        except TypeError:
            continue
    raise TypeError("Unable to instantiate MoCoLSKNet with prescribed scale/features/blocks/stages")


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype)
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def _average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary, unweighted AP equivalent to sklearn for this fixed metric contract."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0 or not np.any(labels):
        raise ValueError("average precision requires aligned scores and a positive label")
    order = np.argsort(scores, kind="mergesort")[::-1]
    labels, scores = labels[order], scores[order]
    threshold_indices = np.r_[np.flatnonzero(np.diff(scores)), labels.size - 1]
    true_positives = np.cumsum(labels, dtype=np.float64)[threshold_indices]
    precision = true_positives / (threshold_indices + 1.0)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _repository_metrics() -> tuple[Any, Any]:
    code_root = str(REPO_ROOT / "code")
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    try:
        module = importlib.import_module("metrics")
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "sklearn.metrics"}:
            raise
        sklearn = types.ModuleType("sklearn")
        sklearn_metrics = types.ModuleType("sklearn.metrics")
        sklearn_metrics.average_precision_score = _average_precision_score
        sklearn.metrics = sklearn_metrics
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = sklearn_metrics
        sys.modules.pop("metrics", None)
        module = importlib.import_module("metrics")
    return module.evaluate_field, module.macro_city_aggregate


def evaluate(model: nn.Module, loader: DataLoader, stats: SourceStats, device: torch.device) -> dict[str, Any]:
    evaluate_field, macro_city_aggregate = _repository_metrics()
    model.eval()
    fields: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["coarse"].to(device), batch["guidance"].to(device))
            pred = pred.cpu().numpy()[0, 0] * stats.thermal_std + stats.thermal_mean
            truth = batch["target"].numpy()[0, 0] * stats.thermal_std + stats.thermal_mean
            mask = batch["mask"].numpy()[0, 0].astype(bool)
            field = evaluate_field(truth, pred, mask, score=pred)
            field.update({"city": str(batch["city"][0]), "year": int(batch["year"][0]), "scene_id": str(batch["scene_id"][0])})
            fields.append(field)
    metric_keys = sorted(key for key in fields[0] if key not in {"city", "year", "scene_id"} and not key.startswith("n_"))
    aggregate = macro_city_aggregate(fields, metric_keys=metric_keys)
    return {"scenes": fields, "cities": aggregate["per_city"], "macro": aggregate["macro"]}


def select_validation_checkpoint(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """The frozen train_v2 joint eligibility and hotspot selection rule."""
    rows = []
    for record in records:
        macro = record["validation"]["macro"]
        row = {
            "step": int(record["step"]),
            "epoch": int(record["epoch"]),
            **{
                key: float(macro[key])
                for key in ("rmse", "r2", "auprc_q90", "top_iou_q90")
            },
        }
        row["hotspot_score"] = 0.5 * (row["auprc_q90"] + row["top_iou_q90"])
        rows.append(row)
    min_rmse, max_r2 = min(row["rmse"] for row in rows), max(row["r2"] for row in rows)
    tolerance = max(0.10, 0.02 * min_rmse)
    eligible = [row for row in rows if row["rmse"] <= min_rmse + tolerance and row["r2"] >= max_r2 - 0.01]
    return min(eligible, key=lambda row: (-row["hotspot_score"], row["rmse"], row["step"]))


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed = args.seed
    if seed != FROZEN_SEED:
        raise ValueError(f"frozen MoCoLSK seed must be {FROZEN_SEED}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    contract = validate_training_contract(args.source_manifest, args.validation_manifest)
    contract["dplus"] = validate_dplus_interface(
        args.dplus_root,
        contract["source_manifest_sha256"],
        contract["validation_manifest_sha256"],
    )
    source_paths, validation_paths = manifest_paths(args.source_manifest), manifest_paths(args.validation_manifest)
    stats = source_statistics_from_manifest(args.source_manifest)
    source_loader = DataLoader(SceneDataset(source_paths, stats, manifest_metadata(args.source_manifest)), batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"))
    validation_loader = DataLoader(SceneDataset(validation_paths, stats, manifest_metadata(args.validation_manifest)), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # The official 10k schedule restarts every 2500 updates.  This runner uses
    # the same first-cycle cosine behavior while retaining epoch-based stopping.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500)
    output = reject_sealed(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix a new run with existing outputs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    epoch_weights = output / ".epoch_weights"
    epoch_weights.mkdir(exist_ok=False)
    best_rmse, best_selected_hotspot, stale, history = math.inf, -math.inf, 0, []
    optimizer_updates = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train(); losses = []
        for batch in source_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["coarse"].to(device), batch["guidance"].to(device))
            loss = masked_l1(pred, batch["target"].to(device), batch["mask"].to(device))
            loss.backward(); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
            optimizer_updates += 1
        metrics = evaluate(model, validation_loader, stats, device)
        row = {"step": epoch, "epoch": epoch, "optimizer_updates": optimizer_updates, "train_l1_normalized": float(np.mean(losses)), "validation": metrics}
        history.append(row)
        torch.save(
            {"model": model.state_dict(), "stats": asdict(stats), "epoch": epoch, "metrics": metrics},
            epoch_weights / f"epoch_{epoch:04d}.pt",
        )
        selected = select_validation_checkpoint(history)
        rmse_improved = float(metrics["macro"]["rmse"]) < best_rmse - args.delta
        hotspot_improved = float(selected["hotspot_score"]) > best_selected_hotspot + 1e-12
        if rmse_improved: best_rmse = float(metrics["macro"]["rmse"])
        if hotspot_improved: best_selected_hotspot = float(selected["hotspot_score"])
        stale = 0 if (rmse_improved or hotspot_improved) else stale + 1
        payload = {"baseline": "MoCoLSK-Net published x4 configuration with matched 7-channel guidance", "runner_sha256": sha256_file(Path(__file__)), "upstream_commit": UPSTREAM_COMMIT, "upstream_x4_config_sha256": UPSTREAM_X4_CONFIG_SHA256, "gradient_fix": "functional_conv2d_dynamic_weights", "sealed": False, "contract": contract, "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 1e-5}, "scheduler": {"name": "CosineAnnealingLR", "period_updates": 2500}, "source_manifest": str(Path(args.source_manifest).resolve()), "validation_manifest": str(Path(args.validation_manifest).resolve()), "stats": asdict(stats), "training_unit": "one complete source scene per optimizer update", "early_stop_unit": "all 12 fixed validation scenes with city-macro aggregation once per epoch", "history": history, "selected": selected}
        (output / "history.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "best_rmse": best_rmse, "selected_hotspot_score": best_selected_hotspot, "stale": stale, "macro": metrics["macro"], "selected": selected}), flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    selected = select_validation_checkpoint(history)
    selected_epoch = int(selected["epoch"])
    selected_checkpoint = torch.load(epoch_weights / f"epoch_{selected_epoch:04d}.pt", map_location="cpu", weights_only=False)
    selected_checkpoint["selection"] = selected
    selected_checkpoint["metrics"] = history[selected_epoch - 1]["validation"]
    temporary_best = output / ".best_weights.pt.tmp"
    torch.save(selected_checkpoint, temporary_best)
    os.replace(temporary_best, output / "best_weights.pt")
    shutil.rmtree(epoch_weights)
    payload["selected"] = selected
    payload["optimizer_updates"] = optimizer_updates
    (output / "history.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"epochs": len(history), "optimizer_updates": optimizer_updates, "best_rmse": best_rmse, "selected": selected, "output": str(output)}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only contract and one complete-scene x4 forward check."""
    if args.seed != FROZEN_SEED:
        raise ValueError(f"frozen MoCoLSK seed must be {FROZEN_SEED}")
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    contract = validate_training_contract(args.source_manifest, args.validation_manifest)
    contract["dplus"] = validate_dplus_interface(
        args.dplus_root,
        contract["source_manifest_sha256"],
        contract["validation_manifest_sha256"],
    )
    stats = source_statistics_from_manifest(args.source_manifest)
    source_paths = manifest_paths(args.source_manifest)
    sample = SceneDataset(source_paths, stats, manifest_metadata(args.source_manifest))[0]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model().to(device).eval()
    with torch.no_grad():
        prediction = model(sample["coarse"][None].to(device), sample["guidance"][None].to(device))
    if tuple(prediction.shape) != (1, 1, 160, 160):
        raise RuntimeError(f"exact x4 full-scene forward returned {tuple(prediction.shape)}")
    return {
        "go": True,
        "runner_sha256": sha256_file(Path(__file__)),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "contract": contract,
        "source_stats": asdict(stats),
        "coarse_shape": list(sample["coarse"].shape),
        "guidance_shape": list(sample["guidance"].shape),
        "prediction_shape": list(prediction.shape),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--dplus-root", type=Path, default=DEFAULT_DPLUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        print(json.dumps(preflight(parsed), indent=2), flush=True)
    else:
        train(parsed)
