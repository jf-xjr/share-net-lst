#!/usr/bin/env python3
"""Metric-first deterministic G246 trainer for Residual U-Net and OCNIR."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


WORKSPACE = Path(__file__).resolve().parents[1]
CODE_ROOT = WORKSPACE / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from g246_data import (  # noqa: E402
    DEFAULT_SPLIT_RECEIPT,
    G246Dataset,
    G246Splits,
    GlobalNormalization,
    fit_normalization,
    load_splits,
    normalize_role,
    reject_forbidden_path,
)
from models import ResidualUNet as _ResidualCore  # noqa: E402
from ocnir import OCNIR, support_project  # noqa: E402


DEFAULT_OUTPUT = WORKSPACE / "artifacts/g246_metric_campaign_v1/run"
EFFECTIVE_BATCH = 32
WARMUP_UPDATES = 2_000
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
MIN_LEARNING_RATE = 1e-6
RMSE_TIE_TOLERANCE_K = 0.001


class ResidualUNet(nn.Module):
    """GlobalCore Kelvin residual U-Net with the same public API as OCNIR."""

    widths = (64, 96)

    def __init__(self, width: int, projection: bool = True,
                 activation_checkpointing: bool = False) -> None:
        super().__init__()
        if int(width) not in self.widths:
            raise ValueError(f"ResidualUNet width must be one of {self.widths}")
        self.width = int(width)
        self.projection = bool(projection)
        self.activation_checkpointing = bool(activation_checkpointing)
        # 11 fine maps plus five broadcast scene-context maps.
        self.core = _ResidualCore(in_channels=16, width=self.width)
        # Start exactly at the physical interpolated-K baseline.
        nn.init.zeros_(self.core.head.weight)
        if self.core.head.bias is not None:
            nn.init.zeros_(self.core.head.bias)

    def forward(self, fine: Tensor, coarse_k: Tensor, support120: Tensor,
                context: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 11:
            raise ValueError("fine must have shape [B,11,H,W]")
        if context.ndim != 2 or context.shape != (fine.shape[0], 5):
            raise ValueError("context must have shape [B,5]")
        spatial_context = context[:, :, None, None].expand(-1, -1, fine.shape[-2], fine.shape[-1])
        network_fine = torch.cat(((fine[:, :1] - 300.0) / 20.0, fine[:, 1:]), dim=1)
        inputs = torch.cat((network_fine, spatial_context), dim=1)
        if self.activation_checkpointing and self.training:
            residual = gradient_checkpoint(self.core, inputs, use_reentrant=False)
        else:
            residual = self.core(inputs)
        raw_k = fine[:, :1] + residual
        return support_project(raw_k, coarse_k, support120) if self.projection else raw_k


def build_model(model: str, width: int, projection: bool = True,
                activation_checkpointing: bool = False) -> nn.Module:
    """Build either registered architecture behind one four-input interface."""
    name = str(model).strip().casefold()
    if name == "resunet":
        return ResidualUNet(width, projection=projection,
                            activation_checkpointing=activation_checkpointing)
    if name == "ocnir":
        if not projection:
            raise ValueError("OCNIR always applies its registered Q projection")
        if int(width) not in OCNIR.widths:
            raise ValueError(f"OCNIR width must be one of {OCNIR.widths}")
        return OCNIR(width=int(width), fine_channels=11, context_dim=5,
                     activation_checkpointing=activation_checkpointing)
    raise ValueError(f"unsupported model: {model!r}")


class EMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            source = value.detach()
            if self.shadow[key].is_floating_point():
                self.shadow[key].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(source)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if float(payload.get("decay", -1)) != self.decay:
            raise ValueError("EMA decay differs from checkpoint")
        shadow = payload.get("shadow")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise ValueError("EMA state differs from model")
        self.shadow = {str(k): torch.as_tensor(v, device=self.shadow[str(k)].device).detach().clone()
                       for k, v in shadow.items()}

    def apply(self, model: nn.Module) -> dict[str, Tensor]:
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup


def masked_metric_loss(prediction: Tensor, target: Tensor, valid: Tensor,
                       eligible: Tensor) -> tuple[Tensor, dict[str, float]]:
    primary = valid & eligible
    if not bool(primary.any().detach().cpu()) or not bool(valid.any().detach().cpu()):
        raise ValueError("training batch has an empty registered loss mask")
    primary_mse = torch.mean(torch.square(prediction[primary] - target[primary]))
    valid_mse = torch.mean(torch.square(prediction[valid] - target[valid]))
    loss = 0.8 * primary_mse + 0.2 * valid_mse
    return loss, {"eligible_valid_mse": float(primary_mse.detach().cpu()),
                  "valid_mse": float(valid_mse.detach().cpu())}


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, bool).reshape(-1)
    scores = np.asarray(scores, np.float64).reshape(-1)
    positives = int(labels.sum())
    if labels.size == 0 or positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / positives)


def _top_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    flat = np.asarray(values, np.float64).reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.size)))
    order = np.argsort(-flat, kind="stable")
    result = np.zeros(flat.size, dtype=bool)
    result[order[:count]] = True
    return result


def scene_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(mask, bool)
    pred = np.asarray(prediction, np.float64)[selected]
    truth = np.asarray(target, np.float64)[selected]
    if pred.size == 0 or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("validation field is empty or non-finite")
    error = pred - truth
    truth_hot = _top_mask(truth)
    pred_hot = _top_mask(pred)
    union = np.logical_or(truth_hot, pred_hot).sum()
    return {"rmse_k": float(np.sqrt(np.mean(error * error))),
            "mae_k": float(np.mean(np.abs(error))),
            "true_hotspot_mae_q90_k": float(
                np.mean(np.abs(error[truth_hot]))
            ),
            "auprc_q90": _average_precision(truth_hot, pred),
            "iou_q90": float(np.logical_and(truth_hot, pred_hot).sum() / union),
            "n_pixels": int(pred.size)}


def _mean_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = ["rmse_k", "mae_k", "auprc_q90", "iou_q90"]
    if all("true_hotspot_mae_q90_k" in record for record in records):
        names.append("true_hotspot_mae_q90_k")
    return {name: float(np.mean([float(record[name]) for record in records])) for name in names}


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


def _autocast(device: torch.device, enabled: bool):
    return (torch.autocast(device_type="cuda", dtype=torch.float16)
            if enabled and device.type == "cuda" else nullcontext())


@torch.inference_mode()
def evaluate_validation(model: nn.Module, dataset: G246Dataset, device: torch.device,
                        *, batch_size: int, amp: bool) -> dict[str, Any]:
    model.eval()
    per_scene: dict[str, dict[str, Any]] = {}
    for start in range(0, len(dataset), batch_size):
        batch = _device_batch(dataset.validation_batch(start, batch_size), device)
        with _autocast(device, amp):
            prediction = model(batch["fine"], batch["coarse_k"], batch["support120"], batch["context"])
        for offset, scene_id in enumerate(batch["scene_id"]):
            metrics = scene_metrics(prediction[offset, 0].float().cpu().numpy(),
                                    batch["target_k"][offset, 0].float().cpu().numpy(),
                                    (batch["valid"] & batch["eligible"])[offset, 0].cpu().numpy())
            per_scene[str(scene_id)] = {"city": batch["city"][offset],
                                        "region": batch["region"][offset], **metrics}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    city_region: dict[str, str] = {}
    for record in per_scene.values():
        grouped.setdefault(str(record["city"]), []).append(record)
        city_region[str(record["city"])] = str(record["region"])
    per_city = {city: {"region": city_region[city], **_mean_metrics(records),
                       "scene_count": len(records)} for city, records in sorted(grouped.items())}
    per_region: dict[str, dict[str, Any]] = {}
    for region in ("us", "china", "europe"):
        records = [record for record in per_city.values() if record["region"] == region]
        if not records:
            raise ValueError(f"validation has no {region} cities")
        per_region[region] = {**_mean_metrics(records), "city_count": len(records)}
    equal_region = _mean_metrics(list(per_region.values()))
    return {"aggregation": "scene_to_city_to_region_equal_region",
            "equal_region": equal_region, "per_region": per_region,
            "per_city": per_city, "per_scene": per_scene}


def select_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records:
        raise ValueError("checkpoint selector requires validation records")
    minimum = min(float(x["validation"]["equal_region"]["rmse_k"]) for x in records)
    eligible = [x for x in records
                if float(x["validation"]["equal_region"]["rmse_k"])
                <= minimum + RMSE_TIE_TOLERANCE_K + 1e-12]
    return min(eligible, key=lambda x: (
        -(float(x["validation"]["equal_region"]["auprc_q90"])
          + float(x["validation"]["equal_region"]["iou_q90"])) / 2.0,
        float(x["validation"]["equal_region"]["rmse_k"]), int(x["update"])))


def selected_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    equal = dict(record["validation"]["equal_region"])
    return {"update": int(record["update"]), "checkpoint": "best.pt",
            "rmse_tolerance_k": RMSE_TIE_TOLERANCE_K, "equal_region": equal,
            "selector_hotspot_score": (float(equal["auprc_q90"]) + float(equal["iou_q90"])) / 2.0}


def _require_resume_progress(requested_max_updates: int, checkpoint_updates: int) -> None:
    """Reject a resume that cannot execute even one optimizer update."""

    if int(requested_max_updates) <= int(checkpoint_updates):
        raise ValueError(
            "resume requires requested max_updates > checkpoint optimizer_updates "
            f"({int(requested_max_updates)} <= {int(checkpoint_updates)})"
        )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scaler(device: torch.device, amp: bool):
    enabled = amp and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def learning_rate(update: int, maximum: int, peak: float) -> float:
    if update <= WARMUP_UPDATES:
        return peak * update / WARMUP_UPDATES
    progress = (update - WARMUP_UPDATES) / max(1, maximum - WARMUP_UPDATES)
    progress = min(max(progress, 0.0), 1.0)
    return MIN_LEARNING_RATE + 0.5 * (peak - MIN_LEARNING_RATE) * (1.0 + math.cos(math.pi * progress))


def _config(args: argparse.Namespace, splits: G246Splits, normalization_sha256: str) -> dict[str, Any]:
    return {"model": args.model, "width": args.width, "projection": args.projection,
            "activation_checkpointing": args.activation_checkpointing,
            "learning_rate": args.learning_rate, "weight_decay": WEIGHT_DECAY,
            "ema_decay": EMA_DECAY, "seed": args.seed, "max_updates": args.max_updates,
            "validation_interval": args.validation_interval,
            "physical_batch_size": args.physical_batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "effective_batch_size": args.physical_batch_size * args.grad_accumulation_steps,
            "split_receipt_sha256": splits.receipt_sha256,
            "normalization_sha256": normalization_sha256,
            "campaign_id": splits.campaign_id}


def _state(config: Mapping[str, Any], *, status: str, optimizer_updates: int,
           run_updates: int, actual_wall: float, run_wall: float,
           records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best = select_record(records) if records else None
    return {"schema_version": "uhi-cdc-g246-metric-state-v1", "status": status,
            **dict(config), "optimizer_updates": optimizer_updates,
            "run_optimizer_updates": run_updates,
            "actual_training_wall_seconds": actual_wall,
            "run_training_wall_seconds": run_wall,
            "last_validation_update": int(records[-1]["update"]) if records else None,
            "best_update": int(best["update"]) if best else None,
            "updated_utc": _utc_now(), "locked_test_opened": False}


def _history(config: Mapping[str, Any], *, status: str, optimizer_updates: int,
             run_updates: int, actual_wall: float, run_wall: float,
             records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "uhi-cdc-g246-metric-history-v1", "status": status,
            "config": dict(config), "optimizer_updates": optimizer_updates,
            "run_optimizer_updates": run_updates,
            "actual_training_wall_seconds": actual_wall,
            "run_training_wall_seconds": run_wall, "records": list(records),
            "selected": selected_summary(select_record(records)) if records else None,
            "locked_test_opened": False}


def _checkpoint(config: Mapping[str, Any], model: nn.Module, optimizer: torch.optim.Optimizer,
                scaler: Any, ema: EMA, *, update: int, draw: int, actual_wall: float,
                records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "uhi-cdc-g246-metric-checkpoint-v1", "config": dict(config),
            "model_state": model.state_dict(), "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema.shadow,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(), "ema_state": ema.state_dict(),
            "optimizer_updates": update, "sample_draw": draw,
            "actual_training_wall_seconds": actual_wall, "records": list(records),
            "locked_test_opened": False}


def _validate_args(args: argparse.Namespace) -> None:
    normalize_role(args.role)  # Before any path operation.
    reject_forbidden_path(args.split_receipt)
    reject_forbidden_path(args.output)
    if args.model == "resunet" and args.width not in ResidualUNet.widths:
        raise ValueError(f"resunet width must be one of {ResidualUNet.widths}")
    if args.model == "ocnir" and args.width not in OCNIR.widths:
        raise ValueError(f"ocnir width must be one of {OCNIR.widths}")
    if args.model == "ocnir" and not args.projection:
        raise ValueError("OCNIR cannot disable projection")
    if args.physical_batch_size <= 0 or args.grad_accumulation_steps <= 0:
        raise ValueError("batch and accumulation must be positive")
    if args.physical_batch_size * args.grad_accumulation_steps != EFFECTIVE_BATCH:
        raise ValueError("physical_batch_size * grad_accumulation_steps must equal 32")
    if args.max_updates <= 0 or args.validation_interval <= 0 or args.max_wall_minutes < 0:
        raise ValueError("update/validation bounds must be positive and wall bound nonnegative")


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    splits = load_splits(args.split_receipt, args.role)
    output = reject_forbidden_path(args.output).resolve()
    state_path = output / "state.json"
    last_path = output / "last.pt"
    if output.exists() and not args.resume and any(output.iterdir()):
        raise FileExistsError("non-empty output requires --resume; refusing to overwrite")
    if args.resume and (not state_path.is_file() or not last_path.is_file()):
        raise FileNotFoundError("--resume requires state.json and last.pt")
    output.mkdir(parents=True, exist_ok=True)
    normalization_path = output / "normalization.json"
    if args.resume:
        normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
        normalization = GlobalNormalization.from_dict(normalization_payload)
        if normalization.fit_view_sha256 != splits.fit_view_sha256:
            raise ValueError("resume normalization is not bound to the current fit view")
    else:
        normalization = fit_normalization(splits.fit, splits.fit_view_sha256)
        atomic_json(normalization_path, normalization.to_dict())
    if (normalization.fit_city_count != len({entry.city for entry in splits.fit})
            or normalization.fit_scene_count != len(splits.fit)):
        raise ValueError("normalization provenance counts differ from the fit view")
    normalization_sha = hashlib.sha256(normalization_path.read_bytes()).hexdigest()
    config = _config(args, splits, normalization_sha)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(args.seed)
    model = build_model(args.model, args.width, args.projection,
                        args.activation_checkpointing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=WEIGHT_DECAY)
    scaler = _scaler(device, args.amp)
    ema = EMA(model)
    update = draw = 0
    cumulative_wall = 0.0
    records: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "uhi-cdc-g246-metric-checkpoint-v1":
            raise ValueError("unsupported resume checkpoint")
        if checkpoint.get("config") != config:
            raise ValueError("resume configuration differs (max-wall may change; scientific config may not)")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint["model_state"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        ema.load_state_dict(checkpoint["ema_state"])
        update, draw = int(checkpoint["optimizer_updates"]), int(checkpoint["sample_draw"])
        _require_resume_progress(args.max_updates, update)
        cumulative_wall = float(checkpoint["actual_training_wall_seconds"])
        records = list(checkpoint.get("records", []))
    fit_data = G246Dataset(splits.fit, normalization, seed=args.seed, augment=True)
    validation_data = G246Dataset(splits.validation, normalization, seed=args.seed, augment=False)
    run_updates = 0
    run_wall = 0.0
    atomic_json(state_path, _state(config, status="running", optimizer_updates=update,
                                   run_updates=0, actual_wall=cumulative_wall,
                                   run_wall=0.0, records=records))
    status = "complete"
    latest_loss: float | None = None

    def validate_and_publish() -> None:
        nonlocal records
        backup = ema.apply(model)
        try:
            validation = evaluate_validation(model, validation_data, device,
                                             batch_size=args.physical_batch_size, amp=args.amp)
        finally:
            model.load_state_dict(backup, strict=True)
        record = {"update": update, "run_optimizer_updates": run_updates,
                  "actual_training_wall_seconds": cumulative_wall + run_wall,
                  "train_loss": latest_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                  "validation": validation}
        records = [x for x in records if int(x["update"]) != update] + [record]
        records.sort(key=lambda x: int(x["update"]))
        payload = _checkpoint(config, model, optimizer, scaler, ema, update=update, draw=draw,
                              actual_wall=cumulative_wall + run_wall, records=records)
        atomic_torch_save(last_path, payload)
        if int(select_record(records)["update"]) == update:
            atomic_torch_save(output / "best.pt", payload)
        atomic_json(output / "history.json", _history(config, status="running",
                    optimizer_updates=update, run_updates=run_updates,
                    actual_wall=cumulative_wall + run_wall, run_wall=run_wall, records=records))

    try:
        while update < args.max_updates:
            if args.max_wall_minutes and run_wall >= args.max_wall_minutes * 60.0:
                status = "wall_time_reached"
                break
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            segment_start = time.monotonic()
            for _ in range(args.grad_accumulation_steps):
                batch = _device_batch(fit_data.batch(draw, args.physical_batch_size), device)
                draw += args.physical_batch_size
                with _autocast(device, args.amp):
                    prediction = model(batch["fine"], batch["coarse_k"],
                                       batch["support120"], batch["context"])
                    loss, _ = masked_metric_loss(prediction, batch["target_k"],
                                                 batch["valid"], batch["eligible"])
                    scaled_loss = loss / args.grad_accumulation_steps
                scaler.scale(scaled_loss).backward()
                total_loss += float(loss.detach().cpu())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            next_update = update + 1
            lr = learning_rate(next_update, args.max_updates, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            prior_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = not scaler.is_enabled() or float(scaler.get_scale()) >= prior_scale
            if step_succeeded:
                ema.update(model)
            else:
                run_wall += time.monotonic() - segment_start
                continue
            update = next_update
            run_updates += 1
            latest_loss = total_loss / args.grad_accumulation_steps
            run_wall += time.monotonic() - segment_start
            if update % args.validation_interval == 0:
                validate_and_publish()
        # A short wall-bounded anchor still always produces a selected best.pt.
        if not records or int(records[-1]["update"]) != update:
            validate_and_publish()
    except KeyboardInterrupt:
        status = "interrupted"
        if update > 0 and (not records or int(records[-1]["update"]) != update):
            validate_and_publish()
    except BaseException:
        atomic_json(state_path, _state(config, status="failed", optimizer_updates=update,
                    run_updates=run_updates, actual_wall=cumulative_wall + run_wall,
                    run_wall=run_wall, records=records))
        raise
    final_actual = cumulative_wall + run_wall
    if records:
        payload = _checkpoint(config, model, optimizer, scaler, ema, update=update, draw=draw,
                              actual_wall=final_actual, records=records)
        atomic_torch_save(last_path, payload)
    atomic_json(output / "history.json", _history(config, status=status,
                optimizer_updates=update, run_updates=run_updates, actual_wall=final_actual,
                run_wall=run_wall, records=records))
    final_state = _state(config, status=status, optimizer_updates=update,
                         run_updates=run_updates, actual_wall=final_actual,
                         run_wall=run_wall, records=records)
    atomic_json(state_path, final_state)
    return final_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-receipt", type=Path, default=DEFAULT_SPLIT_RECEIPT)
    parser.add_argument("--role", default="fit+validation", choices=("fit+validation",))
    parser.add_argument("--model", choices=("resunet", "ocnir"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, choices=(1e-4, 2e-4), default=2e-4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0,
                        help="training-loop minutes for this invocation; excludes normalization/validation")
    parser.add_argument("--max-updates", type=int, default=20_000,
                        help="cumulative optimizer-update target")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-interval", type=int, default=2_000)
    parser.add_argument("--physical-batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation-steps", type=int, default=16)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = train(args)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
