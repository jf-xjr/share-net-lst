#!/usr/bin/env python3
"""Metric-first trainer for G246 R2 deterministic Q-field models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


WORKSPACE = Path(__file__).resolve().parents[1]
CODE_ROOT = WORKSPACE / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from g246_data import (  # noqa: E402
    DEFAULT_SPLIT_RECEIPT,
    G246Scene,
    load_splits,
    normalize_role,
    reject_forbidden_path,
)
from g246_contrast_q import ContrastQParent  # noqa: E402
from g246_calibrated_continuous_q import SceneCalibratedContinuousQ  # noqa: E402
from g246_ipmr_q import IPMRQ, IPMRQComponents  # noqa: E402
from g246_ipmr_q_pcqm import (  # noqa: E402
    IPMRQPCQM,
    IPMRQPCQMComponents,
)
from g246_ipmr_q_v2 import IPMRQV2, IPMRQV2Components  # noqa: E402
from g246_dcf_q import DCFQ, DCFQComponents  # noqa: E402
from g246_aom_q import AOMQ, AOMQComponents  # noqa: E402
from g246_continuous_scale_bridge_q import ContinuousScaleBridgeQ  # noqa: E402
from g246_dexchange_q import DExchangeContinuousScaleBridgeQ  # noqa: E402
from g246_parent_router_q import ParentRouterQ  # noqa: E402
from g246_q_bands import orthogonal_q_bands  # noqa: E402
from g246_qparent import MTStyleQParent  # noqa: E402
from g246_r2_data import (  # noqa: E402
    CONTEXT_DIM,
    FINE_CHANNELS,
    R2Normalization,
    R2TemporalDataset,
    build_fit201_dev12,
    fit_r2_normalization,
    region_equal_per_scene_loss,
)
from g246_r2_multisource import (  # noqa: E402
    MULTISOURCE_CONTEXT_DIM,
    MULTISOURCE_FINE_CHANNELS,
    MultiSourceTemporalDataset,
)
from ocnir import OCNIR, support_project  # noqa: E402
from train_g246_metric import (  # noqa: E402
    EMA,
    _mean_metrics,
    atomic_json,
    atomic_torch_save,
    scene_metrics,
)


DEFAULT_OUTPUT = WORKSPACE / "artifacts/g246_r2/runs/dev"
EFFECTIVE_BATCH_SIZE = 32
EMA_DECAY = 0.999
# Registered defaults from the metric-first campaign.  Keep these configurable
# so legacy engineering checkpoints (500 / 1e-2) remain explicitly resumable,
# while every new run uses the approved 2000 / 1e-4 optimization contract.
WARMUP_UPDATES = 2_000
WEIGHT_DECAY = 1.0e-4
FULL_SCENE_START = 0
VALIDATION_INTERVAL = 2_000
MIN_LEARNING_RATE = 1.0e-6
RMSE_TIE_TOLERANCE_K = 0.01
PRIMARY_LOSS_WEIGHT = 0.8
EMA_STRATEGY = "first_successful_step_copy_then_fixed_decay_v1"
LEGACY_EMA_STRATEGY = "legacy_fixed_decay_v1"
# At 5 / (1-decay), the first optimized state contributes about exp(-5)<0.7%.
EMA_SELECTOR_MATURITY_UPDATES = 5_000
EARLY_STOP_INTERVALS = 3
EARLY_STOP_MIN_IMPROVEMENT_K = 0.01
STATE_HEARTBEAT_INTERVAL_UPDATES = 100
MAX_CONSECUTIVE_AMP_SKIPS = 2
Q_SHAPE_MAX_WEIGHT = 0.15
Q_SHAPE_SUGGESTED_WEIGHT = 0.05
IPMR_MIDDLE_AUX_WEIGHT = 0.05
IPMR_HIGH_AUX_WEIGHT = 0.025
IPMR_AUX_DECAY_START_FRACTION = 0.40
IPMR_AUX_DECAY_END_FRACTION = 0.75
PCQM_MODEL_NAME = "ipmr_q_pcqm"
PCQM_PARAMETER_BUDGET = 19_373
IPMR_MODEL_NAMES = frozenset(("ipmr_q", "ipmr_q_v2", PCQM_MODEL_NAME))
PARENT_ROUTER_MODEL_NAME = "parent_router_q"
DCF_MODEL_NAME = "dcf_q"
AOM_MODEL_NAME = "aom_q"
U1LITE_MODEL_NAME = "u1lite_q"
U1LITE_DISTILL_CHECKPOINT_SCHEMA = (
    "uhi-cdc-g246-u1lite-distill-checkpoint-v1"
)
U1LITE_D1_QUALIFICATION_SCHEMA = "g246-u1lite-d1-qualification-v1"
U1LITE_TARGET_INITIALIZATION_SCHEMA = (
    "g246-u1lite-target-weights-only-initialization-v1"
)
U1LITE_DIRECT_INITIALIZATION_SCHEMA = (
    "g246-u1lite-direct-r6a-initialization-v1"
)
U1LITE_DIRECT_NORMALIZATION_SCHEMA = (
    "g246-u1lite-direct-r6a-normalization-reuse-v1"
)
U1LITE_DIRECT_RESUME_POLICY = (
    "fresh_registered_r6a_raw_u2000_exact_u0_once_then_resume_only_from_last_pt"
)
BAND_MODEL_NAMES = frozenset((*IPMR_MODEL_NAMES, DCF_MODEL_NAME))
DUAL_EXPERT_MODEL_NAMES = frozenset((
    PARENT_ROUTER_MODEL_NAME, DCF_MODEL_NAME, AOM_MODEL_NAME,
))
PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256 = (
    "7ef7a24c251f9496f025152c8267a17d4b0af525df1f0bb76a13c3c2b40c5a8c"
)
PARENT_ROUTER_DEEP_CHECKPOINT_SHA256 = (
    "54b0bb580ae7ff0c2fdb6e95fe892dbf3d35c69c9a168617e7443b9dc5a5c667"
)
PARENT_ROUTER_NORMALIZATION_SHA256 = (
    "452d909e209b9e022067a5f0b9240f4ac7f684e51f9be04b6501beb8631de36c"
)
PARENT_ROUTER_SHALLOW_STATE_SHA256 = (
    "c6581d1aec8d6dcacbe1a17a3067657c029027a925df9fe11c5e705d449117e4"
)
PARENT_ROUTER_DEEP_STATE_SHA256 = (
    "5b9bf263005e58543d93a0d23380e1bb81fd5ffad17d082ce5d0abc50ba3e12d"
)
# PCQM is not a generic IPMR-Q extension screen.  Its registered u0 is the
# independently selected r9 *EMA* anchor below; accepting a merely compatible
# IPMR-Q checkpoint would change both the hypothesis and every absolute gate.
PCQM_ANCHOR_CHECKPOINT_SHA256 = PARENT_ROUTER_DEEP_CHECKPOINT_SHA256
PCQM_ANCHOR_STATE_SHA256 = PARENT_ROUTER_DEEP_STATE_SHA256
PCQM_ANCHOR_CHECKPOINT_ROLE = "best"
PCQM_ANCHOR_OPTIMIZER_UPDATES = 8_000
PCQM_ANCHOR_STATE_KEY = "ema_model_state_dict"
INITIALIZATION_SCHEMA = "g246-r2-weights-only-initialization-v1"
ALLOCATION_ADAPTER_SCHEMA = "g246-support-aware-allocation-v2"
ALLOCATION_ADAPTER_CONTRACT = {
    "schema_version": ALLOCATION_ADAPTER_SCHEMA,
    "enabled": True,
    "phase_parameters": "none_shared_pointwise_token_rule",
    "base_score": "detached_old_raw_q_support_parent_demean_fp32_rms",
    "delta_score": "shared_1x1_zero_initialized",
    "competition": "support_masked_softmax_within_parent4x4_fp32",
    "normalization": "fp32_support_parent_demean_rms_count_ge_2",
    "degenerate_support": "count_lt_2_exact_zero_torch_where",
    "symmetry": "D4_equivariant_shared_within_parent_rule",
    "max_amplitude_k": 4.0,
    "amplitude_initialization": "parent_local_channel_zero_weight_bias",
    "temperature_bounds": [0.25, 4.0],
    "initial_temperature": 1.0,
    "global_residual_gate": "absent",
}
T3_FUSION_SCHEMA = "g246-query-relative-t3-fusion-v1"
T3_FUSION_CONTRACT = {
    "schema_version": T3_FUSION_SCHEMA,
    "enabled": True,
    "baseline": "literal_query_only_registered_single_path",
    "aggregation": "query_relative_auxiliary_permutation_invariant",
    "availability": "strict_mask_before_normalization_and_reduction",
    "similarity": "shared_descriptor_context_doy_cosine_distance_gate",
    "residuals": "fine_parent_scene_independent_zero_gates",
    "phase_parameters": "none_shared_1x1_rules",
    "initial_residual_gates": [0.0, 0.0, 0.0],
}
Q_REFINER_SCHEMA = "g246-q-residual-refiner-v2"
Q_REFINER_CONTRACT = {
    "schema_version": Q_REFINER_SCHEMA,
    "enabled": True,
    "placement": "learned_raw_q_before_existing_support_aware_q_projection",
    "inputs": (
        "support_aware_delivered_q0_parent_contrast_rms_centered_fine52_support_"
        "coarse_valid_fine_parent_scene_features"
    ),
    "rms_use": "input_only_no_division_no_softmax",
    "spatial_rule": "shared_D4_symmetric_depthwise_residual_no_phase_parameters",
    "amplitude": "parent_local_half_bounded_gain_times_centered_q0_plus_signed_bounded_residual",
    "old_q_total_scale_bounds": [0.5, 1.5],
    "max_residual_k": 4.0,
    "identity_initialization": "exact_zero_parent_gain_and_signed_residual_heads",
    "coarse_closure": "existing_support_project_after_refiner",
}
CONTENT_Q_PYRAMID_SCHEMA = "g246-content-only-q-pyramid-v1"
CONTENT_Q_PYRAMID_CONTRACT = {
    "schema_version": CONTENT_Q_PYRAMID_SCHEMA,
    "enabled": True,
    "coordinate_zeroing": {
        "context": "Fine52/Context19",
        "indices": [5, 6, 7, 8],
        "names": [
            "latitude_sin",
            "latitude_cos",
            "longitude_sin",
            "longitude_cos",
        ],
        "timing": "before_any_core_operation",
    },
    "module_inputs": "content_only_no_context_scene_or_ID_input",
    "forbidden_module_inputs": [
        "context",
        "scene",
        "region_id",
        "city_id",
        "latitude",
        "longitude",
    ],
    "pyramid_scales": [160, 80, 40],
    "D4_kernel_sizes": [7, 9, 7],
    "content_decomposition": "G13/G5-G13/I-G5",
    "extension_initialization": "signed_head_40_80_160_exact_zero",
    "full_model_initialization": (
        "no_geolocation_rebase_not_numerically_identical_to_r2k"
    ),
    "coarse_closure": "final_support_weighted_parent_Q_exact_zero",
}
PARENT_DCT15_SCHEMA = "g246-parent-dct15-q-v1"
PARENT_DCT15_CONTRACT = {
    "schema_version": PARENT_DCT15_SCHEMA,
    "enabled": True,
    "coordinate_zeroing": {
        "context": "Fine52/Context19",
        "indices": [5, 6, 7, 8],
        "names": [
            "latitude_sin",
            "latitude_cos",
            "longitude_sin",
            "longitude_cos",
        ],
        "timing": "before_any_core_operation",
    },
    "module_inputs": ["Fine52", "delivered_q0", "support", "coarse_valid"],
    "forbidden_module_inputs": [
        "context",
        "scene",
        "region_id",
        "city_id",
        "latitude",
        "longitude",
    ],
    "parent_geometry": "fixed_nonoverlapping_4x4_children",
    "basis": "fixed_orthonormal_4x4_DCT_15_non_DC_persistent_buffer",
    "decoder_scales": [40, 20, 10],
    "fine_compression_width": 20,
    "max_coefficient_k": 4.0,
    "extension_initialization": "coefficient_head_weight_and_bias_exact_zero",
    "full_model_initialization": (
        "no_geolocation_rebase_not_numerically_identical_to_r2k"
    ),
    "correction_projection": "support_weighted_parent_Pi_s_shape_only",
    "coarse_closure": "coarse_valid_parent_Q_only_after_correction",
    "D4_equivariance": "not_claimed_orientation_specific_DCT_coefficients",
}
NO_GEO_CORE_SCHEMA = "g246-physical-context15-core-v1"
NO_GEO_CONTEXT_INDICES = (5, 6, 7, 8)
NO_GEO_DESCRIPTOR_WEIGHT_KEY = "date_encoder.descriptor.0.weight"
NO_GEO_RESET_PREFIXES = (
    "date_encoder.descriptor.",
    "date_encoder.parent_film.",
    "scene_fusion.",
    "continuous_head.scene_film.",
    "continuous_head.scene_gate.",
    "continuous_head.scene_gain.",
)
NO_GEO_PROBE_SCHEMA = "g246-no-geo-head-backbone-probe-v1"
NO_GEO_PROBE_HEAD_PREFIXES = (
    "date_encoder.descriptor.",
    "date_encoder.parent_film.",
    "scene_fusion.",
    "continuous_head.",
)
NO_GEO_PROBE_BACKBONE_PREFIXES = (
    "date_encoder.fine_stem.",
    "date_encoder.pack.",
    "date_encoder.physical.",
    "date_encoder.merge.",
    "date_encoder.parent_blocks.",
    "temporal_fusion.",
    "context_unet.",
)
NO_GEO_PROBE_EXPECTED = {
    "head_conditioning": {"tensor_count": 65, "parameter_count": 313_308},
    "encoder_backbone": {"tensor_count": 159, "parameter_count": 3_683_680},
    "full": {"tensor_count": 224, "parameter_count": 3_996_988},
}
NO_GEO_PACK_SCHEMA = "g246-no-geo-hierpack-screen-v1"
NO_GEO_LEGACY_PACK_KEYS = (
    "date_encoder.pack.0.weight",
    "date_encoder.pack.1.weight",
    "date_encoder.pack.1.bias",
)
NO_GEO_LEGACY_PACK_SHAPES = {
    "date_encoder.pack.0.weight": (144, 1024, 1, 1),
    "date_encoder.pack.1.weight": (144,),
    "date_encoder.pack.1.bias": (144,),
}
NO_GEO_HIERARCHICAL_PACK_KEYS = (
    "date_encoder.hierarchical_pack.down_half.0.weight",
    "date_encoder.hierarchical_pack.down_half.1.weight",
    "date_encoder.hierarchical_pack.down_half.2.weight",
    "date_encoder.hierarchical_pack.down_half.2.bias",
    "date_encoder.hierarchical_pack.blocks_half.0.scale",
    "date_encoder.hierarchical_pack.blocks_half.0.depthwise.weight",
    "date_encoder.hierarchical_pack.blocks_half.0.norm.weight",
    "date_encoder.hierarchical_pack.blocks_half.0.norm.bias",
    "date_encoder.hierarchical_pack.blocks_half.0.expand.weight",
    "date_encoder.hierarchical_pack.blocks_half.0.expand.bias",
    "date_encoder.hierarchical_pack.blocks_half.0.contract.weight",
    "date_encoder.hierarchical_pack.blocks_half.0.contract.bias",
    "date_encoder.hierarchical_pack.blocks_half.1.scale",
    "date_encoder.hierarchical_pack.blocks_half.1.depthwise.weight",
    "date_encoder.hierarchical_pack.blocks_half.1.norm.weight",
    "date_encoder.hierarchical_pack.blocks_half.1.norm.bias",
    "date_encoder.hierarchical_pack.blocks_half.1.expand.weight",
    "date_encoder.hierarchical_pack.blocks_half.1.expand.bias",
    "date_encoder.hierarchical_pack.blocks_half.1.contract.weight",
    "date_encoder.hierarchical_pack.blocks_half.1.contract.bias",
    "date_encoder.hierarchical_pack.down_parent.0.weight",
    "date_encoder.hierarchical_pack.down_parent.1.weight",
    "date_encoder.hierarchical_pack.down_parent.2.weight",
    "date_encoder.hierarchical_pack.down_parent.2.bias",
)
NO_GEO_PACK_EXPECTED = {
    "legacy_reset_control": {"tensor_count": 224, "parameter_count": 3_996_988},
    "hierarchical": {"tensor_count": 245, "parameter_count": 3_987_868},
}


def _no_geo_core_contract(
    rebase_mode: str, *, probe_mode: str | None = None
) -> dict[str, Any]:
    """Return the immutable contract for a physical Context15 core arm."""

    if rebase_mode not in {"surgery", "reset_conditioning"}:
        raise ValueError("unsupported no-geolocation rebase mode")
    if probe_mode not in {None, "head_conditioning", "encoder_backbone"}:
        raise ValueError("unsupported no-geolocation probe mode")
    return {
        "schema_version": NO_GEO_CORE_SCHEMA,
        "enabled": True,
        "stored_input_contract": "Fine52/Context19",
        "physical_core_input_contract": "Fine52/Context15",
        "physical_context_dim": 15,
        "removed_context_indices": list(NO_GEO_CONTEXT_INDICES),
        "removed_context_names": [
            "latitude_sin",
            "latitude_cos",
            "longitude_sin",
            "longitude_cos",
        ],
        "removal": "physical_column_removal_before_any_learned_core_operation",
        "descriptor_layout": ["mean52", "std52", "context15", "coarse6"],
        "rebase_mode": rebase_mode,
        "reset_prefixes": (
            list(NO_GEO_RESET_PREFIXES)
            if rebase_mode == "reset_conditioning" else []
        ),
        "optimizer": (
            "single_group_all_parameters_full_scheduled_learning_rate"
            if probe_mode is None
            else "delegated_to_no_geo_probe_optimization_contract"
        ),
    }


class WarmStartEMA(EMA):
    """EMA without a long-lived contribution from the random initialization.

    The first successful optimizer update is copied exactly.  Every later
    update uses the registered fixed ``0.999`` decay.  ``num_updates`` and the
    strategy are checkpointed, so an interrupted run is bit-for-bit equivalent
    to an uninterrupted update sequence.

    A v1 checkpoint produced before this strategy existed has neither field.
    Such a checkpoint stays on its original fixed-decay rule after resume; this
    avoids silently changing the meaning of an existing EMA trajectory.
    """

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY) -> None:
        super().__init__(model, decay=decay)
        self.strategy = EMA_STRATEGY
        self.num_updates = 0

    @property
    def effective_decay(self) -> float:
        if self.strategy == LEGACY_EMA_STRATEGY:
            return self.decay
        return 0.0 if self.num_updates == 0 else self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        effective_decay = self.effective_decay
        for key, value in model.state_dict().items():
            source = value.detach()
            if self.shadow[key].is_floating_point():
                self.shadow[key].mul_(effective_decay).add_(
                    source, alpha=1.0 - effective_decay
                )
            else:
                self.shadow[key].copy_(source)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "strategy": self.strategy,
            "num_updates": self.num_updates,
        }

    def load_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        optimizer_updates: int | None = None,
    ) -> None:
        strategy = payload.get("strategy")
        if strategy is None:
            # Backward-compatible continuation of pre-warm-start checkpoints.
            super().load_state_dict(payload)
            self.strategy = LEGACY_EMA_STRATEGY
            self.num_updates = int(optimizer_updates or 0)
            return
        if strategy not in {EMA_STRATEGY, LEGACY_EMA_STRATEGY}:
            raise ValueError(f"unsupported EMA strategy: {strategy!r}")
        super().load_state_dict(payload)
        num_updates = int(payload.get("num_updates", -1))
        if num_updates < 0:
            raise ValueError("EMA checkpoint is missing a nonnegative num_updates")
        if optimizer_updates is not None and num_updates != int(optimizer_updates):
            raise ValueError("EMA update count differs from optimizer checkpoint")
        self.strategy = str(strategy)
        self.num_updates = num_updates


class ParentRouterEMA(WarmStartEMA):
    """EMA only the learned router; copy both registered experts bit-exactly."""

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not isinstance(model, ParentRouterQ):
            raise TypeError("ParentRouterEMA requires ParentRouterQ")
        effective_decay = self.effective_decay
        for key, value in model.state_dict().items():
            source = value.detach()
            if key.startswith("router.") and self.shadow[key].is_floating_point():
                self.shadow[key].mul_(effective_decay).add_(
                    source, alpha=1.0 - effective_decay
                )
            else:
                # Even x*.999+x*.001 can move a float by one ULP.  Exact copy
                # is required for immutable source-expert semantics.
                self.shadow[key].copy_(source)
        self.num_updates += 1

    @torch.no_grad()
    def validate_bound_model(self, model: nn.Module) -> None:
        """Require every non-router shadow tensor to match the model bitwise."""

        if not isinstance(model, ParentRouterQ):
            raise TypeError("ParentRouterEMA validation requires ParentRouterQ")
        state = model.state_dict()
        if set(state) != set(self.shadow):
            raise ValueError("ParentRouterEMA state keys differ from the model")
        for key, value in state.items():
            if key.startswith("router."):
                continue
            shadow = self.shadow[key]
            source = value.detach()
            if shadow.shape != source.shape or shadow.dtype != source.dtype \
                    or not torch.equal(shadow, source):
                raise ValueError(
                    f"ParentRouterEMA immutable state drifted at {key!r}"
                )


class DCFEMA(WarmStartEMA):
    """EMA only DCF-Q's cross-decoder; copy both experts bit-exactly."""

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not isinstance(model, DCFQ):
            raise TypeError("DCFEMA requires DCFQ")
        effective_decay = self.effective_decay
        for key, value in model.state_dict().items():
            source = value.detach()
            if key.startswith("decoder.") and self.shadow[key].is_floating_point():
                self.shadow[key].mul_(effective_decay).add_(
                    source, alpha=1.0 - effective_decay
                )
            else:
                self.shadow[key].copy_(source)
        self.num_updates += 1

    @torch.no_grad()
    def validate_bound_model(self, model: nn.Module) -> None:
        if not isinstance(model, DCFQ):
            raise TypeError("DCFEMA validation requires DCFQ")
        state = model.state_dict()
        if set(state) != set(self.shadow):
            raise ValueError("DCFEMA state keys differ from the model")
        for key, value in state.items():
            if key.startswith("decoder."):
                continue
            shadow = self.shadow[key]
            source = value.detach()
            if shadow.shape != source.shape or shadow.dtype != source.dtype \
                    or not torch.equal(shadow, source):
                raise ValueError(f"DCFEMA immutable state drifted at {key!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrainingWallTimer:
    """Accumulate only explicitly marked training-batch wall-clock segments."""

    def __init__(self, monotonic: Callable[[], float] | None = None) -> None:
        self._monotonic = monotonic or time.monotonic
        self.elapsed_seconds = 0.0

    @contextmanager
    def measure(self) -> Iterator[None]:
        started = self._monotonic()
        try:
            yield
        finally:
            self.elapsed_seconds += self._monotonic() - started


@contextmanager
def _exclusive_training_run_lock(output: Path) -> Iterator[Path]:
    """Reserve one output name for exactly one trainer process.

    The lock lives beside the output directory, so its existence does not make
    a fresh output look non-empty. ``O_EXCL`` is the cross-platform
    compare-and-set: two launches cannot both pass the emptiness check or
    overwrite each other's normalization/checkpoint files.
    """

    resolved_output = reject_forbidden_path(output).resolve()
    reject_forbidden_path(resolved_output)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = reject_forbidden_path(
        resolved_output.parent / f".{resolved_output.name}.training.lock"
    )
    if lock_path.is_symlink():
        raise ValueError("training run lock may not be a symlink")
    token = os.urandom(16).hex()
    document = {
        "schema_version": "g246-r2-exclusive-training-lock-v1",
        "token": token,
        "pid": os.getpid(),
        "output": str(resolved_output),
        "created_utc": _utc_now(),
    }
    raw = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"another trainer owns the output reservation: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            if lock_path.read_bytes() == raw:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _regular_public_artifact(path: Path, label: str) -> Path:
    """Resolve one trainer artifact without following a hidden role symlink."""

    candidate = reject_forbidden_path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    resolved = candidate.resolve()
    reject_forbidden_path(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 used by fail-closed run bindings."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish exact bytes without JSON reserialization drift."""

    target = reject_forbidden_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _parent_router_reuse_normalization(
    shallow_checkpoint: Path,
    deep_checkpoint: Path,
    target_path: Path,
) -> tuple[R2Normalization, dict[str, Any]]:
    """Verify two sibling normalizations and copy their identical bytes once."""

    rows: list[tuple[Path, bytes, str]] = []
    for checkpoint, label in (
        (shallow_checkpoint, "shallow"),
        (deep_checkpoint, "deep"),
    ):
        checkpoint_path = _regular_public_artifact(
            checkpoint, f"ParentRouterQ {label} source checkpoint"
        )
        normalization_path = _regular_public_artifact(
            checkpoint_path.parent / "normalization.json",
            f"ParentRouterQ {label} source normalization",
        )
        with normalization_path.open("rb") as handle:
            payload = handle.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != PARENT_ROUTER_NORMALIZATION_SHA256:
            raise ValueError(
                f"ParentRouterQ {label} normalization SHA-256 differs"
            )
        rows.append((normalization_path, payload, digest))
    if rows[0][1] != rows[1][1]:
        raise ValueError("ParentRouterQ expert normalization bytes differ")
    try:
        normalization_payload = json.loads(rows[0][1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ParentRouterQ source normalization JSON is malformed") from exc
    if not isinstance(normalization_payload, Mapping):
        raise ValueError("ParentRouterQ source normalization root is malformed")
    normalization = R2Normalization.from_dict(dict(normalization_payload))
    _atomic_bytes(target_path, rows[0][1])
    return normalization, {
        "schema_version": "g246-parent-router-normalization-reuse-v1",
        "source_normalization_sha256": rows[0][2],
        "sources_byte_identical": True,
        "target_exact_byte_copy": True,
        "source_files": [str(row[0]) for row in rows],
        "locked_test_opened": False,
    }


def _resume_checkpoint_config(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the resume envelope before exposing any state or records."""

    if checkpoint.get("schema_version") != "uhi-cdc-g246-r2-checkpoint-v1":
        raise ValueError("unsupported R2 checkpoint schema")
    checkpoint_role = checkpoint.get("checkpoint_role")
    if checkpoint_role not in {None, "last"}:
        raise ValueError("resume requires a last-role or legacy pre-role checkpoint")
    if checkpoint.get("locked_test_opened") is not False:
        raise ValueError("resume checkpoint is not locked-test closed")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("resume checkpoint config is missing")
    if config.get("locked_test_opened") is not False:
        raise ValueError("resume checkpoint config is not locked-test closed")
    return config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _autocast(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if enabled and device.type == "cuda"
        else nullcontext()
    )


def _scaler(device: torch.device, amp: bool):
    enabled = amp and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_step_with_finite_guard(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    consecutive_amp_skips: int,
    max_grad_norm: float = 1.0,
) -> tuple[bool, int, float]:
    """Perform one guarded optimizer attempt without ever inventing an update.

    The helper unscales gradients, then clips and explicitly checks them.  An
    enabled GradScaler is considered successful only
    when it starts and ends at a finite positive scale and does not reduce its
    scale (PyTorch's documented overflow/step-skip signal).  One isolated AMP
    overflow is retried at the same logical update; consecutive skips fail
    fast because deterministic replay would otherwise spin forever.
    """

    if consecutive_amp_skips < 0:
        raise ValueError("consecutive_amp_skips must be nonnegative")
    scaler_enabled = bool(scaler.is_enabled())
    prior_scale = float(scaler.get_scale())
    if scaler_enabled and (
        not math.isfinite(prior_scale) or prior_scale <= 0.0
    ):
        raise FloatingPointError(
            f"GradScaler has invalid pre-step scale: {prior_scale!r}"
        )
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise FloatingPointError("optimizer attempt has no parameter gradients")
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm
    )
    grad_norm_value = float(grad_norm.detach().cpu())
    if not math.isfinite(grad_norm_value) and not scaler_enabled:
        raise FloatingPointError(
            f"non-finite gradient norm after unscale/clip: {grad_norm_value!r}"
        )

    # A finite forward loss can still overflow its *scaled* fp16 gradients.
    # GradScaler records that condition during unscale_(), skips optimizer.step,
    # and lowers the scale in update().  Let it perform that documented
    # recovery before deciding whether the attempt was a real optimizer update.
    # The next loop iteration deterministically replays the same logical batch
    # at the lower scale, while EMA/update counters remain unchanged.
    scaler.step(optimizer)
    scaler.update()
    new_scale = float(scaler.get_scale())
    if scaler_enabled and (
        not math.isfinite(new_scale) or new_scale <= 0.0
    ):
        raise FloatingPointError(
            f"GradScaler has invalid post-step scale: {new_scale!r}"
        )

    optimizer_stepped = not scaler_enabled or (
        math.isfinite(grad_norm_value) and new_scale >= prior_scale
    )
    if optimizer_stepped:
        return True, 0, grad_norm_value
    if scaler_enabled and new_scale >= prior_scale:
        raise FloatingPointError(
            "non-finite gradient norm was not accompanied by an AMP scale "
            f"reduction: grad_norm={grad_norm_value!r}, "
            f"scale={prior_scale!r}->{new_scale!r}"
        )
    consecutive_amp_skips += 1
    if consecutive_amp_skips >= MAX_CONSECUTIVE_AMP_SKIPS:
        raise FloatingPointError(
            "GradScaler skipped consecutive deterministic optimizer attempts"
        )
    return False, consecutive_amp_skips, grad_norm_value


def learning_rate(
    update: int, maximum: int, peak: float, warmup_updates: int = WARMUP_UPDATES
) -> float:
    if warmup_updates < 0:
        raise ValueError("warmup_updates must be nonnegative")
    if warmup_updates and update <= warmup_updates:
        return peak * update / warmup_updates
    progress = (update - warmup_updates) / max(1, maximum - warmup_updates)
    progress = min(max(progress, 0.0), 1.0)
    return MIN_LEARNING_RATE + 0.5 * (peak - MIN_LEARNING_RATE) * (
        1.0 + math.cos(math.pi * progress)
    )


def q_refiner_optimizer_groups(
    model: nn.Module,
    *,
    base_learning_rate: float,
    core_lr_multiplier: float,
    refiner_lr_multiplier: float,
) -> list[dict[str, Any]]:
    """Build disjoint named parameter groups for an explicit Q-refiner run."""

    refiner = getattr(model, "q_refiner", None)
    if refiner is None:
        raise ValueError("Q-refiner optimizer groups require a refiner model")
    for value, label in (
        (base_learning_rate, "base learning rate"),
        (core_lr_multiplier, "core LR multiplier"),
        (refiner_lr_multiplier, "refiner LR multiplier"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    refiner_ids = {id(parameter) for parameter in refiner.parameters()}
    refiner_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in refiner_ids
    ]
    core_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in refiner_ids
    ]
    if not refiner_parameters or not core_parameters:
        raise ValueError("Q-refiner optimizer groups must both be nonempty")
    if len(refiner_ids) != len(refiner_parameters):
        raise ValueError("Q-refiner parameters are aliased in the model")
    return [
        {
            "params": core_parameters,
            "lr": float(base_learning_rate) * float(core_lr_multiplier),
            "lr_multiplier": float(core_lr_multiplier),
            "group_name": "core",
        },
        {
            "params": refiner_parameters,
            "lr": float(base_learning_rate) * float(refiner_lr_multiplier),
            "lr_multiplier": float(refiner_lr_multiplier),
            "group_name": "q_refiner",
        },
    ]


def set_q_refiner_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Apply the global-update freeze schedule without touching module modes."""

    refiner = getattr(model, "q_refiner", None)
    if refiner is None:
        if frozen:
            raise ValueError("cannot freeze a Q-refiner backbone without a refiner")
        return
    refiner_ids = {id(parameter) for parameter in refiner.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in refiner_ids or not frozen)


def set_optimizer_group_learning_rates(
    optimizer: torch.optim.Optimizer, base_learning_rate: float
) -> dict[str, float]:
    """Set scheduled group LRs and return an auditable name-to-rate mapping."""

    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        multiplier = float(group.get("lr_multiplier", 1.0))
        name = str(group.get("group_name", "all" if len(optimizer.param_groups) == 1 else index))
        rate = float(base_learning_rate) * multiplier
        group["lr"] = rate
        rates[name] = rate
    return rates


def validate_q_refiner_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    core_lr_multiplier: float,
    refiner_lr_multiplier: float,
    weight_decay: float,
) -> None:
    """Fail closed before loading Q-refiner optimizer state.

    ``Optimizer.load_state_dict`` restores parameter-group metadata as well as
    moments.  Without this check a damaged checkpoint could silently replace
    the LR multipliers or AdamW weight decay while the immutable scientific
    config continued to report the requested values.
    """

    if not isinstance(optimizer_state, Mapping):
        raise ValueError("Q-refiner optimizer checkpoint state is malformed")
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != 2:
        raise ValueError("Q-refiner resume requires exactly two optimizer groups")
    current_groups = optimizer.param_groups
    if len(current_groups) != 2:
        raise ValueError("constructed Q-refiner optimizer does not have two groups")
    expected = (
        ("core", float(core_lr_multiplier)),
        ("q_refiner", float(refiner_lr_multiplier)),
    )
    parameter_ids: list[Any] = []
    for index, (saved, current, (name, multiplier)) in enumerate(
        zip(saved_groups, current_groups, expected)
    ):
        if not isinstance(saved, Mapping):
            raise ValueError("Q-refiner optimizer group metadata is malformed")
        saved_parameters = saved.get("params")
        if not isinstance(saved_parameters, list) \
                or len(saved_parameters) != len(current["params"]):
            raise ValueError(
                f"Q-refiner optimizer parameter group {index} differs from the model"
            )
        if any(
            isinstance(parameter_id, bool) or not isinstance(parameter_id, int)
            for parameter_id in saved_parameters
        ):
            raise ValueError("Q-refiner optimizer parameter identifiers are malformed")
        parameter_ids.extend(saved_parameters)
        if saved.get("group_name") != name:
            raise ValueError("Q-refiner optimizer group names/order differ from config")
        saved_multiplier = saved.get("lr_multiplier")
        if isinstance(saved_multiplier, bool) \
                or not isinstance(saved_multiplier, (int, float)) \
                or not math.isfinite(float(saved_multiplier)) \
                or not math.isclose(
                    float(saved_multiplier), multiplier, rel_tol=0.0, abs_tol=0.0
                ):
            raise ValueError("Q-refiner optimizer LR multiplier differs from config")
        saved_weight_decay = saved.get("weight_decay")
        if isinstance(saved_weight_decay, bool) \
                or not isinstance(saved_weight_decay, (int, float)) \
                or not math.isfinite(float(saved_weight_decay)) \
                or not math.isclose(
                    float(saved_weight_decay), float(weight_decay),
                    rel_tol=0.0, abs_tol=0.0,
                ):
            raise ValueError("Q-refiner optimizer weight decay differs from config")
        for field in (
            "betas", "eps", "amsgrad", "maximize", "foreach",
            "capturable", "differentiable", "fused",
        ):
            if saved.get(field) != current.get(field):
                raise ValueError(
                    "Q-refiner optimizer AdamW algorithm metadata differs"
                )
        saved_lr = saved.get("lr")
        if isinstance(saved_lr, bool) or not isinstance(saved_lr, (int, float)) \
                or not math.isfinite(float(saved_lr)) or float(saved_lr) < 0.0:
            raise ValueError("Q-refiner optimizer checkpoint LR is invalid")
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Q-refiner optimizer parameter groups overlap")


def content_q_pyramid_optimizer_groups(
    model: nn.Module,
    *,
    base_learning_rate: float,
    core_lr_multiplier: float,
    pyramid_lr_multiplier: float,
) -> list[dict[str, Any]]:
    """Build disjoint named parameter groups for Content-only Q-Pyramid."""

    pyramid = getattr(model, "content_q_pyramid", None)
    if pyramid is None:
        raise ValueError("Q-Pyramid optimizer groups require a pyramid model")
    for value, label in (
        (base_learning_rate, "base learning rate"),
        (core_lr_multiplier, "core LR multiplier"),
        (pyramid_lr_multiplier, "pyramid LR multiplier"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    pyramid_ids = {id(parameter) for parameter in pyramid.parameters()}
    pyramid_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in pyramid_ids
    ]
    core_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in pyramid_ids
    ]
    if not pyramid_parameters or not core_parameters:
        raise ValueError("Q-Pyramid optimizer groups must both be nonempty")
    if len(pyramid_ids) != len(pyramid_parameters):
        raise ValueError("Q-Pyramid parameters are aliased in the model")
    return [
        {
            "params": core_parameters,
            "lr": float(base_learning_rate) * float(core_lr_multiplier),
            "lr_multiplier": float(core_lr_multiplier),
            "group_name": "core",
        },
        {
            "params": pyramid_parameters,
            "lr": float(base_learning_rate) * float(pyramid_lr_multiplier),
            "lr_multiplier": float(pyramid_lr_multiplier),
            "group_name": "content_q_pyramid",
        },
    ]


def set_content_q_pyramid_core_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze only the inherited r2k core on the successful-update clock."""

    pyramid = getattr(model, "content_q_pyramid", None)
    if pyramid is None:
        if frozen:
            raise ValueError("cannot freeze a Q-Pyramid core without a pyramid")
        return
    pyramid_ids = {id(parameter) for parameter in pyramid.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in pyramid_ids or not frozen)


def validate_content_q_pyramid_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    core_lr_multiplier: float,
    pyramid_lr_multiplier: float,
    weight_decay: float,
) -> None:
    """Fail closed before restoring the two-group Q-Pyramid optimizer."""

    if not isinstance(optimizer_state, Mapping):
        raise ValueError("Q-Pyramid optimizer checkpoint state is malformed")
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != 2:
        raise ValueError("Q-Pyramid resume requires exactly two optimizer groups")
    current_groups = optimizer.param_groups
    if len(current_groups) != 2:
        raise ValueError("constructed Q-Pyramid optimizer does not have two groups")
    expected = (
        ("core", float(core_lr_multiplier)),
        ("content_q_pyramid", float(pyramid_lr_multiplier)),
    )
    parameter_ids: list[Any] = []
    for index, (saved, current, (name, multiplier)) in enumerate(
        zip(saved_groups, current_groups, expected)
    ):
        if not isinstance(saved, Mapping):
            raise ValueError("Q-Pyramid optimizer group metadata is malformed")
        saved_parameters = saved.get("params")
        if not isinstance(saved_parameters, list) \
                or len(saved_parameters) != len(current["params"]):
            raise ValueError(
                f"Q-Pyramid optimizer parameter group {index} differs from the model"
            )
        if any(
            isinstance(parameter_id, bool) or not isinstance(parameter_id, int)
            for parameter_id in saved_parameters
        ):
            raise ValueError("Q-Pyramid optimizer parameter identifiers are malformed")
        parameter_ids.extend(saved_parameters)
        if saved.get("group_name") != name:
            raise ValueError("Q-Pyramid optimizer group names/order differ from config")
        saved_multiplier = saved.get("lr_multiplier")
        if isinstance(saved_multiplier, bool) \
                or not isinstance(saved_multiplier, (int, float)) \
                or not math.isfinite(float(saved_multiplier)) \
                or not math.isclose(
                    float(saved_multiplier), multiplier, rel_tol=0.0, abs_tol=0.0
                ):
            raise ValueError("Q-Pyramid optimizer LR multiplier differs from config")
        saved_weight_decay = saved.get("weight_decay")
        if isinstance(saved_weight_decay, bool) \
                or not isinstance(saved_weight_decay, (int, float)) \
                or not math.isfinite(float(saved_weight_decay)) \
                or not math.isclose(
                    float(saved_weight_decay), float(weight_decay),
                    rel_tol=0.0, abs_tol=0.0,
                ):
            raise ValueError("Q-Pyramid optimizer weight decay differs from config")
        for field in (
            "betas", "eps", "amsgrad", "maximize", "foreach",
            "capturable", "differentiable", "fused",
        ):
            if saved.get(field) != current.get(field):
                raise ValueError(
                    "Q-Pyramid optimizer AdamW algorithm metadata differs"
                )
        saved_lr = saved.get("lr")
        if isinstance(saved_lr, bool) or not isinstance(saved_lr, (int, float)) \
                or not math.isfinite(float(saved_lr)) or float(saved_lr) < 0.0:
            raise ValueError("Q-Pyramid optimizer checkpoint LR is invalid")
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Q-Pyramid optimizer parameter groups overlap")


def parent_dct15_optimizer_groups(
    model: nn.Module,
    *,
    base_learning_rate: float,
    core_lr_multiplier: float,
    dct15_lr_multiplier: float,
) -> list[dict[str, Any]]:
    """Build disjoint named groups for an explicit Parent-DCT15 run."""

    decoder = getattr(model, "parent_dct15", None)
    if decoder is None:
        raise ValueError("Parent-DCT15 optimizer groups require a decoder model")
    for value, label in (
        (base_learning_rate, "base learning rate"),
        (core_lr_multiplier, "core LR multiplier"),
        (dct15_lr_multiplier, "Parent-DCT15 LR multiplier"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    decoder_ids = {id(parameter) for parameter in decoder.parameters()}
    decoder_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) in decoder_ids
    ]
    core_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in decoder_ids
    ]
    if not decoder_parameters or not core_parameters:
        raise ValueError("Parent-DCT15 optimizer groups must both be nonempty")
    if len(decoder_ids) != len(decoder_parameters):
        raise ValueError("Parent-DCT15 parameters are aliased in the model")
    return [
        {
            "params": core_parameters,
            "lr": float(base_learning_rate) * float(core_lr_multiplier),
            "lr_multiplier": float(core_lr_multiplier),
            "group_name": "core",
        },
        {
            "params": decoder_parameters,
            "lr": float(base_learning_rate) * float(dct15_lr_multiplier),
            "lr_multiplier": float(dct15_lr_multiplier),
            "group_name": "parent_dct15",
        },
    ]


def set_parent_dct15_core_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze only the inherited core on the successful-update clock."""

    decoder = getattr(model, "parent_dct15", None)
    if decoder is None:
        if frozen:
            raise ValueError("cannot freeze a Parent-DCT15 core without a decoder")
        return
    decoder_ids = {id(parameter) for parameter in decoder.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in decoder_ids or not frozen)


def validate_parent_dct15_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    core_lr_multiplier: float,
    dct15_lr_multiplier: float,
    weight_decay: float,
) -> None:
    """Fail closed before restoring the two-group Parent-DCT15 optimizer."""

    if not isinstance(optimizer_state, Mapping):
        raise ValueError("Parent-DCT15 optimizer checkpoint state is malformed")
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != 2:
        raise ValueError("Parent-DCT15 resume requires exactly two optimizer groups")
    current_groups = optimizer.param_groups
    if len(current_groups) != 2:
        raise ValueError(
            "constructed Parent-DCT15 optimizer does not have two groups"
        )
    expected = (
        ("core", float(core_lr_multiplier)),
        ("parent_dct15", float(dct15_lr_multiplier)),
    )
    parameter_ids: list[Any] = []
    for index, (saved, current, (name, multiplier)) in enumerate(
        zip(saved_groups, current_groups, expected)
    ):
        if not isinstance(saved, Mapping):
            raise ValueError("Parent-DCT15 optimizer group metadata is malformed")
        saved_parameters = saved.get("params")
        if not isinstance(saved_parameters, list) \
                or len(saved_parameters) != len(current["params"]):
            raise ValueError(
                "Parent-DCT15 optimizer parameter group "
                f"{index} differs from the model"
            )
        if any(
            isinstance(parameter_id, bool) or not isinstance(parameter_id, int)
            for parameter_id in saved_parameters
        ):
            raise ValueError(
                "Parent-DCT15 optimizer parameter identifiers are malformed"
            )
        parameter_ids.extend(saved_parameters)
        if saved.get("group_name") != name:
            raise ValueError(
                "Parent-DCT15 optimizer group names/order differ from config"
            )
        saved_multiplier = saved.get("lr_multiplier")
        if isinstance(saved_multiplier, bool) \
                or not isinstance(saved_multiplier, (int, float)) \
                or not math.isfinite(float(saved_multiplier)) \
                or not math.isclose(
                    float(saved_multiplier), multiplier,
                    rel_tol=0.0, abs_tol=0.0,
                ):
            raise ValueError(
                "Parent-DCT15 optimizer LR multiplier differs from config"
            )
        saved_weight_decay = saved.get("weight_decay")
        if isinstance(saved_weight_decay, bool) \
                or not isinstance(saved_weight_decay, (int, float)) \
                or not math.isfinite(float(saved_weight_decay)) \
                or not math.isclose(
                    float(saved_weight_decay), float(weight_decay),
                    rel_tol=0.0, abs_tol=0.0,
                ):
            raise ValueError(
                "Parent-DCT15 optimizer weight decay differs from config"
            )
        for field in (
            "betas", "eps", "amsgrad", "maximize", "foreach",
            "capturable", "differentiable", "fused",
        ):
            if saved.get(field) != current.get(field):
                raise ValueError(
                    "Parent-DCT15 optimizer AdamW algorithm metadata differs"
                )
        saved_lr = saved.get("lr")
        if isinstance(saved_lr, bool) or not isinstance(saved_lr, (int, float)) \
                or not math.isfinite(float(saved_lr)) or float(saved_lr) < 0.0:
            raise ValueError("Parent-DCT15 optimizer checkpoint LR is invalid")
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Parent-DCT15 optimizer parameter groups overlap")


def validate_parent_router_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    expected_parameter_count: int,
    weight_decay: float,
) -> None:
    """Fail closed unless the only optimizer group is the small router."""

    saved_groups = optimizer_state.get("param_groups") \
        if isinstance(optimizer_state, Mapping) else None
    if not isinstance(saved_groups, list) or len(saved_groups) != 1 \
            or len(optimizer.param_groups) != 1:
        raise ValueError("ParentRouterQ resume requires one router optimizer group")
    saved = saved_groups[0]
    current = optimizer.param_groups[0]
    if not isinstance(saved, Mapping) \
            or saved.get("group_name") != "parent_router" \
            or current.get("group_name") != "parent_router":
        raise ValueError("ParentRouterQ optimizer group role differs")
    saved_parameters = saved.get("params")
    if not isinstance(saved_parameters, list) \
            or len(saved_parameters) != len(current["params"]) \
            or len(saved_parameters) != expected_parameter_count \
            or len(set(saved_parameters)) != len(saved_parameters):
        raise ValueError("ParentRouterQ optimizer parameter partition differs")
    saved_weight_decay = saved.get("weight_decay")
    if isinstance(saved_weight_decay, bool) \
            or not isinstance(saved_weight_decay, (int, float)) \
            or not math.isclose(
                float(saved_weight_decay), float(weight_decay),
                rel_tol=0.0, abs_tol=0.0,
            ):
        raise ValueError("ParentRouterQ optimizer weight decay differs")
    for field in (
        "betas", "eps", "amsgrad", "maximize", "foreach",
        "capturable", "differentiable", "fused",
    ):
        if saved.get(field) != current.get(field):
            raise ValueError(
                f"ParentRouterQ optimizer algorithm field {field!r} differs"
            )


def validate_dcf_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    expected_parameter_count: int,
    weight_decay: float,
) -> None:
    """Fail closed unless the only optimizer group is the DCF decoder."""

    saved_groups = optimizer_state.get("param_groups") \
        if isinstance(optimizer_state, Mapping) else None
    if not isinstance(saved_groups, list) or len(saved_groups) != 1 \
            or len(optimizer.param_groups) != 1:
        raise ValueError("DCF-Q resume requires one decoder optimizer group")
    saved = saved_groups[0]
    current = optimizer.param_groups[0]
    if not isinstance(saved, Mapping) \
            or saved.get("group_name") != "dcf_decoder" \
            or current.get("group_name") != "dcf_decoder":
        raise ValueError("DCF-Q optimizer group role differs")
    saved_parameters = saved.get("params")
    if not isinstance(saved_parameters, list) \
            or len(saved_parameters) != len(current["params"]) \
            or len(saved_parameters) != expected_parameter_count \
            or len(set(saved_parameters)) != len(saved_parameters):
        raise ValueError("DCF-Q optimizer parameter partition differs")
    saved_weight_decay = saved.get("weight_decay")
    if isinstance(saved_weight_decay, bool) \
            or not isinstance(saved_weight_decay, (int, float)) \
            or not math.isclose(
                float(saved_weight_decay), float(weight_decay),
                rel_tol=0.0, abs_tol=0.0,
            ):
        raise ValueError("DCF-Q optimizer weight decay differs")
    for field in (
        "betas", "eps", "amsgrad", "maximize", "foreach",
        "capturable", "differentiable", "fused",
    ):
        if saved.get(field) != current.get(field):
            raise ValueError(f"DCF-Q optimizer algorithm field {field!r} differs")


def validate_u1lite_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    expected_parameter_count: int,
    weight_decay: float,
) -> None:
    """Require the one complete T0 parameter group before state restore."""

    saved_groups = optimizer_state.get("param_groups") \
        if isinstance(optimizer_state, Mapping) else None
    if not isinstance(saved_groups, list) or len(saved_groups) != 1 \
            or len(optimizer.param_groups) != 1:
        raise ValueError("U1-Lite resume requires one all-parameter optimizer group")
    saved = saved_groups[0]
    current = optimizer.param_groups[0]
    if not isinstance(saved, Mapping) \
            or saved.get("group_name") != "u1lite_all" \
            or current.get("group_name") != "u1lite_all":
        raise ValueError("U1-Lite optimizer group role differs")
    saved_parameters = saved.get("params")
    if not isinstance(saved_parameters, list) \
            or len(saved_parameters) != len(current["params"]) \
            or len(saved_parameters) != expected_parameter_count \
            or len(set(saved_parameters)) != len(saved_parameters):
        raise ValueError("U1-Lite optimizer parameter coverage differs")
    if not math.isclose(
        float(saved.get("weight_decay", float("nan"))), float(weight_decay),
        rel_tol=0.0, abs_tol=0.0,
    ):
        raise ValueError("U1-Lite optimizer weight decay differs")
    for field in (
        "betas", "eps", "amsgrad", "maximize", "foreach",
        "capturable", "differentiable", "fused",
    ):
        if saved.get(field) != current.get(field):
            raise ValueError(f"U1-Lite optimizer field {field!r} differs")


def pcqm_optimization_contract(model: nn.Module) -> dict[str, Any]:
    """Bind the complete, and only, trainable PCQM parameter partition."""

    if not isinstance(model, IPMRQPCQM):
        raise TypeError("PCQM optimization contract requires IPMRQPCQM")
    model.freeze_anchor()
    named = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith("pcqm_")
    ]
    names = [name for name, _parameter in named]
    parameters = [parameter for _name, parameter in named]
    if tuple(map(id, parameters)) != tuple(
        id(parameter) for parameter in model.pcqm_parameters()
    ):
        raise ValueError("PCQM named parameter order differs from its partition")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("PCQM partition contains a frozen parameter")
    if any(
        parameter.requires_grad and not name.startswith("pcqm_")
        for name, parameter in model.named_parameters()
    ):
        raise ValueError("PCQM anchor contains a trainable parameter")
    count = sum(parameter.numel() for parameter in parameters)
    if count != PCQM_PARAMETER_BUDGET \
            or model.pcqm_parameter_count != PCQM_PARAMETER_BUDGET:
        raise ValueError(
            f"PCQM parameter budget differs ({count} != {PCQM_PARAMETER_BUDGET})"
        )
    return {
        "schema_version": "g246-ipmr-q-pcqm-optimizer-v1",
        "group_names": ["pcqm"],
        "optimizer_scope": "pcqm_parameters_only_anchor_frozen",
        "parameter_names": names,
        "parameter_names_sha256": _parameter_names_sha256(names),
        "parameter_tensor_count": len(parameters),
        "trainable_parameter_count": count,
        "registered_parameter_budget": PCQM_PARAMETER_BUDGET,
        "anchor_frozen": True,
    }


def pcqm_optimizer_groups(
    model: nn.Module, *, base_learning_rate: float
) -> list[dict[str, Any]]:
    """Construct the registered single PCQM-only AdamW parameter group."""

    contract = pcqm_optimization_contract(model)
    if not math.isfinite(base_learning_rate) or base_learning_rate <= 0.0:
        raise ValueError("PCQM learning rate must be positive")
    assert isinstance(model, IPMRQPCQM)
    parameters = list(model.pcqm_parameters())
    if len(parameters) != int(contract["parameter_tensor_count"]):
        raise AssertionError("PCQM optimizer partition changed after validation")
    return [{
        "params": parameters,
        "lr": float(base_learning_rate),
        "group_name": "pcqm",
    }]


def validate_pcqm_optimizer_resume_contract(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    weight_decay: float,
) -> None:
    """Fail closed before restoring a PCQM-only optimizer state."""

    expected = pcqm_optimization_contract(model)
    recorded = config.get("pcqm_optimizer_contract")
    if not isinstance(recorded, Mapping) or dict(recorded) != expected:
        raise ValueError("resume PCQM optimizer/config binding differs")
    saved = optimizer_state.get("param_groups") \
        if isinstance(optimizer_state, Mapping) else None
    if not isinstance(saved, list) or len(saved) != 1 \
            or len(optimizer.param_groups) != 1:
        raise ValueError("PCQM resume requires one optimizer group")
    current_group = optimizer.param_groups[0]
    saved_group = saved[0]
    if current_group.get("group_name") != "pcqm" \
            or saved_group.get("group_name") != "pcqm":
        raise ValueError("PCQM optimizer group role differs")
    expected_tensors = int(expected["parameter_tensor_count"])
    if len(current_group.get("params", ())) != expected_tensors \
            or len(saved_group.get("params", ())) != expected_tensors:
        raise ValueError("PCQM optimizer parameter count differs")
    if not math.isclose(
        float(saved_group.get("weight_decay", -1.0)), float(weight_decay),
        rel_tol=0.0, abs_tol=0.0,
    ):
        raise ValueError("PCQM optimizer weight decay differs")
    for field, expected_value in (("betas", (0.9, 0.999)), ("eps", 1e-8)):
        if saved_group.get(field) != expected_value:
            raise ValueError(f"PCQM optimizer field {field!r} differs")


def _pcqm_inherited_anchor_state(
    state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Extract the immutable r9 partition from a complete PCQM state."""

    inherited = {
        key: value for key, value in state.items()
        if not key.startswith("pcqm_")
    }
    if not inherited or len(inherited) == len(state):
        raise ValueError("PCQM state does not contain both anchor and extension")
    return inherited


def validate_pcqm_anchor_resume_contract(
    model: nn.Module,
    config: Mapping[str, Any],
) -> None:
    """Prove that a resumed PCQM still contains the registered frozen r9 EMA."""

    if not isinstance(model, IPMRQPCQM):
        raise TypeError("PCQM anchor resume contract requires IPMRQPCQM")
    design = config.get("pcqm_design_contract")
    registration = design.get("anchor_registration") \
        if isinstance(design, Mapping) else None
    expected_registration = {
        "checkpoint_sha256": PCQM_ANCHOR_CHECKPOINT_SHA256,
        "checkpoint_role": PCQM_ANCHOR_CHECKPOINT_ROLE,
        "optimizer_updates": PCQM_ANCHOR_OPTIMIZER_UPDATES,
        "selected_state_dict_key": PCQM_ANCHOR_STATE_KEY,
        "selected_tensor_state_sha256": PCQM_ANCHOR_STATE_SHA256,
    }
    if not isinstance(registration, Mapping) \
            or dict(registration) != expected_registration:
        raise ValueError("resume PCQM registered anchor contract differs")
    provenance = config.get("initialization_provenance")
    expected = {
        "initialization_mode": "ipmr_q_anchor_plus_zero_pcqm",
        "source_checkpoint_sha256": PCQM_ANCHOR_CHECKPOINT_SHA256,
        "source_checkpoint_role": PCQM_ANCHOR_CHECKPOINT_ROLE,
        "source_optimizer_updates": PCQM_ANCHOR_OPTIMIZER_UPDATES,
        "requested_weight_source": "selected",
        "loaded_weight_kind": "ema",
        "loaded_state_dict_key": PCQM_ANCHOR_STATE_KEY,
        "source_selector_state_dict_key": PCQM_ANCHOR_STATE_KEY,
        "source_tensor_state_sha256": PCQM_ANCHOR_STATE_SHA256,
        "source_model": "ipmr_q",
        "target_model": PCQM_MODEL_NAME,
        "all_inherited_tensors_loaded_exactly": True,
        "anchor_frozen": True,
        "locked_test_opened": False,
    }
    if not isinstance(provenance, Mapping) or any(
        provenance.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("resume PCQM r9 EMA provenance differs")
    # Requires exactly the registered PCQM trainable partition as well as an
    # exact byte-level match of every inherited tensor after strict loading.
    pcqm_optimization_contract(model)
    anchor = _pcqm_inherited_anchor_state(model.state_dict())
    if _tensor_state_sha256(anchor) != PCQM_ANCHOR_STATE_SHA256:
        raise ValueError("resume PCQM inherited r9 EMA tensor state differs")


def aom_optimizer_groups(
    model: nn.Module,
    *,
    base_learning_rate: float,
    backbone_lr_multiplier: float = 0.1,
) -> list[dict[str, Any]]:
    """Return the complete, non-overlapping AOM-Q three-group partition."""

    if not isinstance(model, AOMQ):
        raise TypeError("AOM-Q optimizer partition received another model")
    if not math.isfinite(backbone_lr_multiplier) \
            or not 0.0 < backbone_lr_multiplier <= 1.0:
        raise ValueError("AOM-Q backbone LR multiplier must be in (0,1]")
    named = list(model.named_parameters())
    backbone = [p for n, p in named if n.startswith(("shallow_expert.", "deep_expert."))]
    exchange_prefixes = (
        "s_from_d40.", "d_from_s40.", "s_from_d80.", "d_from_s80.",
        "s_from_d160.", "d_from_s160.", "shallow_down80.",
    )
    exchange = [p for n, p in named if n.startswith(exchange_prefixes)]
    used = {id(p) for p in (*backbone, *exchange)}
    morphology = [p for _n, p in named if id(p) not in used]
    flattened = [*backbone, *exchange, *morphology]
    if not backbone or not exchange or not morphology \
            or len({id(p) for p in flattened}) != len(flattened) \
            or {id(p) for p in flattened} != {id(p) for _n, p in named}:
        raise ValueError("AOM-Q optimizer parameter partition is incomplete/overlapping")
    return [
        {"params": backbone, "lr": base_learning_rate * backbone_lr_multiplier,
         "group_name": "aom_pretrained_backbones",
         "lr_multiplier": float(backbone_lr_multiplier)},
        {"params": exchange, "lr": base_learning_rate,
         "group_name": "aom_cross_scale_exchange", "lr_multiplier": 1.0},
        {"params": morphology, "lr": base_learning_rate,
         "group_name": "aom_identifiable_morphology", "lr_multiplier": 1.0},
    ]


def validate_aom_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    backbone_lr_multiplier: float,
    weight_decay: float,
) -> None:
    saved = optimizer_state.get("param_groups") \
        if isinstance(optimizer_state, Mapping) else None
    current = optimizer.param_groups
    names = (
        "aom_pretrained_backbones", "aom_cross_scale_exchange",
        "aom_identifiable_morphology",
    )
    if not isinstance(saved, list) or len(saved) != 3 or len(current) != 3:
        raise ValueError("AOM-Q resume requires exactly three optimizer groups")
    saved_ids: list[int] = []
    for index, (old, new, name) in enumerate(zip(saved, current, names)):
        if not isinstance(old, Mapping) or old.get("group_name") != name \
                or new.get("group_name") != name:
            raise ValueError("AOM-Q optimizer group name/order differs")
        expected_multiplier = backbone_lr_multiplier if index == 0 else 1.0
        if old.get("lr_multiplier") != expected_multiplier \
                or new.get("lr_multiplier") != expected_multiplier:
            raise ValueError("AOM-Q optimizer LR multiplier differs")
        ids = old.get("params")
        if not isinstance(ids, list) or len(ids) != len(new["params"]) \
                or len(set(ids)) != len(ids):
            raise ValueError("AOM-Q optimizer parameter coverage differs")
        saved_ids.extend(ids)
        if float(old.get("weight_decay", float("nan"))) != float(weight_decay):
            raise ValueError("AOM-Q optimizer weight decay differs")
        for field in (
            "betas", "eps", "amsgrad", "maximize", "foreach",
            "capturable", "differentiable", "fused",
        ):
            if old.get(field) != new.get(field):
                raise ValueError(f"AOM-Q optimizer field {field!r} differs")
    if len(set(saved_ids)) != len(saved_ids):
        raise ValueError("AOM-Q optimizer groups overlap")


def validate_parent_router_ema_resume_contract(
    ema_state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    optimizer_updates: int,
) -> None:
    """Bind the router EMA clock/strategy before checkpoint state is used."""

    if not isinstance(ema_state, Mapping):
        raise ValueError("ParentRouterQ EMA checkpoint state is malformed")
    expected_strategy = config.get("ema_strategy")
    if expected_strategy != EMA_STRATEGY \
            or ema_state.get("strategy") != expected_strategy:
        raise ValueError("ParentRouterQ EMA strategy differs from its config")
    num_updates = ema_state.get("num_updates")
    if isinstance(num_updates, bool) or not isinstance(num_updates, int) \
            or num_updates != int(optimizer_updates):
        raise ValueError("ParentRouterQ EMA update clock differs from optimizer")
    expected_decay = config.get("ema_decay")
    actual_decay = ema_state.get("decay")
    if isinstance(expected_decay, bool) \
            or not isinstance(expected_decay, (int, float)) \
            or isinstance(actual_decay, bool) \
            or not isinstance(actual_decay, (int, float)) \
            or not math.isfinite(float(actual_decay)) \
            or not math.isclose(
                float(actual_decay), float(expected_decay),
                rel_tol=0.0, abs_tol=0.0,
            ):
        raise ValueError("ParentRouterQ EMA decay differs from its config")
    shadow = ema_state.get("shadow")
    if not isinstance(shadow, Mapping) or not shadow:
        raise ValueError("ParentRouterQ EMA shadow state is missing")


def validate_dcf_ema_resume_contract(
    ema_state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    optimizer_updates: int,
) -> None:
    """Bind DCF-Q decoder EMA clock/strategy before loading its shadow."""

    try:
        validate_parent_router_ema_resume_contract(
            ema_state, config, optimizer_updates=optimizer_updates
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("ParentRouterQ", "DCF-Q")) from exc


def validate_parent_router_expert_state_hashes(
    state: Mapping[str, Tensor],
) -> None:
    """Re-authenticate both frozen expert prefixes on every formal resume."""

    if not isinstance(state, Mapping):
        raise ValueError("ParentRouterQ resume model state is malformed")
    initialized = state.get("experts_initialized")
    if not isinstance(initialized, Tensor) \
            or initialized.numel() != 1 \
            or not bool(initialized.detach().cpu().item()):
        raise ValueError("ParentRouterQ resume experts are not initialized")
    expected = (
        ("shallow_expert.", PARENT_ROUTER_SHALLOW_STATE_SHA256),
        ("deep_expert.", PARENT_ROUTER_DEEP_STATE_SHA256),
    )
    for prefix, expected_sha256 in expected:
        expert_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if not expert_state or _tensor_state_sha256(expert_state) != expected_sha256:
            raise ValueError(
                f"ParentRouterQ frozen expert state differs at {prefix!r}"
            )


def validate_dcf_expert_state_hashes(state: Mapping[str, Tensor]) -> None:
    """Re-authenticate both DCF-Q frozen expert prefixes on resume."""

    try:
        validate_parent_router_expert_state_hashes(state)
    except ValueError as exc:
        raise ValueError(str(exc).replace("ParentRouterQ", "DCF-Q")) from exc


def parent_router_precision_contract(
    device_type: str,
    *,
    amp_requested: bool,
) -> dict[str, Any]:
    """Describe the exact numerical execution path used by the router run."""

    if device_type not in {"cpu", "cuda"}:
        raise ValueError("ParentRouterQ requires an explicit cpu/cuda device type")
    amp_enabled = bool(amp_requested and device_type == "cuda")
    return {
        "execution_device_type": device_type,
        "amp_enabled": amp_enabled,
        "cuda_autocast_dtype": "float16" if amp_enabled else None,
        "router_features_q_blend_and_projection_dtype": "float32",
        "field_loss_dtype": "float32",
        "validation_prediction_and_metric_input_dtype": "float32",
    }


def dcf_precision_contract(
    device_type: str,
    *,
    amp_requested: bool,
) -> dict[str, Any]:
    """Bind source-feature, cross-decoder, Q-band and metric precision."""

    if device_type not in {"cpu", "cuda"}:
        raise ValueError("DCF-Q requires an explicit cpu/cuda device type")
    amp_enabled = bool(amp_requested and device_type == "cuda")
    return {
        "execution_device_type": device_type,
        "amp_enabled": amp_enabled,
        "cuda_autocast_dtype": "float16" if amp_enabled else None,
        "frozen_expert_and_decoder_autocast": amp_enabled,
        "anchor_q_band_projection_and_field_loss_dtype": "float32",
        "validation_prediction_and_metric_input_dtype": "float32",
    }


def _parameter_names_sha256(names: Sequence[str]) -> str:
    payload = "\n".join(names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _no_geo_probe_partition(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Fail closed unless H and B exactly partition the audited Context15 S model."""

    if getattr(model, "no_geo_core_enabled", False) is not True:
        raise ValueError("no-geolocation probes require a physical Context15 model")
    named = list(model.named_parameters(remove_duplicate=False))
    names = [name for name, _parameter in named]
    parameter_ids = [id(parameter) for _name, parameter in named]
    if len(names) != len(set(names)) or len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("no-geolocation probe parameters are aliased or duplicated")

    running_buffers = sorted(
        name for name, _buffer in model.named_buffers()
        if name.rsplit(".", 1)[-1] in {
            "running_mean", "running_var", "num_batches_tracked",
        }
    )
    running_modules = sorted(
        name for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and bool(getattr(module, "track_running_stats", False))
    )
    if running_buffers or running_modules:
        raise ValueError(
            "no-geolocation frozen probes forbid train-mode mutable running buffers"
        )

    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "head_conditioning": [],
        "encoder_backbone": [],
    }
    for name, parameter in named:
        in_head = name.startswith(NO_GEO_PROBE_HEAD_PREFIXES)
        in_backbone = name.startswith(NO_GEO_PROBE_BACKBONE_PREFIXES)
        if in_head == in_backbone:
            raise ValueError(
                f"no-geolocation H/B partition is overlapping or incomplete at {name!r}"
            )
        groups["head_conditioning" if in_head else "encoder_backbone"].append(
            (name, parameter)
        )

    result: dict[str, dict[str, Any]] = {}
    for group_name, rows in groups.items():
        group_names = [name for name, _parameter in rows]
        parameters = [parameter for _name, parameter in rows]
        tensor_count = len(parameters)
        parameter_count = sum(parameter.numel() for parameter in parameters)
        expected = NO_GEO_PROBE_EXPECTED[group_name]
        if tensor_count != expected["tensor_count"] \
                or parameter_count != expected["parameter_count"]:
            raise ValueError(
                f"no-geolocation {group_name} audit count differs: "
                f"{tensor_count}/{parameter_count}"
            )
        result[group_name] = {
            "names": tuple(group_names),
            "parameters": tuple(parameters),
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "names_sha256": _parameter_names_sha256(group_names),
        }
    full_tensor_count = sum(row["tensor_count"] for row in result.values())
    full_parameter_count = sum(row["parameter_count"] for row in result.values())
    expected_full = NO_GEO_PROBE_EXPECTED["full"]
    if full_tensor_count != expected_full["tensor_count"] \
            or full_parameter_count != expected_full["parameter_count"] \
            or full_tensor_count != len(named):
        raise ValueError("no-geolocation H/B union does not exactly cover the model")
    return result


def configure_no_geo_probe(
    model: nn.Module, probe_mode: str
) -> tuple[list[nn.Parameter], dict[str, Any]]:
    """Freeze the complement of H/B and return its strict scientific contract."""

    if probe_mode not in {"head_conditioning", "encoder_backbone"}:
        raise ValueError("unsupported no-geolocation probe mode")
    partition = _no_geo_probe_partition(model)
    frozen_mode = (
        "encoder_backbone"
        if probe_mode == "head_conditioning" else "head_conditioning"
    )
    trainable_names = set(partition[probe_mode]["names"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable_names)
    if any(
        parameter.requires_grad != (name in trainable_names)
        for name, parameter in model.named_parameters()
    ):
        raise AssertionError("no-geolocation probe freeze state changed")
    trainable_parameters = list(partition[probe_mode]["parameters"])
    contract = {
        "schema_version": NO_GEO_PROBE_SCHEMA,
        "probe_mode": probe_mode,
        "trainable_partition": probe_mode,
        "frozen_partition": frozen_mode,
        "head_conditioning_prefixes": list(NO_GEO_PROBE_HEAD_PREFIXES),
        "encoder_backbone_prefixes": list(NO_GEO_PROBE_BACKBONE_PREFIXES),
        "head_conditioning_tensor_count": partition["head_conditioning"][
            "tensor_count"
        ],
        "head_conditioning_parameter_count": partition["head_conditioning"][
            "parameter_count"
        ],
        "head_conditioning_names_sha256": partition["head_conditioning"][
            "names_sha256"
        ],
        "encoder_backbone_tensor_count": partition["encoder_backbone"][
            "tensor_count"
        ],
        "encoder_backbone_parameter_count": partition["encoder_backbone"][
            "parameter_count"
        ],
        "encoder_backbone_names_sha256": partition["encoder_backbone"][
            "names_sha256"
        ],
        "trainable_tensor_count": partition[probe_mode]["tensor_count"],
        "trainable_parameter_count": partition[probe_mode]["parameter_count"],
        "trainable_names_sha256": partition[probe_mode]["names_sha256"],
        "frozen_tensor_count": partition[frozen_mode]["tensor_count"],
        "frozen_parameter_count": partition[frozen_mode]["parameter_count"],
        "frozen_names_sha256": partition[frozen_mode]["names_sha256"],
        "union_tensor_count": NO_GEO_PROBE_EXPECTED["full"]["tensor_count"],
        "union_parameter_count": NO_GEO_PROBE_EXPECTED["full"][
            "parameter_count"
        ],
        "intersection_tensor_count": 0,
        "omitted_tensor_count": 0,
        "normalization": "GroupNorm_only_no_train_mode_running_buffers",
        "optimizer_groups": [f"no_geo_probe_{probe_mode}"],
        "optimizer_scope": "trainable_partition_only_full_scheduled_learning_rate",
    }
    return trainable_parameters, contract


def validate_no_geo_probe_optimizer_resume_contract(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    probe_mode: str,
    weight_decay: float,
) -> None:
    """Validate the single target-only AdamW group before exact resume."""

    expected_name = f"no_geo_probe_{probe_mode}"
    expected_count = NO_GEO_PROBE_EXPECTED[probe_mode]["tensor_count"]
    current_groups = optimizer.param_groups
    saved_groups = optimizer_state.get("param_groups")
    if len(current_groups) != 1 or not isinstance(saved_groups, list) \
            or len(saved_groups) != 1:
        raise ValueError("no-geolocation probe resume requires one optimizer group")
    current = current_groups[0]
    saved = saved_groups[0]
    if current.get("group_name") != expected_name \
            or saved.get("group_name") != expected_name:
        raise ValueError("no-geolocation probe optimizer group name differs")
    if len(current.get("params", ())) != expected_count \
            or len(saved.get("params", ())) != expected_count:
        raise ValueError("no-geolocation probe optimizer parameter count differs")
    saved_weight_decay = saved.get("weight_decay")
    if isinstance(saved_weight_decay, bool) \
            or not isinstance(saved_weight_decay, (int, float)) \
            or not math.isfinite(float(saved_weight_decay)) \
            or not math.isclose(
                float(saved_weight_decay), float(weight_decay),
                rel_tol=0.0, abs_tol=0.0,
            ):
        raise ValueError("no-geolocation probe optimizer weight decay differs")
    for field in (
        "betas", "eps", "amsgrad", "maximize", "foreach",
        "capturable", "differentiable", "fused",
    ):
        if saved.get(field) != current.get(field):
            raise ValueError(
                "no-geolocation probe optimizer AdamW metadata differs"
            )
    saved_lr = saved.get("lr")
    if isinstance(saved_lr, bool) or not isinstance(saved_lr, (int, float)) \
            or not math.isfinite(float(saved_lr)) or float(saved_lr) < 0.0:
        raise ValueError("no-geolocation probe optimizer checkpoint LR is invalid")


def _calibrated_no_geo_probe_mode(config: Mapping[str, Any]) -> str | None:
    contract = config.get("no_geo_probe_optimization")
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("no-geolocation probe optimization config is malformed")
    expected_keys = {
        "schema_version", "probe_mode", "trainable_partition",
        "frozen_partition", "head_conditioning_prefixes",
        "encoder_backbone_prefixes", "head_conditioning_tensor_count",
        "head_conditioning_parameter_count", "head_conditioning_names_sha256",
        "encoder_backbone_tensor_count", "encoder_backbone_parameter_count",
        "encoder_backbone_names_sha256", "trainable_tensor_count",
        "trainable_parameter_count", "trainable_names_sha256",
        "frozen_tensor_count", "frozen_parameter_count", "frozen_names_sha256",
        "union_tensor_count", "union_parameter_count",
        "intersection_tensor_count", "omitted_tensor_count", "normalization",
        "optimizer_groups", "optimizer_scope",
    }
    if set(contract) != expected_keys \
            or contract.get("schema_version") != NO_GEO_PROBE_SCHEMA:
        raise ValueError("unsupported no-geolocation probe optimization contract")
    mode = contract.get("probe_mode")
    if mode not in {"head_conditioning", "encoder_backbone"}:
        raise ValueError("unsupported no-geolocation probe mode")
    frozen = (
        "encoder_backbone" if mode == "head_conditioning" else "head_conditioning"
    )
    if contract.get("trainable_partition") != mode \
            or contract.get("frozen_partition") != frozen \
            or contract.get("head_conditioning_prefixes") \
            != list(NO_GEO_PROBE_HEAD_PREFIXES) \
            or contract.get("encoder_backbone_prefixes") \
            != list(NO_GEO_PROBE_BACKBONE_PREFIXES):
        raise ValueError("no-geolocation probe partition contract differs")
    for group in ("head_conditioning", "encoder_backbone"):
        expected = NO_GEO_PROBE_EXPECTED[group]
        if contract.get(f"{group}_tensor_count") != expected["tensor_count"] \
                or contract.get(f"{group}_parameter_count") \
                != expected["parameter_count"]:
            raise ValueError("no-geolocation probe audit counts differ")
    expected_trainable = NO_GEO_PROBE_EXPECTED[mode]
    expected_frozen = NO_GEO_PROBE_EXPECTED[frozen]
    if contract.get("trainable_tensor_count") != expected_trainable["tensor_count"] \
            or contract.get("trainable_parameter_count") \
            != expected_trainable["parameter_count"] \
            or contract.get("frozen_tensor_count") != expected_frozen["tensor_count"] \
            or contract.get("frozen_parameter_count") \
            != expected_frozen["parameter_count"] \
            or contract.get("union_tensor_count") \
            != NO_GEO_PROBE_EXPECTED["full"]["tensor_count"] \
            or contract.get("union_parameter_count") \
            != NO_GEO_PROBE_EXPECTED["full"]["parameter_count"] \
            or contract.get("intersection_tensor_count") != 0 \
            or contract.get("omitted_tensor_count") != 0:
        raise ValueError("no-geolocation probe coverage contract differs")
    hash_fields = (
        "head_conditioning_names_sha256", "encoder_backbone_names_sha256",
        "trainable_names_sha256", "frozen_names_sha256",
    )
    for field in hash_fields:
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64 \
                or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("no-geolocation probe parameter-name hash is malformed")
    if contract.get("trainable_names_sha256") \
            != contract.get(f"{mode}_names_sha256") \
            or contract.get("frozen_names_sha256") \
            != contract.get(f"{frozen}_names_sha256"):
        raise ValueError("no-geolocation probe trainable/frozen hash binding differs")
    if contract.get("normalization") \
            != "GroupNorm_only_no_train_mode_running_buffers" \
            or contract.get("optimizer_groups") != [f"no_geo_probe_{mode}"] \
            or contract.get("optimizer_scope") \
            != "trainable_partition_only_full_scheduled_learning_rate":
        raise ValueError("no-geolocation probe optimizer/normalization contract differs")
    return str(mode)


def _validate_no_geo_probe_model_binding(
    model: nn.Module, config: Mapping[str, Any]
) -> None:
    mode = _calibrated_no_geo_probe_mode(config)
    if mode is None:
        return
    contract = config["no_geo_probe_optimization"]
    partition = _no_geo_probe_partition(model)
    frozen = (
        "encoder_backbone" if mode == "head_conditioning" else "head_conditioning"
    )
    for group in ("head_conditioning", "encoder_backbone"):
        if contract.get(f"{group}_names_sha256") \
                != partition[group]["names_sha256"]:
            raise ValueError(
                "no-geolocation probe parameter-name hash differs from model"
            )
    trainable_names = set(partition[mode]["names"])
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != (name in trainable_names):
            raise ValueError(
                "no-geolocation probe requires_grad binding differs from contract"
            )
    if contract.get("trainable_names_sha256") \
            != partition[mode]["names_sha256"] \
            or contract.get("frozen_names_sha256") \
            != partition[frozen]["names_sha256"]:
        raise ValueError("no-geolocation probe trainable/frozen binding differs")


def build_no_geo_pack_contract(
    model: nn.Module, arm: str
) -> dict[str, Any]:
    """Bind one matched legacy-vs-HierPack arm to the exact constructed model."""

    if arm not in {"legacy_reset_control", "hierarchical"}:
        raise ValueError("unsupported no-geolocation pack arm")
    if getattr(model, "no_geo_core_enabled", False) is not True:
        raise ValueError("no-geolocation pack arms require a Context15 core")
    hierarchical = bool(getattr(model, "hierarchical_pack_enabled", False))
    if hierarchical != (arm == "hierarchical"):
        raise ValueError("no-geolocation pack arm differs from constructed model")
    named = list(model.named_parameters(remove_duplicate=False))
    names = [name for name, _parameter in named]
    ids = [id(parameter) for _name, parameter in named]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise ValueError("no-geolocation pack parameters are aliased or duplicated")
    if list(model.named_buffers()):
        raise ValueError("no-geolocation pack screen forbids unaudited model buffers")
    expected = NO_GEO_PACK_EXPECTED[arm]
    parameter_count = sum(parameter.numel() for _name, parameter in named)
    if len(named) != expected["tensor_count"] \
            or parameter_count != expected["parameter_count"]:
        raise ValueError("no-geolocation pack model audit counts differ")
    target_pack_keys = tuple(
        name for name in names
        if name.startswith("date_encoder.pack.")
        or name.startswith("date_encoder.hierarchical_pack.")
    )
    expected_target_pack = (
        NO_GEO_HIERARCHICAL_PACK_KEYS
        if arm == "hierarchical" else NO_GEO_LEGACY_PACK_KEYS
    )
    if target_pack_keys != expected_target_pack:
        raise ValueError("no-geolocation target pack key whitelist differs")
    shared_names = tuple(name for name in names if name not in target_pack_keys)
    source_dropped = (
        list(NO_GEO_LEGACY_PACK_KEYS) if arm == "hierarchical" else []
    )
    source_reset = (
        list(NO_GEO_LEGACY_PACK_KEYS)
        if arm == "legacy_reset_control" else []
    )
    target_new = (
        list(NO_GEO_HIERARCHICAL_PACK_KEYS) if arm == "hierarchical" else []
    )
    return {
        "schema_version": NO_GEO_PACK_SCHEMA,
        "arm": arm,
        "source_contract": "extension_free_bare_r2k_Fine52_Context19",
        "target_core_contract": "Fine52_physical_Context15_surgery",
        "source_dropped_keys": source_dropped,
        "source_reset_keys": source_reset,
        "target_new_keys": target_new,
        "target_pack_keys": list(expected_target_pack),
        "target_tensor_count": len(named),
        "target_parameter_count": parameter_count,
        "target_parameter_names_sha256": _parameter_names_sha256(names),
        "shared_nonpack_tensor_count": len(shared_names),
        "shared_nonpack_names_sha256": _parameter_names_sha256(shared_names),
        "optimizer": "single_group_all_parameters_full_scheduled_learning_rate",
        "matched_control_identity": (
            "context15_pack_screen_v1_same_source_seed_schedule_and_exact_nonpack_inheritance"
        ),
        "locked_test_opened": False,
    }


def _calibrated_no_geo_pack_arm(config: Mapping[str, Any]) -> str | None:
    contract = config.get("no_geo_pack_screen")
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("no-geolocation pack screen config is malformed")
    expected_keys = {
        "schema_version", "arm", "source_contract", "target_core_contract",
        "source_dropped_keys", "source_reset_keys", "target_new_keys",
        "target_pack_keys", "target_tensor_count", "target_parameter_count",
        "target_parameter_names_sha256", "shared_nonpack_tensor_count",
        "shared_nonpack_names_sha256", "optimizer", "matched_control_identity",
        "locked_test_opened",
    }
    arm = contract.get("arm")
    if set(contract) != expected_keys or contract.get("schema_version") \
            != NO_GEO_PACK_SCHEMA or arm not in NO_GEO_PACK_EXPECTED:
        raise ValueError("unsupported no-geolocation pack screen contract")
    hierarchical = arm == "hierarchical"
    if contract.get("source_dropped_keys") != (
        list(NO_GEO_LEGACY_PACK_KEYS) if hierarchical else []
    ) or contract.get("source_reset_keys") != (
        [] if hierarchical else list(NO_GEO_LEGACY_PACK_KEYS)
    ) or contract.get("target_new_keys") != (
        list(NO_GEO_HIERARCHICAL_PACK_KEYS) if hierarchical else []
    ) or contract.get("target_pack_keys") != (
        list(NO_GEO_HIERARCHICAL_PACK_KEYS)
        if hierarchical else list(NO_GEO_LEGACY_PACK_KEYS)
    ):
        raise ValueError("no-geolocation pack migration whitelist differs")
    expected = NO_GEO_PACK_EXPECTED[str(arm)]
    if contract.get("target_tensor_count") != expected["tensor_count"] \
            or contract.get("target_parameter_count") != expected["parameter_count"] \
            or contract.get("shared_nonpack_tensor_count") != 221:
        raise ValueError("no-geolocation pack audit counts differ")
    for field in (
        "target_parameter_names_sha256", "shared_nonpack_names_sha256",
    ):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64 \
                or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("no-geolocation pack parameter-name hash is malformed")
    if contract.get("source_contract") \
            != "extension_free_bare_r2k_Fine52_Context19" \
            or contract.get("target_core_contract") \
            != "Fine52_physical_Context15_surgery" \
            or contract.get("optimizer") \
            != "single_group_all_parameters_full_scheduled_learning_rate" \
            or contract.get("matched_control_identity") \
            != "context15_pack_screen_v1_same_source_seed_schedule_and_exact_nonpack_inheritance" \
            or contract.get("locked_test_opened") is not False:
        raise ValueError("no-geolocation pack scientific contract differs")
    return str(arm)


def _validate_no_geo_pack_model_binding(
    model: nn.Module, config: Mapping[str, Any]
) -> None:
    arm = _calibrated_no_geo_pack_arm(config)
    if arm is None:
        return
    expected = build_no_geo_pack_contract(model, arm)
    if dict(config["no_geo_pack_screen"]) != expected:
        raise ValueError("no-geolocation pack config differs from constructed model")


def _pause_ready_at_validation(
    *,
    stop_requested: bool,
    pause_exists: bool,
    update: int,
    records: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true as soon as a pause is requested.

    The training loop stops before starting another optimizer update.  Its
    common epilogue then publishes a validation record when ``update`` is not
    already represented in ``records``.  Consequently a rolling recovery
    checkpoint can pause immediately without either training to the next
    scheduled interval or sacrificing a durable validation point.
    """

    del update, records
    return bool(stop_requested or pause_exists)


def _pause_at_update_reached(update: int, pause_at_update: int | None) -> bool:
    """Return whether the runtime-only optimizer-update boundary was reached."""

    return pause_at_update is not None and update >= pause_at_update


def registered_full_scene(update: int, full_scene_start: int) -> bool:
    """Return the deterministic registered patch96:full160 = 3:1 schedule.

    ``full_scene_start`` starts the mixed phase rather than an all-full phase.
    Before it, every update is patch96.  Afterwards, each four-update block is
    patch, patch, patch, full.  The result depends only on the checkpointed
    global optimizer update, so pause/resume cannot shift the cadence.
    """

    if update < 0 or full_scene_start < 0:
        raise ValueError("scene schedule updates must be nonnegative")
    if update < full_scene_start:
        return False
    return (update - full_scene_start) % 4 == 3


def training_objective(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    eligible_weight: float,
    sample_loss_weight: Tensor | None = None,
) -> Tensor:
    """Build the one registered objective shared by full and sliced batches."""

    return region_equal_per_scene_loss(
        prediction,
        batch["target_k"],
        batch["valid"],
        batch["eligible"],
        batch["region_index"],
        eligible_weight=eligible_weight,
        sample_loss_weight=sample_loss_weight,
    )


def _region_equal_scene_values(
    scene_values: Tensor,
    region_index: Tensor,
    sample_loss_weight: Tensor | None,
) -> Tensor:
    """Aggregate one scalar per scene with the registered logical-batch rule."""

    if scene_values.ndim != 1 or region_index.shape != scene_values.shape:
        raise ValueError("scene values and region indices must share shape [B]")
    invalid_region = torch.any((region_index < 0) | (region_index >= 3))
    if invalid_region.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(~invalid_region, "region index must be in [0,2]")
    elif bool(invalid_region.detach().cpu()):
        raise ValueError("region index must be in [0,2]")
    if sample_loss_weight is not None:
        if sample_loss_weight.shape != scene_values.shape:
            raise ValueError("sample_loss_weight must have shape [B]")
        weights = sample_loss_weight.to(
            device=scene_values.device, dtype=scene_values.dtype
        )
        invalid_weight = torch.any(~torch.isfinite(weights) | (weights < 0))
        if invalid_weight.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                ~invalid_weight,
                "sample_loss_weight must be finite and nonnegative",
            )
        elif bool(invalid_weight.detach().cpu()):
            raise ValueError("sample_loss_weight must be finite and nonnegative")
        return torch.sum(scene_values * weights)
    sums = scene_values.new_zeros(3)
    counts = scene_values.new_zeros(3)
    sums.scatter_add_(0, region_index, scene_values)
    counts.scatter_add_(0, region_index, torch.ones_like(scene_values))
    present = counts > 0
    means = sums / counts.clamp_min(1.0)
    return means[present].mean()


def _masked_operator_mse(error: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-scene masked MSE and whether an operator has any support."""

    batch = error.shape[0]
    flat_error = error.float().reshape(batch, -1)
    flat_mask = mask.bool().reshape(batch, -1)
    counts = flat_mask.sum(dim=1)
    numerator = torch.where(flat_mask, flat_error.square(), 0.0).sum(dim=1)
    return numerator / counts.clamp_min(1).to(numerator.dtype), counts > 0


def _q_shape_at_scale(
    prediction_q: Tensor,
    target_q: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalized first-gradient and four-neighbour high-pass errors."""

    # Dividing by each stencil's L2 norm makes both losses comparable to a
    # per-pixel Kelvin-squared error under uncorrelated residual noise.
    gradient_norm = math.sqrt(2.0)
    horizontal_mask = mask[..., :, 1:] & mask[..., :, :-1]
    vertical_mask = mask[..., 1:, :] & mask[..., :-1, :]
    horizontal_error = (
        (prediction_q[..., :, 1:] - prediction_q[..., :, :-1])
        - (target_q[..., :, 1:] - target_q[..., :, :-1])
    ) / gradient_norm
    vertical_error = (
        (prediction_q[..., 1:, :] - prediction_q[..., :-1, :])
        - (target_q[..., 1:, :] - target_q[..., :-1, :])
    ) / gradient_norm
    gradient_error = torch.cat(
        (horizontal_error.flatten(1), vertical_error.flatten(1)), dim=1
    )
    gradient_mask = torch.cat(
        (horizontal_mask.flatten(1), vertical_mask.flatten(1)), dim=1
    )
    gradient_loss, gradient_available = _masked_operator_mse(
        gradient_error, gradient_mask
    )

    centre_mask = mask[..., 1:-1, 1:-1]
    highpass_mask = (
        centre_mask
        & mask[..., :-2, 1:-1]
        & mask[..., 2:, 1:-1]
        & mask[..., 1:-1, :-2]
        & mask[..., 1:-1, 2:]
    )

    def highpass(value: Tensor) -> Tensor:
        neighbours = (
            value[..., :-2, 1:-1]
            + value[..., 2:, 1:-1]
            + value[..., 1:-1, :-2]
            + value[..., 1:-1, 2:]
        ) * 0.25
        # ||[1,-1/4,-1/4,-1/4,-1/4]||_2 = sqrt(1.25).
        return (value[..., 1:-1, 1:-1] - neighbours) / math.sqrt(1.25)

    highpass_loss, highpass_available = _masked_operator_mse(
        highpass(prediction_q) - highpass(target_q), highpass_mask
    )
    return (
        gradient_loss,
        gradient_available,
        highpass_loss,
        highpass_available,
    )


def q_shape_auxiliary_loss(
    prediction_k: Tensor,
    base_k: Tensor,
    target_k: Tensor,
    mask: Tensor,
    region_index: Tensor,
    *,
    sample_loss_weight: Tensor | None = None,
) -> Tensor:
    """Two-scale Q-gradient/high-pass matching without invalid-boundary edges."""

    if prediction_k.shape != base_k.shape or target_k.shape != prediction_k.shape:
        raise ValueError("prediction, interpolation base, and target must share [B,1,H,W]")
    if mask.shape != prediction_k.shape:
        raise ValueError("Q-shape mask must match the prediction")
    if prediction_k.ndim != 4 or prediction_k.shape[1] != 1:
        raise ValueError("Q-shape fields must have shape [B,1,H,W]")
    if prediction_k.shape[-2] < 6 or prediction_k.shape[-1] < 6:
        raise ValueError("Q-shape fields are too small for two-scale high-pass support")
    prediction_q = prediction_k.float() - base_k.float()
    target_q = target_k.float() - base_k.float()
    support = mask.bool()
    component_losses: list[Tensor] = []
    component_available: list[Tensor] = []
    for scale in (1, 2):
        if scale == 1:
            scaled_prediction = prediction_q
            scaled_target = target_q
            scaled_support = support
        else:
            # A strict cell is valid only when its full 2x2 footprint is valid.
            # Multiplication by two compensates the 2x2 mean filter's L2 norm
            # (1/2), keeping the scale-2 operator on the native MSE scale.
            scaled_prediction = 2.0 * torch.nn.functional.avg_pool2d(
                torch.where(support, prediction_q, 0.0), 2, 2
            )
            scaled_target = 2.0 * torch.nn.functional.avg_pool2d(
                torch.where(support, target_q, 0.0), 2, 2
            )
            scaled_support = (
                torch.nn.functional.avg_pool2d(support.float(), 2, 2) == 1.0
            )
        gradient, gradient_ok, highpass, highpass_ok = _q_shape_at_scale(
            scaled_prediction, scaled_target, scaled_support
        )
        component_losses.extend((gradient, highpass))
        component_available.extend((gradient_ok, highpass_ok))
    losses = torch.stack(component_losses, dim=1)
    available = torch.stack(component_available, dim=1)
    count = available.sum(dim=1)
    scene_loss = torch.where(available, losses, 0.0).sum(dim=1)
    scene_loss = scene_loss / count.clamp_min(1).to(scene_loss.dtype)
    return _region_equal_scene_values(
        scene_loss, region_index, sample_loss_weight
    )


def training_objective_with_q_shape(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    eligible_weight: float,
    q_shape_weight: float,
    sample_loss_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convex pixel/shape objective; caller uses this only when weight > 0."""

    if (
        not math.isfinite(q_shape_weight)
        or not 0.0 < q_shape_weight <= Q_SHAPE_MAX_WEIGHT
    ):
        raise ValueError("Q-shape weight must be finite in (0,0.15]")
    pixel = training_objective(
        prediction,
        batch,
        eligible_weight=eligible_weight,
        sample_loss_weight=sample_loss_weight,
    )
    query_support = _query(batch["support"], batch["query_index"]).bool()
    query_fine = _query(batch["fine"], batch["query_index"])
    query_coarse = _query(batch["coarse_k"], batch["query_index"])
    base = support_project(query_fine[:, :1], query_coarse, query_support)
    shape = q_shape_auxiliary_loss(
        prediction,
        base,
        batch["target_k"],
        batch["valid"].bool() & batch["eligible"].bool() & query_support,
        batch["region_index"],
        sample_loss_weight=sample_loss_weight,
    )
    total = (1.0 - q_shape_weight) * pixel + q_shape_weight * shape
    return total, pixel, shape


def ipmr_auxiliary_scale(update: int, max_updates: int) -> float:
    """Fixed late-decay schedule for IPMR-Q gradient-routing supervision."""

    if update < 0 or max_updates <= 0 or update > max_updates:
        raise ValueError("IPMR auxiliary schedule received invalid updates")
    start = IPMR_AUX_DECAY_START_FRACTION * max_updates
    stop = IPMR_AUX_DECAY_END_FRACTION * max_updates
    if update <= start:
        return 1.0
    if update >= stop:
        return 0.0
    progress = (update - start) / (stop - start)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _ipmr_complete_parent_mask(
    support: Tensor, target_valid: Tensor
) -> Tensor:
    """Return 4x4 parents whose every supported pixel has a real target."""

    if support.shape != target_valid.shape or support.ndim != 4:
        raise ValueError("IPMR support and target-valid masks must share N1HW")
    if support.shape[1] != 1 or support.shape[-2] % 4 or support.shape[-1] % 4:
        raise ValueError("IPMR masks require N1HW geometry divisible by four")
    support_bool = support.bool()
    valid_bool = target_valid.bool()
    unsupported_label = support_bool & ~valid_bool
    missing = torch.nn.functional.max_pool2d(
        unsupported_label.float(), 4, 4
    ) > 0
    nonempty = torch.nn.functional.max_pool2d(
        support_bool.float(), 4, 4
    ) > 0
    return nonempty & ~missing


def _ipmr_band_diagnostic_mask(
    support: Tensor, target_valid: Tensor
) -> Tensor:
    """Historical r9 QM/QH mask: complete target-valid physical parents.

    Eligibility belongs to the formal field objective, not to the orthogonal
    band decomposition.  Keeping it out here preserves direct comparability
    with the registered r9 QM/QH baseline.
    """

    complete40 = _ipmr_complete_parent_mask(support, target_valid)
    complete = complete40.repeat_interleave(4, -2).repeat_interleave(4, -1)
    return complete & support.bool()


def ipmr_band_auxiliary_loss(
    components: (
        IPMRQComponents | IPMRQV2Components | IPMRQPCQMComponents
        | DCFQComponents
    ),
    batch: Mapping[str, Any],
    *,
    sample_loss_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Degree-normalized QM/QH deep supervision on complete target blocks."""

    support = _query(batch["support"], batch["query_index"]).bool()
    coarse = _query(batch["coarse_k"], batch["query_index"])
    coarse_valid = torch.isfinite(coarse)
    target_valid = batch["valid"].bool()
    complete40 = _ipmr_complete_parent_mask(support, target_valid)
    complete_fine = complete40.repeat_interleave(4, -2).repeat_interleave(4, -1)
    complete_support = complete_fine & support

    target_q = batch["target_k"].float() - components.base_k.float()
    target_bands = orthogonal_q_bands(
        target_q, support, coarse_valid
    )
    middle_error = components.q_middle_k.float() - target_bands.q_middle
    high_error = components.q_high_k.float() - target_bands.q_high
    middle_energy = torch.where(
        complete_support, middle_error.square(), 0.0
    ).sum(dim=(1, 2, 3))
    high_energy = torch.where(
        complete_support, high_error.square(), 0.0
    ).sum(dim=(1, 2, 3))

    count80 = 4.0 * torch.nn.functional.avg_pool2d(
        support.float(), 2, 2
    )
    active80 = count80 > 0
    complete80 = complete40.repeat_interleave(2, -2).repeat_interleave(2, -1)
    active_complete80 = active80 & complete80
    active_children40 = 4.0 * torch.nn.functional.avg_pool2d(
        active_complete80.float(), 2, 2
    )
    middle_dof40 = (
        active_children40 - coarse_valid.float()
    ).clamp_min(0.0) * complete40.float()
    middle_dof = middle_dof40.sum(dim=(1, 2, 3))
    high_dof = torch.where(
        active_complete80,
        (count80 - 1.0).clamp_min(0.0),
        torch.zeros_like(count80),
    ).sum(dim=(1, 2, 3))

    invalid = (middle_dof <= 0) | (high_dof <= 0)
    message = "each IPMR training scene requires complete QM and QH target degrees"
    if invalid.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(~torch.any(invalid), message)
    elif bool(torch.any(invalid).detach().cpu()):
        raise ValueError(message)
    middle_scene = middle_energy / middle_dof.clamp_min(1.0)
    high_scene = high_energy / high_dof.clamp_min(1.0)
    middle = _region_equal_scene_values(
        middle_scene, batch["region_index"], sample_loss_weight
    )
    high = _region_equal_scene_values(
        high_scene, batch["region_index"], sample_loss_weight
    )
    return middle, high


def training_objective_with_ipmr_bands(
    components: (
        IPMRQComponents | IPMRQV2Components | IPMRQPCQMComponents
        | DCFQComponents
    ),
    batch: Mapping[str, Any],
    *,
    eligible_weight: float,
    middle_weight: float,
    high_weight: float,
    auxiliary_scale: float,
    sample_loss_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Formal field loss plus small, decaying orthogonal-band supervision."""

    for value, name in (
        (middle_weight, "middle"),
        (high_weight, "high"),
        (auxiliary_scale, "scale"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"IPMR {name} auxiliary value must be finite/nonnegative")
    field = training_objective(
        components.prediction_k,
        batch,
        eligible_weight=eligible_weight,
        sample_loss_weight=sample_loss_weight,
    )
    middle, high = ipmr_band_auxiliary_loss(
        components, batch, sample_loss_weight=sample_loss_weight
    )
    total = field + auxiliary_scale * (
        middle_weight * middle + high_weight * high
    )
    return total, field, middle, high


def _query(values: Tensor, query_index: Tensor) -> Tensor:
    shape = (values.shape[0], 1, *values.shape[2:])
    index = query_index.long().reshape(values.shape[0], 1, *([1] * (values.ndim - 2)))
    return torch.gather(values, 1, index.expand(shape)).squeeze(1)


class R2OCNIRControl(nn.Module):
    """Single-date repaired OCNIR behind the R2 six-input model interface."""

    def __init__(
        self, width: int = 48, *, fine_channels: int = FINE_CHANNELS,
        context_dim: int = CONTEXT_DIM,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.core = OCNIR(
            width=self.width, fine_channels=int(fine_channels),
            context_dim=int(context_dim),
        )

    def forward(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> Tensor:
        del temporal_available
        return self.core(
            _query(fine, query_index),
            _query(coarse_k, query_index),
            _query(support, query_index),
            _query(context, query_index),
        )


def build_model(
    name: str,
    *,
    width: int = 48,
    contrast_d4_average: bool = False,
    contrast_activation_checkpointing: bool = False,
    calibrated_activation_checkpointing: bool | None = None,
    calibrated_allocation_adapter: bool = False,
    calibrated_t3_fusion: bool = False,
    calibrated_q_refiner: bool = False,
    calibrated_content_q_pyramid: bool = False,
    calibrated_parent_dct15: bool = False,
    calibrated_no_geo_core: bool = False,
    calibrated_hierarchical_pack: bool = False,
    ipmr_activation_checkpointing: bool | None = None,
    u1lite_dexchange: bool = False,
    fine_channels: int = FINE_CHANNELS,
    context_dim: int = CONTEXT_DIM,
) -> nn.Module:
    value = str(name).strip().casefold()
    if value != "contrast_q" and (
        contrast_d4_average or contrast_activation_checkpointing
    ):
        raise ValueError("contrast-specific model options require model='contrast_q'")
    if value != "calibrated_q" and (
        calibrated_activation_checkpointing is not None
        or calibrated_allocation_adapter
        or calibrated_t3_fusion
        or calibrated_q_refiner
        or calibrated_content_q_pyramid
        or calibrated_parent_dct15
        or calibrated_no_geo_core
        or calibrated_hierarchical_pack
    ):
        raise ValueError(
            "calibrated-specific model options require model='calibrated_q'"
        )
    if value not in IPMR_MODEL_NAMES and ipmr_activation_checkpointing is not None:
        raise ValueError(
            "IPMR-specific model options require an IPMR model"
        )
    if value != U1LITE_MODEL_NAME and u1lite_dexchange:
        raise ValueError("--u1lite-dexchange requires model='u1lite_q'")
    if value == "ocnir_control":
        return R2OCNIRControl(
            width=width, fine_channels=fine_channels, context_dim=context_dim
        )
    if value == "qparent":
        return MTStyleQParent(
            fine_channels=fine_channels, context_dim=context_dim
        )
    if value == "contrast_q":
        if width != 48:
            raise ValueError("ContrastQParent uses its registered fixed widths; --width must be 48")
        return ContrastQParent(
            fine_channels=fine_channels,
            context_dim=context_dim,
            d4_average=bool(contrast_d4_average),
            activation_checkpointing=bool(contrast_activation_checkpointing),
        )
    if value == "calibrated_q":
        return SceneCalibratedContinuousQ(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=(
                True
                if calibrated_activation_checkpointing is None
                else bool(calibrated_activation_checkpointing)
            ),
            allocation_adapter=bool(calibrated_allocation_adapter),
            t3_fusion=bool(calibrated_t3_fusion),
            q_refiner=bool(calibrated_q_refiner),
            content_q_pyramid=bool(calibrated_content_q_pyramid),
            parent_dct15=bool(calibrated_parent_dct15),
            no_geo_core=bool(calibrated_no_geo_core),
            hierarchical_pack=bool(calibrated_hierarchical_pack),
        )
    if value == "ipmr_q":
        return IPMRQ(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=(
                True
                if ipmr_activation_checkpointing is None
                else bool(ipmr_activation_checkpointing)
            ),
        )
    if value == "ipmr_q_v2":
        return IPMRQV2(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=(
                True
                if ipmr_activation_checkpointing is None
                else bool(ipmr_activation_checkpointing)
            ),
        )
    if value == PCQM_MODEL_NAME:
        if (fine_channels, context_dim, width) != (52, 19, 48):
            raise ValueError(
                "PCQM requires the registered Fine52/Context19/width48 anchor"
            )
        return IPMRQPCQM(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=(
                True
                if ipmr_activation_checkpointing is None
                else bool(ipmr_activation_checkpointing)
            ),
        ).freeze_anchor()
    if value == PARENT_ROUTER_MODEL_NAME:
        return ParentRouterQ(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
        )
    if value == DCF_MODEL_NAME:
        return DCFQ(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
        )
    if value == U1LITE_MODEL_NAME:
        if (fine_channels, context_dim, width) != (52, 19, 48):
            raise ValueError(
                "U1-Lite requires registered Fine52/Context19/width48"
            )
        model_type = (
            DExchangeContinuousScaleBridgeQ
            if u1lite_dexchange else ContinuousScaleBridgeQ
        )
        return model_type(activation_checkpointing=False)
    if value == AOM_MODEL_NAME:
        if (fine_channels, context_dim, width) != (52, 19, 48):
            raise ValueError("AOM-Q requires registered Fine52/Context19/width48")
        return AOMQ()
    raise ValueError(f"unsupported R2 model: {name!r}")


def apply_temporal_mode(batch: dict[str, Any], mode: str) -> None:
    if mode == "multi":
        return
    if mode != "single":
        raise ValueError(f"unsupported temporal mode: {mode!r}")
    # R2TemporalDataset always materialises the query date in slot zero.  Slice
    # every temporal predictor so single-date controls do not encode two masked
    # auxiliary dates.  Query-only supervision has no T axis and stays intact.
    for key in ("fine", "coarse_k", "support", "context", "temporal_available"):
        batch[key] = batch[key][:, :1]
    batch["query_index"] = torch.zeros_like(batch["query_index"])


_BAND_METRIC_KEYS = (
    "error_rmse_k", "prediction_rms_k", "target_rms_k", "cosine",
    "optimal_scale",
)


def _band_row(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, Any]:
    """Return finite band diagnostics plus sufficient statistics for pooling."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("band prediction, target, and mask shapes differ")
    active = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    predicted = prediction.float()[active]
    expected = target.float()[active]
    if predicted.numel() == 0:
        raise ValueError("band diagnostics require at least one finite active pixel")
    error = predicted - expected
    error_ss = float(error.double().square().sum().cpu())
    prediction_ss = float(predicted.double().square().sum().cpu())
    target_ss = float(expected.double().square().sum().cpu())
    cross = float((predicted.double() * expected.double()).sum().cpu())
    count = int(predicted.numel())
    cosine_denominator = math.sqrt(prediction_ss * target_ss)
    cosine = cross / cosine_denominator if cosine_denominator > 0.0 else 0.0
    optimal_scale = cross / prediction_ss if prediction_ss > 0.0 else 0.0
    return {
        "error_rmse_k": math.sqrt(error_ss / count),
        "prediction_rms_k": math.sqrt(prediction_ss / count),
        "target_rms_k": math.sqrt(target_ss / count),
        "cosine": float(cosine),
        "optimal_scale": float(optimal_scale),
        "pixel_count": count,
        "error_squared_sum_k2": error_ss,
        "prediction_squared_sum_k2": prediction_ss,
        "target_squared_sum_k2": target_ss,
        "prediction_target_sum_k2": cross,
    }


def _pooled_band_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("pooled band diagnostics require records")
    count = sum(int(row["pixel_count"]) for row in rows)
    if count <= 0:
        raise ValueError("pooled band diagnostics have no active pixels")
    error_ss = sum(float(row["error_squared_sum_k2"]) for row in rows)
    prediction_ss = sum(
        float(row["prediction_squared_sum_k2"]) for row in rows
    )
    target_ss = sum(float(row["target_squared_sum_k2"]) for row in rows)
    cross = sum(float(row["prediction_target_sum_k2"]) for row in rows)
    denominator = math.sqrt(prediction_ss * target_ss)
    return {
        "error_rmse_k": math.sqrt(error_ss / count),
        "prediction_rms_k": math.sqrt(prediction_ss / count),
        "target_rms_k": math.sqrt(target_ss / count),
        "cosine": cross / denominator if denominator > 0.0 else 0.0,
        "optimal_scale": cross / prediction_ss if prediction_ss > 0.0 else 0.0,
        "pixel_count": count,
    }


def _mean_band_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("macro band diagnostics require records")
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in _BAND_METRIC_KEYS
    }


def aggregate_band_metrics(
    records: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate band metrics scene -> city -> region -> equal region.

    The formal ``equal_region`` values are hierarchical macro averages, like
    the field selector. ``pooled`` is published separately and never replaces
    that registered comparison.
    """

    if isinstance(records, Mapping):
        items = [(str(scene_id), row) for scene_id, row in records.items()]
    else:
        items = [
            (str(row.get("scene_id", index)), row)
            for index, row in enumerate(records)
        ]
    if not items:
        raise ValueError("band aggregation requires at least one scene")
    per_scene = {scene_id: dict(row) for scene_id, row in items}
    grouped_city: dict[str, list[Mapping[str, Any]]] = {}
    city_region: dict[str, str] = {}
    for _scene_id, row in items:
        city = str(row["city"])
        region = str(row["region"])
        if city in city_region and city_region[city] != region:
            raise ValueError("a band-diagnostic city spans multiple regions")
        city_region[city] = region
        grouped_city.setdefault(city, []).append(row)
    per_city: dict[str, dict[str, Any]] = {}
    for city, rows in sorted(grouped_city.items()):
        per_city[city] = {
            "region": city_region[city],
            **_mean_band_metrics(rows),
            "scene_count": len(rows),
            "pooled": _pooled_band_metrics(rows),
        }
    regions = sorted({str(row["region"]) for row in per_city.values()})
    per_region: dict[str, dict[str, Any]] = {}
    for region in regions:
        cities = [row for row in per_city.values() if row["region"] == region]
        scene_rows = [row for _scene_id, row in items if row["region"] == region]
        per_region[region] = {
            **_mean_band_metrics(cities),
            "city_count": len(cities),
            "scene_count": len(scene_rows),
            "pooled": _pooled_band_metrics(scene_rows),
        }
    return {
        "aggregation": "scene_to_city_to_region_equal_region",
        "equal_region": _mean_band_metrics(list(per_region.values())),
        "pooled": _pooled_band_metrics([row for _scene_id, row in items]),
        "per_region": per_region,
        "per_city": per_city,
        "per_scene": per_scene,
    }


@torch.inference_mode()
def _evaluate_in_eval_mode(
    model: nn.Module,
    dataset: R2TemporalDataset,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
    temporal_mode: str,
) -> dict[str, Any]:
    model.eval()
    per_scene: dict[str, dict[str, Any]] = {}
    collect_bands = isinstance(model, (IPMRQ, IPMRQV2, IPMRQPCQM, DCFQ))
    middle_records: dict[str, dict[str, Any]] = {}
    high_records: dict[str, dict[str, Any]] = {}
    for start in range(0, dataset.evaluation_size, batch_size):
        size = min(batch_size, dataset.evaluation_size - start)
        batch = _device_batch(
            dataset.evaluation_batch(
                start, size, query_only=(temporal_mode == "single")
            ),
            device,
        )
        apply_temporal_mode(batch, temporal_mode)
        with _autocast(device, amp):
            if collect_bands:
                components = model.forward_components(
                    batch["fine"], batch["coarse_k"], batch["support"],
                    batch["context"], batch["temporal_available"],
                    batch["query_index"],
                )
                prediction = components.prediction_k
            else:
                prediction = model(
                    batch["fine"], batch["coarse_k"], batch["support"],
                    batch["context"], batch["temporal_available"],
                    batch["query_index"],
                )
        if collect_bands:
            support = _query(batch["support"], batch["query_index"]).bool()
            coarse = _query(batch["coarse_k"], batch["query_index"])
            coarse_valid = torch.isfinite(coarse)
            diagnostic_mask = _ipmr_band_diagnostic_mask(
                support, batch["valid"].bool()
            )
            target_bands = orthogonal_q_bands(
                batch["target_k"].float() - components.base_k.float(),
                support,
                coarse_valid,
            )
        for offset, scene_id in enumerate(batch["scene_id"]):
            metrics = scene_metrics(
                prediction[offset, 0].float().cpu().numpy(),
                batch["target_k"][offset, 0].float().cpu().numpy(),
                (batch["valid"] & batch["eligible"])[offset, 0].cpu().numpy(),
            )
            per_scene[str(scene_id)] = {
                "city": batch["city"][offset],
                "region": batch["region"][offset],
                **metrics,
            }
            if collect_bands:
                common = {
                    "city": str(batch["city"][offset]),
                    "region": str(batch["region"][offset]),
                }
                middle_records[str(scene_id)] = {
                    **common,
                    **_band_row(
                        components.q_middle_k[offset],
                        target_bands.q_middle[offset],
                        diagnostic_mask[offset],
                    ),
                }
                high_records[str(scene_id)] = {
                    **common,
                    **_band_row(
                        components.q_high_k[offset],
                        target_bands.q_high[offset],
                        diagnostic_mask[offset],
                    ),
                }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    city_region: dict[str, str] = {}
    for record in per_scene.values():
        grouped.setdefault(str(record["city"]), []).append(record)
        city_region[str(record["city"])] = str(record["region"])
    per_city = {
        city: {
            "region": city_region[city],
            **_mean_metrics(records),
            "scene_count": len(records),
        }
        for city, records in sorted(grouped.items())
    }
    per_region: dict[str, dict[str, Any]] = {}
    for region in ("us", "china", "europe"):
        rows = [row for row in per_city.values() if row["region"] == region]
        if not rows:
            raise ValueError(f"evaluation has no {region} cities")
        per_region[region] = {**_mean_metrics(rows), "city_count": len(rows)}
    result = {
        "aggregation": "scene_to_city_to_region_equal_region",
        "equal_region": _mean_metrics(list(per_region.values())),
        "per_region": per_region,
        "per_city": per_city,
        "per_scene": per_scene,
    }
    if collect_bands:
        result["band_diagnostics"] = {
            "mask_contract": (
                "complete_physical_4x4_parent_support_and_target_valid"
            ),
            "middle": aggregate_band_metrics(middle_records),
            "high": aggregate_band_metrics(high_records),
        }
    return result


def evaluate(
    model: nn.Module,
    dataset: R2TemporalDataset,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
    temporal_mode: str,
) -> dict[str, Any]:
    """Evaluate without changing the caller's train/eval mode."""

    was_training = model.training
    try:
        return _evaluate_in_eval_mode(
            model,
            dataset,
            device,
            batch_size=batch_size,
            amp=amp,
            temporal_mode=temporal_mode,
        )
    finally:
        model.train(was_training)


def evaluate_raw_and_ema(
    model: nn.Module,
    ema: EMA,
    dataset: R2TemporalDataset,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
    temporal_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate raw weights first, then EMA weights, restoring raw weights exactly."""

    if isinstance(ema, (ParentRouterEMA, DCFEMA)):
        ema.validate_bound_model(model)
    was_training = model.training
    raw_validation = evaluate(
        model, dataset, device,
        batch_size=batch_size, amp=amp, temporal_mode=temporal_mode,
    )
    backup = ema.apply(model)
    try:
        ema_validation = evaluate(
            model, dataset, device,
            batch_size=batch_size, amp=amp, temporal_mode=temporal_mode,
        )
    finally:
        model.load_state_dict(backup, strict=True)
        model.train(was_training)
    return raw_validation, ema_validation


def select_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records:
        raise ValueError("checkpoint selector requires validation records")
    minimum = min(float(row["validation"]["equal_region"]["rmse_k"]) for row in records)
    tied = [
        row for row in records
        if float(row["validation"]["equal_region"]["rmse_k"])
        <= minimum + RMSE_TIE_TOLERANCE_K + 1e-12
    ]
    return min(
        tied,
        key=lambda row: (
            -(float(row["validation"]["equal_region"]["auprc_q90"])
              + float(row["validation"]["equal_region"]["iou_q90"])) / 2.0,
            float(row["validation"]["equal_region"]["rmse_k"]),
            int(row["update"]),
        ),
    )


def validation_weight_source(record: Mapping[str, Any]) -> str:
    sources = record.get("validation_metric_sources", {})
    if isinstance(sources, Mapping) and isinstance(sources.get("validation"), str):
        return str(sources["validation"])
    # Every pre-contract history used EMA for the canonical validation field.
    return "ema_model_legacy_selector"


def choose_validation_for_selector(
    ema: WarmStartEMA,
    raw_validation: Mapping[str, Any],
    ema_validation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, bool]:
    """Use raw weights until fixed-decay EMA has shed its early-step tail."""

    mature = (
        ema.strategy == LEGACY_EMA_STRATEGY
        or ema.num_updates >= EMA_SELECTOR_MATURITY_UPDATES
    )
    if mature:
        return ema_validation, "ema_model_mature_selector", True
    return raw_validation, "raw_model_pre_ema_maturity_selector", False


def selected_state_dict_key(record: Mapping[str, Any]) -> str:
    source = validation_weight_source(record)
    if source.startswith("raw_model"):
        return "raw_model_state_dict"
    if source == "ema_model_legacy_selector":
        return "ema_state_dict.shadow"
    return "ema_model_state_dict"


def selected_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    equal = dict(record["validation"]["equal_region"])
    return {
        "update": int(record["update"]),
        "checkpoint": "best.pt",
        "rmse_tolerance_k": RMSE_TIE_TOLERANCE_K,
        "validation_weight_source": validation_weight_source(record),
        "inference_state_dict_key": selected_state_dict_key(record),
        "equal_region": equal,
    }


def early_stop_plateau(records: Sequence[Mapping[str, Any]]) -> bool:
    if len(records) < EARLY_STOP_INTERVALS + 1:
        return False
    recent = records[-(EARLY_STOP_INTERVALS + 1):]
    if len({validation_weight_source(row) for row in recent}) != 1:
        # Do not infer a plateau across the one-time raw-to-EMA selector switch.
        return False
    first = float(recent[0]["validation"]["equal_region"]["rmse_k"])
    later_best = min(
        float(row["validation"]["equal_region"]["rmse_k"]) for row in recent[1:]
    )
    return first - later_best < EARLY_STOP_MIN_IMPROVEMENT_K


def automatic_early_stop_enabled(model_name: str) -> bool:
    """Manual-gate candidates never use the legacy four-point rule."""

    return str(model_name).strip().casefold() not in {
        "ipmr_q_v2", PCQM_MODEL_NAME, PARENT_ROUTER_MODEL_NAME, DCF_MODEL_NAME,
        AOM_MODEL_NAME, U1LITE_MODEL_NAME,
    }


def _resolved_eligible_loss_weight(args: argparse.Namespace) -> float:
    value = getattr(args, "eligible_loss_weight", PRIMARY_LOSS_WEIGHT)
    return PRIMARY_LOSS_WEIGHT if value is None else float(value)


def _resolved_ipmr_middle_weight(args: argparse.Namespace) -> float:
    value = (
        getattr(args, "dcf_middle_loss_weight", None)
        if getattr(args, "model", None) == DCF_MODEL_NAME
        else getattr(args, "ipmr_middle_loss_weight", None)
    )
    if getattr(args, "model", None) == PCQM_MODEL_NAME and value is None:
        return 0.0
    return IPMR_MIDDLE_AUX_WEIGHT if value is None else float(value)


def _resolved_ipmr_high_weight(args: argparse.Namespace) -> float:
    value = (
        getattr(args, "dcf_high_loss_weight", None)
        if getattr(args, "model", None) == DCF_MODEL_NAME
        else getattr(args, "ipmr_high_loss_weight", None)
    )
    if getattr(args, "model", None) == PCQM_MODEL_NAME and value is None:
        return 0.0
    return IPMR_HIGH_AUX_WEIGHT if value is None else float(value)


def _loss_contract(eligible_weight: float) -> dict[str, Any]:
    registered = math.isclose(
        float(eligible_weight), PRIMARY_LOSS_WEIGHT, rel_tol=0.0, abs_tol=1.0e-12
    )
    return {
        "schema_version": "g246-r2-loss-contract-v1",
        "eligible_valid_mse_weight": float(eligible_weight),
        "all_valid_mse_weight": round(1.0 - float(eligible_weight), 12),
        "registered_0p8_0p2": registered,
    }


def _config(
    args: argparse.Namespace,
    *,
    receipt_sha256: str,
    fit_view_sha256: str,
    train_city_count: int,
    evaluation_city_count: int,
    split_identity: str,
    normalization_sha256: str,
    parameter_count: int,
    input_provenance: str,
    scientific_status: str,
    fine_channels: int = FINE_CHANNELS,
    context_dim: int = CONTEXT_DIM,
    multisource_provenance: Mapping[str, Any] | None = None,
    physical_batch_size: int = EFFECTIVE_BATCH_SIZE,
    effective_evaluation_batch_size: int | None = None,
    no_geo_probe_optimization: Mapping[str, Any] | None = None,
    no_geo_pack_screen: Mapping[str, Any] | None = None,
    pcqm_optimization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "model": args.model,
        "width": args.width,
        "temporal_mode": args.temporal_mode,
        "data_scope": args.data_scope,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": float(getattr(args, "weight_decay", WEIGHT_DECAY)),
        "ema_decay": EMA_DECAY,
        "ema_strategy": EMA_STRATEGY,
        "ema_selector_maturity_updates": EMA_SELECTOR_MATURITY_UPDATES,
        "warmup_updates": int(getattr(args, "warmup_updates", WARMUP_UPDATES)),
        "max_updates": args.max_updates,
        "full_scene_start": args.full_scene_start,
        "validation_interval": args.validation_interval,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_size": args.evaluation_batch_size,
        # Keep the established scalar key for old consumers; the structured
        # contract below makes accidental objective drift fail visibly.
        "primary_loss_weight": _resolved_eligible_loss_weight(args),
        "loss_contract": _loss_contract(_resolved_eligible_loss_weight(args)),
        "input_provenance": input_provenance,
        "scientific_status": scientific_status,
        "provisional_optical_authorized": bool(args.allow_provisional_optical),
        "fine_channels": int(fine_channels),
        "context_dim": int(context_dim),
        "parameter_count": parameter_count,
        "receipt_sha256": receipt_sha256,
        "fit_view_sha256": fit_view_sha256,
        "split_identity": split_identity,
        "train_city_count": train_city_count,
        "evaluation_city_count": evaluation_city_count,
        "normalization_sha256": normalization_sha256,
        "locked_test_opened": False,
    }
    region_sampling = str(getattr(args, "region_sampling", "equal_region"))
    if region_sampling != "dataset_proportional":
        # Omit only the legacy mode so old provisional checkpoints can still
        # be resumed with an explicit --region-sampling dataset_proportional.
        config["region_sampling"] = region_sampling
    # Preserve byte-for-byte-equivalent scientific config dictionaries for
    # existing OCNIR/QParent checkpoints.  Contrast-specific execution choices
    # exist only in a ContrastQ checkpoint contract.
    q_shape_weight = float(getattr(args, "q_shape_loss_weight", 0.0))
    if q_shape_weight > 0.0:
        config["q_shape_loss"] = {
            "weight": q_shape_weight,
            "pixel_mse_weight": 1.0 - q_shape_weight,
            "field": "query_Q=field-support_project(query_base,query_coarse)",
            "mask": "query_support&valid&eligible",
            "scales": [1, 2],
            "scale2_pool": "strict_all_valid_2x2_mean_l2_normalized",
            "gradient": "forward_xy_difference_l2_normalized",
            "highpass": "four_neighbour_laplacian_l2_normalized",
            "aggregation": "scene_then_region_equal",
        }
    if args.model == "contrast_q":
        config.update({
            "contrast_d4_average": bool(getattr(args, "contrast_d4_average", False)),
            "contrast_activation_checkpointing": bool(
                getattr(args, "contrast_activation_checkpointing", False)
            ),
        })
    if args.model == "calibrated_q":
        calibrated_checkpointing = getattr(
            args, "calibrated_activation_checkpointing", None
        )
        calibrated_channels = SceneCalibratedContinuousQ.registered_channels(
            int(args.width)
        )
        config.update({
            "calibrated_size": SceneCalibratedContinuousQ.size_labels[int(args.width)],
            "calibrated_channels": calibrated_channels,
            "calibrated_width_scaling": (
                "S_reference_x(base_width/48)_nearest_multiple_of_8"
            ),
            "calibrated_activation_checkpointing": (
                True
                if calibrated_checkpointing is None
                else bool(calibrated_checkpointing)
            ),
            "physical_batch_size": int(physical_batch_size),
            "gradient_accumulation_steps": math.ceil(
                EFFECTIVE_BATCH_SIZE / int(physical_batch_size)
            ),
            "evaluation_batch_size": int(
                effective_evaluation_batch_size
                if effective_evaluation_batch_size is not None
                else args.evaluation_batch_size
            ),
            "gradient_accumulation": (
                "registered_logical_batch32_region_weighted_batch_slice"
            ),
            "scene_schedule": (
                "patch_only_before_full_scene_start_then_patch96x3_full160x1"
            ),
        })
        if bool(getattr(args, "calibrated_allocation_adapter", False)):
            config["calibrated_allocation_adapter"] = copy.deepcopy(
                ALLOCATION_ADAPTER_CONTRACT
            )
        if bool(getattr(args, "calibrated_t3_fusion", False)):
            config["calibrated_t3_fusion"] = copy.deepcopy(T3_FUSION_CONTRACT)
        if bool(getattr(args, "calibrated_q_refiner", False)):
            config["calibrated_q_refiner"] = copy.deepcopy(Q_REFINER_CONTRACT)
            config["q_refiner_optimization"] = {
                "freeze_backbone_updates": int(
                    getattr(args, "q_refiner_freeze_backbone_updates", 0)
                ),
                "core_lr_multiplier": float(
                    getattr(args, "q_refiner_core_lr_multiplier", 1.0)
                ),
                "refiner_lr_multiplier": float(
                    getattr(args, "q_refiner_lr_multiplier", 1.0)
                ),
                "freeze_clock": "global_successful_optimizer_updates",
                "frozen_parameters": "all_except_q_refiner_prefix",
                "optimizer_groups": ["core", "q_refiner"],
            }
        if bool(getattr(args, "calibrated_content_q_pyramid", False)):
            config["calibrated_content_q_pyramid"] = copy.deepcopy(
                CONTENT_Q_PYRAMID_CONTRACT
            )
            config["content_q_pyramid_optimization"] = {
                "freeze_core_updates": int(
                    getattr(args, "q_pyramid_freeze_core_updates", 0)
                ),
                "core_lr_multiplier": float(
                    getattr(args, "q_pyramid_core_lr_multiplier", 1.0)
                ),
                "pyramid_lr_multiplier": float(
                    getattr(args, "q_pyramid_lr_multiplier", 1.0)
                ),
                "freeze_clock": "global_successful_optimizer_updates",
                "frozen_parameters": "all_except_content_q_pyramid_prefix",
                "optimizer_groups": ["core", "content_q_pyramid"],
            }
        if bool(getattr(args, "calibrated_parent_dct15", False)):
            config["calibrated_parent_dct15"] = copy.deepcopy(
                PARENT_DCT15_CONTRACT
            )
            config["parent_dct15_optimization"] = {
                "freeze_core_updates": int(
                    getattr(args, "dct15_freeze_core_updates", 0)
                ),
                "core_lr_multiplier": float(
                    getattr(args, "dct15_core_lr_multiplier", 1.0)
                ),
                "dct15_lr_multiplier": float(
                    getattr(args, "dct15_lr_multiplier", 1.0)
                ),
                "freeze_clock": "global_successful_optimizer_updates",
                "frozen_parameters": "all_except_parent_dct15_prefix",
                "optimizer_groups": ["core", "parent_dct15"],
            }
        if bool(getattr(args, "calibrated_no_geo_core", False)):
            rebase_mode = getattr(args, "no_geo_rebase_mode", None)
            probe_mode = getattr(args, "no_geo_probe", None)
            if not isinstance(rebase_mode, str):
                raise ValueError(
                    "--calibrated-no-geo-core requires an explicit "
                    "--no-geo-rebase-mode"
                )
            config["calibrated_no_geo_core"] = _no_geo_core_contract(
                rebase_mode, probe_mode=probe_mode
            )
            if probe_mode is not None:
                if no_geo_probe_optimization is None:
                    raise ValueError(
                        "no-geolocation probe optimization binding is missing"
                    )
                config["no_geo_probe_optimization"] = copy.deepcopy(
                    dict(no_geo_probe_optimization)
                )
                if _calibrated_no_geo_probe_mode(config) != probe_mode:
                    raise ValueError(
                        "no-geolocation probe optimization mode differs from CLI"
                    )
            elif no_geo_probe_optimization is not None:
                raise ValueError(
                    "no-geolocation probe optimization was supplied without a probe"
                )
            pack_arm = getattr(args, "no_geo_pack_arm", None)
            if pack_arm is not None:
                if no_geo_pack_screen is None:
                    raise ValueError("no-geolocation pack screen binding is missing")
                config["no_geo_pack_screen"] = copy.deepcopy(
                    dict(no_geo_pack_screen)
                )
                if _calibrated_no_geo_pack_arm(config) != pack_arm:
                    raise ValueError(
                        "no-geolocation pack screen arm differs from CLI"
                    )
            elif no_geo_pack_screen is not None:
                raise ValueError(
                    "no-geolocation pack screen binding was supplied without an arm"
                )
        elif no_geo_probe_optimization is not None:
            raise ValueError(
                "no-geolocation probe optimization requires a no-geo core"
            )
        elif no_geo_pack_screen is not None:
            raise ValueError("no-geolocation pack screen requires a no-geo core")
    if args.model in IPMR_MODEL_NAMES:
        ipmr_checkpointing = getattr(args, "ipmr_activation_checkpointing", None)
        ipmr_class = (
            IPMRQV2 if args.model == "ipmr_q_v2"
            else IPMRQPCQM if args.model == PCQM_MODEL_NAME
            else IPMRQ
        )
        config.update({
            "ipmr_schema_version": (
                "g246-ipmr-q-v2" if args.model == "ipmr_q_v2"
                else IPMRQPCQM.schema_version
                if args.model == PCQM_MODEL_NAME else "g246-ipmr-q-v1"
            ),
            "ipmr_channels": ipmr_class.registered_channels(int(args.width)),
            "ipmr_scale_path": ipmr_class.scale_path,
            "ipmr_band_contract": ipmr_class.band_contract,
            "ipmr_context_contract": {
                "stored": "Context19",
                "learned": "Context15",
                "physically_removed_indices": [5, 6, 7, 8],
                "coordinates_used": False,
            },
            "ipmr_activation_checkpointing": (
                True if ipmr_checkpointing is None else bool(ipmr_checkpointing)
            ),
            "ipmr_auxiliary_loss": {
                "middle_weight": _resolved_ipmr_middle_weight(args),
                "high_weight": _resolved_ipmr_high_weight(args),
                "normalization": "per_scene_active_band_degrees_then_region_equal",
                "target_mask": "complete_physical_support_within_active_4x4_parent",
                "eligible_used": False,
                "decay_start_fraction": IPMR_AUX_DECAY_START_FRACTION,
                "decay_end_fraction": IPMR_AUX_DECAY_END_FRACTION,
                "checkpoint_selection": "formal_field_metric_only",
            },
            "physical_batch_size": int(physical_batch_size),
            "gradient_accumulation_steps": math.ceil(
                EFFECTIVE_BATCH_SIZE / int(physical_batch_size)
            ),
            "evaluation_batch_size": int(
                effective_evaluation_batch_size
                if effective_evaluation_batch_size is not None
                else args.evaluation_batch_size
            ),
            "gradient_accumulation": (
                "registered_logical_batch32_region_weighted_batch_slice"
            ),
            "scene_schedule": (
                "patch_only_before_full_scene_start_then_patch96x3_full160x1"
            ),
        })
        if args.model == "ipmr_q_v2":
            config.update({
                "ipmr_content_contract": IPMRQV2.content_contract,
                "ipmr_physical_contract": IPMRQV2.physical_contract,
                "ipmr_precision_contract": {
                    "amp_enabled": bool(args.amp),
                    "cuda_autocast_dtype": "float16" if bool(args.amp) else None,
                    "q_band_and_field_loss_dtype": "float32",
                },
                "ipmr_implementation_sha256": {
                    "trainer": _sha256_file(Path(__file__).resolve()),
                    "model": _sha256_file(CODE_ROOT / "g246_ipmr_q_v2.py"),
                    "q_bands": _sha256_file(CODE_ROOT / "g246_q_bands.py"),
                    "support_projection": _sha256_file(CODE_ROOT / "ocnir.py"),
                    "split_loader": _sha256_file(CODE_ROOT / "g246_data.py"),
                    "core_data_and_objective": _sha256_file(
                        CODE_ROOT / "g246_r2_data.py"
                    ),
                    "multisource_adapter": _sha256_file(
                        CODE_ROOT / "g246_r2_multisource.py"
                    ),
                    "metric_and_checkpoint": _sha256_file(
                        CODE_ROOT / "train_g246_metric.py"
                    ),
                },
                "ipmr_resume_policy": (
                    "fail_closed_on_precision_or_bound_implementation_byte_drift"
                ),
                "early_stop_policy": (
                    "manual_metric_gates_generic_four_point_plateau_disabled"
                ),
            })
        if args.model == PCQM_MODEL_NAME:
            if not isinstance(pcqm_optimization, Mapping):
                raise ValueError("PCQM optimizer binding is missing")
            if _resolved_ipmr_middle_weight(args) != 0.0 \
                    or _resolved_ipmr_high_weight(args) != 0.0:
                raise ValueError("PCQM uses the field objective only")
            config.update({
                "pcqm_schema_version": IPMRQPCQM.schema_version,
                "pcqm_design_contract": {
                    "pcqm": IPMRQPCQM.pcqm_contract,
                    "anchor": IPMRQPCQM.anchor_contract,
                    "equivariance": IPMRQPCQM.equivariance_contract,
                    "zero_initialization": (
                        "pcqm_output_weight_and_bias_exact_zero"
                    ),
                    "inherited_anchor": "ipmr_q_Fine52_Context19",
                    "anchor_registration": {
                        "checkpoint_sha256": PCQM_ANCHOR_CHECKPOINT_SHA256,
                        "checkpoint_role": PCQM_ANCHOR_CHECKPOINT_ROLE,
                        "optimizer_updates": PCQM_ANCHOR_OPTIMIZER_UPDATES,
                        "selected_state_dict_key": PCQM_ANCHOR_STATE_KEY,
                        "selected_tensor_state_sha256": (
                            PCQM_ANCHOR_STATE_SHA256
                        ),
                    },
                    "anchor_frozen": True,
                    "field_objective_only": True,
                    "explicit_geolocation_city_region_used": False,
                    "locked_test_opened": False,
                },
                "pcqm_optimizer_contract": copy.deepcopy(
                    dict(pcqm_optimization)
                ),
                "pcqm_implementation_sha256": {
                    "trainer": _sha256_file(Path(__file__).resolve()),
                    "model": _sha256_file(CODE_ROOT / "g246_ipmr_q_pcqm.py"),
                    "anchor_model": _sha256_file(CODE_ROOT / "g246_ipmr_q.py"),
                    "q_bands": _sha256_file(CODE_ROOT / "g246_q_bands.py"),
                    "support_projection": _sha256_file(CODE_ROOT / "ocnir.py"),
                    "split_loader": _sha256_file(CODE_ROOT / "g246_data.py"),
                    "core_data_and_objective": _sha256_file(
                        CODE_ROOT / "g246_r2_data.py"
                    ),
                    "multisource_adapter": _sha256_file(
                        CODE_ROOT / "g246_r2_multisource.py"
                    ),
                    "metric_and_checkpoint": _sha256_file(
                        CODE_ROOT / "train_g246_metric.py"
                    ),
                },
                "pcqm_resume_policy": (
                    "fresh_ipmr_anchor_once_then_resume_only_from_last_pt;_"
                    "fail_closed_on_registered_r9_ema_anchor_hash,_"
                    "optimizer_partition_and_config_binding"
                ),
                "early_stop_policy": (
                    "manual_mechanism_gate_generic_four_point_plateau_disabled"
                ),
            })
            config["ipmr_auxiliary_loss"].update({
                "enabled": False,
                "computed_during_training": False,
                "objective": "field_only_registered_0p8_eligible_0p2_valid",
            })
    if args.model == DCF_MODEL_NAME:
        config.update({
            "dcf_schema_version": DCFQ.schema_version,
            "dcf_scale_path": DCFQ.scale_path,
            "dcf_band_contract": DCFQ.band_contract,
            "dcf_context_contract": {
                "stored": "Context19",
                "learned": "Context15_in_both_frozen_experts",
                "physically_removed_indices": [5, 6, 7, 8],
                "coordinates_used": False,
            },
            "dcf_auxiliary_loss": {
                "middle_weight": _resolved_ipmr_middle_weight(args),
                "high_weight": _resolved_ipmr_high_weight(args),
                "supervised_value": "anchor_plus_correction_total_QM_QH",
                "normalization": "per_scene_active_band_degrees_then_region_equal",
                "target_mask": "complete_physical_support_within_active_4x4_parent",
                "eligible_used": False,
                "decay_start_fraction": IPMR_AUX_DECAY_START_FRACTION,
                "decay_end_fraction": IPMR_AUX_DECAY_END_FRACTION,
                "checkpoint_selection": "formal_field_metric_only",
            },
            "physical_batch_size": int(physical_batch_size),
            "gradient_accumulation_steps": math.ceil(
                EFFECTIVE_BATCH_SIZE / int(physical_batch_size)
            ),
            "evaluation_batch_size": int(
                effective_evaluation_batch_size
                if effective_evaluation_batch_size is not None
                else args.evaluation_batch_size
            ),
            "gradient_accumulation": (
                "registered_logical_batch32_region_weighted_batch_slice"
            ),
            "scene_schedule": (
                "patch_only_before_full_scene_start_then_patch96x3_full160x1"
            ),
        })
    if args.model == U1LITE_MODEL_NAME:
        u1lite_type = (
            DExchangeContinuousScaleBridgeQ
            if bool(getattr(args, "u1lite_dexchange", False))
            else ContinuousScaleBridgeQ
        )
        config.update({
            "u1lite_schema_version": u1lite_type.schema_version,
            "u1lite_stage": "target",
            "u1lite_scale_path": u1lite_type.scale_path,
            "u1lite_band_contract": u1lite_type.band_contract,
            "u1lite_context_contract": {
                "stored": "Context19",
                "learned": "Context15",
                "physically_removed_indices": [5, 6, 7, 8],
                "coordinates_used": False,
            },
            "u1lite_loss_contract": {
                "field": "registered_0p8_eligible_plus_0p2_valid",
                "qm_qh_auxiliary": False,
                "q_shape_auxiliary": False,
                "teacher_or_cache_attached": False,
            },
            "u1lite_optimizer_contract": {
                "group_names": ["u1lite_all"],
                "all_parameters_trainable": True,
                "source_optimizer_scaler_ema_imported": False,
            },
            "physical_batch_size": int(physical_batch_size),
            "gradient_accumulation_steps": math.ceil(
                EFFECTIVE_BATCH_SIZE / int(physical_batch_size)
            ),
            "evaluation_batch_size": int(
                effective_evaluation_batch_size
                if effective_evaluation_batch_size is not None
                else args.evaluation_batch_size
            ),
            "gradient_accumulation": (
                "registered_logical_batch32_region_weighted_batch_slice"
            ),
            "scene_schedule": (
                "patch_only_before_full_scene_start_then_patch96x3_full160x1"
            ),
            "early_stop_policy": (
                "manual_u150_learning_u750_performance_u1200_curve_exception"
                if u1lite_type is DExchangeContinuousScaleBridgeQ
                else "manual_u500_gate_generic_plateau_disabled"
            ),
        })
        if u1lite_type is DExchangeContinuousScaleBridgeQ:
            config["u1lite_dexchange"] = True
    if args.model == AOM_MODEL_NAME:
        config.update({
            "aom_schema_version": AOMQ.schema_version,
            "aom_scale_path": AOMQ.scale_path,
            "aom_context_contract": {
                "stored": "Context19", "learned": "Context15_in_both_backbones",
                "physically_removed_indices": [5, 6, 7, 8],
                "coordinates_used": False,
            },
            "aom_loss_contract": {
                "field": "registered_0p8_eligible_plus_0p2_valid",
                "branch_mse": False, "qm_qh_auxiliary": False,
                "extra_training_arms": False,
            },
            "aom_optimizer_contract": {
                "group_names": ["aom_pretrained_backbones", "aom_cross_scale_exchange",
                                "aom_identifiable_morphology"],
                "backbone_lr_multiplier": float(
                    getattr(args, "aom_backbone_lr_multiplier", 0.1)
                ),
                "exchange_lr_multiplier": 1.0, "morphology_lr_multiplier": 1.0,
            },
            "aom_formal_long_train_max_updates": max(8000, int(args.max_updates)),
            "physical_batch_size": int(physical_batch_size),
            "evaluation_batch_size": int(
                effective_evaluation_batch_size or args.evaluation_batch_size
            ),
            "gradient_accumulation_steps": math.ceil(
                EFFECTIVE_BATCH_SIZE / int(physical_batch_size)
            ),
            "scene_schedule": "patch_only_before_full_scene_start_then_patch96x3_full160x1",
        })
    if multisource_provenance is not None:
        config.update({
            "input_mode": "multisource",
            "multisource_provenance": dict(multisource_provenance),
            "physical_batch_size": int(physical_batch_size),
            "evaluation_batch_size": int(
                effective_evaluation_batch_size
                if effective_evaluation_batch_size is not None
                else args.evaluation_batch_size
            ),
            "gradient_accumulation": (
                "registered_logical_batch32_region_weighted_batch_slice"
            ),
            "scene_schedule": (
                "patch_only_before_full_scene_start_then_patch96x3_full160x1"
            ),
        })
    return config


def _validate_u1lite_d1_receipt(receipt: Any) -> dict[str, Any]:
    """Validate the independently produced D1 qualification receipt."""

    if not isinstance(receipt, Mapping) \
            or receipt.get("schema_version") != U1LITE_D1_QUALIFICATION_SCHEMA \
            or receipt.get("qualified") is not True \
            or receipt.get("locked_test_opened") is not False:
        raise ValueError("U1-Lite D1 qualification receipt is malformed")
    for field in ("fit_total_q_rmse_k", "dev15_total_q_rmse_k"):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) \
                or not 0.0 <= float(value) <= 0.020:
            raise ValueError(
                f"U1-Lite D1 qualification metric {field!r} did not pass"
            )
    return copy.deepcopy(dict(receipt))


def _validate_u1lite_initialization_provenance(value: Any) -> None:
    """Fail closed on the immutable D0-to-T0 transition receipt."""

    if not isinstance(value, Mapping) \
            or value.get("schema_version") \
            != U1LITE_TARGET_INITIALIZATION_SCHEMA \
            or value.get("source_checkpoint_schema") \
            != U1LITE_DISTILL_CHECKPOINT_SCHEMA \
            or value.get("source_checkpoint_role") != "qualified_d1" \
            or value.get("selected_state_dict_key") \
            != "raw_model_state_dict" \
            or value.get("source_model") != U1LITE_MODEL_NAME \
            or value.get("source_stage") != "distill" \
            or value.get("target_stage") != "target" \
            or value.get("imported_training_state") != [] \
            or value.get("all_parameters_trainable") is not True \
            or value.get("locked_test_opened") is not False:
        raise ValueError("resume U1-Lite initialization provenance is malformed")
    for field in (
        "source_checkpoint_sha256", "source_tensor_state_sha256",
        "source_config_sha256", "d1_qualification_receipt_sha256",
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"resume U1-Lite initialization provenance {field!r} differs"
            )
    receipt = _validate_u1lite_d1_receipt(
        value.get("d1_qualification_receipt")
    )
    if _canonical_mapping_sha256(receipt) \
            != value.get("d1_qualification_receipt_sha256"):
        raise ValueError("resume U1-Lite D1 receipt hash differs")
    normalization = value.get("normalization_reuse")
    if not isinstance(normalization, Mapping) \
            or normalization.get("schema_version") \
            != "g246-u1lite-distilled-normalization-reuse-v1" \
            or normalization.get("copy_mode") != "verified_exact_bytes" \
            or normalization.get("target_exact_byte_copy") is not True \
            or normalization.get("source_checkpoint_sha256") \
            != value.get("source_checkpoint_sha256") \
            or normalization.get("locked_test_opened") is not False:
        raise ValueError("resume U1-Lite normalization provenance differs")
    normalization_sha256 = normalization.get("source_normalization_sha256")
    if not isinstance(normalization_sha256, str) \
            or len(normalization_sha256) != 64 \
            or any(character not in "0123456789abcdef"
                   for character in normalization_sha256):
        raise ValueError("resume U1-Lite normalization SHA-256 differs")


def _validate_u1lite_direct_initialization_provenance(value: Any) -> None:
    """Validate the distinct, non-D1 registered-r6a migration receipt."""

    fresh_state = [
        "optimizer", "scaler", "ema", "update_clock", "wall_clock",
        "validation_records",
    ]
    if not isinstance(value, Mapping) \
            or value.get("schema_version") \
            != U1LITE_DIRECT_INITIALIZATION_SCHEMA \
            or value.get("initialization_mode") != "direct_r6a_exact_u0" \
            or value.get("source_checkpoint_schema") \
            != "uhi-cdc-g246-r2-checkpoint-v1" \
            or value.get("source_checkpoint_role") != "best" \
            or value.get("source_checkpoint_update") != 2000 \
            or value.get("selected_state_dict_key") \
            != "raw_model_state_dict" \
            or value.get("source_model") != "calibrated_q" \
            or value.get("target_model") != U1LITE_MODEL_NAME \
            or value.get("d1_artifact_used") is not False \
            or "d1_qualification_receipt" in value \
            or value.get("imported_training_state") != [] \
            or value.get("fresh_training_state") != fresh_state \
            or value.get("all_parameters_trainable") is not True \
            or value.get("u0_function_contract") \
            != "exact_r6a_raw_u2000_zero_bridge_zero_qm_qh_heads" \
            or value.get("locked_test_opened") is not False:
        raise ValueError("resume U1-Lite direct provenance is malformed")
    if value.get("source_checkpoint_sha256") \
            != PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256 \
            or value.get("source_tensor_state_sha256") \
            != PARENT_ROUTER_SHALLOW_STATE_SHA256:
        raise ValueError("resume U1-Lite direct registered r6a hash differs")
    for field in ("source_config_sha256", "full_u0_state_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"resume U1-Lite direct {field!r} differs")
    normalization = value.get("normalization_reuse")
    if not isinstance(normalization, Mapping) \
            or normalization.get("schema_version") \
            != U1LITE_DIRECT_NORMALIZATION_SCHEMA \
            or normalization.get("source_checkpoint_sha256") \
            != value.get("source_checkpoint_sha256") \
            or normalization.get("source_normalization_sha256") \
            != PARENT_ROUTER_NORMALIZATION_SHA256 \
            or normalization.get("copy_mode") != "verified_exact_bytes" \
            or normalization.get("target_exact_byte_copy") is not True \
            or normalization.get("locked_test_opened") is not False:
        raise ValueError("resume U1-Lite direct normalization provenance differs")
    fit_view = normalization.get("fit_view_sha256")
    if not isinstance(fit_view, str) or len(fit_view) != 64 \
            or any(character not in "0123456789abcdef" for character in fit_view):
        raise ValueError("resume U1-Lite direct normalization Fit view differs")


def _resume_scientific_config(
    checkpoint_config: Mapping[str, Any],
    requested_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate resume config while accepting pre-contract R2 checkpoints.

    Old checkpoints did not record the EMA strategy or the explicit loss
    contract.  If *only* those derived fields are absent, retain the old config
    verbatim and let :class:`WarmStartEMA` continue its legacy fixed-decay path.
    No scientific field may otherwise change.
    """

    checkpoint = dict(checkpoint_config)
    requested = dict(requested_config)
    if checkpoint == requested:
        return requested
    # A weights-only initialization is performed once on a fresh run.  Resume
    # must retain its immutable provenance without requiring (or allowing) the
    # source checkpoint to be read again.
    checkpoint_for_compare = dict(checkpoint)
    initialization = checkpoint_for_compare.pop("initialization_provenance", None)
    if initialization is not None:
        if not isinstance(initialization, Mapping) \
                or initialization.get("schema_version") != INITIALIZATION_SCHEMA \
                or initialization.get("locked_test_opened") is not False:
            raise ValueError("resume initialization provenance is malformed")
        if checkpoint_for_compare == requested:
            return checkpoint
    dcf_initialization = checkpoint_for_compare.pop(
        "dcf_initialization_provenance", None
    )
    if dcf_initialization is not None:
        if not isinstance(dcf_initialization, Mapping) \
                or dcf_initialization.get("schema_version") \
                != DCFQ.schema_version \
                or dcf_initialization.get("locked_test_opened") is not False:
            raise ValueError("resume DCF-Q initialization provenance is malformed")
        normalization_reuse = dcf_initialization.get("normalization_reuse")
        if not isinstance(normalization_reuse, Mapping) \
                or normalization_reuse.get("sources_byte_identical") is not True \
                or normalization_reuse.get("target_exact_byte_copy") is not True \
                or normalization_reuse.get("source_normalization_sha256") \
                != PARENT_ROUTER_NORMALIZATION_SHA256:
            raise ValueError("resume DCF-Q normalization provenance differs")
        for label, expected in (
            ("shallow", {
                "checkpoint_sha256": PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
                "checkpoint_update": 2000,
                "selected_state_dict_key": "raw_model_state_dict",
                "selected_tensor_state_sha256": PARENT_ROUTER_SHALLOW_STATE_SHA256,
                "source_model": "calibrated_q",
            }),
            ("deep", {
                "checkpoint_sha256": PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
                "checkpoint_update": 8000,
                "selected_state_dict_key": "ema_model_state_dict",
                "selected_tensor_state_sha256": PARENT_ROUTER_DEEP_STATE_SHA256,
                "source_model": "ipmr_q",
            }),
        ):
            source = dcf_initialization.get(label)
            if not isinstance(source, Mapping) \
                    or source.get("locked_test_opened") is not False \
                    or any(source.get(key) != value for key, value in expected.items()):
                raise ValueError(f"resume DCF-Q {label} source provenance differs")
        if checkpoint_for_compare == requested:
            return checkpoint
    aom_initialization = checkpoint_for_compare.pop(
        "aom_initialization_provenance", None
    )
    if aom_initialization is not None:
        if not isinstance(aom_initialization, Mapping) \
                or aom_initialization.get("schema_version") != AOMQ.schema_version \
                or aom_initialization.get("locked_test_opened") is not False:
            raise ValueError("resume AOM-Q initialization provenance is malformed")
        if checkpoint_for_compare == requested:
            return checkpoint
    u1lite_initialization = checkpoint_for_compare.pop(
        "u1lite_distilled_initialization_provenance", None
    )
    u1lite_direct_initialization = checkpoint_for_compare.pop(
        "u1lite_direct_initialization_provenance", None
    )
    if u1lite_initialization is not None \
            and u1lite_direct_initialization is not None:
        raise ValueError("resume U1-Lite carries two initialization modes")
    if u1lite_initialization is not None:
        _validate_u1lite_initialization_provenance(u1lite_initialization)
        normalization = u1lite_initialization["normalization_reuse"]
        if normalization.get("source_normalization_sha256") \
                != checkpoint_for_compare.get("normalization_sha256") \
                or normalization.get("fit_view_sha256") \
                != checkpoint_for_compare.get("fit_view_sha256"):
            raise ValueError(
                "resume U1-Lite normalization/scientific config binding differs"
            )
        if checkpoint_for_compare == requested:
            return checkpoint
    if u1lite_direct_initialization is not None:
        _validate_u1lite_direct_initialization_provenance(
            u1lite_direct_initialization
        )
        normalization = u1lite_direct_initialization["normalization_reuse"]
        if checkpoint_for_compare.get("model") != U1LITE_MODEL_NAME \
                or requested.get("model") != U1LITE_MODEL_NAME \
                or normalization.get("source_normalization_sha256") \
                != checkpoint_for_compare.get("normalization_sha256") \
                or normalization.get("fit_view_sha256") \
                != checkpoint_for_compare.get("fit_view_sha256") \
                or checkpoint_for_compare.get("u1lite_resume_policy") \
                != U1LITE_DIRECT_RESUME_POLICY:
            raise ValueError(
                "resume U1-Lite direct scientific config binding differs"
            )
        # A direct resume deliberately has no source-path CLI.  Its validated
        # immutable provenance is the authority for reconstructing this one
        # mode-specific config field before the ordinary exact comparison.
        requested["u1lite_resume_policy"] = U1LITE_DIRECT_RESUME_POLICY
        if checkpoint_for_compare == requested:
            return checkpoint
    parent_router_initialization = checkpoint_for_compare.pop(
        "parent_router_initialization_provenance", None
    )
    if parent_router_initialization is not None:
        if not isinstance(parent_router_initialization, Mapping) \
                or parent_router_initialization.get("schema_version") \
                != ParentRouterQ.schema_version \
                or parent_router_initialization.get("locked_test_opened") is not False:
            raise ValueError(
                "resume ParentRouterQ initialization provenance is malformed"
            )
        normalization_reuse = parent_router_initialization.get(
            "normalization_reuse"
        )
        if not isinstance(normalization_reuse, Mapping) \
                or normalization_reuse.get("sources_byte_identical") is not True \
                or normalization_reuse.get("target_exact_byte_copy") is not True:
            raise ValueError(
                "resume ParentRouterQ normalization provenance is malformed"
            )
        if normalization_reuse.get("source_normalization_sha256") \
                != PARENT_ROUTER_NORMALIZATION_SHA256:
            raise ValueError(
                "resume ParentRouterQ normalization SHA-256 differs"
            )
        for label, expected in (
            ("shallow", {
                "checkpoint_sha256": PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
                "checkpoint_update": 2000,
                "selected_state_dict_key": "raw_model_state_dict",
                "selected_tensor_state_sha256": (
                    PARENT_ROUTER_SHALLOW_STATE_SHA256
                ),
                "source_model": "calibrated_q",
            }),
            ("deep", {
                "checkpoint_sha256": PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
                "checkpoint_update": 8000,
                "selected_state_dict_key": "ema_model_state_dict",
                "selected_tensor_state_sha256": PARENT_ROUTER_DEEP_STATE_SHA256,
                "source_model": "ipmr_q",
            }),
        ):
            source = parent_router_initialization.get(label)
            if not isinstance(source, Mapping) \
                    or source.get("locked_test_opened") is not False \
                    or any(source.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    f"resume ParentRouterQ {label} source provenance differs"
                )
        if checkpoint_for_compare == requested:
            return checkpoint
    for field in (
        "ema_strategy", "ema_selector_maturity_updates", "loss_contract"
    ):
        requested.pop(field, None)
    if checkpoint_for_compare == requested:
        return checkpoint
    raise ValueError("resume scientific config differs from the checkpoint")


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reuse_initialization_normalization(
    checkpoint_path: str | os.PathLike[str],
    destination: Path,
    *,
    expected_source_model: str | None = None,
) -> tuple[R2Normalization, dict[str, Any]]:
    """Byte-copy the normalization bound to a weights-only source checkpoint.

    Re-fitting the same statistics can serialize the final floating-point
    values differently across NumPy/Python platforms.  A weights-only
    extension must use the source checkpoint's exact predictor scale, so this
    helper verifies the sibling ``normalization.json`` against the SHA stored
    in the source scientific config and atomically copies those exact bytes.
    It never imports optimizer, EMA, scheduler, updates, or validation records.
    """

    checkpoint_candidate = reject_forbidden_path(checkpoint_path)
    if checkpoint_candidate.is_symlink():
        raise ValueError("initialization checkpoint may not be a symlink")
    checkpoint = checkpoint_candidate.resolve()
    reject_forbidden_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"initialization checkpoint does not exist: {checkpoint}"
        )
    with checkpoint.open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) \
            or payload.get("schema_version") != "uhi-cdc-g246-r2-checkpoint-v1":
        raise ValueError("unsupported R2 initialization checkpoint schema")
    if payload.get("locked_test_opened") is not False:
        raise ValueError("initialization checkpoint is not locked-test closed")
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("initialization checkpoint config is missing")
    if source_config.get("data_scope") != "public_validation" \
            or source_config.get("locked_test_opened") is not False:
        raise ValueError(
            "normalization reuse requires a locked-test-closed public_validation source"
        )
    if source_config.get("model") not in {"calibrated_q", "ipmr_q"} \
            or source_config.get("input_mode") != "multisource" \
            or source_config.get("fine_channels") != MULTISOURCE_FINE_CHANNELS \
            or source_config.get("context_dim") != MULTISOURCE_CONTEXT_DIM:
        raise ValueError(
            "normalization reuse requires an exact registered Fine52/Context19 source"
        )
    if expected_source_model is not None \
            and source_config.get("model") != expected_source_model:
        raise ValueError(
            "normalization reuse source model differs from the target migration"
        )
    if source_config.get("provisional_optical_authorized") is not False:
        raise ValueError("normalization reuse forbids provisional optical inputs")
    multisource = source_config.get("multisource_provenance")
    if not isinstance(multisource, Mapping) \
            or multisource.get("locked_test_opened") is not False \
            or multisource.get("target_arrays_opened_by_multisource_adapter") is not False:
        raise ValueError("normalization reuse source is not target-free")
    expected_sha = source_config.get("normalization_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 \
            or any(character not in "0123456789abcdef" for character in expected_sha):
        raise ValueError("source normalization SHA-256 contract is malformed")

    source_candidate = reject_forbidden_path(
        checkpoint.parent / "normalization.json"
    )
    if source_candidate.is_symlink():
        raise ValueError("source normalization.json may not be a symlink")
    source = source_candidate.resolve()
    reject_forbidden_path(source)
    if not source.is_file():
        raise FileNotFoundError(
            "weights-only initialization requires the source checkpoint's "
            "sibling normalization.json"
        )
    with source.open("rb") as handle:
        raw = handle.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "source normalization bytes differ from the checkpoint contract"
        )
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source normalization JSON is malformed") from exc
    if not isinstance(document, Mapping):
        raise ValueError("source normalization JSON root must be an object")
    normalization = R2Normalization.from_dict(document)
    if normalization.fit_view_sha256 != source_config.get("fit_view_sha256"):
        raise ValueError("source normalization fit view differs from the checkpoint")

    destination_candidate = reject_forbidden_path(destination)
    if destination_candidate.is_symlink():
        raise ValueError("destination normalization.json may not be a symlink")
    destination = destination_candidate.resolve()
    reject_forbidden_path(destination)
    if destination.exists():
        raise FileExistsError(
            "normalization reuse destination already exists on a fresh run"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    created_by_this_call = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        created_by_this_call = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created_by_this_call:
            destination.unlink(missing_ok=True)
        raise
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha:
        raise OSError("atomic normalization reuse did not preserve exact bytes")
    return normalization, {
        "schema_version": "g246-r2-initialization-normalization-reuse-v1",
        "source_path": str(source),
        "source_sha256": expected_sha,
        "byte_count": len(raw),
        "copy_mode": "verified_exact_bytes",
        "fit_view_sha256": normalization.fit_view_sha256,
        "locked_test_opened": False,
    }


def _allocation_adapter_enabled(config: Mapping[str, Any]) -> bool:
    contract = config.get("calibrated_allocation_adapter")
    if contract is None:
        return False
    if not isinstance(contract, Mapping):
        raise ValueError("calibrated allocation-adapter config is malformed")
    if dict(contract) != ALLOCATION_ADAPTER_CONTRACT:
        raise ValueError("unsupported calibrated allocation-adapter contract")
    return True


def _calibrated_t3_enabled(config: Mapping[str, Any]) -> bool:
    contract = config.get("calibrated_t3_fusion")
    if contract is None:
        return False
    if not isinstance(contract, Mapping):
        raise ValueError("calibrated T3-fusion config is malformed")
    if dict(contract) != T3_FUSION_CONTRACT:
        raise ValueError("unsupported calibrated T3-fusion contract")
    return True


def _calibrated_q_refiner_enabled(config: Mapping[str, Any]) -> bool:
    contract = config.get("calibrated_q_refiner")
    if contract is None:
        return False
    if not isinstance(contract, Mapping):
        raise ValueError("calibrated Q-refiner config is malformed")
    if dict(contract) != Q_REFINER_CONTRACT:
        raise ValueError("unsupported calibrated Q-refiner contract")
    optimization = config.get("q_refiner_optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("calibrated Q-refiner optimization contract is missing")
    expected_keys = {
        "freeze_backbone_updates",
        "core_lr_multiplier",
        "refiner_lr_multiplier",
        "freeze_clock",
        "frozen_parameters",
        "optimizer_groups",
    }
    if set(optimization) != expected_keys:
        raise ValueError("calibrated Q-refiner optimization contract is malformed")
    freeze = optimization.get("freeze_backbone_updates")
    core_multiplier = optimization.get("core_lr_multiplier")
    refiner_multiplier = optimization.get("refiner_lr_multiplier")
    if isinstance(freeze, bool) or not isinstance(freeze, int) or freeze < 0:
        raise ValueError("Q-refiner freeze updates must be nonnegative")
    for value, label in (
        (core_multiplier, "core"), (refiner_multiplier, "refiner")
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"Q-refiner {label} LR multiplier must be positive")
    if optimization.get("freeze_clock") != "global_successful_optimizer_updates" \
            or optimization.get("frozen_parameters") != "all_except_q_refiner_prefix" \
            or optimization.get("optimizer_groups") != ["core", "q_refiner"]:
        raise ValueError("unsupported calibrated Q-refiner optimization contract")
    return True


def _calibrated_content_q_pyramid_enabled(config: Mapping[str, Any]) -> bool:
    contract = config.get("calibrated_content_q_pyramid")
    if contract is None:
        return False
    if not isinstance(contract, Mapping):
        raise ValueError("calibrated Content-only Q-Pyramid config is malformed")
    if dict(contract) != CONTENT_Q_PYRAMID_CONTRACT:
        raise ValueError("unsupported calibrated Content-only Q-Pyramid contract")
    optimization = config.get("content_q_pyramid_optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("calibrated Q-Pyramid optimization contract is missing")
    expected_keys = {
        "freeze_core_updates",
        "core_lr_multiplier",
        "pyramid_lr_multiplier",
        "freeze_clock",
        "frozen_parameters",
        "optimizer_groups",
    }
    if set(optimization) != expected_keys:
        raise ValueError("calibrated Q-Pyramid optimization contract is malformed")
    freeze = optimization.get("freeze_core_updates")
    core_multiplier = optimization.get("core_lr_multiplier")
    pyramid_multiplier = optimization.get("pyramid_lr_multiplier")
    if isinstance(freeze, bool) or not isinstance(freeze, int) or freeze < 0:
        raise ValueError("Q-Pyramid freeze updates must be nonnegative")
    for value, label in (
        (core_multiplier, "core"), (pyramid_multiplier, "pyramid")
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"Q-Pyramid {label} LR multiplier must be positive")
    if optimization.get("freeze_clock") != "global_successful_optimizer_updates" \
            or optimization.get("frozen_parameters") \
            != "all_except_content_q_pyramid_prefix" \
            or optimization.get("optimizer_groups") \
            != ["core", "content_q_pyramid"]:
        raise ValueError("unsupported calibrated Q-Pyramid optimization contract")
    return True


def _calibrated_parent_dct15_enabled(config: Mapping[str, Any]) -> bool:
    contract = config.get("calibrated_parent_dct15")
    if contract is None:
        return False
    if not isinstance(contract, Mapping):
        raise ValueError("calibrated Parent-DCT15 config is malformed")
    if dict(contract) != PARENT_DCT15_CONTRACT:
        raise ValueError("unsupported calibrated Parent-DCT15 contract")
    optimization = config.get("parent_dct15_optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("calibrated Parent-DCT15 optimization contract is missing")
    expected_keys = {
        "freeze_core_updates",
        "core_lr_multiplier",
        "dct15_lr_multiplier",
        "freeze_clock",
        "frozen_parameters",
        "optimizer_groups",
    }
    if set(optimization) != expected_keys:
        raise ValueError("calibrated Parent-DCT15 optimization contract is malformed")
    freeze = optimization.get("freeze_core_updates")
    core_multiplier = optimization.get("core_lr_multiplier")
    dct15_multiplier = optimization.get("dct15_lr_multiplier")
    if isinstance(freeze, bool) or not isinstance(freeze, int) or freeze < 0:
        raise ValueError("Parent-DCT15 freeze updates must be nonnegative")
    for value, label in (
        (core_multiplier, "core"), (dct15_multiplier, "decoder")
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                f"Parent-DCT15 {label} LR multiplier must be positive"
            )
    if optimization.get("freeze_clock") \
            != "global_successful_optimizer_updates" \
            or optimization.get("frozen_parameters") \
            != "all_except_parent_dct15_prefix" \
            or optimization.get("optimizer_groups") \
            != ["core", "parent_dct15"]:
        raise ValueError("unsupported calibrated Parent-DCT15 optimization contract")
    return True


def _calibrated_no_geo_core_mode(config: Mapping[str, Any]) -> str | None:
    """Validate and return the registered physical-Context15 rebase mode."""

    contract = config.get("calibrated_no_geo_core")
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("calibrated no-geolocation core config is malformed")
    mode = contract.get("rebase_mode")
    probe_mode = _calibrated_no_geo_probe_mode(config)
    if not isinstance(mode, str) or dict(contract) != _no_geo_core_contract(
        mode, probe_mode=probe_mode
    ):
        raise ValueError("unsupported calibrated no-geolocation core contract")
    return mode


_INITIALIZATION_SCIENTIFIC_FIELDS = (
    "model",
    "width",
    "data_scope",
    "region_sampling",
    "primary_loss_weight",
    "loss_contract",
    "q_shape_loss",
    "input_provenance",
    "scientific_status",
    "provisional_optical_authorized",
    "fine_channels",
    "context_dim",
    "receipt_sha256",
    "fit_view_sha256",
    "split_identity",
    "train_city_count",
    "evaluation_city_count",
    "normalization_sha256",
    "locked_test_opened",
    "input_mode",
    "multisource_provenance",
    "calibrated_size",
    "calibrated_channels",
    "calibrated_width_scaling",
    "full_scene_start",
    "scene_schedule",
)


def _validate_initialization_scientific_contract(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[
    bool, bool, bool, bool, bool, bool, bool, bool, bool, bool,
    str | None, str | None,
]:
    """Fail closed outside the explicitly registered extension migrations.

    Optimizer, scheduler, seed, batch-size and activation-checkpoint execution
    fields are deliberately not compared because an initialization is a fresh
    run.  Every field affecting the public data view, predictors, objective or
    scene schedule is compared exactly.  The Content-Q and Parent-DCT15 routes
    additionally mask geolocation before the inherited core and therefore
    declare, rather than conceal, their non-identical no-geolocation rebase.
    """

    if source.get("data_scope") != "public_validation" \
            or target.get("data_scope") != "public_validation":
        raise ValueError("initialization requires public_validation scope")
    if source.get("model") != "calibrated_q" \
            or target.get("model") != "calibrated_q":
        raise ValueError("initialization supports only calibrated_q checkpoints")
    for label, config in (("source", source), ("target", target)):
        if config.get("locked_test_opened") is not False:
            raise ValueError(f"{label} initialization config is not locked-test closed")
        if config.get("provisional_optical_authorized") is not False:
            raise ValueError(f"{label} initialization config uses provisional optical")
        if config.get("input_mode") != "multisource" \
                or config.get("fine_channels") != MULTISOURCE_FINE_CHANNELS \
                or config.get("context_dim") != MULTISOURCE_CONTEXT_DIM:
            raise ValueError("initialization requires the exact Fine52/Context19 contract")
        provenance = config.get("multisource_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{label} Fine52 manifest provenance is missing")
        if provenance.get("locked_test_opened") is not False \
                or provenance.get("target_arrays_opened_by_multisource_adapter") is not False:
            raise ValueError(f"{label} Fine52 provenance is not target-free")
        for field in (
            "binding_sha256",
            "texture_manifest_sha256",
            "weather_manifest_sha256",
        ):
            value = provenance.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{label} Fine52 provenance lacks {field}")
    for field in _INITIALIZATION_SCIENTIFIC_FIELDS:
        if source.get(field) != target.get(field):
            raise ValueError(
                f"initialization scientific contract differs at {field!r}"
            )
    source_adapter = _allocation_adapter_enabled(source)
    target_adapter = _allocation_adapter_enabled(target)
    source_t3 = _calibrated_t3_enabled(source)
    target_t3 = _calibrated_t3_enabled(target)
    source_refiner = _calibrated_q_refiner_enabled(source)
    target_refiner = _calibrated_q_refiner_enabled(target)
    source_pyramid = _calibrated_content_q_pyramid_enabled(source)
    target_pyramid = _calibrated_content_q_pyramid_enabled(target)
    source_dct15 = _calibrated_parent_dct15_enabled(source)
    target_dct15 = _calibrated_parent_dct15_enabled(target)
    source_no_geo = _calibrated_no_geo_core_mode(source)
    target_no_geo = _calibrated_no_geo_core_mode(target)
    source_probe = _calibrated_no_geo_probe_mode(source)
    target_probe = _calibrated_no_geo_probe_mode(target)
    source_pack_arm = _calibrated_no_geo_pack_arm(source)
    target_pack_arm = _calibrated_no_geo_pack_arm(target)
    source_temporal = source.get("temporal_mode")
    target_temporal = target.get("temporal_mode")
    if source_t3 and not target_t3:
        raise ValueError("cannot initialize a non-T3 model from T3 weights")
    if source_refiner and not target_refiner:
        raise ValueError("cannot initialize a non-refiner model from refiner weights")
    if source_pyramid:
        raise ValueError(
            "weights-only Q-Pyramid initialization source must be the bare r2k anchor"
        )
    if source_dct15:
        raise ValueError(
            "weights-only Parent-DCT15 initialization source must be the bare r2k anchor"
        )
    if source_no_geo is not None:
        raise ValueError(
            "weights-only no-geolocation initialization source must be the bare r2k anchor"
        )
    if source_probe is not None:
        raise ValueError(
            "weights-only no-geolocation probe source must be the bare r2k anchor"
        )
    if source_pack_arm is not None:
        raise ValueError(
            "weights-only no-geolocation pack source must be the bare r2k anchor"
        )
    if target_pyramid and any((target_adapter, target_t3, target_refiner)):
        raise ValueError(
            "Content-only Q-Pyramid is mutually exclusive with adapter, T3 and Q-refiner"
        )
    if target_pyramid and any((
        source_adapter, source_t3, source_refiner, source_pyramid,
    )):
        raise ValueError(
            "Content-only Q-Pyramid initialization requires the extension-free r2k anchor"
        )
    if target_pyramid and target_temporal != "single":
        raise ValueError("Content-only Q-Pyramid initialization requires single temporal mode")
    if target_dct15 and any((
        target_adapter, target_t3, target_refiner, target_pyramid,
    )):
        raise ValueError(
            "Parent-DCT15 is mutually exclusive with adapter, T3, Q-refiner "
            "and Content-only Q-Pyramid"
        )
    if target_dct15 and any((
        source_adapter, source_t3, source_refiner, source_pyramid, source_dct15,
    )):
        raise ValueError(
            "Parent-DCT15 initialization requires the extension-free bare r2k anchor"
        )
    if target_dct15 and target_temporal != "single":
        raise ValueError("Parent-DCT15 initialization requires single temporal mode")
    if target_no_geo is not None and any((
        target_adapter, target_t3, target_refiner, target_pyramid, target_dct15,
    )):
        raise ValueError(
            "physical Context15 core is mutually exclusive with every calibrated extension"
        )
    if target_no_geo is not None and any((
        source_adapter, source_t3, source_refiner, source_pyramid, source_dct15,
    )):
        raise ValueError(
            "physical Context15 initialization requires the extension-free bare r2k anchor"
        )
    if target_no_geo is not None and (
        source_temporal != "single" or target_temporal != "single"
    ):
        raise ValueError(
            "physical Context15 initialization requires unchanged single temporal mode"
        )
    if target_probe is not None and target_no_geo != "surgery":
        raise ValueError(
            "no-geolocation H/B probes require the surgery Context15 rebase"
        )
    if target_pack_arm is not None and target_no_geo != "surgery":
        raise ValueError(
            "no-geolocation pack screen requires the surgery Context15 rebase"
        )
    if target_pack_arm is not None and target_probe is not None:
        raise ValueError("no-geolocation pack screen and H/B probe are mutually exclusive")
    if target_t3 and target_temporal != "multi":
        raise ValueError("calibrated T3 initialization requires target temporal_mode=multi")
    if source_t3:
        if source_temporal != "multi" or target_temporal != "multi":
            raise ValueError("T3 checkpoints require an unchanged multi-temporal contract")
    elif target_t3:
        if source_temporal != "single" or target_temporal != "multi":
            raise ValueError(
                "new T3 initialization permits only single-to-multi temporal migration"
            )
    elif source_temporal != target_temporal:
        raise ValueError("initialization scientific contract differs at 'temporal_mode'")
    if source_adapter:
        raise ValueError(
            "weights-only initialization source must be the nonadapter r2k anchor"
        )
    return (
        source_adapter, target_adapter, source_t3, target_t3,
        source_refiner, target_refiner, source_pyramid, target_pyramid,
        source_dct15, target_dct15, source_no_geo, target_no_geo,
    )


def _selected_initialization_state(
    checkpoint: Mapping[str, Any], requested_source: str
) -> tuple[Mapping[str, Tensor], str, str | None]:
    """Return exactly one raw/EMA inference state and its unambiguous key."""

    if requested_source not in {"selected", "raw", "ema"}:
        raise ValueError("initialization weight source must be selected, raw, or ema")
    contract = checkpoint.get("weight_contract")
    selector: str | None = None
    if requested_source == "selected":
        if not isinstance(contract, Mapping):
            raise ValueError("selected initialization requires a weight contract")
        selector = contract.get("best_inference_state_dict_key")
        if selector not in {"raw_model_state_dict", "ema_model_state_dict"}:
            raise ValueError("checkpoint has no selected raw/EMA inference weights")
        requested_source = "raw" if selector == "raw_model_state_dict" else "ema"
    if requested_source == "raw":
        state = checkpoint.get("raw_model_state_dict")
        state_key = "raw_model_state_dict"
    else:
        state = checkpoint.get("ema_model_state_dict")
        state_key = "ema_model_state_dict"
        if state is None:
            ema_payload = checkpoint.get("ema_state_dict")
            if isinstance(ema_payload, Mapping):
                state = ema_payload.get("shadow")
                state_key = "ema_state_dict.shadow"
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint is missing {requested_source} model weights")
    if any(not isinstance(key, str) or not isinstance(value, Tensor)
           for key, value in state.items()):
        raise ValueError("initialization state dict must map string keys to tensors")
    return state, state_key, selector


def _tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash tensor names, dtype/shape metadata and exact contiguous bytes."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("tensor state for hashing must be a nonempty mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("tensor state must map string names to tensors")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _load_u1lite_distilled_checkpoint(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Authenticate one independently produced, D1-qualified student."""

    source_path = _regular_public_artifact(
        Path(checkpoint_path), "U1-Lite distilled source checkpoint"
    )
    # Hash and deserialize the same inode so provenance cannot describe a
    # different file from the tensors that are subsequently checked.
    with source_path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) \
            or payload.get("schema_version") \
            != U1LITE_DISTILL_CHECKPOINT_SCHEMA \
            or payload.get("checkpoint_role") != "qualified_d1" \
            or payload.get("locked_test_opened") is not False:
        raise ValueError(
            "U1-Lite distilled source is not a locked-test-closed qualified D1 artifact"
        )
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping) \
            or source_config.get("model") != U1LITE_MODEL_NAME \
            or source_config.get("u1lite_stage") != "distill" \
            or source_config.get("locked_test_opened") is not False:
        raise ValueError("U1-Lite distilled source config is malformed")
    required_config = {
        "width": 48,
        "temporal_mode": "single",
        "data_scope": "public_validation",
        "input_mode": "multisource",
        "fine_channels": MULTISOURCE_FINE_CHANNELS,
        "context_dim": MULTISOURCE_CONTEXT_DIM,
        "provisional_optical_authorized": False,
    }
    for field, expected in required_config.items():
        if source_config.get(field) != expected:
            raise ValueError(
                f"U1-Lite distilled source config differs at {field!r}"
            )
    for field in ("receipt_sha256", "fit_view_sha256", "normalization_sha256"):
        value = source_config.get(field)
        if not isinstance(value, str) or len(value) != 64 \
                or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(
                f"U1-Lite distilled source config {field!r} is malformed"
            )
    multisource = source_config.get("multisource_provenance")
    if not isinstance(multisource, Mapping) \
            or multisource.get("locked_test_opened") is not False \
            or multisource.get("target_arrays_opened_by_multisource_adapter") \
            is not False:
        raise ValueError("U1-Lite distilled source multisource binding is malformed")
    binding_sha256 = multisource.get("binding_sha256")
    if not isinstance(binding_sha256, str) or len(binding_sha256) != 64 \
            or any(character not in "0123456789abcdef" for character in binding_sha256):
        raise ValueError("U1-Lite distilled source input binding SHA-256 is malformed")

    state = payload.get("raw_model_state_dict")
    if not isinstance(state, Mapping) or not state \
            or any(not isinstance(key, str) or not isinstance(value, Tensor)
                   for key, value in state.items()):
        raise ValueError("U1-Lite distilled source raw model state is malformed")
    tensor_sha256 = _tensor_state_sha256(state)
    if payload.get("raw_model_state_sha256") != tensor_sha256:
        raise ValueError("U1-Lite distilled source raw tensor SHA-256 differs")
    receipt = _validate_u1lite_d1_receipt(
        payload.get("d1_qualification_receipt")
    )
    return {
        "path": source_path,
        "checkpoint_sha256": checkpoint_sha256,
        "payload": payload,
        "config": dict(source_config),
        "state": state,
        "tensor_sha256": tensor_sha256,
        "d1_qualification_receipt": receipt,
    }


def _reuse_u1lite_distilled_normalization(
    checkpoint_path: str | os.PathLike[str],
    destination: Path,
) -> tuple[R2Normalization, dict[str, Any]]:
    """Copy the D0 sibling normalization byte-exactly for fresh T0."""

    source = _load_u1lite_distilled_checkpoint(checkpoint_path)
    source_config = source["config"]
    normalization_path = _regular_public_artifact(
        source["path"].parent / "normalization.json",
        "U1-Lite distilled source normalization",
    )
    raw = normalization_path.read_bytes()
    normalization_sha256 = hashlib.sha256(raw).hexdigest()
    if normalization_sha256 != source_config["normalization_sha256"]:
        raise ValueError(
            "U1-Lite distilled source normalization differs from its config"
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "U1-Lite distilled source normalization JSON is malformed"
        ) from exc
    if not isinstance(document, Mapping):
        raise ValueError("U1-Lite distilled normalization root is malformed")
    normalization = R2Normalization.from_dict(document)
    if normalization.fit_view_sha256 != source_config["fit_view_sha256"]:
        raise ValueError("U1-Lite distilled normalization fit view differs")
    target = reject_forbidden_path(destination)
    if target.is_symlink():
        raise ValueError("U1-Lite destination normalization may not be a symlink")
    if target.exists():
        raise FileExistsError(
            "U1-Lite destination normalization already exists on a fresh run"
        )
    _atomic_bytes(target, raw)
    if target.read_bytes() != raw:
        raise OSError("U1-Lite normalization copy was not byte exact")
    return normalization, {
        "schema_version": "g246-u1lite-distilled-normalization-reuse-v1",
        "source_path": str(normalization_path),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_normalization_sha256": normalization_sha256,
        "byte_count": len(raw),
        "copy_mode": "verified_exact_bytes",
        "target_exact_byte_copy": True,
        "fit_view_sha256": normalization.fit_view_sha256,
        "locked_test_opened": False,
    }


def initialize_u1lite_from_distilled_checkpoint(
    model: nn.Module,
    checkpoint_path: str | os.PathLike[str],
    target_config: Mapping[str, Any],
    normalization_reuse: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly import only the complete D0 raw student state into fresh T0."""

    if not isinstance(model, ContinuousScaleBridgeQ):
        raise TypeError("U1-Lite initialization requires ContinuousScaleBridgeQ")
    if target_config.get("model") != U1LITE_MODEL_NAME \
            or target_config.get("u1lite_stage") != "target" \
            or target_config.get("locked_test_opened") is not False:
        raise ValueError("U1-Lite target config is malformed")
    source = _load_u1lite_distilled_checkpoint(checkpoint_path)
    if normalization_reuse.get("source_checkpoint_sha256") \
            != source["checkpoint_sha256"]:
        raise ValueError(
            "U1-Lite distilled checkpoint changed after normalization reuse"
        )
    if normalization_reuse.get("source_normalization_sha256") \
            != target_config.get("normalization_sha256") \
            or normalization_reuse.get("target_exact_byte_copy") is not True:
        raise ValueError("U1-Lite target normalization binding differs")
    source_config = source["config"]
    for field in (
        "width", "temporal_mode", "data_scope", "input_mode",
        "fine_channels", "context_dim", "receipt_sha256",
        "fit_view_sha256", "normalization_sha256", "split_identity",
    ):
        if source_config.get(field) != target_config.get(field):
            raise ValueError(
                f"U1-Lite D0/T0 scientific identity differs at {field!r}"
            )
    source_multisource = source_config.get("multisource_provenance")
    target_multisource = target_config.get("multisource_provenance")
    if not isinstance(source_multisource, Mapping) \
            or not isinstance(target_multisource, Mapping) \
            or source_multisource.get("binding_sha256") \
            != target_multisource.get("binding_sha256"):
        raise ValueError("U1-Lite D0/T0 multisource binding differs")
    state = source["state"]
    reference = model.state_dict()
    if set(state) != set(reference):
        missing = sorted(set(reference) - set(state))
        unexpected = sorted(set(state) - set(reference))
        raise ValueError(
            "U1-Lite distilled full-state keys differ; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for key, expected in reference.items():
        value = state[key]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(
                f"U1-Lite distilled full-state tensor differs at {key!r}"
            )
    model.load_state_dict(state, strict=True)
    if _tensor_state_sha256(model.state_dict()) != source["tensor_sha256"]:
        raise ValueError("U1-Lite loaded target state differs from D0 raw state")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("U1-Lite T0 requires every model parameter trainable")
    receipt = source["d1_qualification_receipt"]
    provenance = {
        "schema_version": U1LITE_TARGET_INITIALIZATION_SCHEMA,
        "source_path": str(source["path"]),
        "source_checkpoint_schema": U1LITE_DISTILL_CHECKPOINT_SCHEMA,
        "source_checkpoint_role": "qualified_d1",
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_model": U1LITE_MODEL_NAME,
        "source_stage": "distill",
        "target_stage": "target",
        "selected_state_dict_key": "raw_model_state_dict",
        "source_tensor_state_sha256": source["tensor_sha256"],
        "source_config_sha256": _canonical_mapping_sha256(source_config),
        "d1_qualification_receipt": receipt,
        "d1_qualification_receipt_sha256": _canonical_mapping_sha256(receipt),
        "normalization_reuse": copy.deepcopy(dict(normalization_reuse)),
        "imported_training_state": [],
        "fresh_training_state": ["optimizer", "scaler", "ema", "update_clock",
                                 "wall_clock", "validation_records"],
        "all_parameters_trainable": True,
        "locked_test_opened": False,
    }
    _validate_u1lite_initialization_provenance(provenance)
    return provenance


def _parent_router_source_state(
    path: Path,
    *,
    label: str,
    expected_file_sha256: str,
    expected_model: str,
    expected_update: int,
    expected_state_key: str,
    expected_state_sha256: str,
) -> tuple[Mapping[str, Tensor], Mapping[str, Any], dict[str, Any]]:
    """Load one independently selected frozen expert and fail closed on drift."""

    source_path = _regular_public_artifact(path, f"ParentRouterQ {label} source")
    # Hash and deserialize the same opened inode.  A concurrently replaced
    # best.pt can therefore never yield a hash for A and tensors from B.
    with source_path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        file_sha256 = digest.hexdigest()
        if file_sha256 != expected_file_sha256:
            raise ValueError(
                f"ParentRouterQ {label} checkpoint SHA-256 differs from the "
                "registered formal source"
            )
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"ParentRouterQ {label} checkpoint root is malformed")
    if checkpoint.get("schema_version") != "uhi-cdc-g246-r2-checkpoint-v1" \
            or checkpoint.get("checkpoint_role") != "best":
        raise ValueError(f"ParentRouterQ {label} requires a formal best checkpoint")
    if checkpoint.get("locked_test_opened") is not False:
        raise ValueError(f"ParentRouterQ {label} source opened the locked test")
    update = checkpoint.get("optimizer_updates")
    if update != expected_update:
        raise ValueError(f"ParentRouterQ {label} source update differs")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"ParentRouterQ {label} source config is missing")
    if config.get("model") != expected_model \
            or config.get("locked_test_opened") is not False:
        raise ValueError(f"ParentRouterQ {label} source scientific role differs")
    weight_contract = checkpoint.get("weight_contract")
    if not isinstance(weight_contract, Mapping) \
            or weight_contract.get("best_inference_state_dict_key") \
            != expected_state_key:
        raise ValueError(
            f"ParentRouterQ {label} independently selected weight key differs"
        )
    state = checkpoint.get(expected_state_key)
    if not isinstance(state, Mapping) or not state \
            or any(not isinstance(key, str) or not isinstance(value, Tensor)
                   for key, value in state.items()):
        raise ValueError(f"ParentRouterQ {label} selected tensor state is malformed")
    state_sha256 = _tensor_state_sha256(state)
    if state_sha256 != expected_state_sha256:
        raise ValueError(
            f"ParentRouterQ {label} selected tensor state SHA-256 differs"
        )
    provenance = {
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256,
        "checkpoint_update": int(update),
        "checkpoint_role": "best",
        "selected_state_dict_key": expected_state_key,
        "selected_tensor_state_sha256": state_sha256,
        "source_config_sha256": _canonical_mapping_sha256(config),
        "source_model": expected_model,
        "locked_test_opened": False,
    }
    return state, config, provenance


def _u1lite_direct_r6a_state(
    path: Path,
) -> tuple[Mapping[str, Tensor], Mapping[str, Any], dict[str, Any]]:
    return _parent_router_source_state(
        path,
        label="U1-Lite direct r6a",
        expected_file_sha256=PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
        expected_model="calibrated_q",
        expected_update=2000,
        expected_state_key="raw_model_state_dict",
        expected_state_sha256=PARENT_ROUTER_SHALLOW_STATE_SHA256,
    )


def _reuse_u1lite_direct_normalization(
    checkpoint_path: Path, destination: Path,
) -> tuple[R2Normalization, dict[str, Any]]:
    """Copy the fixed r6a normalization and bind it to the exact source."""

    _state, source_config, source = _u1lite_direct_r6a_state(checkpoint_path)
    normalization, reused = _reuse_initialization_normalization(
        checkpoint_path, destination
    )
    if source_config.get("normalization_sha256") \
            != PARENT_ROUTER_NORMALIZATION_SHA256 \
            or reused.get("source_sha256") \
            != PARENT_ROUTER_NORMALIZATION_SHA256 \
            or source_config.get("fit_view_sha256") \
            != normalization.fit_view_sha256:
        raise ValueError("U1-Lite direct r6a normalization binding differs")
    return normalization, {
        "schema_version": U1LITE_DIRECT_NORMALIZATION_SCHEMA,
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_path": reused["source_path"],
        "source_normalization_sha256": reused["source_sha256"],
        "byte_count": reused["byte_count"],
        "copy_mode": "verified_exact_bytes",
        "target_exact_byte_copy": True,
        "fit_view_sha256": normalization.fit_view_sha256,
        "locked_test_opened": False,
    }


def initialize_u1lite_from_direct_r6a_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    target_config: Mapping[str, Any],
    normalization_reuse: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate registered r6a raw weights into an exact-u0 trainable U1."""

    if not isinstance(model, ContinuousScaleBridgeQ):
        raise TypeError("U1-Lite direct initialization requires ContinuousScaleBridgeQ")
    if target_config.get("model") != U1LITE_MODEL_NAME \
            or target_config.get("u1lite_stage") != "target" \
            or target_config.get("locked_test_opened") is not False:
        raise ValueError("U1-Lite direct target config is malformed")
    state, source_config, source = _u1lite_direct_r6a_state(checkpoint_path)
    if normalization_reuse.get("schema_version") \
            != U1LITE_DIRECT_NORMALIZATION_SCHEMA \
            or normalization_reuse.get("source_checkpoint_sha256") \
            != source["checkpoint_sha256"] \
            or normalization_reuse.get("source_normalization_sha256") \
            != target_config.get("normalization_sha256") \
            or normalization_reuse.get("fit_view_sha256") \
            != target_config.get("fit_view_sha256") \
            or normalization_reuse.get("target_exact_byte_copy") is not True:
        raise ValueError("U1-Lite direct normalization/source binding differs")
    no_geo = source_config.get("calibrated_no_geo_core")
    if not isinstance(no_geo, Mapping) \
            or no_geo.get("enabled") is not True \
            or no_geo.get("physical_context_dim") != 15 \
            or no_geo.get("removed_context_indices") != [5, 6, 7, 8]:
        raise ValueError("U1-Lite direct source is not registered Context15 r6a")
    common_fields = (
        "width", "data_scope", "region_sampling", "primary_loss_weight",
        "loss_contract", "input_provenance", "scientific_status",
        "provisional_optical_authorized", "fine_channels", "context_dim",
        "receipt_sha256", "fit_view_sha256", "split_identity",
        "train_city_count", "evaluation_city_count", "normalization_sha256",
        "locked_test_opened", "input_mode", "multisource_provenance",
        "temporal_mode", "full_scene_start", "scene_schedule",
    )
    for field in common_fields:
        if source_config.get(field) != target_config.get(field):
            raise ValueError(
                f"U1-Lite direct source/target differs at {field!r}"
            )
    source_count = sum(value.numel() for value in state.values())
    target_count = sum(parameter.numel() for parameter in model.parameters())
    if source_config.get("parameter_count") != source_count \
            or target_config.get("parameter_count") != target_count:
        raise ValueError("U1-Lite direct parameter-count contract differs")
    model.load_r6a_state_dict(state)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[key].detach().cpu(), value.detach().cpu())
           for key, value in state.items()):
        raise ValueError("U1-Lite direct inherited r6a tensor changed")
    zero_modules = (model.bridge40, model.middle_head[-1], model.high_head[-1])
    if any(not torch.equal(parameter, torch.zeros_like(parameter))
           for module in zero_modules for parameter in module.parameters()):
        raise ValueError("U1-Lite direct u0 bridge/QM/QH heads are not zero")
    model.requires_grad_(True)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("U1-Lite direct requires every parameter trainable")
    provenance = {
        "schema_version": U1LITE_DIRECT_INITIALIZATION_SCHEMA,
        "initialization_mode": "direct_r6a_exact_u0",
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_schema": "uhi-cdc-g246-r2-checkpoint-v1",
        "source_checkpoint_role": "best",
        "source_checkpoint_update": 2000,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "selected_state_dict_key": "raw_model_state_dict",
        "source_tensor_state_sha256": source["selected_tensor_state_sha256"],
        "source_config_sha256": source["source_config_sha256"],
        "source_model": "calibrated_q",
        "target_model": U1LITE_MODEL_NAME,
        "full_u0_state_sha256": _tensor_state_sha256(model.state_dict()),
        "u0_function_contract": (
            "exact_r6a_raw_u2000_zero_bridge_zero_qm_qh_heads"
        ),
        "normalization_reuse": copy.deepcopy(dict(normalization_reuse)),
        "d1_artifact_used": False,
        "imported_training_state": [],
        "fresh_training_state": [
            "optimizer", "scaler", "ema", "update_clock", "wall_clock",
            "validation_records",
        ],
        "all_parameters_trainable": True,
        "locked_test_opened": False,
    }
    _validate_u1lite_direct_initialization_provenance(provenance)
    return provenance


def initialize_parent_router_experts(
    model: nn.Module,
    shallow_checkpoint: Path,
    deep_checkpoint: Path,
    target_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Install the registered r6a/r9 selected states and bind their data view."""

    if not isinstance(model, ParentRouterQ):
        raise TypeError("ParentRouterQ expert initialization received another model")
    shallow_state, shallow_config, shallow_provenance = (
        _parent_router_source_state(
            shallow_checkpoint,
            label="shallow",
            expected_file_sha256=PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
            expected_model="calibrated_q",
            expected_update=2000,
            expected_state_key="raw_model_state_dict",
            expected_state_sha256=PARENT_ROUTER_SHALLOW_STATE_SHA256,
        )
    )
    deep_state, deep_config, deep_provenance = _parent_router_source_state(
        deep_checkpoint,
        label="deep",
        expected_file_sha256=PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
        expected_model="ipmr_q",
        expected_update=8000,
        expected_state_key="ema_model_state_dict",
        expected_state_sha256=PARENT_ROUTER_DEEP_STATE_SHA256,
    )
    shallow_no_geo = shallow_config.get("calibrated_no_geo_core")
    if not isinstance(shallow_no_geo, Mapping) \
            or shallow_no_geo.get("enabled") is not True \
            or shallow_no_geo.get("physical_context_dim") != 15 \
            or shallow_no_geo.get("removed_context_indices") != [5, 6, 7, 8]:
        raise ValueError("ParentRouterQ shallow expert is not physical Context15")
    deep_context = deep_config.get("ipmr_context_contract")
    if not isinstance(deep_context, Mapping) \
            or deep_context.get("learned") != "Context15" \
            or deep_context.get("coordinates_used") is not False \
            or deep_context.get("physically_removed_indices") != [5, 6, 7, 8]:
        raise ValueError("ParentRouterQ deep expert is not physical Context15")
    common_fields = (
        "data_scope",
        "region_sampling",
        "primary_loss_weight",
        "loss_contract",
        "input_provenance",
        "scientific_status",
        "provisional_optical_authorized",
        "fine_channels",
        "context_dim",
        "receipt_sha256",
        "fit_view_sha256",
        "split_identity",
        "train_city_count",
        "evaluation_city_count",
        "normalization_sha256",
        "locked_test_opened",
        "input_mode",
        "multisource_provenance",
        "temporal_mode",
        "scene_schedule",
    )
    for field in common_fields:
        shallow_value = shallow_config.get(field)
        deep_value = deep_config.get(field)
        target_value = target_config.get(field)
        if shallow_value != deep_value or shallow_value != target_value:
            raise ValueError(
                f"ParentRouterQ source/target scientific contract differs at {field!r}"
            )
    if shallow_config.get("seed") != deep_config.get("seed"):
        raise ValueError("ParentRouterQ source expert training seeds differ")
    model.load_expert_state_dicts(shallow_state, deep_state)
    if model.trainable_parameter_count > 20_000:
        raise ValueError("ParentRouterQ trainable router exceeds its low-capacity cap")
    expert_parameter_ids = {
        id(parameter)
        for expert in (model.shallow_expert, model.deep_expert)
        for parameter in expert.parameters()
    }
    router_parameters = model.router_parameters()
    if any(id(parameter) in expert_parameter_ids for parameter in router_parameters) \
            or any(parameter.requires_grad for expert in (
                model.shallow_expert, model.deep_expert
            ) for parameter in expert.parameters()):
        raise ValueError("ParentRouterQ expert/router freeze partition drifted")
    return {
        "schema_version": ParentRouterQ.schema_version,
        "shallow": shallow_provenance,
        "deep": deep_provenance,
        "gate_lattice": "one shared scalar per 4x4 physical parent",
        "gate_range": list(ParentRouterQ.gate_range),
        "gate_initialization": "exact_fixed_0p5_average",
        "gate_parameterization": "0.5+0.25*softsign(logit)",
        "router_feature_contract": {
            "channels": ParentRouterQ.router_input_channels,
            "layout": [
                "support_weighted_parent_mean_Fine52",
                "support_weighted_within_parent_RMS_Fine52",
                "explicit_Context15",
                "seven_expert_Q_energy_disagreement_statistics",
                "coarse_z",
                "coarse_valid",
                "support_fraction",
                "base_minus_coarse",
            ],
            "lst_and_coarse_scaling": "(kelvin-300)/20",
            "expert_q_scaling_before_statistics": "kelvin/2",
            "router_width": model.router_width,
            "spatial_operator": "shared_parentwise_1x1_only_no_location_template",
        },
        "explicit_geolocation_city_region_used": False,
        "physically_removed_context_indices": [5, 6, 7, 8],
        "solar_geometry_retained": True,
        "solar_geometry_context_indices": [9, 10, 11],
        "frozen_experts": True,
        "expert_forward": "permanent_eval_no_grad",
        "optimizer_scope": "router_parameters_only",
        "ema_scope": "router_floating_state_only_experts_exact_copy",
        "source_expert_training_seed": shallow_config.get("seed"),
        "historical_source_limitation": (
            "r6a_and_r9_v1_checkpoints_precede_training_implementation_hashes;_"
            "weights_configs_and_current_inference_bytes_are_bound_but_original_"
            "training_source_bytes_cannot_be_retroactively_proven"
        ),
        "trainable_parameter_count": model.trainable_parameter_count,
        "total_parameter_count": model.parameter_count,
        "locked_test_opened": False,
    }


def initialize_dcf_experts(
    model: nn.Module,
    shallow_checkpoint: Path,
    deep_checkpoint: Path,
    target_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Install the exact r6a/r9 states behind DCF-Q's feature interface."""

    if not isinstance(model, DCFQ):
        raise TypeError("DCF-Q expert initialization received another model")
    shallow_state, shallow_config, shallow_provenance = _parent_router_source_state(
        shallow_checkpoint,
        label="shallow",
        expected_file_sha256=PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
        expected_model="calibrated_q",
        expected_update=2000,
        expected_state_key="raw_model_state_dict",
        expected_state_sha256=PARENT_ROUTER_SHALLOW_STATE_SHA256,
    )
    deep_state, deep_config, deep_provenance = _parent_router_source_state(
        deep_checkpoint,
        label="deep",
        expected_file_sha256=PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
        expected_model="ipmr_q",
        expected_update=8000,
        expected_state_key="ema_model_state_dict",
        expected_state_sha256=PARENT_ROUTER_DEEP_STATE_SHA256,
    )
    shallow_no_geo = shallow_config.get("calibrated_no_geo_core")
    if not isinstance(shallow_no_geo, Mapping) \
            or shallow_no_geo.get("enabled") is not True \
            or shallow_no_geo.get("physical_context_dim") != 15 \
            or shallow_no_geo.get("removed_context_indices") != [5, 6, 7, 8]:
        raise ValueError("DCF-Q shallow expert is not physical Context15")
    deep_context = deep_config.get("ipmr_context_contract")
    if not isinstance(deep_context, Mapping) \
            or deep_context.get("learned") != "Context15" \
            or deep_context.get("coordinates_used") is not False \
            or deep_context.get("physically_removed_indices") != [5, 6, 7, 8]:
        raise ValueError("DCF-Q deep expert is not physical Context15")
    common_fields = (
        "data_scope", "region_sampling", "primary_loss_weight", "loss_contract",
        "input_provenance", "scientific_status", "provisional_optical_authorized",
        "fine_channels", "context_dim", "receipt_sha256", "fit_view_sha256",
        "split_identity", "train_city_count", "evaluation_city_count",
        "normalization_sha256", "locked_test_opened", "input_mode",
        "multisource_provenance", "temporal_mode", "scene_schedule",
    )
    for field in common_fields:
        shallow_value = shallow_config.get(field)
        deep_value = deep_config.get(field)
        target_value = target_config.get(field)
        if shallow_value != deep_value or shallow_value != target_value:
            raise ValueError(
                f"DCF-Q source/target scientific contract differs at {field!r}"
            )
    if shallow_config.get("seed") != deep_config.get("seed"):
        raise ValueError("DCF-Q source expert training seeds differ")
    model.load_expert_state_dicts(shallow_state, deep_state)
    if model.trainable_parameter_count > 650_000 \
            or model.parameter_count > 7_791_582:
        raise ValueError("DCF-Q parameter cap exceeded")
    expert_parameter_ids = {
        id(parameter)
        for expert in (model.shallow_expert, model.deep_expert)
        for parameter in expert.parameters()
    }
    decoder_parameters = model.decoder_parameters()
    if any(id(parameter) in expert_parameter_ids for parameter in decoder_parameters) \
            or any(parameter.requires_grad for expert in (
                model.shallow_expert, model.deep_expert
            ) for parameter in expert.parameters()):
        raise ValueError("DCF-Q expert/decoder freeze partition drifted")
    return {
        "schema_version": DCFQ.schema_version,
        "shallow": shallow_provenance,
        "deep": deep_provenance,
        "anchor": "fixed_0p5_q_space_average_never_trainable",
        "source_q_visible_to_decoder": False,
        "feature_contract": {
            "shallow": {"S160": 64, "S40": 144},
            "deep": {"F160": 64, "D80": 96, "D40": 144},
            "fusion_lattices": [40, 80, 160],
            "fusion": "sum(projected)/sqrt(n)+spatial_concat_residual",
            "head_inputs": ["G80", "G160"],
        },
        "q_corrections": ["QM=P80-P40", "QH=I-P80"],
        "explicit_geolocation_city_region_used": False,
        "physically_removed_context_indices": [5, 6, 7, 8],
        "solar_geometry_retained": True,
        "solar_geometry_context_indices": [9, 10, 11],
        "frozen_experts": True,
        "expert_forward": "permanent_eval_no_grad",
        "optimizer_scope": "decoder_parameters_only",
        "ema_scope": "decoder_floating_state_only_experts_exact_copy",
        "source_expert_training_seed": shallow_config.get("seed"),
        "trainable_parameter_count": model.trainable_parameter_count,
        "total_parameter_count": model.parameter_count,
        "locked_test_opened": False,
    }


def initialize_aom_experts(
    model: nn.Module,
    shallow_checkpoint: Path,
    deep_checkpoint: Path,
    target_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly bind and install the registered r6a/r9 AOM-Q initial states."""

    if not isinstance(model, AOMQ):
        raise TypeError("AOM-Q expert initialization received another model")
    shallow_state, shallow_config, shallow_provenance = _parent_router_source_state(
        shallow_checkpoint, label="shallow",
        expected_file_sha256=PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
        expected_model="calibrated_q", expected_update=2000,
        expected_state_key="raw_model_state_dict",
        expected_state_sha256=PARENT_ROUTER_SHALLOW_STATE_SHA256,
    )
    deep_state, deep_config, deep_provenance = _parent_router_source_state(
        deep_checkpoint, label="deep",
        expected_file_sha256=PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
        expected_model="ipmr_q", expected_update=8000,
        expected_state_key="ema_model_state_dict",
        expected_state_sha256=PARENT_ROUTER_DEEP_STATE_SHA256,
    )
    shallow_no_geo = shallow_config.get("calibrated_no_geo_core")
    deep_context = deep_config.get("ipmr_context_contract")
    if not isinstance(shallow_no_geo, Mapping) \
            or shallow_no_geo.get("physical_context_dim") != 15 \
            or shallow_no_geo.get("removed_context_indices") != [5, 6, 7, 8]:
        raise ValueError("AOM-Q shallow source is not registered Context15")
    if not isinstance(deep_context, Mapping) \
            or deep_context.get("learned") != "Context15" \
            or deep_context.get("coordinates_used") is not False:
        raise ValueError("AOM-Q deep source is not registered Context15")
    fields = (
        "data_scope", "region_sampling", "primary_loss_weight", "loss_contract",
        "input_provenance", "fine_channels", "context_dim", "receipt_sha256",
        "fit_view_sha256", "split_identity", "train_city_count",
        "evaluation_city_count", "normalization_sha256", "locked_test_opened",
        "input_mode", "multisource_provenance", "temporal_mode", "scene_schedule",
    )
    for field in fields:
        if shallow_config.get(field) != deep_config.get(field) \
                or shallow_config.get(field) != target_config.get(field):
            raise ValueError(f"AOM-Q source/target contract differs at {field!r}")
    if shallow_config.get("locked_test_opened") is not False:
        raise ValueError("AOM-Q source checkpoint opened locked test")
    model.load_expert_state_dicts(shallow_state, deep_state)
    if model.trainable_parameter_count > 10_000_000 \
            or not all(p.requires_grad for p in model.parameters()):
        raise ValueError("AOM-Q trainable parameter/all-trainable contract differs")
    return {
        "schema_version": AOMQ.schema_version,
        "shallow": shallow_provenance,
        "deep": deep_provenance,
        "source_state_keys": {
            "shallow": "raw_model_state_dict", "deep": "ema_model_state_dict",
        },
        "source_state_sha256": {
            "shallow": PARENT_ROUTER_SHALLOW_STATE_SHA256,
            "deep": PARENT_ROUTER_DEEP_STATE_SHA256,
        },
        "source_file_sha256": {
            "shallow": PARENT_ROUTER_SHALLOW_CHECKPOINT_SHA256,
            "deep": PARENT_ROUTER_DEEP_CHECKPOINT_SHA256,
        },
        "frozen_experts": False,
        "optimizer_scope": "all_parameters_three_differential_lr_groups",
        "locked_test_opened": False,
    }


def _no_geo_descriptor_surgery(
    source_weight: Tensor, target_weight: Tensor
) -> Tensor:
    """Delete only Context19 coordinate columns from the r2k descriptor.

    The registered S anchor concatenates ``mean52,std52,context19,coarse6``.
    Consequently the four coordinate columns occupy absolute columns
    ``109:113`` and their removal maps ``144x129`` to ``144x125`` without
    changing any other learned coefficient.
    """

    expected_source_shape = (144, 2 * MULTISOURCE_FINE_CHANNELS + 19 + 6)
    expected_target_shape = (144, 2 * MULTISOURCE_FINE_CHANNELS + 15 + 6)
    if tuple(source_weight.shape) != expected_source_shape:
        raise ValueError(
            "no-geolocation descriptor surgery requires source shape 144x129"
        )
    if tuple(target_weight.shape) != expected_target_shape:
        raise ValueError(
            "no-geolocation descriptor surgery requires target shape 144x125"
        )
    if source_weight.dtype != target_weight.dtype:
        raise ValueError("no-geolocation descriptor surgery dtype differs")
    context_offset = 2 * MULTISOURCE_FINE_CHANNELS
    dropped = {
        context_offset + index for index in NO_GEO_CONTEXT_INDICES
    }
    keep = [
        index for index in range(expected_source_shape[1])
        if index not in dropped
    ]
    migrated = source_weight.index_select(
        1, torch.tensor(keep, dtype=torch.long, device=source_weight.device)
    )
    if tuple(migrated.shape) != expected_target_shape:
        raise AssertionError("no-geolocation descriptor surgery shape changed")
    return migrated


def _no_geo_conditioning_reset_keys(keys: Sequence[str]) -> list[str]:
    """Resolve the complete, exact conditioning reset key set."""

    key_set = set(keys)
    reset: set[str] = set()
    for prefix in NO_GEO_RESET_PREFIXES:
        matched = {key for key in key_set if key.startswith(prefix)}
        if not matched:
            raise ValueError(
                f"no-geolocation conditioning reset prefix has no keys: {prefix}"
            )
        reset.update(matched)
    return sorted(reset)


def _initialize_pcqm_from_loaded_checkpoint(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    source_config: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    path: Path,
    checkpoint_sha256: str,
    weight_source: str,
) -> dict[str, Any]:
    """Load an IPMR-Q anchor exactly, leaving only the zero PCQM extension new."""

    if not isinstance(model, IPMRQPCQM):
        raise TypeError("PCQM initialization target is not IPMRQPCQM")
    if source_config.get("model") != "ipmr_q" \
            or target_config.get("model") != PCQM_MODEL_NAME:
        raise ValueError("PCQM initialization requires ipmr_q -> ipmr_q_pcqm")
    if checkpoint_sha256 != PCQM_ANCHOR_CHECKPOINT_SHA256:
        raise ValueError(
            "PCQM checkpoint SHA-256 differs from the registered r9 anchor"
        )
    if checkpoint.get("checkpoint_role") != PCQM_ANCHOR_CHECKPOINT_ROLE \
            or checkpoint.get("optimizer_updates") \
            != PCQM_ANCHOR_OPTIMIZER_UPDATES:
        raise ValueError("PCQM requires the registered r9 best checkpoint at u8000")
    if weight_source != "selected":
        raise ValueError("PCQM requires the registered selected r9 EMA weights")
    compatibility_fields = (
        "width", "data_scope", "region_sampling", "primary_loss_weight",
        "loss_contract", "q_shape_loss", "input_provenance",
        "scientific_status", "provisional_optical_authorized",
        "fine_channels", "context_dim", "receipt_sha256",
        "fit_view_sha256", "split_identity", "train_city_count",
        "evaluation_city_count", "normalization_sha256", "locked_test_opened",
        "input_mode", "multisource_provenance", "temporal_mode",
    )
    for label, config in (("source", source_config), ("target", target_config)):
        if config.get("data_scope") != "public_validation" \
                or config.get("locked_test_opened") is not False:
            raise ValueError(
                f"{label} PCQM initialization is not locked-test-closed "
                "public_validation"
            )
        if config.get("input_mode") != "multisource" \
                or config.get("fine_channels") != MULTISOURCE_FINE_CHANNELS \
                or config.get("context_dim") != MULTISOURCE_CONTEXT_DIM:
            raise ValueError("PCQM initialization requires Fine52/Context19")
        if config.get("provisional_optical_authorized") is not False:
            raise ValueError("PCQM initialization forbids provisional optical")
        provenance = config.get("multisource_provenance")
        if not isinstance(provenance, Mapping) \
                or provenance.get("locked_test_opened") is not False \
                or provenance.get(
                    "target_arrays_opened_by_multisource_adapter"
                ) is not False:
            raise ValueError(f"{label} PCQM multisource provenance is unsafe")
    for field in compatibility_fields:
        if source_config.get(field) != target_config.get(field):
            raise ValueError(
                f"PCQM initialization scientific contract differs at {field!r}"
            )
    if target_config.get("ipmr_auxiliary_loss", {}).get("middle_weight") != 0.0 \
            or target_config.get("ipmr_auxiliary_loss", {}).get(
                "high_weight"
            ) != 0.0:
        raise ValueError("PCQM target must use field-only supervision")

    state, state_key, selector = _selected_initialization_state(
        checkpoint, weight_source
    )
    state_sha256 = _tensor_state_sha256(state)
    if state_key != PCQM_ANCHOR_STATE_KEY \
            or selector != PCQM_ANCHOR_STATE_KEY \
            or state_sha256 != PCQM_ANCHOR_STATE_SHA256:
        raise ValueError("PCQM selected tensor state differs from registered r9 EMA")
    target_state = model.state_dict()
    source_keys = set(state)
    target_keys = set(target_state)
    unexpected = sorted(source_keys - target_keys)
    missing = sorted(target_keys - source_keys)
    allowed_missing = sorted(
        key for key in target_keys if key.startswith("pcqm_")
    )
    if unexpected or not allowed_missing or missing != allowed_missing:
        raise ValueError(
            "PCQM initialization state differs outside its extension: "
            f"unexpected={unexpected[:5]}, missing={missing[:5]}"
        )
    for key in sorted(source_keys):
        source_value = state[key]
        target_value = target_state[key]
        if source_value.shape != target_value.shape \
                or source_value.dtype != target_value.dtype:
            raise ValueError(
                f"PCQM inherited tensor contract differs at {key!r}"
            )
    source_parameter_count = sum(value.numel() for value in state.values())
    target_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if source_config.get("parameter_count") != source_parameter_count:
        raise ValueError("PCQM source parameter-count contract differs")
    if target_config.get("parameter_count") != target_parameter_count:
        raise ValueError("PCQM target parameter-count contract differs")

    incompatible = model.load_state_dict(state, strict=False)
    if sorted(incompatible.missing_keys) != allowed_missing \
            or incompatible.unexpected_keys:
        raise AssertionError("validated PCQM migration keys changed during load")
    loaded_state = model.state_dict()
    for key in sorted(source_keys):
        if not torch.equal(
            loaded_state[key].detach().cpu(), state[key].detach().cpu()
        ):
            raise ValueError(
                f"PCQM inherited tensor was not loaded exactly at {key!r}"
            )
    for key in ("pcqm_output.weight", "pcqm_output.bias"):
        value = loaded_state.get(key)
        if value is None or not torch.equal(value, torch.zeros_like(value)):
            raise ValueError("PCQM output projection is not exactly zero initialized")
    model.freeze_anchor()
    optimization = pcqm_optimization_contract(model)
    if target_config.get("pcqm_optimizer_contract") != optimization:
        raise ValueError("PCQM target optimizer/config binding differs")

    selected_kind = "ema" if state_key.startswith("ema") else "raw"
    multisource = source_config["multisource_provenance"]
    return {
        "schema_version": INITIALIZATION_SCHEMA,
        "initialization_mode": "ipmr_q_anchor_plus_zero_pcqm",
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_role": checkpoint.get("checkpoint_role"),
        "source_optimizer_updates": int(checkpoint.get("optimizer_updates", -1)),
        "requested_weight_source": weight_source,
        "loaded_weight_kind": selected_kind,
        "loaded_state_dict_key": state_key,
        "source_selector_state_dict_key": selector,
        "source_tensor_state_sha256": state_sha256,
        "source_config_sha256": _canonical_mapping_sha256(source_config),
        "source_model": "ipmr_q",
        "target_model": PCQM_MODEL_NAME,
        "source_receipt_sha256": source_config["receipt_sha256"],
        "source_fit_view_sha256": source_config["fit_view_sha256"],
        "source_multisource_binding_sha256": multisource["binding_sha256"],
        "missing_state_keys": allowed_missing,
        "missing_state_prefix": "pcqm_",
        "all_inherited_tensors_loaded_exactly": True,
        "pcqm_output_exact_zero": True,
        "anchor_frozen": True,
        "trainable_parameter_count": PCQM_PARAMETER_BUDGET,
        "reset_contract": {
            "optimizer": "fresh_pcqm_only",
            "scaler": "fresh",
            "ema": "fresh_from_initialized_model_num_updates_0",
            "optimizer_updates": 0,
            "scheduler_update": 0,
            "training_wall_seconds": 0.0,
            "validation_records": "empty",
        },
        "locked_test_opened": False,
    }


def initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | os.PathLike[str],
    target_config: Mapping[str, Any],
    *,
    weight_source: str = "selected",
) -> dict[str, Any]:
    """Migrate inference weights only into a scientifically compatible model.

    This is intentionally separate from ``--resume``: no optimizer, scaler,
    EMA, update count, scheduler phase, validation record, or wall time is
    imported.  The caller constructs all of those objects *after* this helper.
    """

    path = reject_forbidden_path(checkpoint_path).resolve()
    reject_forbidden_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"initialization checkpoint does not exist: {path}")
    # Hash and deserialize the same open inode so an atomic best.pt replacement
    # cannot make provenance describe different bytes than the loaded weights.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        checkpoint_sha256 = digest.hexdigest()
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) \
            or checkpoint.get("schema_version") != "uhi-cdc-g246-r2-checkpoint-v1":
        raise ValueError("unsupported R2 initialization checkpoint schema")
    if checkpoint.get("locked_test_opened") is not False:
        raise ValueError("initialization checkpoint is not locked-test closed")
    source_config = checkpoint.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("initialization checkpoint config is missing")
    if target_config.get("model") == PCQM_MODEL_NAME:
        return _initialize_pcqm_from_loaded_checkpoint(
            model,
            checkpoint,
            source_config,
            target_config,
            path=path,
            checkpoint_sha256=checkpoint_sha256,
            weight_source=weight_source,
        )
    (
        source_adapter, target_adapter, source_t3, target_t3,
        source_refiner, target_refiner, source_pyramid, target_pyramid,
        source_dct15, target_dct15, source_no_geo, target_no_geo,
    ) = _validate_initialization_scientific_contract(source_config, target_config)
    state, state_key, selector = _selected_initialization_state(
        checkpoint, weight_source
    )
    target_state = model.state_dict()
    _validate_no_geo_probe_model_binding(model, target_config)
    _validate_no_geo_pack_model_binding(model, target_config)
    source_parameter_count = sum(value.numel() for value in state.values())
    target_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if source_config.get("parameter_count") != source_parameter_count:
        raise ValueError("source parameter-count contract differs from selected weights")
    if target_config.get("parameter_count") != target_parameter_count:
        raise ValueError("target parameter-count contract differs from constructed model")
    source_keys = set(state)
    target_keys = set(target_state)
    target_pack_arm = _calibrated_no_geo_pack_arm(target_config)
    if target_pack_arm is not None:
        for key, expected_shape in NO_GEO_LEGACY_PACK_SHAPES.items():
            value = state.get(key)
            if not isinstance(value, Tensor) \
                    or tuple(value.shape) != expected_shape \
                    or value.dtype != torch.float32:
                raise ValueError(
                    f"bare-r2k source legacy pack tensor contract differs at {key!r}"
                )
    pack_source_dropped_keys = (
        list(NO_GEO_LEGACY_PACK_KEYS)
        if target_pack_arm == "hierarchical" else []
    )
    pack_source_reset_keys = (
        list(NO_GEO_LEGACY_PACK_KEYS)
        if target_pack_arm == "legacy_reset_control" else []
    )
    pack_target_new_keys = (
        list(NO_GEO_HIERARCHICAL_PACK_KEYS)
        if target_pack_arm == "hierarchical" else []
    )
    pack_target_random_keys = sorted(
        pack_source_reset_keys + pack_target_new_keys
    )
    unexpected = sorted(source_keys - target_keys)
    missing = sorted(target_keys - source_keys)
    allowed_missing = sorted(set(pack_target_new_keys) | {
        key for key in target_keys
        if (
            target_adapter and not source_adapter
            and key.startswith("allocation_adapter.")
        ) or (
            target_t3 and not source_t3 and key.startswith("t3_fusion.")
        ) or (
            target_refiner and not source_refiner and key.startswith("q_refiner.")
        ) or (
            target_pyramid and not source_pyramid
            and key.startswith("content_q_pyramid.")
        ) or (
            target_dct15 and not source_dct15
            and key.startswith("parent_dct15.")
        )
    })
    if unexpected != sorted(pack_source_dropped_keys):
        raise ValueError(
            "initialization state has incompatible keys: "
            f"unexpected={unexpected[:5]}, allowed_pack_dropped="
            f"{pack_source_dropped_keys[:5]}"
        )
    if missing != allowed_missing:
        raise ValueError(
            "initialization state has incompatible keys: "
            f"missing={missing[:5]}, allowed_extension_missing={allowed_missing[:5]}"
        )
    load_state: dict[str, Tensor] = dict(state)
    no_geo_reset_keys: list[str] = []
    initial_reset_values: dict[str, Tensor] = {}
    descriptor_source_shape: list[int] | None = None
    descriptor_target_shape: list[int] | None = None
    initial_pack_values = {
        key: target_state[key].detach().cpu().clone()
        for key in pack_target_random_keys
    }
    for key in pack_source_dropped_keys + pack_source_reset_keys:
        load_state.pop(key, None)
    if target_no_geo is not None:
        if source_no_geo is not None:
            raise AssertionError("validated no-geolocation source changed")
        if NO_GEO_DESCRIPTOR_WEIGHT_KEY not in state \
                or NO_GEO_DESCRIPTOR_WEIGHT_KEY not in target_state:
            raise ValueError("no-geolocation descriptor weight is missing")
        descriptor_source_shape = list(
            state[NO_GEO_DESCRIPTOR_WEIGHT_KEY].shape
        )
        descriptor_target_shape = list(
            target_state[NO_GEO_DESCRIPTOR_WEIGHT_KEY].shape
        )
        if target_no_geo == "surgery":
            load_state[NO_GEO_DESCRIPTOR_WEIGHT_KEY] = (
                _no_geo_descriptor_surgery(
                    state[NO_GEO_DESCRIPTOR_WEIGHT_KEY],
                    target_state[NO_GEO_DESCRIPTOR_WEIGHT_KEY],
                )
            )
        elif target_no_geo == "reset_conditioning":
            no_geo_reset_keys = _no_geo_conditioning_reset_keys(
                sorted(target_keys)
            )
            initial_reset_values = {
                key: target_state[key].detach().cpu().clone()
                for key in no_geo_reset_keys
            }
            for key in no_geo_reset_keys:
                load_state.pop(key, None)
        else:  # guarded by the immutable config contract
            raise AssertionError("validated no-geolocation mode changed")
    for key in sorted(source_keys):
        if key in pack_source_dropped_keys or key in pack_source_reset_keys:
            continue
        source_value = state[key]
        target_value = target_state[key]
        if target_no_geo == "surgery" and key == NO_GEO_DESCRIPTOR_WEIGHT_KEY:
            # The helper above validates both exact registered shapes/dtypes.
            continue
        if key in no_geo_reset_keys:
            continue
        if source_value.shape != target_value.shape or source_value.dtype != target_value.dtype:
            raise ValueError(
                f"initialization tensor contract differs at {key!r}: "
                f"{tuple(source_value.shape)}/{source_value.dtype} vs "
                f"{tuple(target_value.shape)}/{target_value.dtype}"
            )
    expected_load_missing = sorted(
        set(allowed_missing) | set(no_geo_reset_keys) | set(pack_source_reset_keys)
    )
    incompatible = model.load_state_dict(load_state, strict=False)
    if sorted(incompatible.missing_keys) != expected_load_missing \
            or incompatible.unexpected_keys:
        raise AssertionError("validated initialization keys changed during load")
    if target_no_geo is not None:
        loaded_state = model.state_dict()
        if target_no_geo == "surgery":
            expected_descriptor = _no_geo_descriptor_surgery(
                state[NO_GEO_DESCRIPTOR_WEIGHT_KEY],
                target_state[NO_GEO_DESCRIPTOR_WEIGHT_KEY],
            ).detach().cpu()
            if not torch.equal(
                loaded_state[NO_GEO_DESCRIPTOR_WEIGHT_KEY].detach().cpu(),
                expected_descriptor,
            ):
                raise ValueError(
                    "no-geolocation descriptor surgery was not loaded exactly"
                )
        for key in sorted(source_keys):
            if key in pack_source_dropped_keys:
                continue
            loaded = loaded_state[key].detach().cpu()
            if key in pack_source_reset_keys:
                if not torch.equal(loaded, initial_pack_values[key]):
                    raise ValueError(
                        f"legacy control pack random initialization changed at {key!r}"
                    )
                continue
            if key in no_geo_reset_keys:
                if not torch.equal(loaded, initial_reset_values[key]):
                    raise ValueError(
                        f"no-geolocation conditioning reset changed at {key!r}"
                    )
                continue
            expected = load_state[key].detach().cpu()
            if not torch.equal(loaded, expected):
                raise ValueError(
                    f"no-geolocation inherited tensor was not loaded exactly at {key!r}"
                )
        for key in pack_target_new_keys:
            if not torch.equal(
                loaded_state[key].detach().cpu(), initial_pack_values[key]
            ):
                raise ValueError(
                    f"hierarchical target pack random initialization changed at {key!r}"
                )
    if target_adapter and not source_adapter:
        loaded_state = model.state_dict()
        zero_slices = (
            ("allocation_adapter.parent_controls.weight", 0),
            ("allocation_adapter.parent_controls.bias", 0),
        )
        if any(
            key not in loaded_state
            or not torch.equal(
                loaded_state[key][index],
                torch.zeros_like(loaded_state[key][index]),
            )
            for key, index in zero_slices
        ):
            raise ValueError(
                "new allocation-v2 parent amplitude is not zero-initialized"
            )
        for key in (
            "allocation_adapter.delta_score.3.weight",
            "allocation_adapter.delta_score.3.bias",
        ):
            value = loaded_state.get(key)
            if value is None or not torch.equal(value, torch.zeros_like(value)):
                raise ValueError(
                    "new allocation-v2 delta score is not zero-initialized"
                )
        controls_weight = loaded_state.get(
            "allocation_adapter.parent_controls.weight"
        )
        controls_bias = loaded_state.get(
            "allocation_adapter.parent_controls.bias"
        )
        if controls_weight is None or controls_bias is None \
                or not torch.equal(
                    controls_weight[1], torch.zeros_like(controls_weight[1])
                ):
            raise ValueError(
                "new allocation-v2 temperature is not spatially constant"
            )
        adapter_module = getattr(model, "allocation_adapter", None)
        if adapter_module is None:
            raise ValueError("target allocation-v2 module is missing")
        temperature = (
            adapter_module.min_temperature
            + (adapter_module.max_temperature - adapter_module.min_temperature)
            * torch.sigmoid(controls_bias[1].float())
        )
        if not torch.isclose(
            temperature,
            temperature.new_tensor(adapter_module.initial_temperature),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "new allocation-v2 temperature does not start at the safe constant"
            )
    if target_t3 and not source_t3:
        loaded_state = model.state_dict()
        gate_keys = (
            "t3_fusion.fine_residual_gate",
            "t3_fusion.parent_residual_gate",
            "t3_fusion.scene_residual_gate",
        )
        if any(
            key not in loaded_state
            or not torch.equal(loaded_state[key], torch.zeros_like(loaded_state[key]))
            for key in gate_keys
        ):
            raise ValueError("new T3 fusion is not zero-gated at initialization")
    if target_refiner and not source_refiner:
        loaded_state = model.state_dict()
        for key in (
            "q_refiner.parent_gain.2.weight",
            "q_refiner.parent_gain.2.bias",
            "q_refiner.signed_residual.weight",
            "q_refiner.signed_residual.bias",
        ):
            value = loaded_state.get(key)
            if value is None or not torch.equal(value, torch.zeros_like(value)):
                raise ValueError("new Q-refiner is not identity-initialized")
    if target_pyramid and not source_pyramid:
        loaded_state = model.state_dict()
        for scale in (40, 80, 160):
            for suffix in ("weight", "bias"):
                key = f"content_q_pyramid.signed_head_{scale}.{suffix}"
                value = loaded_state.get(key)
                if value is None or not torch.equal(value, torch.zeros_like(value)):
                    raise ValueError(
                        "new Content-only Q-Pyramid heads are not zero-initialized"
                    )
    if target_dct15 and not source_dct15:
        loaded_state = model.state_dict()
        for key in (
            "parent_dct15.coefficient_head.weight",
            "parent_dct15.coefficient_head.bias",
        ):
            value = loaded_state.get(key)
            if value is None or not torch.equal(value, torch.zeros_like(value)):
                raise ValueError(
                    "new Parent-DCT15 coefficient head is not zero-initialized"
                )
        # ``load_state_dict`` is keyed, but verify each inherited tensor after
        # the partial extension load so provenance never claims an exact r2k
        # migration on the strength of key/shape agreement alone.
        for key in sorted(source_keys):
            loaded = loaded_state[key].detach().cpu()
            expected = state[key].detach().cpu()
            if not torch.equal(loaded, expected):
                raise ValueError(
                    f"Parent-DCT15 inherited tensor was not loaded exactly at {key!r}"
                )
    selected_kind = "ema" if state_key.startswith("ema") else "raw"
    return {
        "schema_version": INITIALIZATION_SCHEMA,
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_role": checkpoint.get("checkpoint_role"),
        "source_optimizer_updates": int(checkpoint.get("optimizer_updates", -1)),
        "requested_weight_source": weight_source,
        "loaded_weight_kind": selected_kind,
        "loaded_state_dict_key": state_key,
        "source_selector_state_dict_key": selector,
        "source_config_sha256": _canonical_mapping_sha256(source_config),
        "source_receipt_sha256": source_config["receipt_sha256"],
        "source_fit_view_sha256": source_config["fit_view_sha256"],
        "source_multisource_binding_sha256": source_config[
            "multisource_provenance"
        ]["binding_sha256"],
        "source_adapter_enabled": source_adapter,
        "target_adapter_enabled": target_adapter,
        "source_t3_fusion_enabled": source_t3,
        "target_t3_fusion_enabled": target_t3,
        "source_q_refiner_enabled": source_refiner,
        "target_q_refiner_enabled": target_refiner,
        "source_content_q_pyramid_enabled": source_pyramid,
        "target_content_q_pyramid_enabled": target_pyramid,
        "source_parent_dct15_enabled": source_dct15,
        "target_parent_dct15_enabled": target_dct15,
        "source_no_geo_core_enabled": source_no_geo is not None,
        "target_no_geo_core_enabled": target_no_geo is not None,
        "temporal_migration": {
            "source_temporal_mode": source_config.get("temporal_mode"),
            "target_temporal_mode": target_config.get("temporal_mode"),
            "kind": (
                "single_query_to_permutation_invariant_t3_zero_gate"
                if target_t3 and not source_t3 else "none"
            ),
        },
        "q_refiner_migration": {
            "kind": (
                "learned_raw_q_to_identity_initialized_D4_q_refiner"
                if target_refiner and not source_refiner else "none"
            ),
            "old_keys_loaded_exactly": True,
        },
        "content_q_pyramid_migration": {
            "kind": (
                "bare_r2k_to_no_geolocation_rebase_plus_zero_q_pyramid"
                if target_pyramid and not source_pyramid else "none"
            ),
            "old_keys_loaded_exactly": True,
            "numerically_identical_to_source": not bool(target_pyramid),
            "coordinate_masked_before_core": bool(target_pyramid),
        },
        "parent_dct15_migration": {
            "kind": (
                "bare_r2k_to_no_geolocation_rebase_plus_zero_parent_dct15"
                if target_dct15 and not source_dct15 else "none"
            ),
            "old_keys_loaded_exactly": True,
            "numerically_identical_to_source": not bool(target_dct15),
            "coordinate_masked_before_core": bool(target_dct15),
        },
        "no_geo_core_migration": {
            "kind": (
                "bare_r2k_context19_to_physical_context15_core"
                if target_no_geo is not None else "none"
            ),
            "rebase_mode": target_no_geo,
            "stored_input_contract": "Fine52/Context19",
            "target_physical_core_input_contract": (
                "Fine52/Context15" if target_no_geo is not None else None
            ),
            "removed_context_indices": (
                list(NO_GEO_CONTEXT_INDICES)
                if target_no_geo is not None else []
            ),
            "descriptor_weight_key": (
                NO_GEO_DESCRIPTOR_WEIGHT_KEY
                if target_no_geo is not None else None
            ),
            "descriptor_source_shape": descriptor_source_shape,
            "descriptor_target_shape": descriptor_target_shape,
            "surgically_mapped_keys": (
                [NO_GEO_DESCRIPTOR_WEIGHT_KEY]
                if target_no_geo == "surgery" else []
            ),
            "reset_prefixes": (
                list(NO_GEO_RESET_PREFIXES)
                if target_no_geo == "reset_conditioning" else []
            ),
            "reset_keys": no_geo_reset_keys,
            "all_non_reset_tensors_loaded_exactly": (
                target_no_geo == "reset_conditioning"
            ),
            "all_unmodified_tensors_loaded_exactly": bool(target_no_geo),
            "numerically_identical_to_source": target_no_geo is None,
        },
        "no_geo_pack_migration": {
            "arm": target_pack_arm,
            "kind": (
                "bare_r2k_to_context15_legacy_pack_random_reset_control"
                if target_pack_arm == "legacy_reset_control"
                else (
                    "bare_r2k_drop_legacy_pack_add_random_hierarchical_pack"
                    if target_pack_arm == "hierarchical" else "none"
                )
            ),
            "source_dropped_keys": pack_source_dropped_keys,
            "source_reset_keys": pack_source_reset_keys,
            "target_new_keys": pack_target_new_keys,
            "source_dropped_names_sha256": _parameter_names_sha256(
                pack_source_dropped_keys
            ),
            "source_reset_names_sha256": _parameter_names_sha256(
                pack_source_reset_keys
            ),
            "target_new_names_sha256": _parameter_names_sha256(
                pack_target_new_keys
            ),
            "target_tensor_count": (
                target_config["no_geo_pack_screen"]["target_tensor_count"]
                if target_pack_arm is not None else None
            ),
            "target_parameter_count": (
                target_config["no_geo_pack_screen"]["target_parameter_count"]
                if target_pack_arm is not None else None
            ),
            "target_parameter_names_sha256": (
                target_config["no_geo_pack_screen"][
                    "target_parameter_names_sha256"
                ] if target_pack_arm is not None else None
            ),
            "all_shared_nonpack_tensors_loaded_exactly": bool(target_pack_arm),
            "matched_control_identity": (
                target_config["no_geo_pack_screen"]["matched_control_identity"]
                if target_pack_arm is not None else None
            ),
            "fresh_optimizer": bool(target_pack_arm),
            "fresh_scaler": bool(target_pack_arm),
            "fresh_ema": bool(target_pack_arm),
            "fresh_scheduler_update_zero": bool(target_pack_arm),
        },
        "reset_contract": {
            "optimizer": "fresh",
            "scaler": "fresh",
            "ema": "fresh_from_initialized_model_num_updates_0",
            "optimizer_updates": 0,
            "scheduler_update": 0,
            "training_wall_seconds": 0.0,
            "validation_records": "empty",
        },
        "locked_test_opened": False,
    }


def _state(
    config: Mapping[str, Any], *, status: str, update: int, run_updates: int,
    training_wall: float, run_wall: float, records: Sequence[Mapping[str, Any]],
    pause_pending: bool = False,
) -> dict[str, Any]:
    best = select_record(records) if records else None
    rmse = (
        float(best["validation"]["equal_region"]["rmse_k"])
        if best is not None else None
    )
    return {
        "schema_version": "uhi-cdc-g246-r2-state-v1",
        "status": status,
        "config": dict(config),
        "optimizer_updates": update,
        "run_optimizer_updates": run_updates,
        "actual_training_wall_seconds": training_wall,
        "run_training_wall_seconds": run_wall,
        "last_validation_update": int(records[-1]["update"]) if records else None,
        "best_update": int(best["update"]) if best else None,
        "best_equal_region_rmse_k": rmse,
        "best_validation_weight_source": (
            validation_weight_source(best) if best is not None else None
        ),
        "best_inference_state_dict_key": (
            selected_state_dict_key(best) if best is not None else None
        ),
        "threshold_grade": threshold_grade(rmse),
        "pause_pending": bool(pause_pending),
        "input_provenance": config["input_provenance"],
        "scientific_status": config["scientific_status"],
        "updated_utc": _utc_now(),
        "locked_test_opened": False,
    }


def _write_state_heartbeat(
    path: Path,
    config: Mapping[str, Any],
    *,
    update: int,
    run_updates: int,
    training_wall: float,
    run_wall: float,
    records: Sequence[Mapping[str, Any]],
    pause_pending: bool,
) -> None:
    """Atomically publish progress without validation or checkpoint side effects."""

    payload = _state(
        config,
        status="running",
        update=update,
        run_updates=run_updates,
        training_wall=training_wall,
        run_wall=run_wall,
        records=records,
        pause_pending=pause_pending,
    )
    payload.update({
        "state_update_kind": "heartbeat",
        "heartbeat_optimizer_update": int(update),
        "heartbeat_interval_updates": STATE_HEARTBEAT_INTERVAL_UPDATES,
    })
    atomic_json(path, payload)


def threshold_grade(rmse: float | None) -> str:
    if rmse is None:
        return "unmeasured"
    if rmse < 0.50:
        return "main_goal_achieved"
    if rmse < 0.55:
        return "effective"
    if rmse < 0.60:
        return "meaningful_transition"
    return "design_not_yet_effective"


def _checkpoint(
    config: Mapping[str, Any], model: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: Any, ema: EMA, *, update: int, training_wall: float,
    records: Sequence[Mapping[str, Any]], checkpoint_role: str = "last",
    best_inference_state_dict_key: str | None = None,
) -> dict[str, Any]:
    if checkpoint_role not in {"last", "best"}:
        raise ValueError("checkpoint_role must be 'last' or 'best'")
    if checkpoint_role == "best":
        if best_inference_state_dict_key not in {
            "raw_model_state_dict", "ema_model_state_dict"
        }:
            raise ValueError(
                "best checkpoint requires an explicit selected inference state dict"
            )
    elif best_inference_state_dict_key is not None:
        raise ValueError("last checkpoint cannot declare selected inference weights")
    raw_state = model.state_dict()
    ema_payload = dict(ema.state_dict())
    ema_snapshot = {
        str(key): value.detach().cpu().clone()
        for key, value in ema.shadow.items()
    }
    ema_payload["shadow"] = ema_snapshot
    payload = {
        "schema_version": "uhi-cdc-g246-r2-checkpoint-v1",
        "checkpoint_role": checkpoint_role,
        "config": dict(config),
        # Keep the established raw key for exact optimizer resume and old
        # consumers.  Selected inference weights have their own unambiguous key.
        "model_state_dict": raw_state,
        "raw_model_state_dict": raw_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema_payload,
        "weight_contract": {
            "resume_state_dict_key": "model_state_dict",
            "raw_state_dict_key": "raw_model_state_dict",
            "validation_selector_state_dict_key": best_inference_state_dict_key,
            "best_inference_state_dict_key": best_inference_state_dict_key,
            "validation_metric_record_key": "validation",
            "raw_validation_metric_record_key": "raw_validation",
        },
        "optimizer_updates": update,
        "actual_training_wall_seconds": training_wall,
        "records": list(records),
        "locked_test_opened": False,
    }
    initialization = config.get("initialization_provenance")
    if initialization is not None:
        if not isinstance(initialization, Mapping) \
                or initialization.get("schema_version") != INITIALIZATION_SCHEMA:
            raise ValueError("checkpoint initialization provenance is malformed")
        payload["initialization_provenance"] = copy.deepcopy(dict(initialization))
        payload["weight_contract"]["training_initialization"] = (
            "inference_weights_only_fresh_optimizer_scaler_ema_scheduler"
        )
    parent_router_initialization = config.get(
        "parent_router_initialization_provenance"
    )
    if parent_router_initialization is not None:
        if not isinstance(parent_router_initialization, Mapping) \
                or parent_router_initialization.get("schema_version") \
                != ParentRouterQ.schema_version \
                or parent_router_initialization.get("locked_test_opened") is not False:
            raise ValueError(
                "checkpoint ParentRouterQ initialization provenance is malformed"
            )
        payload["parent_router_initialization_provenance"] = copy.deepcopy(
            dict(parent_router_initialization)
        )
        payload["weight_contract"]["training_initialization"] = (
            "two_selected_frozen_experts_plus_fresh_router_optimizer_scaler_ema"
        )
    dcf_initialization = config.get("dcf_initialization_provenance")
    if dcf_initialization is not None:
        if not isinstance(dcf_initialization, Mapping) \
                or dcf_initialization.get("schema_version") != DCFQ.schema_version \
                or dcf_initialization.get("locked_test_opened") is not False:
            raise ValueError("checkpoint DCF-Q initialization provenance is malformed")
        payload["dcf_initialization_provenance"] = copy.deepcopy(
            dict(dcf_initialization)
        )
        payload["weight_contract"]["training_initialization"] = (
            "two_selected_frozen_experts_plus_fresh_cross_decoder_optimizer_scaler_ema"
        )
    u1lite_initialization = config.get(
        "u1lite_distilled_initialization_provenance"
    )
    u1lite_direct_initialization = config.get(
        "u1lite_direct_initialization_provenance"
    )
    if u1lite_initialization is not None \
            and u1lite_direct_initialization is not None:
        raise ValueError("checkpoint U1-Lite has two initialization modes")
    if u1lite_initialization is not None:
        _validate_u1lite_initialization_provenance(u1lite_initialization)
        payload["u1lite_distilled_initialization_provenance"] = copy.deepcopy(
            dict(u1lite_initialization)
        )
        payload["weight_contract"]["training_initialization"] = (
            "qualified_d1_full_raw_weights_only_fresh_optimizer_scaler_ema"
        )
    if u1lite_direct_initialization is not None:
        _validate_u1lite_direct_initialization_provenance(
            u1lite_direct_initialization
        )
        payload["u1lite_direct_initialization_provenance"] = copy.deepcopy(
            dict(u1lite_direct_initialization)
        )
        payload["weight_contract"]["training_initialization"] = (
            "registered_r6a_raw_exact_u0_weights_only_fresh_optimizer_scaler_ema"
        )
    # This flat compatibility key is present only when EMA is the selected
    # inference state.  Legacy downstream loaders that prefer it therefore load
    # EMA for mature best.pt, but correctly fall through to model_state_dict for
    # a raw-selected short-screen best.pt.  The complete EMA is always retained
    # under ema_state_dict["shadow"] regardless of checkpoint role.
    if (
        checkpoint_role == "best"
        and best_inference_state_dict_key == "ema_model_state_dict"
    ):
        payload["ema_model_state_dict"] = ema_snapshot
    return payload


def _write_history(
    path: Path, config: Mapping[str, Any], *, status: str, update: int,
    run_updates: int, training_wall: float, run_wall: float,
    records: Sequence[Mapping[str, Any]],
) -> None:
    atomic_json(path, {
        "schema_version": "uhi-cdc-g246-r2-history-v1",
        "status": status,
        "config": dict(config),
        "optimizer_updates": update,
        "run_optimizer_updates": run_updates,
        "actual_training_wall_seconds": training_wall,
        "run_training_wall_seconds": run_wall,
        "records": list(records),
        "selected": selected_summary(select_record(records)) if records else None,
        "input_provenance": config["input_provenance"],
        "scientific_status": config["scientific_status"],
        "locked_test_opened": False,
    })


def _validate_args(args: argparse.Namespace) -> None:
    normalize_role(args.role)
    reject_forbidden_path(args.split_receipt)
    reject_forbidden_path(args.output)
    if args.texture_manifest is not None:
        reject_forbidden_path(args.texture_manifest)
    if getattr(args, "weather_manifest", None) is not None:
        reject_forbidden_path(args.weather_manifest)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint is not None:
        reject_forbidden_path(init_checkpoint)
    elif str(getattr(args, "init_weight_source", "selected")) != "selected":
        raise ValueError("--init-weight-source requires --init-checkpoint")
    if bool(getattr(args, "resume", False)) and init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    u1lite_distilled_checkpoint = getattr(
        args, "u1lite_distilled_checkpoint", None
    )
    u1lite_direct_checkpoint = getattr(
        args, "u1lite_direct_r6a_checkpoint", None
    )
    u1lite_physical = int(
        getattr(args, "u1lite_physical_batch_size", 0)
    )
    u1lite_dexchange = bool(getattr(args, "u1lite_dexchange", False))
    if u1lite_distilled_checkpoint is not None:
        reject_forbidden_path(u1lite_distilled_checkpoint)
    if u1lite_direct_checkpoint is not None:
        reject_forbidden_path(u1lite_direct_checkpoint)
    is_resume = bool(getattr(args, "resume", False))
    if u1lite_dexchange and args.model != U1LITE_MODEL_NAME:
        raise ValueError("--u1lite-dexchange requires --model u1lite_q")
    if args.model == U1LITE_MODEL_NAME:
        source_count = sum(source is not None for source in (
            u1lite_distilled_checkpoint, u1lite_direct_checkpoint,
        ))
        if is_resume and source_count:
            raise ValueError(
                "U1-Lite resume forbids distilled/direct source checkpoints"
            )
        if not is_resume and source_count != 1:
            raise ValueError(
                "fresh U1-Lite requires exactly one distilled/direct source"
            )
        if u1lite_dexchange and not is_resume and (
            u1lite_direct_checkpoint is None
            or u1lite_distilled_checkpoint is not None
        ):
            raise ValueError(
                "fresh --u1lite-dexchange requires only the direct r6a source"
            )
        if init_checkpoint is not None:
            raise ValueError(
                "U1-Lite uses only its distilled/direct source on a fresh run"
            )
    elif u1lite_distilled_checkpoint is not None \
            or u1lite_direct_checkpoint is not None or u1lite_physical != 0:
        raise ValueError("U1-Lite-specific options require --model u1lite_q")
    parent_router_shallow = getattr(
        args, "parent_router_shallow_checkpoint", None
    )
    parent_router_deep = getattr(args, "parent_router_deep_checkpoint", None)
    parent_router_physical = int(
        getattr(args, "parent_router_physical_batch_size", 0)
    )
    dcf_shallow = getattr(args, "dcf_shallow_checkpoint", None)
    dcf_deep = getattr(args, "dcf_deep_checkpoint", None)
    dcf_physical = int(getattr(args, "dcf_physical_batch_size", 0))
    dcf_middle_explicit = getattr(args, "dcf_middle_loss_weight", None)
    dcf_high_explicit = getattr(args, "dcf_high_loss_weight", None)
    aom_shallow = getattr(args, "aom_shallow_checkpoint", None)
    aom_deep = getattr(args, "aom_deep_checkpoint", None)
    aom_physical = int(getattr(args, "aom_physical_batch_size", 0))
    for value, label in (
        (parent_router_shallow, "shallow"),
        (parent_router_deep, "deep"),
    ):
        if value is not None:
            reject_forbidden_path(value)
        if args.model == PARENT_ROUTER_MODEL_NAME and not is_resume \
                and not isinstance(value, Path):
            raise ValueError(
                f"ParentRouterQ requires --parent-router-{label}-checkpoint"
            )
        if args.model == PARENT_ROUTER_MODEL_NAME and is_resume \
                and value is not None:
            raise ValueError(
                "ParentRouterQ resume restores experts only from last.pt and "
                "forbids source checkpoint arguments"
            )
    if args.model != PARENT_ROUTER_MODEL_NAME and (
        parent_router_shallow is not None
        or parent_router_deep is not None
        or parent_router_physical != 0
    ):
        raise ValueError(
            "ParentRouterQ-specific options require --model parent_router_q"
        )
    for value, label in ((dcf_shallow, "shallow"), (dcf_deep, "deep")):
        if value is not None:
            reject_forbidden_path(value)
        if args.model == DCF_MODEL_NAME and not is_resume \
                and not isinstance(value, Path):
            raise ValueError(f"DCF-Q requires --dcf-{label}-checkpoint")
        if args.model == DCF_MODEL_NAME and is_resume and value is not None:
            raise ValueError(
                "DCF-Q resume restores experts only from last.pt and forbids "
                "source checkpoint arguments"
            )
    if args.model != DCF_MODEL_NAME and (
        dcf_shallow is not None or dcf_deep is not None or dcf_physical != 0
        or dcf_middle_explicit is not None or dcf_high_explicit is not None
    ):
        raise ValueError("DCF-specific options require --model dcf_q")
    for value, label in ((aom_shallow, "shallow"), (aom_deep, "deep")):
        if value is not None:
            reject_forbidden_path(value)
        if args.model == AOM_MODEL_NAME and not is_resume and not isinstance(value, Path):
            raise ValueError(f"AOM-Q requires --aom-{label}-checkpoint")
        if args.model == AOM_MODEL_NAME and is_resume and value is not None:
            raise ValueError("AOM-Q resume forbids source checkpoint arguments")
    if args.model != AOM_MODEL_NAME and (
        aom_shallow is not None or aom_deep is not None or aom_physical != 0
        or float(getattr(args, "aom_backbone_lr_multiplier", 0.1)) != 0.1
    ):
        raise ValueError("AOM-specific options require --model aom_q")
    input_mode = str(getattr(args, "input_mode", "core22"))
    contrast_option_was_explicit = bool(
        getattr(args, "contrast_d4_average", False)
        or getattr(args, "no_contrast_d4_average", False)
        or getattr(args, "contrast_activation_checkpointing", False)
    )
    if args.model != "contrast_q" and contrast_option_was_explicit:
        raise ValueError("contrast-specific options require --model contrast_q")
    calibrated_checkpointing = getattr(
        args, "calibrated_activation_checkpointing", None
    )
    calibrated_physical = int(
        getattr(args, "calibrated_physical_batch_size", 0)
    )
    ipmr_checkpointing = getattr(args, "ipmr_activation_checkpointing", None)
    ipmr_physical = int(getattr(args, "ipmr_physical_batch_size", 0))
    ipmr_middle_explicit = getattr(args, "ipmr_middle_loss_weight", None)
    ipmr_high_explicit = getattr(args, "ipmr_high_loss_weight", None)
    calibrated_adapter = bool(
        getattr(args, "calibrated_allocation_adapter", False)
    )
    calibrated_t3 = bool(getattr(args, "calibrated_t3_fusion", False))
    calibrated_refiner = bool(getattr(args, "calibrated_q_refiner", False))
    calibrated_pyramid = bool(
        getattr(args, "calibrated_content_q_pyramid", False)
    )
    calibrated_dct15 = bool(
        getattr(args, "calibrated_parent_dct15", False)
    )
    calibrated_no_geo = bool(
        getattr(args, "calibrated_no_geo_core", False)
    )
    no_geo_mode = getattr(args, "no_geo_rebase_mode", None)
    no_geo_probe = getattr(args, "no_geo_probe", None)
    no_geo_pack_arm = getattr(args, "no_geo_pack_arm", None)
    refiner_freeze = int(
        getattr(args, "q_refiner_freeze_backbone_updates", 0)
    )
    refiner_core_multiplier = float(
        getattr(args, "q_refiner_core_lr_multiplier", 1.0)
    )
    refiner_multiplier = float(
        getattr(args, "q_refiner_lr_multiplier", 1.0)
    )
    refiner_optimization_was_explicit = (
        refiner_freeze != 0
        or refiner_core_multiplier != 1.0
        or refiner_multiplier != 1.0
    )
    pyramid_freeze = int(getattr(args, "q_pyramid_freeze_core_updates", 0))
    pyramid_core_multiplier = float(
        getattr(args, "q_pyramid_core_lr_multiplier", 1.0)
    )
    pyramid_multiplier = float(getattr(args, "q_pyramid_lr_multiplier", 1.0))
    pyramid_optimization_was_explicit = (
        pyramid_freeze != 0
        or pyramid_core_multiplier != 1.0
        or pyramid_multiplier != 1.0
    )
    dct15_freeze = int(getattr(args, "dct15_freeze_core_updates", 0))
    dct15_core_multiplier = float(
        getattr(args, "dct15_core_lr_multiplier", 1.0)
    )
    dct15_multiplier = float(getattr(args, "dct15_lr_multiplier", 1.0))
    dct15_optimization_was_explicit = (
        dct15_freeze != 0
        or dct15_core_multiplier != 1.0
        or dct15_multiplier != 1.0
    )
    if args.model != "calibrated_q" and (
        calibrated_checkpointing is not None or calibrated_physical != 0
        or calibrated_adapter or calibrated_t3 or calibrated_refiner
        or calibrated_pyramid or calibrated_dct15
        or calibrated_no_geo or no_geo_mode is not None or no_geo_probe is not None
        or no_geo_pack_arm is not None
        or refiner_optimization_was_explicit
        or pyramid_optimization_was_explicit
        or dct15_optimization_was_explicit
    ):
        raise ValueError("calibrated-specific options require --model calibrated_q")
    if args.model not in IPMR_MODEL_NAMES and (
        ipmr_checkpointing is not None
        or ipmr_physical != 0
        or ipmr_middle_explicit is not None
        or ipmr_high_explicit is not None
    ):
        raise ValueError(
            "IPMR-specific options require --model ipmr_q or ipmr_q_v2"
        )
    if args.model == "contrast_q" and args.width != 48:
        raise ValueError("ContrastQParent uses registered fixed widths; --width must be 48")
    if args.model == "calibrated_q" and args.width not in SceneCalibratedContinuousQ.widths:
        raise ValueError(
            "SceneCalibratedContinuousQ width must be one of 48, 64, 96"
        )
    if args.model in IPMR_MODEL_NAMES and args.width not in IPMRQ.widths:
        raise ValueError("IPMR-Q width must be one of 48, 64, 96")
    if args.model in IPMR_MODEL_NAMES and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
    ):
        raise ValueError(
            "IPMR-Q requires single-temporal public_validation Fine52/Context19 "
            "multisource input"
        )
    if args.model == PCQM_MODEL_NAME:
        if args.width != 48:
            raise ValueError("PCQM requires the registered width48 anchor")
        if not is_resume and init_checkpoint is None:
            raise ValueError(
                "fresh PCQM requires --init-checkpoint from an IPMR-Q anchor"
            )
        if _resolved_ipmr_middle_weight(args) != 0.0 \
                or _resolved_ipmr_high_weight(args) != 0.0:
            raise ValueError(
                "PCQM requires zero IPMR middle/high auxiliary weights"
            )
        if str(getattr(args, "init_weight_source", "selected")) != "selected":
            raise ValueError("PCQM requires the registered selected r9 EMA weights")
    if args.model == PARENT_ROUTER_MODEL_NAME and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
        or args.width != 48
    ):
        raise ValueError(
            "ParentRouterQ requires width48, single-temporal public_validation "
            "Fine52/Context19 multisource input"
        )
    if args.model == DCF_MODEL_NAME and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
        or args.width != 48
    ):
        raise ValueError(
            "DCF-Q requires width48, single-temporal public_validation "
            "Fine52/Context19 multisource input"
        )
    if args.model == U1LITE_MODEL_NAME and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
        or args.width != 48
    ):
        raise ValueError(
            "U1-Lite requires width48, single-temporal public_validation "
            "Fine52/Context19 multisource input"
        )
    if calibrated_no_geo and no_geo_mode is None:
        raise ValueError(
            "--calibrated-no-geo-core requires an explicit --no-geo-rebase-mode"
        )
    if not calibrated_no_geo and no_geo_mode is not None:
        raise ValueError(
            "--no-geo-rebase-mode requires --calibrated-no-geo-core"
        )
    if no_geo_probe is not None and not calibrated_no_geo:
        raise ValueError(
            "--no-geo-probe requires --calibrated-no-geo-core"
        )
    if no_geo_probe is not None and no_geo_mode != "surgery":
        raise ValueError(
            "--no-geo-probe requires --no-geo-rebase-mode surgery"
        )
    if no_geo_pack_arm is not None and not calibrated_no_geo:
        raise ValueError(
            "--no-geo-pack-arm requires --calibrated-no-geo-core"
        )
    if no_geo_pack_arm is not None and no_geo_mode != "surgery":
        raise ValueError(
            "--no-geo-pack-arm requires --no-geo-rebase-mode surgery"
        )
    if no_geo_pack_arm is not None and no_geo_probe is not None:
        raise ValueError(
            "--no-geo-pack-arm and --no-geo-probe are mutually exclusive"
        )
    if calibrated_no_geo and any((
        calibrated_adapter, calibrated_t3, calibrated_refiner,
        calibrated_pyramid, calibrated_dct15,
    )):
        raise ValueError(
            "--calibrated-no-geo-core is mutually exclusive with every calibrated extension"
        )
    if calibrated_no_geo and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
    ):
        raise ValueError(
            "--calibrated-no-geo-core requires single-temporal public_validation "
            "Fine52/Context19 multisource input"
        )
    if calibrated_no_geo and args.width != 48:
        raise ValueError(
            "--calibrated-no-geo-core requires the registered width48 r2k anchor"
        )
    if calibrated_no_geo and not bool(getattr(args, "resume", False)) \
            and init_checkpoint is None:
        raise ValueError(
            "fresh physical Context15 core runs require --init-checkpoint from bare r2k"
        )
    if calibrated_pyramid and any((
        calibrated_adapter, calibrated_t3, calibrated_refiner,
        calibrated_dct15,
    )):
        raise ValueError(
            "--calibrated-content-q-pyramid is mutually exclusive with allocation "
            "adapter, T3 fusion and Q-refiner"
        )
    if calibrated_dct15 and any((
        calibrated_adapter, calibrated_t3, calibrated_refiner,
        calibrated_pyramid,
    )):
        raise ValueError(
            "--calibrated-parent-dct15 is mutually exclusive with allocation "
            "adapter, T3 fusion, Q-refiner and Content-only Q-Pyramid"
        )
    if calibrated_adapter and calibrated_refiner:
        raise ValueError(
            "--calibrated-allocation-adapter and --calibrated-q-refiner are "
            "mutually exclusive"
        )
    if calibrated_t3 and args.temporal_mode != "multi":
        raise ValueError("--calibrated-t3-fusion requires --temporal-mode multi")
    if calibrated_refiner and input_mode != "multisource":
        raise ValueError("--calibrated-q-refiner requires Fine52 multisource input")
    if calibrated_pyramid and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
    ):
        raise ValueError(
            "--calibrated-content-q-pyramid requires single-temporal public_validation "
            "Fine52/Context19 multisource input"
        )
    if calibrated_pyramid and not bool(getattr(args, "resume", False)) \
            and init_checkpoint is None:
        raise ValueError(
            "fresh Content-only Q-Pyramid runs require --init-checkpoint from bare r2k"
        )
    if calibrated_dct15 and (
        args.temporal_mode != "single"
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
    ):
        raise ValueError(
            "--calibrated-parent-dct15 requires single-temporal public_validation "
            "Fine52/Context19 multisource input"
        )
    if calibrated_dct15 and not bool(getattr(args, "resume", False)) \
            and init_checkpoint is None:
        raise ValueError(
            "fresh Parent-DCT15 runs require --init-checkpoint from bare r2k"
        )
    if refiner_optimization_was_explicit and not calibrated_refiner:
        raise ValueError("Q-refiner optimization options require --calibrated-q-refiner")
    if pyramid_optimization_was_explicit and not calibrated_pyramid:
        raise ValueError(
            "Q-Pyramid optimization options require --calibrated-content-q-pyramid"
        )
    if dct15_optimization_was_explicit and not calibrated_dct15:
        raise ValueError(
            "Parent-DCT15 optimization options require --calibrated-parent-dct15"
        )
    if refiner_freeze < 0:
        raise ValueError("Q-refiner freeze updates must be nonnegative")
    for value, label in (
        (refiner_core_multiplier, "core"),
        (refiner_multiplier, "refiner"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Q-refiner {label} LR multiplier must be positive")
    if pyramid_freeze < 0:
        raise ValueError("Q-Pyramid freeze updates must be nonnegative")
    for value, label in (
        (pyramid_core_multiplier, "core"),
        (pyramid_multiplier, "pyramid"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Q-Pyramid {label} LR multiplier must be positive")
    if dct15_freeze < 0:
        raise ValueError("Parent-DCT15 freeze updates must be nonnegative")
    for value, label in (
        (dct15_core_multiplier, "core"),
        (dct15_multiplier, "decoder"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Parent-DCT15 {label} LR multiplier must be positive"
            )
    calibrated_initialization = (
        args.model == "calibrated_q"
        and (
            calibrated_adapter or calibrated_t3 or calibrated_refiner
            or calibrated_pyramid or calibrated_dct15 or calibrated_no_geo
        )
    )
    pcqm_initialization = args.model == PCQM_MODEL_NAME
    if init_checkpoint is not None and (
        not (calibrated_initialization or pcqm_initialization)
        or input_mode != "multisource"
        or args.data_scope != "public_validation"
    ):
        raise ValueError(
            "--init-checkpoint requires a registered calibrated_q extension or "
            "ipmr_q_pcqm on public_validation Fine52 multisource"
        )
    if args.texture_manifest is not None and args.allow_provisional_optical:
        raise ValueError("strict optical sidecars and provisional optical are mutually exclusive")
    if args.texture_manifest is None and not args.allow_provisional_optical:
        raise ValueError(
            "formal R2 requires --texture-manifest pointing to the complete target-free v2 "
            "manifest; engineering smoke instead requires --allow-provisional-optical"
        )
    if args.allow_provisional_optical and args.data_scope != "internal_dev":
        raise ValueError(
            "--allow-provisional-optical is restricted to internal_dev and cannot score "
            "public validation"
        )
    if input_mode == "multisource":
        if args.model not in {
            "qparent", "contrast_q", "calibrated_q",
            PARENT_ROUTER_MODEL_NAME, DCF_MODEL_NAME, U1LITE_MODEL_NAME,
            *IPMR_MODEL_NAMES,
        }:
            raise ValueError(
                "Stage-C multisource supports qparent, contrast_q, calibrated_q, "
                "ipmr_q, ipmr_q_v2, or parent_router_q; dcf_q and u1lite_q "
                "are additionally registered"
            )
        if args.texture_manifest is None or getattr(args, "weather_manifest", None) is None:
            raise ValueError(
                "Stage-C multisource requires both --texture-manifest and "
                "--weather-manifest"
            )
        if args.allow_provisional_optical:
            raise ValueError("Stage-C multisource forbids provisional optical inputs")
    elif input_mode == "core22":
        if getattr(args, "weather_manifest", None) is not None:
            raise ValueError("--weather-manifest requires --input-mode multisource")
        if int(getattr(args, "multisource_physical_batch_size", 0)) != 0:
            raise ValueError(
                "--multisource-physical-batch-size requires --input-mode multisource"
            )
    else:  # defensive for programmatic Namespace callers
        raise ValueError(f"unsupported input mode: {input_mode!r}")
    physical = int(getattr(args, "multisource_physical_batch_size", 0))
    if physical < 0 or physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("multisource physical batch size must be in [0,32]")
    if calibrated_physical < 0 or calibrated_physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("calibrated physical batch size must be in [0,32]")
    if ipmr_physical < 0 or ipmr_physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("IPMR physical batch size must be in [0,32]")
    if parent_router_physical < 0 \
            or parent_router_physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("ParentRouterQ physical batch size must be in [0,32]")
    if dcf_physical < 0 or dcf_physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("DCF-Q physical batch size must be in [0,32]")
    if aom_physical < 0 or aom_physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("AOM-Q physical batch size must be in [0,32]")
    if u1lite_physical < 0 or u1lite_physical > EFFECTIVE_BATCH_SIZE:
        raise ValueError("U1-Lite physical batch size must be in [0,32]")
    if args.model == "calibrated_q" and physical != 0:
        raise ValueError(
            "calibrated_q uses --calibrated-physical-batch-size in every input mode"
        )
    if args.model in IPMR_MODEL_NAMES and physical != 0:
        raise ValueError(
            "IPMR models use --ipmr-physical-batch-size in every input mode"
        )
    if args.model == PARENT_ROUTER_MODEL_NAME and physical != 0:
        raise ValueError(
            "ParentRouterQ uses --parent-router-physical-batch-size"
        )
    if args.model == DCF_MODEL_NAME and physical != 0:
        raise ValueError("DCF-Q uses --dcf-physical-batch-size")
    if args.model == AOM_MODEL_NAME and physical != 0:
        raise ValueError("AOM-Q uses --aom-physical-batch-size")
    if args.model == U1LITE_MODEL_NAME and physical != 0:
        raise ValueError("U1-Lite uses --u1lite-physical-batch-size")
    if args.max_updates <= 0 or args.validation_interval <= 0:
        raise ValueError("update intervals must be positive")
    pause_at_update = getattr(args, "pause_at_update", None)
    if pause_at_update is not None:
        if isinstance(pause_at_update, bool) \
                or not isinstance(pause_at_update, int) \
                or pause_at_update <= 0 \
                or pause_at_update > args.max_updates:
            raise ValueError(
                "--pause-at-update must be a positive integer no greater than "
                "--max-updates"
            )
    warmup_updates = int(getattr(args, "warmup_updates", WARMUP_UPDATES))
    weight_decay = float(getattr(args, "weight_decay", WEIGHT_DECAY))
    if warmup_updates < 0:
        raise ValueError("warmup updates must be nonnegative")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight decay must be finite and nonnegative")
    if args.full_scene_start < 0 or args.full_scene_start > args.max_updates:
        raise ValueError("full_scene_start must be in [0,max_updates]")
    if args.evaluation_batch_size <= 0 or args.max_wall_minutes < 0:
        raise ValueError("evaluation batch must be positive and wall limit nonnegative")
    eligible_weight = _resolved_eligible_loss_weight(args)
    if not math.isfinite(eligible_weight) or not 0.0 <= eligible_weight <= 1.0:
        raise ValueError("eligible loss weight must be finite and in [0,1]")
    registered_loss = math.isclose(
        eligible_weight, PRIMARY_LOSS_WEIGHT, rel_tol=0.0, abs_tol=1.0e-12
    )
    if (
        not registered_loss
        and not bool(getattr(args, "resume", False))
        and not bool(getattr(args, "allow_nonstandard_loss_objective", False))
    ):
        raise ValueError(
            "new R2 runs require the registered 0.8 eligible-valid + 0.2 all-valid "
            "loss; a deliberate ablation must pass --allow-nonstandard-loss-objective"
        )
    q_shape_weight = float(getattr(args, "q_shape_loss_weight", 0.0))
    if (
        not math.isfinite(q_shape_weight)
        or not 0.0 <= q_shape_weight <= Q_SHAPE_MAX_WEIGHT
    ):
        raise ValueError(
            "Q-shape loss weight must be finite in [0,0.15] so pixel MSE weight is >=0.85"
        )
    if args.model in IPMR_MODEL_NAMES and q_shape_weight != 0.0:
        raise ValueError("IPMR-Q uses orthogonal-band supervision, not Q-shape loss")
    if args.model == PARENT_ROUTER_MODEL_NAME and q_shape_weight != 0.0:
        raise ValueError(
            "ParentRouterQ uses the registered field loss and does not support "
            "Q-shape loss"
        )
    if args.model == DCF_MODEL_NAME and q_shape_weight != 0.0:
        raise ValueError(
            "DCF-Q uses orthogonal-band supervision and does not support Q-shape loss"
        )
    if args.model == AOM_MODEL_NAME and q_shape_weight != 0.0:
        raise ValueError("AOM-Q uses final field loss only; Q-shape loss is disabled")
    if args.model == U1LITE_MODEL_NAME and q_shape_weight != 0.0:
        raise ValueError(
            "U1-Lite target uses final field loss only; Q-shape loss is disabled"
        )
    if args.model in BAND_MODEL_NAMES:
        middle_weight = _resolved_ipmr_middle_weight(args)
        high_weight = _resolved_ipmr_high_weight(args)
        if (
            not math.isfinite(middle_weight)
            or not math.isfinite(high_weight)
            or middle_weight < 0.0
            or high_weight < 0.0
            or middle_weight + high_weight > 0.15
        ):
            raise ValueError(
                "orthogonal-band weights must be finite/nonnegative with sum <= 0.15"
            )
    if args.model == "ocnir_control" and args.width != 48:
        raise ValueError("R2 repaired OCNIR control is fixed at width 48")


def _training_entries(
    args: argparse.Namespace,
    fit: Sequence[G246Scene],
    validation: Sequence[G246Scene],
    fit_view_sha256: str,
) -> tuple[tuple[G246Scene, ...], tuple[G246Scene, ...], str]:
    if args.data_scope == "internal_dev":
        split = build_fit201_dev12(fit, fit_view_sha256)
        return split.train_entries, split.dev_entries, split.split_sha256
    if args.data_scope == "public_validation":
        return tuple(fit), tuple(validation), "public-validation-v1"
    raise ValueError(f"unsupported data scope: {args.data_scope!r}")


def _prepare_input_datasets(
    args: argparse.Namespace,
    fit_base: R2TemporalDataset,
    evaluation_base: R2TemporalDataset,
) -> tuple[
    R2TemporalDataset | MultiSourceTemporalDataset,
    R2TemporalDataset | MultiSourceTemporalDataset,
    int,
    int,
    dict[str, Any] | None,
    int,
    int,
]:
    """Resolve input wrapper and a memory-safe physical batch contract."""

    if str(getattr(args, "input_mode", "core22")) == "core22":
        if args.model == AOM_MODEL_NAME:
            requested = int(getattr(args, "aom_physical_batch_size", 0))
            physical_batch_size = requested or 1
            return (
                fit_base, evaluation_base, FINE_CHANNELS, CONTEXT_DIM, None,
                physical_batch_size,
                min(int(args.evaluation_batch_size), physical_batch_size),
            )
        if args.model == "calibrated_q":
            requested = int(
                getattr(args, "calibrated_physical_batch_size", 0)
            )
            physical_batch_size = requested or (2 if int(args.width) == 48 else 1)
            evaluation_batch_size = min(
                int(args.evaluation_batch_size), physical_batch_size
            )
            return (
                fit_base,
                evaluation_base,
                FINE_CHANNELS,
                CONTEXT_DIM,
                None,
                physical_batch_size,
                evaluation_batch_size,
            )
        return (
            fit_base, evaluation_base, FINE_CHANNELS, CONTEXT_DIM, None,
            EFFECTIVE_BATCH_SIZE, int(args.evaluation_batch_size),
        )
    if args.model == PARENT_ROUTER_MODEL_NAME:
        requested = int(getattr(args, "parent_router_physical_batch_size", 0))
        physical_batch_size = requested or 8
    elif args.model == DCF_MODEL_NAME:
        requested = int(getattr(args, "dcf_physical_batch_size", 0))
        physical_batch_size = requested or 2
    elif args.model == U1LITE_MODEL_NAME:
        requested = int(getattr(args, "u1lite_physical_batch_size", 0))
        physical_batch_size = requested or 1
    elif args.model == AOM_MODEL_NAME:
        requested = int(getattr(args, "aom_physical_batch_size", 0))
        physical_batch_size = requested or 1
    elif args.model == "calibrated_q" or args.model in IPMR_MODEL_NAMES:
        requested = int(
            getattr(
                args,
                "calibrated_physical_batch_size"
                if args.model == "calibrated_q"
                else "ipmr_physical_batch_size",
                0,
            )
        )
        physical_batch_size = requested or 1
    else:
        requested = int(getattr(args, "multisource_physical_batch_size", 0))
        physical_batch_size = requested or (2 if args.model == "contrast_q" else 4)
    # Stage-C sidecars are immutable local assets.  Materialise each requested
    # scene once before training so equal-region random sampling never falls
    # back to repeated NPZ/JSON decompression.  Capacity is tied to the actual
    # view (603 fit / 45 validation in the formal campaign) and the adapters
    # fail closed if a requested scene cannot remain resident.
    fit_cache_size = max(128, len(tuple(getattr(fit_base, "entries", ()))))
    evaluation_cache_size = max(
        128, len(tuple(getattr(evaluation_base, "entries", ())))
    )
    fit_data = MultiSourceTemporalDataset(
        fit_base,
        texture_manifest=args.texture_manifest,
        weather_manifest=args.weather_manifest,
        texture_cache_size=fit_cache_size,
        weather_cache_size=fit_cache_size,
        preload_texture_cache=True,
        preload_weather_cache=True,
    )
    evaluation_data = MultiSourceTemporalDataset(
        evaluation_base,
        texture_manifest=args.texture_manifest,
        weather_manifest=args.weather_manifest,
        texture_cache_size=evaluation_cache_size,
        weather_cache_size=evaluation_cache_size,
        preload_texture_cache=True,
        preload_weather_cache=True,
    )
    fit_provenance = fit_data.provenance_record()
    evaluation_provenance = evaluation_data.provenance_record()
    if fit_provenance != evaluation_provenance:
        raise ValueError("Stage-C fit/evaluation source bindings differ")
    evaluation_batch_size = min(
        int(args.evaluation_batch_size), physical_batch_size
    )
    return (
        fit_data, evaluation_data,
        MULTISOURCE_FINE_CHANNELS, MULTISOURCE_CONTEXT_DIM,
        fit_provenance, physical_batch_size, evaluation_batch_size,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Validate and execute one exclusively reserved training run."""

    _validate_args(args)
    output = reject_forbidden_path(args.output).resolve()
    reject_forbidden_path(output)
    with _exclusive_training_run_lock(output):
        return _train_with_lock_held(args)


def _train_with_lock_held(args: argparse.Namespace) -> dict[str, Any]:
    splits = load_splits(args.split_receipt, args.role)
    train_entries, evaluation_entries, split_identity = _training_entries(
        args, splits.fit, splits.validation, splits.fit_view_sha256
    )
    output = reject_forbidden_path(args.output).resolve()
    state_path = output / "state.json"
    history_path = output / "history.json"
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    normalization_path = output / "normalization.json"
    pause_path = output / "pause.request"
    if output.exists() and not args.resume and any(output.iterdir()):
        raise FileExistsError("non-empty R2 output requires --resume")
    resume_state_path: Path | None = None
    resume_last_path: Path | None = None
    resume_normalization_path: Path | None = None
    if args.resume:
        try:
            resume_state_path = _regular_public_artifact(
                state_path, "resume state.json"
            )
            resume_last_path = _regular_public_artifact(
                last_path, "resume last.pt"
            )
            resume_normalization_path = _regular_public_artifact(
                normalization_path, "resume normalization.json"
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "--resume requires state.json, last.pt, and normalization.json"
            ) from exc
    output.mkdir(parents=True, exist_ok=True)

    normalization_reuse: dict[str, Any] | None = None
    parent_router_normalization_reuse: dict[str, Any] | None = None
    dcf_normalization_reuse: dict[str, Any] | None = None
    aom_normalization_reuse: dict[str, Any] | None = None
    u1lite_normalization_reuse: dict[str, Any] | None = None
    if args.resume:
        if resume_normalization_path is None:
            raise AssertionError("resume normalization path was not validated")
        normalization = R2Normalization.from_dict(
            json.loads(resume_normalization_path.read_text(encoding="utf-8"))
        )
    elif args.model == PARENT_ROUTER_MODEL_NAME:
        shallow_checkpoint = getattr(
            args, "parent_router_shallow_checkpoint", None
        )
        deep_checkpoint = getattr(args, "parent_router_deep_checkpoint", None)
        if not isinstance(shallow_checkpoint, Path) \
                or not isinstance(deep_checkpoint, Path):
            raise ValueError(
                "fresh ParentRouterQ requires two expert checkpoints"
            )
        normalization, parent_router_normalization_reuse = (
            _parent_router_reuse_normalization(
                shallow_checkpoint, deep_checkpoint, normalization_path
            )
        )
    elif args.model == DCF_MODEL_NAME:
        shallow_checkpoint = getattr(args, "dcf_shallow_checkpoint", None)
        deep_checkpoint = getattr(args, "dcf_deep_checkpoint", None)
        if not isinstance(shallow_checkpoint, Path) \
                or not isinstance(deep_checkpoint, Path):
            raise ValueError("fresh DCF-Q requires two expert checkpoints")
        normalization, dcf_normalization_reuse = (
            _parent_router_reuse_normalization(
                shallow_checkpoint, deep_checkpoint, normalization_path
            )
        )
    elif args.model == U1LITE_MODEL_NAME:
        distilled_checkpoint = getattr(
            args, "u1lite_distilled_checkpoint", None
        )
        direct_checkpoint = getattr(
            args, "u1lite_direct_r6a_checkpoint", None
        )
        if direct_checkpoint is not None:
            normalization, u1lite_normalization_reuse = (
                _reuse_u1lite_direct_normalization(
                    direct_checkpoint, normalization_path
                )
            )
        elif distilled_checkpoint is not None:
            normalization, u1lite_normalization_reuse = (
                _reuse_u1lite_distilled_normalization(
                    distilled_checkpoint, normalization_path
                )
            )
        else:
            raise ValueError("fresh U1-Lite source checkpoint is missing")
    elif args.model == AOM_MODEL_NAME:
        shallow_checkpoint = getattr(args, "aom_shallow_checkpoint", None)
        deep_checkpoint = getattr(args, "aom_deep_checkpoint", None)
        if not isinstance(shallow_checkpoint, Path) \
                or not isinstance(deep_checkpoint, Path):
            raise ValueError("fresh AOM-Q requires two source checkpoints")
        normalization, aom_normalization_reuse = _parent_router_reuse_normalization(
            shallow_checkpoint, deep_checkpoint, normalization_path
        )
    elif getattr(args, "init_checkpoint", None) is not None:
        normalization, normalization_reuse = _reuse_initialization_normalization(
            args.init_checkpoint,
            normalization_path,
            expected_source_model=(
                "ipmr_q" if args.model == PCQM_MODEL_NAME else "calibrated_q"
            ),
        )
    else:
        normalization = fit_r2_normalization(
            train_entries, splits.fit_view_sha256,
            texture_manifest=args.texture_manifest,
            allow_provisional=args.allow_provisional_optical,
        )
        atomic_json(normalization_path, normalization.to_dict())
    normalization_bytes_path = (
        resume_normalization_path if args.resume else normalization_path
    )
    if normalization_bytes_path is None:
        raise AssertionError("normalization bytes path was not validated")
    normalization_sha = hashlib.sha256(
        normalization_bytes_path.read_bytes()
    ).hexdigest()

    fit_base = R2TemporalDataset(
        train_entries, normalization, seed=args.seed, augment=True,
        texture_manifest=args.texture_manifest,
        allow_provisional=args.allow_provisional_optical,
        region_sampling=args.region_sampling,
    )
    evaluation_base = R2TemporalDataset(
        evaluation_entries, normalization, seed=args.seed, augment=False,
        texture_manifest=args.texture_manifest,
        allow_provisional=args.allow_provisional_optical,
        region_sampling=args.region_sampling,
    )
    (
        fit_data, evaluation_data, input_fine_channels, input_context_dim,
        multisource_provenance, physical_batch_size, evaluation_batch_size,
    ) = _prepare_input_datasets(args, fit_base, evaluation_base)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    _seed_everything(args.seed)
    model = build_model(
        args.model,
        width=args.width,
        contrast_d4_average=getattr(args, "contrast_d4_average", False),
        contrast_activation_checkpointing=getattr(
            args, "contrast_activation_checkpointing", False
        ),
        calibrated_activation_checkpointing=(
            True
            if getattr(args, "calibrated_activation_checkpointing", None) is None
            else bool(args.calibrated_activation_checkpointing)
        ) if args.model == "calibrated_q" else None,
        calibrated_allocation_adapter=bool(
            getattr(args, "calibrated_allocation_adapter", False)
        ),
        calibrated_t3_fusion=bool(
            getattr(args, "calibrated_t3_fusion", False)
        ),
        calibrated_q_refiner=bool(
            getattr(args, "calibrated_q_refiner", False)
        ),
        calibrated_content_q_pyramid=bool(
            getattr(args, "calibrated_content_q_pyramid", False)
        ),
        calibrated_parent_dct15=bool(
            getattr(args, "calibrated_parent_dct15", False)
        ),
        calibrated_no_geo_core=bool(
            getattr(args, "calibrated_no_geo_core", False)
        ),
        calibrated_hierarchical_pack=(
            getattr(args, "no_geo_pack_arm", None) == "hierarchical"
        ),
        ipmr_activation_checkpointing=(
            True
            if getattr(args, "ipmr_activation_checkpointing", None) is None
            else bool(args.ipmr_activation_checkpointing)
        ) if args.model in IPMR_MODEL_NAMES else None,
        u1lite_dexchange=bool(
            getattr(args, "u1lite_dexchange", False)
        ),
        fine_channels=input_fine_channels,
        context_dim=input_context_dim,
    ).to(device)
    no_geo_probe_parameters: list[nn.Parameter] | None = None
    no_geo_probe_contract: dict[str, Any] | None = None
    no_geo_probe_mode = getattr(args, "no_geo_probe", None)
    if no_geo_probe_mode is not None:
        no_geo_probe_parameters, no_geo_probe_contract = configure_no_geo_probe(
            model, str(no_geo_probe_mode)
        )
    no_geo_pack_arm = getattr(args, "no_geo_pack_arm", None)
    no_geo_pack_contract: dict[str, Any] | None = None
    if no_geo_pack_arm is not None:
        no_geo_pack_contract = build_no_geo_pack_contract(
            model, str(no_geo_pack_arm)
        )
    pcqm_contract = (
        pcqm_optimization_contract(model)
        if args.model == PCQM_MODEL_NAME else None
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    eligible_loss_weight = _resolved_eligible_loss_weight(args)
    q_shape_loss_weight = float(getattr(args, "q_shape_loss_weight", 0.0))
    ipmr_middle_loss_weight = _resolved_ipmr_middle_weight(args)
    ipmr_high_loss_weight = _resolved_ipmr_high_weight(args)
    config = _config(
        args,
        receipt_sha256=splits.receipt_sha256,
        fit_view_sha256=splits.fit_view_sha256,
        train_city_count=len({entry.city for entry in train_entries}),
        evaluation_city_count=len({entry.city for entry in evaluation_entries}),
        split_identity=split_identity,
        normalization_sha256=normalization_sha,
        parameter_count=parameter_count,
        input_provenance=(
            f"g246_r2_multisource:{multisource_provenance['binding_sha256']}"
            if multisource_provenance is not None else normalization.optical_source
        ),
        scientific_status=(
            MultiSourceTemporalDataset.scientific_status
            if multisource_provenance is not None else normalization.scientific_status
        ),
        fine_channels=input_fine_channels,
        context_dim=input_context_dim,
        multisource_provenance=multisource_provenance,
        physical_batch_size=physical_batch_size,
        effective_evaluation_batch_size=evaluation_batch_size,
        no_geo_probe_optimization=no_geo_probe_contract,
        no_geo_pack_screen=no_geo_pack_contract,
        pcqm_optimization=pcqm_contract,
    )
    if args.model == PARENT_ROUTER_MODEL_NAME:
        if not isinstance(model, ParentRouterQ):
            raise AssertionError("ParentRouterQ model dispatch drifted")
        router_named = [
            (name, parameter) for name, parameter in model.named_parameters()
            if name.startswith("router.")
        ]
        router_names = [name for name, _parameter in router_named]
        if tuple(parameter for _name, parameter in router_named) \
                != model.router_parameters():
            raise ValueError("ParentRouterQ named router parameter order drifted")
        config.update({
            "parent_router_design_contract": {
                "schema_version": ParentRouterQ.schema_version,
                "gate_lattice": "one_shared_scalar_per_4x4_physical_parent",
                "gate_range": list(ParentRouterQ.gate_range),
                "gate_initialization": "exact_fixed_0p5_average",
                "gate_parameterization": "0.5+0.25*softsign(logit)",
                "router_feature_channels": ParentRouterQ.router_input_channels,
                "router_width": model.router_width,
                "router_parameter_names": router_names,
                "router_parameter_names_sha256": _parameter_names_sha256(
                    router_names
                ),
                "router_parameter_tensor_count": len(router_named),
                "router_trainable_parameter_count": (
                    model.trainable_parameter_count
                ),
                "total_parameter_count": model.parameter_count,
                "optimizer_scope": "router_parameters_only",
                "ema_scope": "router_floating_state_only_experts_exact_copy",
                "explicit_geolocation_city_region_used": False,
                "physically_removed_context_indices": [5, 6, 7, 8],
                "solar_geometry_retained": True,
                "locked_test_opened": False,
            },
            "parent_router_precision_contract": parent_router_precision_contract(
                device.type, amp_requested=bool(args.amp)
            ),
            "parent_router_implementation_sha256": {
                "trainer": _sha256_file(Path(__file__).resolve()),
                "model": _sha256_file(CODE_ROOT / "g246_parent_router_q.py"),
                "shallow_expert_model": _sha256_file(
                    CODE_ROOT / "g246_calibrated_continuous_q.py"
                ),
                "deep_expert_model": _sha256_file(
                    CODE_ROOT / "g246_ipmr_q.py"
                ),
                "q_bands": _sha256_file(CODE_ROOT / "g246_q_bands.py"),
                "support_projection": _sha256_file(CODE_ROOT / "ocnir.py"),
                "split_loader": _sha256_file(CODE_ROOT / "g246_data.py"),
                "core_data_and_objective": _sha256_file(
                    CODE_ROOT / "g246_r2_data.py"
                ),
                "multisource_adapter": _sha256_file(
                    CODE_ROOT / "g246_r2_multisource.py"
                ),
                "metric_and_checkpoint": _sha256_file(
                    CODE_ROOT / "train_g246_metric.py"
                ),
            },
            "parent_router_resume_policy": (
                "fresh_sources_bound_once_then_resume_only_from_last_pt;_"
                "fail_closed_on_bound_implementation_byte_drift"
            ),
            "early_stop_policy": (
                "manual_u0_u500_u1000_metric_gates_generic_plateau_disabled"
            ),
        })
        if not args.resume:
            shallow_checkpoint = getattr(
                args, "parent_router_shallow_checkpoint", None
            )
            deep_checkpoint = getattr(
                args, "parent_router_deep_checkpoint", None
            )
            if not isinstance(shallow_checkpoint, Path) \
                    or not isinstance(deep_checkpoint, Path):
                raise ValueError(
                    "fresh ParentRouterQ requires both registered expert checkpoints"
                )
            parent_router_provenance = initialize_parent_router_experts(
                model,
                shallow_checkpoint,
                deep_checkpoint,
                config,
            )
            if parent_router_normalization_reuse is None:
                raise AssertionError(
                    "ParentRouterQ initialization omitted normalization reuse"
                )
            parent_router_provenance["normalization_reuse"] = copy.deepcopy(
                parent_router_normalization_reuse
            )
            config[
                "parent_router_initialization_provenance"
            ] = parent_router_provenance
    if args.model == DCF_MODEL_NAME:
        if not isinstance(model, DCFQ):
            raise AssertionError("DCF-Q model dispatch drifted")
        decoder_named = [
            (name, parameter) for name, parameter in model.named_parameters()
            if name.startswith("decoder.")
        ]
        decoder_names = [name for name, _parameter in decoder_named]
        if tuple(parameter for _name, parameter in decoder_named) \
                != model.decoder_parameters():
            raise ValueError("DCF-Q named decoder parameter order drifted")
        config.update({
            "dcf_design_contract": {
                "schema_version": DCFQ.schema_version,
                "anchor": "fixed_0p5_q_space_average",
                "anchor_trainable": False,
                "source_q_visible_to_decoder": False,
                "feature_lattices": [40, 80, 160],
                "feature_channels": {
                    "r6a_S160": 64, "r6a_S40": 144,
                    "r9_F160": 64, "r9_D80": 96, "r9_D40": 144,
                },
                "fusion": "balanced_additive_plus_concat_spatial_residual",
                "decoder_channels": [112, 112, 64],
                "head_outputs": ["orthogonal_QM", "orthogonal_QH"],
                "decoder_parameter_names": decoder_names,
                "decoder_parameter_names_sha256": _parameter_names_sha256(
                    decoder_names
                ),
                "decoder_parameter_tensor_count": len(decoder_named),
                "decoder_trainable_parameter_count": model.trainable_parameter_count,
                "total_parameter_count": model.parameter_count,
                "optimizer_scope": "decoder_parameters_only",
                "ema_scope": "decoder_floating_state_only_experts_exact_copy",
                "explicit_geolocation_city_region_used": False,
                "physically_removed_context_indices": [5, 6, 7, 8],
                "solar_geometry_retained": True,
                "locked_test_opened": False,
            },
            "dcf_precision_contract": dcf_precision_contract(
                device.type, amp_requested=bool(args.amp)
            ),
            "dcf_implementation_sha256": {
                "trainer": _sha256_file(Path(__file__).resolve()),
                "model": _sha256_file(CODE_ROOT / "g246_dcf_q.py"),
                "shallow_expert_model": _sha256_file(
                    CODE_ROOT / "g246_calibrated_continuous_q.py"
                ),
                "deep_expert_model": _sha256_file(CODE_ROOT / "g246_ipmr_q.py"),
                "q_bands": _sha256_file(CODE_ROOT / "g246_q_bands.py"),
                "support_projection": _sha256_file(CODE_ROOT / "ocnir.py"),
                "split_loader": _sha256_file(CODE_ROOT / "g246_data.py"),
                "core_data_and_objective": _sha256_file(CODE_ROOT / "g246_r2_data.py"),
                "multisource_adapter": _sha256_file(CODE_ROOT / "g246_r2_multisource.py"),
                "metric_and_checkpoint": _sha256_file(CODE_ROOT / "train_g246_metric.py"),
            },
            "dcf_resume_policy": (
                "fresh_sources_bound_once_then_resume_only_from_last_pt;_"
                "fail_closed_on_precision_and_bound_implementation_byte_drift"
            ),
            "early_stop_policy": "manual_u0_u500_gate_generic_plateau_disabled",
        })
        if not args.resume:
            shallow_checkpoint = getattr(args, "dcf_shallow_checkpoint", None)
            deep_checkpoint = getattr(args, "dcf_deep_checkpoint", None)
            if not isinstance(shallow_checkpoint, Path) \
                    or not isinstance(deep_checkpoint, Path):
                raise ValueError("fresh DCF-Q requires both expert checkpoints")
            dcf_provenance = initialize_dcf_experts(
                model, shallow_checkpoint, deep_checkpoint, config
            )
            if dcf_normalization_reuse is None:
                raise AssertionError("DCF-Q initialization omitted normalization reuse")
            dcf_provenance["normalization_reuse"] = copy.deepcopy(
                dcf_normalization_reuse
            )
            config["dcf_initialization_provenance"] = dcf_provenance
    if args.model == U1LITE_MODEL_NAME:
        if not isinstance(model, ContinuousScaleBridgeQ):
            raise AssertionError("U1-Lite model dispatch drifted")
        u1lite_dexchange = bool(getattr(args, "u1lite_dexchange", False))
        if u1lite_dexchange != isinstance(
            model, DExchangeContinuousScaleBridgeQ
        ):
            raise AssertionError("U1-Lite D-Exchange dispatch drifted")
        direct_checkpoint = getattr(
            args, "u1lite_direct_r6a_checkpoint", None
        )
        parameter_names = [name for name, _parameter in model.named_parameters()]
        if u1lite_dexchange and (
            model.exchange_parameter_count != 76_800
            or model.added_parameter_count > 650_000
            or parameter_count >= 4_650_000
        ):
            raise ValueError("U1-Lite D-Exchange parameter budget was exceeded")
        if not u1lite_dexchange and (
            model.added_parameter_count >= 550_000
            or parameter_count >= 4_550_000
        ):
            raise ValueError("U1-Lite parameter budget was exceeded")
        config.update({
            "u1lite_design_contract": {
                "single_backbone": True,
                "scale_path": type(model).scale_path,
                "direct_residual": (
                    "q=r6a_q+QM(delta80)+QH(delta160);final_Q40_repair"
                ),
                "second_expert": False,
                "aom_path": False,
                "band_auxiliary_loss": False,
                "all_parameters_trainable": True,
                "optimizer_parameter_names_sha256": _parameter_names_sha256(
                    parameter_names
                ),
                "optimizer_parameter_tensor_count": len(parameter_names),
                "added_parameter_count": model.added_parameter_count,
                "total_parameter_count": parameter_count,
                "explicit_geolocation_city_region_used": False,
                "physically_removed_context_indices": [5, 6, 7, 8],
                "locked_test_opened": False,
            },
            "u1lite_implementation_sha256": {
                "trainer": _sha256_file(Path(__file__).resolve()),
                "model": _sha256_file(
                    CODE_ROOT / (
                        "g246_dexchange_q.py" if u1lite_dexchange
                        else "g246_continuous_scale_bridge_q.py"
                    )
                ),
                "source_backbone_model": _sha256_file(
                    CODE_ROOT / "g246_calibrated_continuous_q.py"
                ),
                "q_bands": _sha256_file(CODE_ROOT / "g246_q_bands.py"),
                "support_projection": _sha256_file(CODE_ROOT / "ocnir.py"),
                "core_data_and_objective": _sha256_file(
                    CODE_ROOT / "g246_r2_data.py"
                ),
                "multisource_adapter": _sha256_file(
                    CODE_ROOT / "g246_r2_multisource.py"
                ),
                "metric_and_checkpoint": _sha256_file(
                    CODE_ROOT / "train_g246_metric.py"
                ),
            },
            "u1lite_resume_policy": (
                U1LITE_DIRECT_RESUME_POLICY
                if not args.resume and direct_checkpoint is not None
                else "fresh_qualified_d1_full_raw_state_once_then_resume_only_from_last_pt"
            ),
        })
        if u1lite_dexchange:
            config["u1lite_design_contract"].update({
                "candidate": "dexchange",
                "exchange_prefixes": list(model.exchange_prefixes),
                "exchange_parameter_count": model.exchange_parameter_count,
            })
            config["u1lite_implementation_sha256"].update({
                "u1_parent_model": _sha256_file(
                    CODE_ROOT / "g246_continuous_scale_bridge_q.py"
                ),
                "ipmr_primitives": _sha256_file(CODE_ROOT / "g246_ipmr_q.py"),
            })
        if not args.resume:
            distilled_checkpoint = getattr(
                args, "u1lite_distilled_checkpoint", None
            )
            if u1lite_normalization_reuse is None:
                raise AssertionError(
                    "U1-Lite initialization omitted normalization reuse"
                )
            if direct_checkpoint is not None:
                config["u1lite_direct_initialization_provenance"] = (
                    initialize_u1lite_from_direct_r6a_checkpoint(
                        model, direct_checkpoint, config,
                        u1lite_normalization_reuse,
                    )
                )
                if u1lite_dexchange and any(
                    torch.count_nonzero(parameter).item() != 0
                    for prefix in model.exchange_prefixes
                    for parameter in getattr(model, prefix[:-1]).parameters()
                ):
                    raise ValueError(
                        "U1-Lite D-Exchange edges are not zero at direct r6a u0"
                    )
            elif distilled_checkpoint is not None:
                config["u1lite_distilled_initialization_provenance"] = (
                    initialize_u1lite_from_distilled_checkpoint(
                        model, distilled_checkpoint, config,
                        u1lite_normalization_reuse,
                    )
                )
            else:
                raise ValueError("fresh U1-Lite source checkpoint is missing")
    if args.model == AOM_MODEL_NAME:
        if not isinstance(model, AOMQ):
            raise AssertionError("AOM-Q model dispatch drifted")
        groups = aom_optimizer_groups(
            model, base_learning_rate=float(args.learning_rate),
            backbone_lr_multiplier=float(
                getattr(args, "aom_backbone_lr_multiplier", 0.1)
            ),
        )
        config.update({
            "aom_design_contract": {
                "trainable_parameter_count": model.trainable_parameter_count,
                "all_backbone_parameters_trainable": True,
                "exchange_lattices": [40, 80, 160],
                "anchor": "current_per_band_0p5_shallow_plus_deep",
                "u0_anchor_checkpoint_is_recoverable_not_duplicated_in_memory": True,
                "evaluation_outputs": [
                    "final", "shallow", "deep", "current_half", "gain_only",
                    "shape_only", "QM", "QH", "error_cosine", "disagreement",
                ],
                "optimizer_parameter_names_sha256": _parameter_names_sha256(
                    [name for name, _p in model.named_parameters()]
                ),
                "optimizer_group_tensor_counts": [len(g["params"]) for g in groups],
                "locked_test_opened": False,
            },
            "aom_implementation_sha256": {
                "trainer": _sha256_file(Path(__file__).resolve()),
                "model": _sha256_file(CODE_ROOT / "g246_aom_q.py"),
                "shallow_expert_model": _sha256_file(
                    CODE_ROOT / "g246_calibrated_continuous_q.py"
                ),
                "deep_expert_model": _sha256_file(CODE_ROOT / "g246_ipmr_q.py"),
                "q_bands": _sha256_file(CODE_ROOT / "g246_q_bands.py"),
                "support_projection": _sha256_file(CODE_ROOT / "ocnir.py"),
            },
            "early_stop_policy": "manual_u500_mechanism_gate_no_generic_plateau",
        })
        if not args.resume:
            shallow_checkpoint = getattr(args, "aom_shallow_checkpoint", None)
            deep_checkpoint = getattr(args, "aom_deep_checkpoint", None)
            if not isinstance(shallow_checkpoint, Path) \
                    or not isinstance(deep_checkpoint, Path):
                raise ValueError("fresh AOM-Q requires both source checkpoints")
            provenance = initialize_aom_experts(
                model, shallow_checkpoint, deep_checkpoint, config
            )
            if aom_normalization_reuse is None:
                raise AssertionError("AOM-Q initialization omitted normalization reuse")
            provenance["normalization_reuse"] = copy.deepcopy(aom_normalization_reuse)
            config["aom_initialization_provenance"] = provenance
    if getattr(args, "init_checkpoint", None) is not None:
        initialization_provenance = initialize_model_from_checkpoint(
            model,
            args.init_checkpoint,
            config,
            weight_source=str(getattr(args, "init_weight_source", "selected")),
        )
        if normalization_reuse is None:
            raise AssertionError(
                "weights-only initialization omitted normalization reuse provenance"
            )
        initialization_provenance["normalization_reuse"] = copy.deepcopy(
            normalization_reuse
        )
        config["initialization_provenance"] = initialization_provenance
    if args.model == PCQM_MODEL_NAME:
        optimizer_parameters: Any = pcqm_optimizer_groups(
            model, base_learning_rate=float(args.learning_rate)
        )
    elif bool(getattr(args, "calibrated_q_refiner", False)):
        optimizer_parameters = q_refiner_optimizer_groups(
            model,
            base_learning_rate=float(args.learning_rate),
            core_lr_multiplier=float(
                getattr(args, "q_refiner_core_lr_multiplier", 1.0)
            ),
            refiner_lr_multiplier=float(
                getattr(args, "q_refiner_lr_multiplier", 1.0)
            ),
        )
    elif bool(getattr(args, "calibrated_content_q_pyramid", False)):
        optimizer_parameters = content_q_pyramid_optimizer_groups(
            model,
            base_learning_rate=float(args.learning_rate),
            core_lr_multiplier=float(
                getattr(args, "q_pyramid_core_lr_multiplier", 1.0)
            ),
            pyramid_lr_multiplier=float(
                getattr(args, "q_pyramid_lr_multiplier", 1.0)
            ),
        )
    elif bool(getattr(args, "calibrated_parent_dct15", False)):
        optimizer_parameters = parent_dct15_optimizer_groups(
            model,
            base_learning_rate=float(args.learning_rate),
            core_lr_multiplier=float(
                getattr(args, "dct15_core_lr_multiplier", 1.0)
            ),
            dct15_lr_multiplier=float(
                getattr(args, "dct15_lr_multiplier", 1.0)
            ),
        )
    elif no_geo_probe_parameters is not None:
        if no_geo_probe_mode not in {"head_conditioning", "encoder_backbone"}:
            raise AssertionError("validated no-geolocation probe mode changed")
        optimizer_parameters = [{
            "params": no_geo_probe_parameters,
            "lr": float(args.learning_rate),
            "group_name": f"no_geo_probe_{no_geo_probe_mode}",
        }]
    elif args.model == PARENT_ROUTER_MODEL_NAME:
        if not isinstance(model, ParentRouterQ):
            raise AssertionError("ParentRouterQ model dispatch drifted")
        optimizer_parameters = [{
            "params": list(model.router_parameters()),
            "lr": float(args.learning_rate),
            "group_name": "parent_router",
        }]
    elif args.model == DCF_MODEL_NAME:
        if not isinstance(model, DCFQ):
            raise AssertionError("DCF-Q model dispatch drifted")
        optimizer_parameters = [{
            "params": list(model.decoder_parameters()),
            "lr": float(args.learning_rate),
            "group_name": "dcf_decoder",
        }]
    elif args.model == U1LITE_MODEL_NAME:
        if not isinstance(model, ContinuousScaleBridgeQ):
            raise AssertionError("U1-Lite optimizer model dispatch drifted")
        u1lite_parameters = list(model.parameters())
        if not u1lite_parameters \
                or any(not parameter.requires_grad for parameter in u1lite_parameters):
            raise ValueError("U1-Lite T0 optimizer requires every parameter")
        optimizer_parameters = [{
            "params": u1lite_parameters,
            "lr": float(args.learning_rate),
            "group_name": "u1lite_all",
        }]
    elif args.model == AOM_MODEL_NAME:
        optimizer_parameters = aom_optimizer_groups(
            model,
            base_learning_rate=float(args.learning_rate),
            backbone_lr_multiplier=float(
                getattr(args, "aom_backbone_lr_multiplier", 0.1)
            ),
        )
    else:
        # Preserve the exact single-group optimizer state contract for every
        # existing non-refiner checkpoint and resume path.
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        weight_decay=float(getattr(args, "weight_decay", WEIGHT_DECAY)),
    )
    scaler = _scaler(device, args.amp)
    if args.model == PARENT_ROUTER_MODEL_NAME:
        ema: WarmStartEMA = ParentRouterEMA(model, decay=EMA_DECAY)
    elif args.model == DCF_MODEL_NAME:
        ema = DCFEMA(model, decay=EMA_DECAY)
    else:
        ema = WarmStartEMA(model, decay=EMA_DECAY)
    update = run_updates = 0
    cumulative_wall = 0.0
    training_wall = TrainingWallTimer()
    records: list[dict[str, Any]] = []
    if args.resume:
        if resume_last_path is None or resume_state_path is None:
            raise AssertionError("resume artifacts were not validated")
        checkpoint = torch.load(
            resume_last_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("resume checkpoint root is malformed")
        checkpoint_config = _resume_checkpoint_config(checkpoint)
        config = _resume_scientific_config(checkpoint_config, config)
        if config.get("initialization_provenance") != checkpoint.get(
            "initialization_provenance"
        ):
            raise ValueError(
                "resume checkpoint initialization provenance binding differs"
            )
        if config.get("parent_router_initialization_provenance") \
                != checkpoint.get("parent_router_initialization_provenance"):
            raise ValueError(
                "resume ParentRouterQ initialization provenance binding differs"
            )
        if config.get("dcf_initialization_provenance") \
                != checkpoint.get("dcf_initialization_provenance"):
            raise ValueError("resume DCF-Q initialization provenance binding differs")
        if config.get("u1lite_distilled_initialization_provenance") \
                != checkpoint.get(
                    "u1lite_distilled_initialization_provenance"
                ):
            raise ValueError(
                "resume U1-Lite initialization provenance binding differs"
            )
        if config.get("u1lite_direct_initialization_provenance") \
                != checkpoint.get("u1lite_direct_initialization_provenance"):
            raise ValueError(
                "resume U1-Lite direct provenance binding differs"
            )
        if config.get("aom_initialization_provenance") \
                != checkpoint.get("aom_initialization_provenance"):
            raise ValueError("resume AOM-Q initialization provenance binding differs")
        update = int(checkpoint["optimizer_updates"])
        if args.max_updates <= update:
            raise ValueError(
                "resume requires requested max_updates > checkpoint optimizer_updates "
                f"({args.max_updates} <= {update})"
            )
        resume_model_state = checkpoint["model_state_dict"]
        if args.model == PARENT_ROUTER_MODEL_NAME:
            validate_parent_router_expert_state_hashes(resume_model_state)
        elif args.model == DCF_MODEL_NAME:
            validate_dcf_expert_state_hashes(resume_model_state)
        model.load_state_dict(resume_model_state, strict=True)
        if args.model == PCQM_MODEL_NAME:
            validate_pcqm_anchor_resume_contract(model, config)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("resume checkpoint optimizer state is missing")
        if args.model == PCQM_MODEL_NAME:
            validate_pcqm_optimizer_resume_contract(
                model,
                optimizer,
                optimizer_state,
                config,
                weight_decay=float(config["weight_decay"]),
            )
        elif bool(getattr(args, "calibrated_q_refiner", False)):
            optimization = config.get("q_refiner_optimization")
            if not isinstance(optimization, Mapping):  # defensive after config validation
                raise ValueError("resume Q-refiner optimization config is missing")
            validate_q_refiner_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                core_lr_multiplier=float(optimization["core_lr_multiplier"]),
                refiner_lr_multiplier=float(optimization["refiner_lr_multiplier"]),
                weight_decay=float(config["weight_decay"]),
            )
        elif bool(getattr(args, "calibrated_content_q_pyramid", False)):
            optimization = config.get("content_q_pyramid_optimization")
            if not isinstance(optimization, Mapping):
                raise ValueError("resume Q-Pyramid optimization config is missing")
            validate_content_q_pyramid_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                core_lr_multiplier=float(optimization["core_lr_multiplier"]),
                pyramid_lr_multiplier=float(optimization["pyramid_lr_multiplier"]),
                weight_decay=float(config["weight_decay"]),
            )
        elif bool(getattr(args, "calibrated_parent_dct15", False)):
            optimization = config.get("parent_dct15_optimization")
            if not isinstance(optimization, Mapping):
                raise ValueError("resume Parent-DCT15 optimization config is missing")
            validate_parent_dct15_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                core_lr_multiplier=float(optimization["core_lr_multiplier"]),
                dct15_lr_multiplier=float(optimization["dct15_lr_multiplier"]),
                weight_decay=float(config["weight_decay"]),
            )
        elif no_geo_probe_mode is not None:
            resumed_probe_mode = _calibrated_no_geo_probe_mode(config)
            if resumed_probe_mode != no_geo_probe_mode:
                raise ValueError(
                    "resume no-geolocation probe mode differs from config"
                )
            validate_no_geo_probe_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                probe_mode=str(no_geo_probe_mode),
                weight_decay=float(config["weight_decay"]),
            )
        elif args.model == PARENT_ROUTER_MODEL_NAME:
            if not isinstance(model, ParentRouterQ):
                raise AssertionError("ParentRouterQ resume model dispatch drifted")
            design = config.get("parent_router_design_contract")
            router_names = [
                name for name, _parameter in model.named_parameters()
                if name.startswith("router.")
            ]
            if not isinstance(design, Mapping) \
                    or design.get("router_parameter_names") != router_names \
                    or design.get("router_parameter_names_sha256") \
                    != _parameter_names_sha256(router_names) \
                    or design.get("router_parameter_tensor_count") \
                    != len(router_names):
                raise ValueError(
                    "ParentRouterQ resume router name/order contract differs"
                )
            validate_parent_router_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                expected_parameter_count=len(model.router_parameters()),
                weight_decay=float(config["weight_decay"]),
            )
        elif args.model == DCF_MODEL_NAME:
            if not isinstance(model, DCFQ):
                raise AssertionError("DCF-Q resume model dispatch drifted")
            design = config.get("dcf_design_contract")
            decoder_names = [
                name for name, _parameter in model.named_parameters()
                if name.startswith("decoder.")
            ]
            if not isinstance(design, Mapping) \
                    or design.get("decoder_parameter_names") != decoder_names \
                    or design.get("decoder_parameter_names_sha256") \
                    != _parameter_names_sha256(decoder_names) \
                    or design.get("decoder_parameter_tensor_count") \
                    != len(decoder_names):
                raise ValueError("DCF-Q resume decoder name/order contract differs")
            validate_dcf_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                expected_parameter_count=len(model.decoder_parameters()),
                weight_decay=float(config["weight_decay"]),
            )
        elif args.model == U1LITE_MODEL_NAME:
            if not isinstance(model, ContinuousScaleBridgeQ):
                raise AssertionError("U1-Lite resume model dispatch drifted")
            design = config.get("u1lite_design_contract")
            names = [name for name, _parameter in model.named_parameters()]
            if not isinstance(design, Mapping) \
                    or design.get("optimizer_parameter_names_sha256") \
                    != _parameter_names_sha256(names) \
                    or design.get("optimizer_parameter_tensor_count") \
                    != len(names) \
                    or design.get("all_parameters_trainable") is not True:
                raise ValueError(
                    "U1-Lite resume parameter name/order contract differs"
                )
            validate_u1lite_optimizer_resume_contract(
                optimizer,
                optimizer_state,
                expected_parameter_count=len(names),
                weight_decay=float(config["weight_decay"]),
            )
        elif args.model == AOM_MODEL_NAME:
            if not isinstance(model, AOMQ):
                raise AssertionError("AOM-Q resume model dispatch drifted")
            design = config.get("aom_design_contract")
            names = [name for name, _p in model.named_parameters()]
            if not isinstance(design, Mapping) \
                    or design.get("optimizer_parameter_names_sha256") \
                    != _parameter_names_sha256(names):
                raise ValueError("AOM-Q resume parameter name/order contract differs")
            optimization = config.get("aom_optimizer_contract")
            if not isinstance(optimization, Mapping):
                raise ValueError("AOM-Q resume optimizer config is missing")
            validate_aom_optimizer_resume_contract(
                optimizer, optimizer_state,
                backbone_lr_multiplier=float(optimization["backbone_lr_multiplier"]),
                weight_decay=float(config["weight_decay"]),
            )
        optimizer.load_state_dict(optimizer_state)
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema_state = checkpoint["ema_state_dict"]
        if args.model == PARENT_ROUTER_MODEL_NAME:
            validate_parent_router_ema_resume_contract(
                ema_state, config, optimizer_updates=update
            )
        elif args.model == DCF_MODEL_NAME:
            validate_dcf_ema_resume_contract(
                ema_state, config, optimizer_updates=update
            )
        ema.load_state_dict(ema_state, optimizer_updates=update)
        if isinstance(ema, ParentRouterEMA):
            ema.validate_bound_model(model)
        elif isinstance(ema, DCFEMA):
            ema.validate_bound_model(model)
        cumulative_wall = float(checkpoint["actual_training_wall_seconds"])
        records = list(checkpoint.get("records", []))

    stop_requested = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prior_sigint = signal.signal(signal.SIGINT, _request_stop)
    status = "running"
    latest_loss: float | None = None
    latest_loss_components: dict[str, float] | None = None
    consecutive_amp_skips = 0
    pause_at_update = getattr(args, "pause_at_update", None)

    def publish_validation() -> None:
        nonlocal records
        raw_validation, ema_validation = evaluate_raw_and_ema(
            model, ema, evaluation_data, device,
            batch_size=evaluation_batch_size,
            amp=args.amp, temporal_mode=args.temporal_mode,
        )
        validation, validation_source, ema_mature = choose_validation_for_selector(
            ema, raw_validation, ema_validation
        )
        run_wall = training_wall.elapsed_seconds
        record = {
            "update": update,
            "run_optimizer_updates": run_updates,
            "actual_training_wall_seconds": cumulative_wall + run_wall,
            "train_loss": latest_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_group_learning_rates": {
                str(group.get("group_name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
            "raw_validation": raw_validation,
            "ema_validation": ema_validation,
            "validation": validation,
            "ema_selector_mature": ema_mature,
            "ema_num_updates": ema.num_updates,
            "validation_metric_sources": {
                "raw_validation": "raw_model",
                "ema_validation": "ema_model",
                "validation": validation_source,
            },
        }
        if latest_loss_components is not None:
            record["train_loss_components"] = dict(latest_loss_components)
        records = [row for row in records if int(row["update"]) != update] + [record]
        records.sort(key=lambda row: int(row["update"]))
        payload = _checkpoint(
            config, model, optimizer, scaler, ema, update=update,
            training_wall=cumulative_wall + run_wall, records=records,
            checkpoint_role="last",
        )
        atomic_torch_save(last_path, payload)
        if int(select_record(records)["update"]) == update:
            best_payload = _checkpoint(
                config, model, optimizer, scaler, ema, update=update,
                training_wall=cumulative_wall + run_wall, records=records,
                checkpoint_role="best",
                best_inference_state_dict_key=selected_state_dict_key(record),
            )
            atomic_torch_save(best_path, best_payload)
        _write_history(
            history_path, config, status="running", update=update,
            run_updates=run_updates, training_wall=cumulative_wall + run_wall,
            run_wall=run_wall, records=records,
        )
        atomic_json(state_path, _state(
            config, status="running", update=update, run_updates=run_updates,
            training_wall=cumulative_wall + run_wall, run_wall=run_wall,
            records=records,
            pause_pending=(
                stop_requested or pause_path.exists()
                or _pause_at_update_reached(update, pause_at_update)
            ),
        ))

    try:
        atomic_json(state_path, _state(
            config, status="running", update=update, run_updates=0,
            training_wall=cumulative_wall, run_wall=0.0, records=records,
        ))
        # A resumed recovery checkpoint may already be beyond a newly supplied
        # runtime boundary.  Re-evaluate and durably save that exact update,
        # then pause without performing another optimizer step.
        if not records or _pause_at_update_reached(update, pause_at_update):
            publish_validation()
        if _pause_at_update_reached(update, pause_at_update):
            status = "paused"
        elif _pause_ready_at_validation(
            stop_requested=stop_requested,
            pause_exists=pause_path.exists(),
            update=update,
            records=records,
        ):
            stop_requested = True
            status = "paused"
        # Evaluation preserves its caller's mode, and this explicit assertion
        # keeps activation-checkpointing/training-only modules active even when
        # resuming a historical checkpoint created by the old loop.
        model.train()
        while status == "running" and update < args.max_updates:
            if _pause_at_update_reached(update, pause_at_update):
                status = "paused"
                break
            if pause_path.exists():
                stop_requested = True
            # Stop before another optimizer update.  If this is not already a
            # validated checkpoint, the common epilogue evaluates the current
            # rolling recovery point before writing the pause receipt.
            if _pause_ready_at_validation(
                stop_requested=stop_requested,
                pause_exists=pause_path.exists(),
                update=update,
                records=records,
            ):
                status = "paused"
                break
            run_wall = training_wall.elapsed_seconds
            if args.max_wall_minutes and run_wall >= args.max_wall_minutes * 60.0:
                status = "wall_time_reached"
                break
            if bool(getattr(args, "calibrated_q_refiner", False)):
                set_q_refiner_backbone_frozen(
                    model,
                    update < int(
                        getattr(args, "q_refiner_freeze_backbone_updates", 0)
                    ),
                )
            elif bool(getattr(args, "calibrated_content_q_pyramid", False)):
                set_content_q_pyramid_core_frozen(
                    model,
                    update < int(
                        getattr(args, "q_pyramid_freeze_core_updates", 0)
                    ),
                )
            elif bool(getattr(args, "calibrated_parent_dct15", False)):
                set_parent_dct15_core_frozen(
                    model,
                    update < int(
                        getattr(args, "dct15_freeze_core_updates", 0)
                    ),
                )
            successful_optimizer_update = False
            with training_wall.measure():
                full = registered_full_scene(update, args.full_scene_start)
                optimizer.zero_grad(set_to_none=True)
                logical_components: dict[str, float] | None = None
                ipmr_scale = (
                    ipmr_auxiliary_scale(update, args.max_updates)
                    if args.model in BAND_MODEL_NAMES
                    and args.model != PCQM_MODEL_NAME else 0.0
                )
                if physical_batch_size == EFFECTIVE_BATCH_SIZE:
                    # Preserve the established unsliced update for models that
                    # qualify at the full physical batch.
                    batch = _device_batch(
                        fit_data.batch(
                            update, EFFECTIVE_BATCH_SIZE, full=full,
                            query_only=(args.temporal_mode == "single"),
                        ),
                        device,
                    )
                    apply_temporal_mode(batch, args.temporal_mode)
                    with _autocast(device, args.amp):
                        if args.model in BAND_MODEL_NAMES:
                            if not isinstance(
                                model, (IPMRQ, IPMRQV2, IPMRQPCQM, DCFQ)
                            ):
                                raise AssertionError("Q-band model binding changed")
                            components = model.forward_components(
                                batch["fine"], batch["coarse_k"], batch["support"],
                                batch["context"], batch["temporal_available"],
                                batch["query_index"],
                            )
                            prediction = components.prediction_k
                            if args.model == PCQM_MODEL_NAME:
                                field_loss = training_objective(
                                    prediction,
                                    batch,
                                    eligible_weight=eligible_loss_weight,
                                )
                                loss = field_loss
                                middle_loss = field_loss.new_zeros(())
                                high_loss = field_loss.new_zeros(())
                            else:
                                loss, field_loss, middle_loss, high_loss = (
                                    training_objective_with_ipmr_bands(
                                        components,
                                        batch,
                                        eligible_weight=eligible_loss_weight,
                                        middle_weight=ipmr_middle_loss_weight,
                                        high_weight=ipmr_high_loss_weight,
                                        auxiliary_scale=ipmr_scale,
                                    )
                                )
                            logical_components = {
                                "field_mse": float(field_loss.detach().cpu()),
                                "q_middle": float(middle_loss.detach().cpu()),
                                "q_high": float(high_loss.detach().cpu()),
                                "auxiliary_scale": float(ipmr_scale),
                            }
                            if isinstance(components, DCFQComponents):
                                logical_components.update({
                                    "dcf_correction_middle_rms_k": float(
                                        components.correction_q_middle_k.detach()
                                        .float().square().mean().sqrt().cpu()
                                    ),
                                    "dcf_correction_high_rms_k": float(
                                        components.correction_q_high_k.detach()
                                        .float().square().mean().sqrt().cpu()
                                    ),
                                    "dcf_q_repair_max_k": float((
                                        components.q_k - components.q_preclosure_k
                                    ).abs().max().detach().cpu()),
                                })
                        elif args.model == PARENT_ROUTER_MODEL_NAME:
                            if not isinstance(model, ParentRouterQ):
                                raise AssertionError("ParentRouterQ binding changed")
                            router_components = model.forward_components(
                                batch["fine"], batch["coarse_k"], batch["support"],
                                batch["context"], batch["temporal_available"],
                                batch["query_index"],
                            )
                            prediction = router_components.prediction_k
                            gate = router_components.gate40.detach()
                            logical_components = {
                                "router_gate_mean": float(gate.mean().cpu()),
                                "router_gate_abs_shift": float(
                                    (gate - 0.5).abs().mean().cpu()
                                ),
                                "router_gate_near_bound_fraction": float(
                                    ((gate - 0.5).abs() >= 0.24).float().mean().cpu()
                                ),
                                "router_q_repair_max_k": float(
                                    (
                                        router_components.q_k
                                        - router_components.q_preclosure_k
                                    ).abs().max().detach().cpu()
                                ),
                            }
                        else:
                            prediction = model(
                                batch["fine"], batch["coarse_k"], batch["support"],
                                batch["context"], batch["temporal_available"],
                                batch["query_index"],
                            )
                        if args.model in BAND_MODEL_NAMES:
                            pass
                        elif q_shape_loss_weight == 0.0:
                            loss = training_objective(
                                prediction, batch,
                                eligible_weight=eligible_loss_weight,
                            )
                        else:
                            loss, pixel_loss, shape_loss = (
                                training_objective_with_q_shape(
                                    prediction,
                                    batch,
                                    eligible_weight=eligible_loss_weight,
                                    q_shape_weight=q_shape_loss_weight,
                                )
                            )
                            logical_components = {
                                "pixel_mse": float(pixel_loss.detach().cpu()),
                                "q_shape": float(shape_loss.detach().cpu()),
                            }
                    scaler.scale(loss).backward()
                    logical_loss = float(loss.detach().cpu())
                else:
                    # Fine52 and the continuous fine-grid candidate do not
                    # safely fit full scenes as physical batch32.  batch_slice
                    # retains the configured logical schedule (registered
                    # equal-region batches rotate 11/11/10 across US/China/
                    # Europe) and supplies weights whose accumulated sum is
                    # exactly the region-equal objective, for Core22 and
                    # multisource alike.
                    logical_loss = 0.0
                    if args.model in BAND_MODEL_NAMES:
                        logical_components = {
                            "field_mse": 0.0,
                            "q_middle": 0.0,
                            "q_high": 0.0,
                            "auxiliary_scale": float(ipmr_scale),
                        }
                        if args.model == DCF_MODEL_NAME:
                            logical_components.update({
                                "dcf_correction_middle_rms_k": 0.0,
                                "dcf_correction_high_rms_k": 0.0,
                                "dcf_q_repair_max_k": 0.0,
                            })
                    elif args.model == PARENT_ROUTER_MODEL_NAME:
                        logical_components = {
                            "router_gate_mean": 0.0,
                            "router_gate_abs_shift": 0.0,
                            "router_gate_near_bound_fraction": 0.0,
                            "router_q_repair_max_k": 0.0,
                        }
                    elif q_shape_loss_weight > 0.0:
                        logical_components = {"pixel_mse": 0.0, "q_shape": 0.0}
                    for start in range(0, EFFECTIVE_BATCH_SIZE, physical_batch_size):
                        stop = min(start + physical_batch_size, EFFECTIVE_BATCH_SIZE)
                        batch = _device_batch(
                            fit_data.batch_slice(
                                update, start, stop, full=full,
                                query_only=(args.temporal_mode == "single"),
                            ),
                            device,
                        )
                        apply_temporal_mode(batch, args.temporal_mode)
                        with _autocast(device, args.amp):
                            if args.model in BAND_MODEL_NAMES:
                                if not isinstance(
                                    model, (IPMRQ, IPMRQV2, IPMRQPCQM, DCFQ)
                                ):
                                    raise AssertionError("Q-band model binding changed")
                                components = model.forward_components(
                                    batch["fine"], batch["coarse_k"], batch["support"],
                                    batch["context"], batch["temporal_available"],
                                    batch["query_index"],
                                )
                                prediction = components.prediction_k
                                if args.model == PCQM_MODEL_NAME:
                                    field_loss = training_objective(
                                        prediction,
                                        batch,
                                        eligible_weight=eligible_loss_weight,
                                        sample_loss_weight=batch[
                                            "sample_loss_weight"
                                        ],
                                    )
                                    slice_loss = field_loss
                                    middle_loss = field_loss.new_zeros(())
                                    high_loss = field_loss.new_zeros(())
                                else:
                                    slice_loss, field_loss, middle_loss, high_loss = (
                                        training_objective_with_ipmr_bands(
                                            components,
                                            batch,
                                            eligible_weight=eligible_loss_weight,
                                            middle_weight=ipmr_middle_loss_weight,
                                            high_weight=ipmr_high_loss_weight,
                                            auxiliary_scale=ipmr_scale,
                                            sample_loss_weight=batch[
                                                "sample_loss_weight"
                                            ],
                                        )
                                    )
                                if logical_components is None:
                                    raise AssertionError("IPMR components were not initialized")
                                logical_components["field_mse"] += float(
                                    field_loss.detach().cpu()
                                )
                                logical_components["q_middle"] += float(
                                    middle_loss.detach().cpu()
                                )
                                logical_components["q_high"] += float(
                                    high_loss.detach().cpu()
                                )
                                if isinstance(components, DCFQComponents):
                                    slices = math.ceil(
                                        EFFECTIVE_BATCH_SIZE / physical_batch_size
                                    )
                                    logical_components[
                                        "dcf_correction_middle_rms_k"
                                    ] += float(
                                        components.correction_q_middle_k.detach()
                                        .float().square().mean().sqrt().cpu()
                                    ) / slices
                                    logical_components[
                                        "dcf_correction_high_rms_k"
                                    ] += float(
                                        components.correction_q_high_k.detach()
                                        .float().square().mean().sqrt().cpu()
                                    ) / slices
                                    logical_components["dcf_q_repair_max_k"] = max(
                                        logical_components["dcf_q_repair_max_k"],
                                        float((
                                            components.q_k
                                            - components.q_preclosure_k
                                        ).abs().max().detach().cpu()),
                                    )
                            elif args.model == PARENT_ROUTER_MODEL_NAME:
                                if not isinstance(model, ParentRouterQ):
                                    raise AssertionError("ParentRouterQ binding changed")
                                router_components = model.forward_components(
                                    batch["fine"], batch["coarse_k"],
                                    batch["support"], batch["context"],
                                    batch["temporal_available"], batch["query_index"],
                                )
                                prediction = router_components.prediction_k
                                if logical_components is None:
                                    raise AssertionError(
                                        "ParentRouterQ diagnostics were not initialized"
                                    )
                                slices = math.ceil(
                                    EFFECTIVE_BATCH_SIZE / physical_batch_size
                                )
                                gate = router_components.gate40.detach()
                                logical_components["router_gate_mean"] += float(
                                    gate.mean().cpu()
                                ) / slices
                                logical_components[
                                    "router_gate_abs_shift"
                                ] += float(
                                    (gate - 0.5).abs().mean().cpu()
                                ) / slices
                                logical_components[
                                    "router_gate_near_bound_fraction"
                                ] += float(
                                    ((gate - 0.5).abs() >= 0.24)
                                    .float().mean().cpu()
                                ) / slices
                                logical_components[
                                    "router_q_repair_max_k"
                                ] = max(
                                    logical_components["router_q_repair_max_k"],
                                    float((
                                        router_components.q_k
                                        - router_components.q_preclosure_k
                                    ).abs().max().detach().cpu()),
                                )
                            else:
                                prediction = model(
                                    batch["fine"], batch["coarse_k"], batch["support"],
                                    batch["context"], batch["temporal_available"],
                                    batch["query_index"],
                                )
                            if args.model in BAND_MODEL_NAMES:
                                pass
                            elif q_shape_loss_weight == 0.0:
                                slice_loss = training_objective(
                                    prediction, batch,
                                    eligible_weight=eligible_loss_weight,
                                    sample_loss_weight=batch["sample_loss_weight"],
                                )
                            else:
                                slice_loss, pixel_loss, shape_loss = (
                                    training_objective_with_q_shape(
                                        prediction,
                                        batch,
                                        eligible_weight=eligible_loss_weight,
                                        q_shape_weight=q_shape_loss_weight,
                                        sample_loss_weight=batch["sample_loss_weight"],
                                    )
                                )
                                if logical_components is None:  # defensive
                                    raise AssertionError("shape components were not initialized")
                                logical_components["pixel_mse"] += float(
                                    pixel_loss.detach().cpu()
                                )
                                logical_components["q_shape"] += float(
                                    shape_loss.detach().cpu()
                                )
                        scaler.scale(slice_loss).backward()
                        logical_loss += float(slice_loss.detach().cpu())
                next_update = update + 1
                lr = learning_rate(
                    next_update, args.max_updates, args.learning_rate,
                    int(getattr(args, "warmup_updates", WARMUP_UPDATES)),
                )
                set_optimizer_group_learning_rates(optimizer, lr)
                (
                    optimizer_stepped,
                    consecutive_amp_skips,
                    _grad_norm,
                ) = optimizer_step_with_finite_guard(
                    model,
                    optimizer,
                    scaler,
                    consecutive_amp_skips=consecutive_amp_skips,
                )
                if optimizer_stepped:
                    ema.update(model)
                    update = next_update
                    run_updates += 1
                    latest_loss = logical_loss
                    latest_loss_components = logical_components
                    successful_optimizer_update = True
            run_wall = training_wall.elapsed_seconds
            if successful_optimizer_update \
                    and update % STATE_HEARTBEAT_INTERVAL_UPDATES == 0:
                # Keep ``last.pt`` aligned with the durable heartbeat rather
                # than waiting up to an entire validation interval.  This is
                # a rolling recovery checkpoint only: validation records and
                # best-model selection remain unchanged.  Saving it before
                # state.json ensures the heartbeat never advertises an update
                # that cannot be resumed after a crash or guarded numerical
                # stop.
                recovery_payload = _checkpoint(
                    config, model, optimizer, scaler, ema, update=update,
                    training_wall=cumulative_wall + run_wall,
                    records=records, checkpoint_role="last",
                )
                atomic_torch_save(last_path, recovery_payload)
                _write_state_heartbeat(
                    state_path, config,
                    update=update,
                    run_updates=run_updates,
                    training_wall=cumulative_wall + run_wall,
                    run_wall=run_wall,
                    records=records,
                    pause_pending=stop_requested or pause_path.exists(),
                )
            runtime_pause_reached = (
                successful_optimizer_update
                and _pause_at_update_reached(update, pause_at_update)
            )
            if successful_optimizer_update and (
                update % args.validation_interval == 0 or runtime_pause_reached
            ):
                publish_validation()
                if runtime_pause_reached:
                    status = "paused"
                    break
                # Re-read the file after evaluation: a request may arrive
                # while validation is running, after the loop-top check.
                if stop_requested or pause_path.exists():
                    stop_requested = True
                    status = "paused"
                    break
                if automatic_early_stop_enabled(args.model) \
                        and update >= max(
                            3_000,
                            (EARLY_STOP_INTERVALS + 1) * args.validation_interval,
                        ) \
                        and early_stop_plateau(records):
                    status = "early_stopped_plateau"
                    break
        if not records or int(records[-1]["update"]) != update:
            publish_validation()
        if status == "running":
            status = "max_updates"
    except BaseException:
        run_wall = training_wall.elapsed_seconds
        atomic_json(state_path, _state(
            config, status="failed", update=update, run_updates=run_updates,
            training_wall=cumulative_wall + run_wall, run_wall=run_wall,
            records=records, pause_pending=stop_requested,
        ))
        raise
    finally:
        signal.signal(signal.SIGINT, prior_sigint)

    run_wall = training_wall.elapsed_seconds
    final_wall = cumulative_wall + run_wall
    payload = _checkpoint(
        config, model, optimizer, scaler, ema, update=update,
        training_wall=final_wall, records=records, checkpoint_role="last",
    )
    atomic_torch_save(last_path, payload)
    _write_history(
        history_path, config, status=status, update=update,
        run_updates=run_updates, training_wall=final_wall,
        run_wall=run_wall, records=records,
    )
    final_state = _state(
        config, status=status, update=update, run_updates=run_updates,
        training_wall=final_wall, run_wall=run_wall, records=records,
        pause_pending=False,
    )
    atomic_json(state_path, final_state)
    if status == "paused":
        atomic_json(output / "pause_receipt.json", {
            "schema_version": "uhi-cdc-g246-r2-pause-v1",
            "status": "ready_to_resume",
            "optimizer_updates": update,
            "best_update": final_state["best_update"],
            "runtime_pause_at_update": (
                pause_at_update
                if _pause_at_update_reached(update, pause_at_update) else None
            ),
            "created_utc": _utc_now(),
            "locked_test_opened": False,
        })
        pause_path.unlink(missing_ok=True)
    return final_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-receipt", type=Path, default=DEFAULT_SPLIT_RECEIPT)
    parser.add_argument("--role", default="fit+validation", choices=("fit+validation",))
    parser.add_argument(
        "--model",
        choices=(
            "ocnir_control", "qparent", "contrast_q", "calibrated_q",
            "ipmr_q", "ipmr_q_v2", PCQM_MODEL_NAME,
            PARENT_ROUTER_MODEL_NAME, DCF_MODEL_NAME,
            AOM_MODEL_NAME, U1LITE_MODEL_NAME,
        ),
        required=True,
    )
    parser.add_argument("--width", type=int, default=48)
    contrast_d4 = parser.add_mutually_exclusive_group()
    contrast_d4.add_argument(
        "--contrast-d4-average",
        action="store_true",
        help="enable exact eight-orientation D4 averaging for ContrastQParent",
    )
    contrast_d4.add_argument(
        "--no-contrast-d4-average",
        action="store_true",
        help="explicitly select the fast non-averaged ContrastQ screening path",
    )
    parser.add_argument(
        "--contrast-activation-checkpointing",
        action="store_true",
        help="checkpoint ContrastQ core activations; disabled by default for patch96",
    )
    parser.add_argument(
        "--calibrated-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("checkpoint SceneCalibratedContinuousQ core activations; defaults "
              "to enabled and is recorded explicitly in its config"),
    )
    parser.add_argument(
        "--calibrated-physical-batch-size",
        type=int,
        default=0,
        help=("physical microbatch in [1,32] for calibrated_q while preserving "
              "logical region-equal batch32; 0 selects Core22 S=2, Core22 M/L=1, "
              "multisource=1"),
    )
    parser.add_argument(
        "--ipmr-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("checkpoint the IPMR-Q full multiresolution proposal route; "
              "defaults to enabled and is recorded in the scientific config"),
    )
    parser.add_argument(
        "--ipmr-physical-batch-size",
        type=int,
        default=0,
        help=("physical microbatch in [1,32] for IPMR-Q while preserving the "
              "logical region-equal batch32; 0 selects the qualified value 1"),
    )
    parser.add_argument(
        "--ipmr-middle-loss-weight",
        type=float,
        default=None,
        help=("IPMR QM deep-supervision weight; omission uses the registered "
              "0.05 value (PCQM requires and defaults to 0)"),
    )
    parser.add_argument(
        "--ipmr-high-loss-weight",
        type=float,
        default=None,
        help=("IPMR QH deep-supervision weight; omission uses the registered "
              "0.025 value (PCQM requires and defaults to 0)"),
    )
    parser.add_argument(
        "--parent-router-shallow-checkpoint",
        type=Path,
        help=("registered r6a Context15 best checkpoint; required only for a "
              "fresh parent_router_q run and forbidden on resume"),
    )
    parser.add_argument(
        "--parent-router-deep-checkpoint",
        type=Path,
        help=("registered r9 IPMR-Q best checkpoint; required only for a fresh "
              "parent_router_q run and forbidden on resume"),
    )
    parser.add_argument(
        "--parent-router-physical-batch-size",
        type=int,
        default=0,
        help=("physical microbatch in [1,32] for the two frozen experts plus "
              "router; 0 selects the conservative qualified value 8"),
    )
    parser.add_argument(
        "--dcf-shallow-checkpoint",
        type=Path,
        help=("registered r6a Context15 best checkpoint; required only for a "
              "fresh dcf_q run and forbidden on resume"),
    )
    parser.add_argument(
        "--dcf-deep-checkpoint",
        type=Path,
        help=("registered r9 IPMR-Q best checkpoint; required only for a fresh "
              "dcf_q run and forbidden on resume"),
    )
    parser.add_argument(
        "--dcf-physical-batch-size",
        type=int,
        default=0,
        help=("physical microbatch in [1,32] for frozen dual experts plus the "
              "cross-decoder; 0 selects the conservative value 2"),
    )
    parser.add_argument(
        "--dcf-middle-loss-weight",
        type=float,
        default=None,
        help="DCF-Q total-QM deep-supervision weight; omission uses 0.05",
    )
    parser.add_argument(
        "--dcf-high-loss-weight",
        type=float,
        default=None,
        help="DCF-Q total-QH deep-supervision weight; omission uses 0.025",
    )
    parser.add_argument("--aom-shallow-checkpoint", type=Path,
                        help="registered r6a raw u2000 source; fresh aom_q only")
    parser.add_argument("--aom-deep-checkpoint", type=Path,
                        help="registered r9 EMA u8000 source; fresh aom_q only")
    parser.add_argument("--aom-physical-batch-size", type=int, default=0,
                        help="AOM-Q physical microbatch; 0 selects 1")
    parser.add_argument("--aom-backbone-lr-multiplier", type=float, default=0.1,
                        help="native r6a/r9 LR multiplier; registered value 0.1")
    parser.add_argument(
        "--u1lite-distilled-checkpoint",
        type=Path,
        help=("independent qualified-D1 student artifact; required only for a "
              "fresh u1lite_q target run and forbidden on resume"),
    )
    parser.add_argument(
        "--u1lite-direct-r6a-checkpoint",
        type=Path,
        help=("registered r6a raw-u2000 exact-u0 source; alternative to the "
              "qualified-D1 source on fresh u1lite_q and forbidden on resume"),
    )
    parser.add_argument(
        "--u1lite-dexchange",
        action="store_true",
        help=("replace only the U1 model body with the lightweight D-Exchange "
              "candidate; requires the direct r6a source on a fresh run"),
    )
    parser.add_argument(
        "--u1lite-physical-batch-size",
        type=int,
        default=0,
        help=("U1-Lite target physical microbatch in [1,32] while preserving "
              "logical region-equal batch32; 0 selects conservative value 1"),
    )
    parser.add_argument(
        "--calibrated-allocation-adapter",
        action="store_true",
        help=("explicitly enable allocation-v2: detached old-Q ranking plus a "
              "shared delta score and zero-initialized parent-local amplitudes; "
              "omitted preserves the calibrated-Q module/state dictionary exactly"),
    )
    parser.add_argument(
        "--calibrated-t3-fusion",
        action="store_true",
        help=("explicitly enable query-relative, availability-masked, "
              "permutation-invariant T3 residual fusion with independently "
              "zero-initialized fine/parent/scene gates; requires temporal-mode multi"),
    )
    parser.add_argument(
        "--calibrated-q-refiner",
        action="store_true",
        help=("explicitly enable the identity-initialized D4-equivariant "
              "learned-Q residual refiner; requires Fine52 multisource input"),
    )
    parser.add_argument(
        "--q-refiner-freeze-backbone-updates",
        type=int,
        default=0,
        help=("freeze every non-refiner parameter for this many successful "
              "global optimizer updates; 0 trains jointly from update one"),
    )
    parser.add_argument(
        "--q-refiner-core-lr-multiplier",
        type=float,
        default=1.0,
        help="positive multiplier on the scheduled backbone learning rate",
    )
    parser.add_argument(
        "--q-refiner-lr-multiplier",
        type=float,
        default=1.0,
        help="positive multiplier on the scheduled Q-refiner learning rate",
    )
    parser.add_argument(
        "--calibrated-content-q-pyramid",
        action="store_true",
        help=("enable the zero-head-initialized content-only 160/80/40 Q-Pyramid; "
              "this no-geolocation rebase requires a bare-r2k initialization "
              "checkpoint and masks Context19 latitude/longitude before the core"),
    )
    parser.add_argument(
        "--q-pyramid-freeze-core-updates",
        type=int,
        default=0,
        help=("freeze every inherited r2k core parameter for this many successful "
              "global optimizer updates"),
    )
    parser.add_argument(
        "--q-pyramid-core-lr-multiplier",
        type=float,
        default=1.0,
        help="positive multiplier on the scheduled inherited-core learning rate",
    )
    parser.add_argument(
        "--q-pyramid-lr-multiplier",
        type=float,
        default=1.0,
        help="positive multiplier on the scheduled Content-only Q-Pyramid learning rate",
    )
    parser.add_argument(
        "--calibrated-parent-dct15",
        action="store_true",
        help=("enable the zero-head-initialized content-only Parent-DCT15 decoder; "
              "requires a bare-r2k initialization and masks Context19 "
              "latitude/longitude before the core"),
    )
    parser.add_argument(
        "--calibrated-no-geo-core",
        action="store_true",
        help=("physically remove Context19 latitude/longitude columns from the "
              "calibrated encoder/core; requires a bare-r2k initialization and "
              "an explicit rebase mode"),
    )
    parser.add_argument(
        "--no-geo-rebase-mode",
        choices=("surgery", "reset_conditioning"),
        default=None,
        help=("surgery deletes only the four descriptor columns; "
              "reset_conditioning freshly initializes the registered "
              "conditioning branches; valid only with --calibrated-no-geo-core"),
    )
    parser.add_argument(
        "--no-geo-probe",
        choices=("head_conditioning", "encoder_backbone"),
        default=None,
        help=("strict frozen H/B probe for a surgery-rebased Context15 model; "
              "omission trains the existing full model unchanged"),
    )
    parser.add_argument(
        "--no-geo-pack-arm",
        choices=("legacy_reset_control", "hierarchical"),
        default=None,
        help=("matched full-training pack screen: reinitialize the legacy pack "
              "control or replace it with the hierarchical H/H2/H4 candidate; "
              "omission preserves every existing run"),
    )
    parser.add_argument(
        "--dct15-freeze-core-updates",
        type=int,
        default=0,
        help=("freeze every inherited r2k core parameter for this many successful "
              "global optimizer updates; first screening suggestion is 500, but "
              "the default 0 preserves explicit opt-in semantics"),
    )
    parser.add_argument(
        "--dct15-core-lr-multiplier",
        type=float,
        default=1.0,
        help=("positive multiplier on the scheduled inherited-core learning rate; "
              "first screening suggestion is 0.25"),
    )
    parser.add_argument(
        "--dct15-lr-multiplier",
        type=float,
        default=1.0,
        help=("positive multiplier on the scheduled Parent-DCT15 learning rate; "
              "first screening suggestion is 2.0"),
    )
    parser.add_argument("--temporal-mode", choices=("single", "multi"), required=True)
    parser.add_argument(
        "--data-scope", choices=("internal_dev", "public_validation"),
        default="internal_dev",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-updates", type=int, default=WARMUP_UPDATES,
        help="linear warm-up length; registered default is 2000 updates",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY,
        help="AdamW weight decay; registered default is 1e-4",
    )
    parser.add_argument(
        "--region-sampling",
        choices=("equal_region", "dataset_proportional"),
        default="equal_region",
        help=("training-city sampling contract; equal_region is registered, "
              "dataset_proportional exists only for legacy checkpoint resume"),
    )
    parser.add_argument(
        "--eligible-loss-weight", type=float, default=PRIMARY_LOSS_WEIGHT,
        help=("weight of valid-and-eligible MSE in the registered objective; "
              "the remaining weight is all-valid MSE"),
    )
    parser.add_argument(
        "--allow-nonstandard-loss-objective",
        action="store_true",
        help=("explicitly authorize a new ablation with a loss other than the "
              "registered 0.8 eligible-valid + 0.2 all-valid objective; resume of "
              "a legacy checkpoint remains compatible without this flag"),
    )
    parser.add_argument(
        "--q-shape-loss-weight",
        type=float,
        default=0.0,
        help=("optional convex weight in [0,0.15] for two-scale query-Q "
              "gradient/high-pass matching; default 0 preserves the legacy "
              "pixel-only path, suggested first gate 0.05"),
    )
    parser.add_argument("--max-updates", type=int, default=12_000)
    parser.add_argument(
        "--pause-at-update",
        type=int,
        default=None,
        help=("runtime-only successful optimizer-update boundary; validate, save, "
              "and pause exactly at N without changing max_updates or the "
              "scientific configuration"),
    )
    parser.add_argument(
        "--full-scene-start", type=int, default=FULL_SCENE_START,
        help=("patch-only before this global update; from here use registered "
              "patch96,patch96,patch96,full160 cadence; the registered default "
              "is 0, enabling the 3:1 mixture immediately"),
    )
    parser.add_argument(
        "--validation-interval", type=int, default=VALIDATION_INTERVAL,
        help="registered validation cadence is every 2000 optimizer updates",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=("fresh-run weights-only initialization from a compatible public "
              "Fine52 calibrated-Q extension or IPMR-Q anchor checkpoint; never "
              "imports optimizer, scaler, "
              "EMA, updates, scheduler phase, wall time, or records"),
    )
    parser.add_argument(
        "--init-weight-source",
        choices=("selected", "raw", "ema"),
        default="selected",
        help=("raw/EMA inference weights to migrate; selected follows the source "
              "best-checkpoint weight contract"),
    )
    parser.add_argument(
        "--input-mode", choices=("core22", "multisource"), default="core22",
        help=("core22 preserves the established input; multisource explicitly enables "
              "Stage-C Fine52/Context19"),
    )
    parser.add_argument("--texture-manifest", type=Path)
    parser.add_argument(
        "--weather-manifest", type=Path,
        help="complete immutable NASA POWER v1 manifest; Stage-C only",
    )
    parser.add_argument(
        "--multisource-physical-batch-size", type=int, default=0,
        help=("Stage-C physical microbatch in [1,32]; 0 selects 4 for qparent "
              "and 2 for contrast_q while preserving logical batch 32"),
    )
    parser.add_argument(
        "--allow-provisional-optical", action="store_true",
        help=("authorize target-QA-conditioned stored optical for internal engineering "
              "smoke only; never valid for formal/public validation"),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
