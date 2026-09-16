"""Target-free parent-local fusion of two frozen, no-coordinate Q experts.

The formal G246 observation is one support-weighted value per 4x4 fine-grid
parent.  Both source experts already return a delivered residual in the
corresponding nullspace.  A spatially varying mixture would normally destroy
that property; this model instead predicts exactly one convex weight per 4x4
parent and repeats it over the parent's fine pixels.  The mixture therefore
remains in Q on every observed parent, followed by the registered projection
only as a fail-closed numerical repair.

The router never receives targets, masks derived from targets, city/region
identity, or the four public latitude/longitude context columns.  Its two
experts are immutable feature generators and are kept in eval/no-grad mode for
the complete lifetime of the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from g246_calibrated_continuous_q import SceneCalibratedContinuousQ
from g246_ipmr_q import IPMRQ
from g246_q_bands import support_block_mean
from ocnir import support_project


__all__ = [
    "ParentRouterQComponents",
    "ParentRouterQ",
]


_PARENT = 4
_FINE_CHANNELS = 52
_CONTEXT_DIM = 19
_CONTEXT15_DIM = 15


def _contains_true(value: Tensor) -> bool:
    return bool(torch.any(value).detach().cpu().item())


def _binary(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if not value.is_floating_point() and value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} must be boolean or numeric binary")
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


class _ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels only, independent of crop height/width."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(value.movedim(1, -1)).movedim(-1, 1)


class _ParentRouter(nn.Module):
    """Low-capacity shared 1x1 parent classifier with no spatial template."""

    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.project = nn.Conv2d(input_channels, width, 1, bias=False)
        self.norm1 = _ChannelLayerNorm2d(width)
        self.hidden = nn.Conv2d(width, width, 1, bias=False)
        self.norm2 = _ChannelLayerNorm2d(width)
        self.logit = nn.Conv2d(width, 1, 1)
        nn.init.kaiming_normal_(
            self.project.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.hidden.weight, mode="fan_out", nonlinearity="relu"
        )
        # A fresh formal run must reproduce the fixed 0.5 anchor exactly.
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = F.silu(self.norm1(self.project(features)), inplace=False)
        hidden = F.silu(self.norm2(self.hidden(hidden)), inplace=False)
        logits = self.logit(hidden)
        # The conservative [.25,.75] range retains the measured useful local
        # oracle gap while preventing sparse-parent winner-take-all fitting.
        # Softsign has polynomial rather than tanh/sigmoid tail gradients and
        # avoids recreating the saturated gate observed in IPMR-Q v1.
        weights = 0.5 + 0.25 * F.softsign(logits)
        return weights, logits


class ParentRouterQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_k: Tensor
    q_preclosure_k: Tensor
    shallow_q_k: Tensor
    deep_q_k: Tensor
    gate40: Tensor
    gate_logits40: Tensor


class ParentRouterQ(nn.Module):
    """Fuse frozen r6a and r9 experts using one target-free weight per parent."""

    schema_version = "g246-parent-router-q-v1"
    band_contract = "parent-constant convex blend in Q=I-P40"
    context_contract = "stored Context19; learned Context15; indices 5:9 removed"
    geolocation_context_indices = (5, 6, 7, 8)
    gate_range = (0.25, 0.75)
    expert_names = ("r6a_context15_continuous", "r9_ipmr_q")

    # mean52 + within-parent RMS52 + Context15 + seven expert-disagreement
    # statistics + coarse-z/valid/support-fraction/base-minus-coarse.
    router_input_channels = 2 * _FINE_CHANNELS + _CONTEXT15_DIM + 7 + 4

    def __init__(
        self,
        *,
        fine_channels: int = _FINE_CHANNELS,
        context_dim: int = _CONTEXT_DIM,
        width: int = 48,
        router_width: int = 48,
    ) -> None:
        super().__init__()
        if fine_channels != _FINE_CHANNELS or context_dim != _CONTEXT_DIM:
            raise ValueError(
                "ParentRouterQ requires the exact Fine52/Context19 contract"
            )
        if width != 48:
            raise ValueError("ParentRouterQ source experts require registered width48")
        if isinstance(router_width, bool) or not isinstance(router_width, int) \
                or router_width < 8:
            raise ValueError("router_width must be an integer >= 8")
        self.fine_channels = int(fine_channels)
        self.context_dim = int(context_dim)
        self.width = int(width)
        self.router_width = int(router_width)

        self.shallow_expert = SceneCalibratedContinuousQ(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=False,
            no_geo_core=True,
        )
        self.deep_expert = IPMRQ(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=False,
        )
        self.router = _ParentRouter(self.router_input_channels, router_width)
        self.register_buffer(
            "experts_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self._freeze_experts()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def router_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.router.parameters())
        if not parameters or any(not value.requires_grad for value in parameters):
            raise RuntimeError("router trainable-parameter contract drifted")
        return parameters

    def _freeze_experts(self) -> None:
        for expert in (self.shallow_expert, self.deep_expert):
            expert.eval()
            for parameter in expert.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ParentRouterQ":
        super().train(mode)
        # ``nn.Module.train`` recurses into children, so restore the immutable
        # expert state after every caller transition.
        self._freeze_experts()
        self.router.train(mode)
        return self

    def load_expert_state_dicts(
        self,
        shallow_state: Mapping[str, Tensor],
        deep_state: Mapping[str, Tensor],
    ) -> None:
        """Strictly install the two independently selected inference states."""

        if bool(self.experts_initialized.detach().cpu().item()):
            raise RuntimeError("ParentRouterQ experts may be initialized only once")
        self.shallow_expert.load_state_dict(shallow_state, strict=True)
        self.deep_expert.load_state_dict(deep_state, strict=True)
        self.experts_initialized.fill_(True)
        self._freeze_experts()

    @staticmethod
    def _context15(context19: Tensor) -> Tensor:
        if context19.shape[-1] != _CONTEXT_DIM:
            raise ValueError("ParentRouterQ physical surgery requires Context19")
        return torch.cat((context19[..., :5], context19[..., 9:]), dim=-1)

    @staticmethod
    def _parent_rms(
        field: Tensor,
        support: Tensor,
        parent_mean: Tensor,
        active: Tensor,
    ) -> Tensor:
        lifted = parent_mean.repeat_interleave(_PARENT, -2).repeat_interleave(
            _PARENT, -1
        )
        centered = (field - lifted) * support.to(dtype=field.dtype)
        variance, _ = support_block_mean(centered.square(), support, _PARENT)
        rms = torch.sqrt(variance.clamp_min(0.0) + 1.0e-8)
        return rms * active.to(dtype=rms.dtype)

    @staticmethod
    def _expert_parent_statistics(
        shallow_q: Tensor,
        deep_q: Tensor,
        support: Tensor,
    ) -> Tensor:
        # Kelvin residuals are naturally O(1); division by two keeps all
        # disagreement statistics on the scale of normalized physical inputs.
        shallow = shallow_q / 2.0
        deep = deep_q / 2.0
        delta = shallow - deep
        fields: Sequence[Tensor] = (
            shallow.square(),
            deep.square(),
            delta.square(),
            shallow.abs(),
            deep.abs(),
            delta.abs(),
            shallow * deep,
        )
        return torch.cat(
            [support_block_mean(value, support, _PARENT)[0] for value in fields],
            dim=1,
        )

    def _router_features(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context15: Tensor,
        shallow_q: Tensor,
        deep_q: Tensor,
    ) -> Tensor:
        normalized_fine = torch.cat(
            (((fine[:, :1].float() - 300.0) / 20.0), fine[:, 1:].float()),
            dim=1,
        )
        support_bool = _binary(support, "query support")
        fine_mean, active = support_block_mean(
            normalized_fine, support_bool, _PARENT
        )
        fine_rms = self._parent_rms(
            normalized_fine, support_bool, fine_mean, active
        )
        expert_stats = self._expert_parent_statistics(
            shallow_q.float(), deep_q.float(), support_bool
        )
        support_fraction = F.avg_pool2d(
            support_bool.to(dtype=torch.float32), _PARENT, _PARENT
        )
        coarse_valid = torch.isfinite(coarse_k)
        coarse_safe = torch.where(
            coarse_valid,
            coarse_k.float(),
            torch.full_like(coarse_k.float(), 300.0),
        )
        coarse_z = (coarse_safe - 300.0) / 20.0
        base_minus_coarse = fine_mean[:, :1] - coarse_z
        context_map = context15.float()[..., None, None].expand(
            -1, -1, coarse_k.shape[-2], coarse_k.shape[-1]
        )
        features = torch.cat(
            (
                fine_mean,
                fine_rms,
                context_map,
                expert_stats,
                coarse_z,
                coarse_valid.to(dtype=torch.float32),
                support_fraction,
                base_minus_coarse,
            ),
            dim=1,
        )
        if features.shape[1] != self.router_input_channels:
            raise AssertionError("ParentRouterQ feature contract drifted")
        if _contains_true(~torch.isfinite(features)):
            raise FloatingPointError("ParentRouterQ router features are non-finite")
        return features

    def _validate_public_inputs(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not bool(self.experts_initialized.detach().cpu().item()):
            raise RuntimeError("ParentRouterQ experts have not been initialized")
        if fine.ndim != 5 or fine.shape[1:3] != (1, self.fine_channels):
            raise ValueError("fine must have shape [B,1,52,H,W]")
        batch, _, _, height, width = fine.shape
        if height % 16 or width % 16:
            raise ValueError("fine geometry must be divisible by sixteen")
        if coarse_k.shape != (batch, 1, 1, height // 4, width // 4):
            raise ValueError("coarse_k must have shape [B,1,1,H/4,W/4]")
        if support.shape != (batch, 1, 1, height, width):
            raise ValueError("support must have shape [B,1,1,H,W]")
        if context.shape != (batch, 1, self.context_dim):
            raise ValueError("context must have shape [B,1,19]")
        if temporal_available.shape != (batch, 1):
            raise ValueError("temporal_available must have shape [B,1]")
        if query_index.shape != (batch,) or _contains_true(query_index != 0):
            raise ValueError("single-date query_index must be zero")
        query_support = _binary(support[:, 0], "support")
        if _contains_true(~_binary(temporal_available, "temporal_available")):
            raise ValueError("the query date must be available")
        return fine[:, 0], coarse_k[:, 0], query_support, context[:, 0]

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> ParentRouterQComponents:
        query_fine, query_coarse, query_support, query_context = (
            self._validate_public_inputs(
                fine,
                coarse_k,
                support,
                context,
                temporal_available,
                query_index,
            )
        )
        # The immutable experts cannot accumulate gradients or train-mode
        # running state through the router run.
        with torch.no_grad():
            shallow = self.shallow_expert.forward_components(
                fine,
                coarse_k,
                support,
                context,
                temporal_available,
                query_index,
            )
            deep = self.deep_expert.forward_components(
                fine,
                coarse_k,
                support,
                context,
                temporal_available,
                query_index,
            )
        shallow_q = shallow.q_k.float()
        deep_q = deep.q_k.float()
        features = self._router_features(
            query_fine,
            query_coarse,
            query_support,
            self._context15(query_context),
            shallow_q,
            deep_q,
        )
        gate40, gate_logits40 = self.router(features)
        gate160 = gate40.repeat_interleave(_PARENT, -2).repeat_interleave(
            _PARENT, -1
        )
        q_preclosure = (
            gate160.float() * shallow_q
            + (1.0 - gate160.float()) * deep_q
        )
        coarse_valid = torch.isfinite(query_coarse)
        q_k = support_project(
            q_preclosure,
            torch.zeros_like(query_coarse, dtype=q_preclosure.dtype),
            query_support,
            coarse_valid,
        ) * query_support.to(dtype=q_preclosure.dtype)
        base_k = support_project(query_fine[:, :1], query_coarse, query_support)
        return ParentRouterQComponents(
            prediction_k=base_k.float() + q_k,
            base_k=base_k.float(),
            q_k=q_k,
            q_preclosure_k=q_preclosure,
            shallow_q_k=shallow_q,
            deep_q_k=deep_q,
            gate40=gate40.float(),
            gate_logits40=gate_logits40.float(),
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
        return self.forward_components(
            fine,
            coarse_k,
            support,
            context,
            temporal_available,
            query_index,
        ).prediction_k
