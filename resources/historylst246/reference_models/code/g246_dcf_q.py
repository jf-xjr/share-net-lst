"""Dual-context feature fusion on a fixed two-expert Q-space anchor.

DCF-Q is deliberately not another output router.  The independently selected
no-coordinate r6a and r9 predictors define a fixed 0.5 Q-space anchor.  Their
intermediate 160/80/40 feature fields are then fused by one bottom-up/top-down
spatial decoder, whose two zero-initialized heads add orthogonal middle- and
high-frequency Q corrections.  No learned module receives either expert's
final Q value, target-derived masks, city/region identity, or explicit
latitude/longitude.

The source experts remain immutable ``eval``/``no_grad`` feature generators.
Their private module graph is evaluated explicitly rather than through forward
hooks so the feature contract is visible, testable, and independent of hook
ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from g246_calibrated_continuous_q import SceneCalibratedContinuousQ
from g246_ipmr_q import IPMRQ
from g246_q_bands import lift_p80_coefficients, orthogonal_q_bands
from ocnir import support_project


__all__ = ["DCFQComponents", "DCFQ"]


_FINE_CHANNELS = 52
_CONTEXT_DIM = 19
_PARENT = 4


def _contains_true(value: Tensor) -> bool:
    return bool(torch.any(value).detach().cpu().item())


def _binary(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if not value.is_floating_point() and value.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise TypeError(f"{name} must be bool or a numeric binary tensor")
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Project(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=False),
        )


class _SpatialResidual(nn.Module):
    """True spatial mixing with no near-zero LayerScale or multiplicative gate."""

    def __init__(self, channels: int, *, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.contract = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        residual = F.silu(self.norm(self.depthwise(value)), inplace=False)
        residual = F.silu(self.expand(residual), inplace=False)
        return value + self.contract(residual)


class _SpatialFuse(nn.Module):
    """Balanced additive entry plus learned spatial cross-source correction.

    Every source has a fixed, nonzero route into the fused representation.
    The learned concatenation branch may refine that consensus, but it cannot
    suppress a source through a scalar/pixel gate before spatial mixing.
    """

    def __init__(self, input_channels: Sequence[int], width: int) -> None:
        super().__init__()
        channels = tuple(int(value) for value in input_channels)
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("SpatialFuse inputs must be positive channel widths")
        self.projects = nn.ModuleList(_Project(value, width) for value in channels)
        self.merge = _Project(len(channels) * width, width)
        self.spatial3 = _SpatialResidual(width, kernel_size=3)
        self.spatial5 = _SpatialResidual(width, kernel_size=5)

    def forward(self, *values: Tensor) -> Tensor:
        if len(values) != len(self.projects):
            raise ValueError("SpatialFuse input count differs from its contract")
        projected = [module(value) for module, value in zip(self.projects, values)]
        geometry = projected[0].shape[-2:]
        if any(value.shape[-2:] != geometry for value in projected):
            raise ValueError("SpatialFuse inputs must share one spatial lattice")
        concatenated = torch.cat(projected, dim=1)
        balanced = concatenated.reshape(
            concatenated.shape[0], len(projected), -1,
            *concatenated.shape[-2:],
        ).sum(dim=1) / math.sqrt(len(projected))
        fused = balanced + self.merge(concatenated)
        return self.spatial5(self.spatial3(fused))


class _CorrectionDecoder(nn.Module):
    """One explicit 40->80->160 semantic/detail synthesis path."""

    def __init__(self, width40: int = 96, width80: int = 96, width160: int = 64):
        super().__init__()
        self.width40 = int(width40)
        self.width80 = int(width80)
        self.width160 = int(width160)
        # r6a S40[144], r9 D40[144], and explicit physical support fraction.
        self.fuse40 = _SpatialFuse((144, 144, 1), width40)
        # r6a S160[64] is spatially downsampled before meeting r9 D80[96]
        # and decoded G40.  This makes the 80 lattice a real synthesis stage.
        self.shallow_down80 = _Project(64, width80, kernel_size=3, stride=2)
        self.from40 = _Project(width40, width80)
        self.fuse80 = _SpatialFuse((width80, 96, width80, 1), width80)
        self.from80 = _Project(width80, width160)
        # No raw feature reaches a head: both r6a S160 and r9 F160 must pass
        # through this spatial fusion with decoded G80 first.
        self.fuse160 = _SpatialFuse((64, 64, width160, 1), width160)
        self.middle_head = nn.Sequential(
            _SpatialResidual(width80, kernel_size=3),
            nn.Conv2d(width80, 1, 3, padding=1),
        )
        self.high_head = nn.Sequential(
            _SpatialResidual(width160, kernel_size=3),
            _SpatialResidual(width160, kernel_size=5),
            # A constant high-head bias lies in P80 and is annihilated exactly
            # by QH=I-P80, so it would be an unidentifiable zero-gradient term.
            nn.Conv2d(width160, 1, 3, padding=1, bias=False),
        )
        self.apply(self._initialize)
        # A new formal run is an exact fixed-anchor model at u0.  Gradients
        # first reach each head and then open the complete spatial path.
        nn.init.zeros_(self.middle_head[-1].weight)
        nn.init.zeros_(self.middle_head[-1].bias)
        nn.init.zeros_(self.high_head[-1].weight)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            if module.groups == 1 \
                    and module.in_channels == module.out_channels \
                    and module.stride == (1, 1):
                nn.init.dirac_(module.weight)
            else:
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        shallow160: Tensor,
        shallow40: Tensor,
        deep160: Tensor,
        deep80: Tensor,
        deep40: Tensor,
        support: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        support_float = support.to(dtype=shallow160.dtype)
        support80 = F.max_pool2d(support_float, 2, 2)
        support40 = F.avg_pool2d(support_float, 4, 4)
        g40 = self.fuse40(shallow40, deep40, support40)
        g40 = g40 * (support40 > 0).to(dtype=g40.dtype)

        shallow80 = self.shallow_down80(shallow160)
        decoded40 = self.from40(F.interpolate(
            g40, size=deep80.shape[-2:], mode="bilinear", align_corners=False
        ))
        g80 = self.fuse80(shallow80, deep80, decoded40, support80)
        g80 = g80 * (support80 > 0).to(dtype=g80.dtype)

        decoded80 = self.from80(F.interpolate(
            g80, size=shallow160.shape[-2:], mode="bilinear", align_corners=False
        ))
        g160 = self.fuse160(shallow160, deep160, decoded80, support_float)
        g160 = g160 * support_float
        return self.middle_head(g80), self.high_head(g160), (g40, g80, g160)


class DCFQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_middle_k: Tensor
    q_high_k: Tensor
    q_k: Tensor
    q_preclosure_k: Tensor
    middle_coefficients: Tensor
    anchor_q_k: Tensor
    correction_q_middle_k: Tensor
    correction_q_high_k: Tensor


class DCFQ(nn.Module):
    """Frozen complementary features plus a band-resolved Q correction decoder."""

    schema_version = "g246-dcf-q-v1"
    scale_path = "r6a/r9 features -> G40 -> G80 -> G160 -> QM/QH correction"
    band_contract = "q=0.5(q_r6a+q_r9)+QM(delta80)+QH(delta160)"
    geolocation_context_indices = (5, 6, 7, 8)
    source_expert_names = ("r6a_context15_continuous", "r9_ipmr_q")

    def __init__(
        self,
        *,
        fine_channels: int = _FINE_CHANNELS,
        context_dim: int = _CONTEXT_DIM,
        width: int = 48,
        decoder_channels: Sequence[int] = (112, 112, 64),
    ) -> None:
        super().__init__()
        if (fine_channels, context_dim, width) != (52, 19, 48):
            raise ValueError("DCF-Q requires registered Fine52/Context19/width48")
        decoder_tuple = tuple(int(value) for value in decoder_channels)
        if decoder_tuple != (112, 112, 64):
            raise ValueError(
                "formal DCF-Q locks decoder_channels=(112,112,64)"
            )
        self.fine_channels = int(fine_channels)
        self.context_dim = int(context_dim)
        self.width = int(width)
        self.shallow_expert = SceneCalibratedContinuousQ(
            fine_channels=52,
            context_dim=19,
            width=48,
            activation_checkpointing=False,
            no_geo_core=True,
        )
        self.deep_expert = IPMRQ(
            fine_channels=52,
            context_dim=19,
            width=48,
            activation_checkpointing=False,
        )
        self.decoder = _CorrectionDecoder(*decoder_tuple)
        self.register_buffer(
            "experts_initialized", torch.tensor(False, dtype=torch.bool), persistent=True
        )
        self._freeze_experts()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )

    def decoder_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.decoder.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError("DCF-Q decoder trainable partition drifted")
        return parameters

    def _freeze_experts(self) -> None:
        for expert in (self.shallow_expert, self.deep_expert):
            expert.eval()
            for parameter in expert.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "DCFQ":
        super().train(mode)
        self._freeze_experts()
        self.decoder.train(mode)
        return self

    def load_expert_state_dicts(
        self,
        shallow_state: Mapping[str, Tensor],
        deep_state: Mapping[str, Tensor],
    ) -> None:
        if bool(self.experts_initialized.detach().cpu().item()):
            raise RuntimeError("DCF-Q experts may be initialized only once")
        self.shallow_expert.load_state_dict(shallow_state, strict=True)
        self.deep_expert.load_state_dict(deep_state, strict=True)
        self.experts_initialized.fill_(True)
        self._freeze_experts()

    @staticmethod
    def _context15(context19: Tensor) -> Tensor:
        if context19.shape[-1] != 19:
            raise ValueError("DCF-Q requires stored Context19")
        return torch.cat((context19[..., :5], context19[..., 9:]), dim=-1)

    def _validate_inputs(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        for value, name in (
            (fine, "fine"),
            (coarse_k, "coarse_k"),
            (support, "support"),
            (context, "context"),
            (temporal_available, "temporal_available"),
            (query_index, "query_index"),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if not bool(self.experts_initialized.detach().cpu().item()):
            raise RuntimeError("DCF-Q experts have not been initialized")
        if fine.ndim != 5 or fine.shape[1:3] != (1, 52):
            raise ValueError("fine must have shape [B,1,52,H,W]")
        batch, _, _, height, width = fine.shape
        if height % 16 or width % 16:
            raise ValueError("fine geometry must be divisible by sixteen")
        if coarse_k.shape != (batch, 1, 1, height // 4, width // 4):
            raise ValueError("coarse_k must have shape [B,1,1,H/4,W/4]")
        if support.shape != (batch, 1, 1, height, width):
            raise ValueError("support must have shape [B,1,1,H,W]")
        if context.shape != (batch, 1, 19):
            raise ValueError("context must have shape [B,1,19]")
        if temporal_available.shape != (batch, 1) \
                or _contains_true(~_binary(temporal_available, "availability")):
            raise ValueError("the single query date must be available")
        if query_index.shape != (batch,) or _contains_true(query_index != 0):
            raise ValueError("single-date query_index must be zero")
        if query_index.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise TypeError("query_index must be integer zero")
        if len({
            fine.device, coarse_k.device, support.device, context.device,
            temporal_available.device, query_index.device,
        }) != 1:
            raise ValueError("all DCF-Q inputs must be on one device")
        if not fine.is_floating_point() or not coarse_k.is_floating_point() \
                or not context.is_floating_point():
            raise TypeError("fine, coarse_k and context must be floating tensors")
        if _contains_true(~torch.isfinite(fine)) \
                or _contains_true(~torch.isfinite(context)) \
                or _contains_true(torch.isinf(coarse_k)):
            raise ValueError("DCF-Q public predictors contain invalid values")
        return fine[:, 0], coarse_k[:, 0], _binary(support[:, 0], "support"), context[:, 0]

    def _shallow_features(
        self,
        fine: Tensor,
        coarse: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        expert = self.shallow_expert
        batch = fine.shape[0]
        available = torch.ones(batch, dtype=torch.bool, device=fine.device)
        query = torch.zeros(batch, dtype=torch.long, device=fine.device)
        support_float = support.to(dtype=fine.dtype)
        fine160, token40, descriptor = expert.date_encoder(
            fine, coarse, support_float, context15, available
        )
        parent40 = expert.context_unet(expert.temporal_fusion(
            token40[:, None], available[:, None], query
        ))
        scene = expert.scene_fusion(torch.cat(
            (descriptor, descriptor, torch.zeros_like(descriptor)), dim=1
        ))

        head = expert.continuous_head
        parent_up = head.fine_projection(F.interpolate(
            parent40, size=fine160.shape[-2:], mode="bilinear", align_corners=False
        ))
        fused160 = head.fuse(torch.cat((fine160, parent_up), dim=1))
        scale, shift = head.scene_film(scene).chunk(2, dim=1)
        fused160 = fused160 * (1.0 + 0.15 * torch.tanh(scale)[..., None, None])
        fused160 = fused160 + 0.15 * shift[..., None, None]
        fused160 = head.blocks(fused160)
        expert_values = torch.cat((
            head.local_expert(fused160),
            head.edge_expert(fused160),
            head.context_expert(parent_up),
        ), dim=1)
        gains = 1.0 + 0.25 * torch.tanh(head.scene_gain(scene))
        expert_values = expert_values * gains[..., None, None]
        logits = head.spatial_gate(fused160) + head.scene_gate(scene)[..., None, None]
        weights = torch.softmax(logits.float(), dim=1).to(dtype=fused160.dtype)
        raw_q = (expert_values * weights).sum(dim=1, keepdim=True)
        coarse_valid = torch.isfinite(coarse)
        q = support_project(
            raw_q, torch.zeros_like(coarse), support, coarse_valid
        ) * support.to(dtype=raw_q.dtype)
        base = support_project(fine[:, :1], coarse, support)
        return q, base, fused160, parent40

    def _deep_features(
        self,
        fine: Tensor,
        coarse: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        expert = self.deep_expert
        support_float = support.to(dtype=fine.dtype)
        local = torch.cat((((fine[:, :1] - 300.0) / 20.0), fine[:, 1:], support_float), dim=1)
        f160 = expert.fine_stem(local)
        f80 = expert.encode80(expert.down80(f160))
        content40 = expert.encode40(expert.down40(f80))
        merged40 = expert.merge40(
            content40,
            expert._physical_token40(fine, coarse, support_float, context15),
        )
        f20 = expert.encode20(expert.down20(merged40))
        f10 = expert.bottleneck10(expert.down10(f20))
        d20 = expert.decode20(f20, expert.up20(F.interpolate(
            f10, size=f20.shape[-2:], mode="bilinear", align_corners=False
        )))
        d40 = expert.decode40(merged40, expert.up40(F.interpolate(
            d20, size=merged40.shape[-2:], mode="bilinear", align_corners=False
        )))
        d80 = expert.decode80(f80, expert.up80(F.interpolate(
            d40, size=f80.shape[-2:], mode="bilinear", align_corners=False
        )))
        middle = expert.middle_head(d80)
        high = expert.high_head(expert._condition_high(f160, d80))
        coarse_valid = torch.isfinite(coarse)
        middle_lift = lift_p80_coefficients(middle, support)
        middle_bands = orthogonal_q_bands(middle_lift.float(), support, coarse_valid)
        high_bands = orthogonal_q_bands(high.float(), support, coarse_valid)
        q_pre = middle_bands.q_middle + high_bands.q_high
        q = support_project(
            q_pre, torch.zeros_like(coarse, dtype=q_pre.dtype), support, coarse_valid
        ) * support.to(dtype=q_pre.dtype)
        return q, f160, d80, d40, middle

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> DCFQComponents:
        query_fine, query_coarse, query_support, query_context = self._validate_inputs(
            fine, coarse_k, support, context, temporal_available, query_index
        )
        context15 = self._context15(query_context)
        with torch.no_grad():
            shallow_q, shallow_base, shallow160, shallow40 = self._shallow_features(
                query_fine, query_coarse, query_support, context15
            )
            deep_q, deep160, deep80, deep40, _deep_middle = self._deep_features(
                query_fine, query_coarse, query_support, context15
            )
        anchor_q = 0.5 * (shallow_q.float() + deep_q.float())
        middle_coefficients, high_proposal, _features = self.decoder(
            shallow160,
            shallow40,
            deep160,
            deep80,
            deep40,
            query_support,
        )
        coarse_valid = torch.isfinite(query_coarse)
        middle_lift = lift_p80_coefficients(middle_coefficients, query_support)
        correction_middle = orthogonal_q_bands(
            middle_lift.float(), query_support, coarse_valid
        ).q_middle
        correction_high = orthogonal_q_bands(
            high_proposal.float(), query_support, coarse_valid
        ).q_high
        anchor_bands = orthogonal_q_bands(
            anchor_q, query_support, coarse_valid
        )
        q_middle = anchor_bands.q_middle + correction_middle
        q_high = anchor_bands.q_high + correction_high
        q_preclosure = anchor_q + correction_middle + correction_high
        q_k = support_project(
            q_preclosure,
            torch.zeros_like(query_coarse, dtype=q_preclosure.dtype),
            query_support,
            coarse_valid,
        ) * query_support.to(dtype=q_preclosure.dtype)
        # Both source models use this same base operation; retain the literal
        # shallow value and assert equivalence in qualification tests.
        base_k = shallow_base.float()
        return DCFQComponents(
            prediction_k=base_k + q_k,
            base_k=base_k,
            q_middle_k=q_middle,
            q_high_k=q_high,
            q_k=q_k,
            q_preclosure_k=q_preclosure,
            middle_coefficients=middle_coefficients,
            anchor_q_k=anchor_q,
            correction_q_middle_k=correction_middle,
            correction_q_high_k=correction_high,
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
            fine, coarse_k, support, context, temporal_available, query_index
        ).prediction_k
