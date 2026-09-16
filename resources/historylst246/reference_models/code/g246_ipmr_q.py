"""Information-Preserving Multiresolution Q network for G246.

IPMR-Q is the metric-first replacement for the legacy late-fusion decoder.  It
routes local content exactly once through

    160 -> 80 -> 40 -> 20 -> 10 -> 20 -> 40 -> 80 -> 160,

merges low-dimensional physical/context evidence only on the 40 lattice, and
uses support-aware orthogonal middle/high Q heads.  The high-resolution branch
cannot bypass decoded semantics: D80 produces a bounded low-rank selector for
F160 before the final fusion.  Geolocation columns are physically removed from
Context19, leaving Context15.

The inference signature intentionally contains no target, validity, eligible,
city, region, or coordinate argument.  The delivered field is closed exactly
against every valid coarse observation by :func:`ocnir.support_project`.
"""

from __future__ import annotations

import math
from math import gcd
from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from g246_q_bands import (
    lift_p80_coefficients,
    orthogonal_q_bands,
    support_block_mean,
)
from ocnir import support_project


__all__ = ["IPMRQ", "IPMRQComponents"]


_SCALE = 4
_REGISTERED_WIDTHS = (48, 64, 96)
_S_CHANNELS = {
    "fine": 64,
    "half": 96,
    "parent": 144,
    "middle": 208,
    "low": 288,
}


def _round_channels(value: float) -> int:
    return max(8, int(math.floor(value / 8.0 + 0.5)) * 8)


def _channels(width: int) -> dict[str, int]:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width not in _REGISTERED_WIDTHS:
        raise ValueError("registered IPMR-Q width must be one of 48, 64, 96")
    if width == 48:
        return dict(_S_CHANNELS)
    factor = width / 48.0
    return {name: _round_channels(value * factor) for name, value in _S_CHANNELS.items()}


def _groups(channels: int) -> int:
    return gcd(channels, min(channels, 8))


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels), channels)


def _contains_true(value: Tensor) -> bool:
    return bool(torch.any(value).detach().cpu().item())


def _binary(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


class _ResidualDW(nn.Module):
    """Depthwise residual block with a deliberately open initial route."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = expansion * channels
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = _norm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.contract = nn.Conv2d(hidden, channels, 1)
        # A fixed, open residual route prevents the long multiresolution path
        # from being effectively dormant early on (the legacy learned 0.01
        # LayerScale was one identified optimization bottleneck).
        self.residual_multiplier = 0.5

    def forward(self, inputs: Tensor) -> Tensor:
        value = F.silu(self.norm(self.depthwise(inputs)))
        value = self.contract(F.silu(self.expand(value)))
        return inputs + self.residual_multiplier * value


class _Project(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=False),
        )


class _Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=False),
        )


class _Merge(nn.Module):
    def __init__(self, skip_channels: int, decoded_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = _Project(skip_channels + decoded_channels, out_channels)
        self.blocks = nn.Sequential(_ResidualDW(out_channels), _ResidualDW(out_channels))

    def forward(self, skip: Tensor, decoded: Tensor) -> Tensor:
        return self.blocks(self.project(torch.cat((skip, decoded), dim=1)))


class IPMRQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_middle_k: Tensor
    q_high_k: Tensor
    q_k: Tensor
    q_preclosure_k: Tensor
    middle_coefficients_k: Tensor


class IPMRQ(nn.Module):
    """Fine52/Context15 single-date support-aware multiresolution Q model."""

    widths = _REGISTERED_WIDTHS
    geolocation_context_indices = (5, 6, 7, 8)
    scale_path = "160->80->40->20->10->20->40->80->160"
    band_contract = "QM=P80-P40;QH=I-P80;Q=QM+QH=I-P40"
    physical_merge_lattice = 40
    local_content_lattices = (160, 80, 40)

    @staticmethod
    def registered_channels(width: int) -> dict[str, int]:
        return _channels(width)

    def __init__(
        self,
        *,
        fine_channels: int = 52,
        context_dim: int = 19,
        width: int = 48,
        activation_checkpointing: bool = True,
        channels: Sequence[int] | None = None,
        blocks_per_scale: int = 2,
        selector_rank: int = 32,
    ) -> None:
        super().__init__()
        if fine_channels != 52 or context_dim != 19:
            raise ValueError("IPMR-Q requires the exact Fine52/Context19 input contract")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be bool")
        if isinstance(blocks_per_scale, bool) or blocks_per_scale <= 0:
            raise ValueError("blocks_per_scale must be positive")
        if channels is None:
            resolved = _channels(width)
            channel_tuple = tuple(resolved[name] for name in (
                "fine", "half", "parent", "middle", "low"
            ))
        else:
            if len(tuple(channels)) != 5 or any(int(value) <= 0 for value in channels):
                raise ValueError("channels must provide five positive widths")
            channel_tuple = tuple(int(value) for value in channels)
        fine_width, half_width, parent_width, middle_width, low_width = channel_tuple
        if selector_rank <= 0 or selector_rank > min(fine_width, half_width):
            raise ValueError("selector_rank must be positive and no larger than fine/half widths")

        self.fine_channels = int(fine_channels)
        self.context_dim = int(context_dim)
        self.effective_context_dim = context_dim - 4
        self.width = int(width)
        self.channels = {
            "fine": fine_width,
            "half": half_width,
            "parent": parent_width,
            "middle": middle_width,
            "low": low_width,
        }
        self.activation_checkpointing = activation_checkpointing
        self.selector_rank = int(selector_rank)

        self.fine_stem = nn.Sequential(
            nn.Conv2d(fine_channels + 1, fine_width, 3, padding=1, bias=False),
            _norm(fine_width),
            nn.SiLU(inplace=False),
            *(_ResidualDW(fine_width) for _ in range(blocks_per_scale)),
        )
        self.down80 = _Downsample(fine_width, half_width)
        self.encode80 = nn.Sequential(
            *(_ResidualDW(half_width) for _ in range(blocks_per_scale))
        )
        self.down40 = _Downsample(half_width, parent_width)
        self.encode40 = nn.Sequential(
            *(_ResidualDW(parent_width) for _ in range(blocks_per_scale))
        )

        # Six support-weighted physical summaries + coarse value/validity,
        # support fraction, base/coarse discrepancy, and physical Context15.
        physical_channels = 6 + 4 + self.effective_context_dim
        self.physical40 = nn.Sequential(
            nn.Conv2d(physical_channels, parent_width, 3, padding=1, bias=False),
            _norm(parent_width),
            nn.SiLU(inplace=False),
            _ResidualDW(parent_width),
        )
        self.merge40 = _Merge(parent_width, parent_width, parent_width)

        self.down20 = _Downsample(parent_width, middle_width)
        self.encode20 = nn.Sequential(
            *(_ResidualDW(middle_width) for _ in range(blocks_per_scale))
        )
        self.down10 = _Downsample(middle_width, low_width)
        self.bottleneck10 = nn.Sequential(
            _ResidualDW(low_width, dilation=1),
            _ResidualDW(low_width, dilation=2),
            _ResidualDW(low_width, dilation=3),
        )
        self.up20 = _Project(low_width, middle_width)
        self.decode20 = _Merge(middle_width, middle_width, middle_width)
        self.up40 = _Project(middle_width, parent_width)
        self.decode40 = _Merge(parent_width, parent_width, parent_width)
        self.up80 = _Project(parent_width, half_width)
        self.decode80 = _Merge(half_width, half_width, half_width)

        self.middle_head = nn.Sequential(
            _ResidualDW(half_width),
            nn.Conv2d(half_width, 1, 3, padding=1),
        )

        # Strict non-bypass conditioner.  F160 is first compressed to a
        # low-rank basis and can reach D160 only through multiplication by a
        # D80-derived tanh gate.  A separate D80 value path keeps semantic
        # content available when the local basis is uninformative.  In
        # particular, D80=0 makes the high proposal independent of F160;
        # there is deliberately no ``1 + gate`` or raw-F160 residual route.
        self.high_basis = nn.Conv2d(fine_width, selector_rank, 1, bias=False)
        self.high_gate = nn.Conv2d(half_width, selector_rank, 1, bias=False)
        self.high_value = nn.Conv2d(half_width, selector_rank, 1, bias=False)
        self.decode160 = nn.Sequential(
            _Project(2 * selector_rank, fine_width),
            _ResidualDW(fine_width, dilation=1),
            _ResidualDW(fine_width, dilation=2),
        )
        self.high_head = nn.Sequential(
            _ResidualDW(fine_width, dilation=1),
            _ResidualDW(fine_width, dilation=2),
            nn.Conv2d(fine_width, 1, 3, padding=1),
        )

        self.apply(self._initialize)
        # Start close to the exact physical base while retaining nonzero
        # gradients through every feature route and both heads.
        nn.init.normal_(self.middle_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.middle_head[-1].bias)
        nn.init.normal_(self.high_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.high_head[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

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
        if fine.ndim != 5 or fine.shape[2] != self.fine_channels:
            raise ValueError("fine must have shape [B,1,52,H,W]")
        batch, times, _, height, width = fine.shape
        if times != 1:
            raise ValueError("IPMR-Q requires a single query date (T=1)")
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
        if query_index.shape != (batch,):
            raise ValueError("query_index must have shape [B]")
        if len({
            fine.device, coarse_k.device, support.device, context.device,
            temporal_available.device, query_index.device,
        }) != 1:
            raise ValueError("all IPMR-Q inputs must be on one device")
        if not fine.is_floating_point() or not coarse_k.is_floating_point() \
                or not context.is_floating_point():
            raise TypeError("fine, coarse_k and context must be floating tensors")
        if _contains_true(~torch.isfinite(fine)) or _contains_true(~torch.isfinite(context)):
            raise ValueError("fine and context must be finite")
        if _contains_true(torch.isinf(coarse_k)):
            raise ValueError("coarse_k may contain NaN but not infinity")
        support_bool = _binary(support[:, 0], "support")
        available = _binary(temporal_available, "temporal_available")
        if _contains_true(~available):
            raise ValueError("the single query date must be available")
        if query_index.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ) or _contains_true(query_index != 0):
            raise ValueError("single-date query_index must be integer zero")
        return fine[:, 0], coarse_k[:, 0], support_bool, context[:, 0]

    @staticmethod
    def _context15(context19: Tensor) -> Tensor:
        if context19.shape[-1] != 19:
            raise ValueError("physical context surgery requires Context19")
        return torch.cat((context19[..., :5], context19[..., 9:]), dim=-1)

    def _physical_token40(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> Tensor:
        # base z, interpolation confidence, optical validity, built, water,
        # and LULC coverage are the low-dimensional physical spatial summary.
        selected = torch.cat(
            (((fine[:, :1] - 300.0) / 20.0), fine[:, 1:2], fine[:, 18:22]),
            dim=1,
        )
        parent_means, _ = support_block_mean(selected, support, 4)
        support_fraction = F.avg_pool2d(support.to(dtype=fine.dtype), 4, 4)
        coarse_valid = torch.isfinite(coarse_k)
        coarse_safe = torch.where(
            coarse_valid,
            coarse_k.to(dtype=fine.dtype),
            torch.full_like(coarse_k, 300.0, dtype=fine.dtype),
        )
        coarse_z = (coarse_safe - 300.0) / 20.0
        base_minus_coarse = parent_means[:, :1] - coarse_z
        context_map = context15[..., None, None].expand(
            -1, -1, coarse_k.shape[-2], coarse_k.shape[-1]
        )
        return self.physical40(torch.cat((
            parent_means,
            coarse_z,
            coarse_valid.to(dtype=fine.dtype),
            support_fraction,
            base_minus_coarse,
            context_map,
        ), dim=1))

    def _band_proposals(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> tuple[Tensor, Tensor]:
        local = torch.cat((((fine[:, :1] - 300.0) / 20.0), fine[:, 1:], support), dim=1)
        f160 = self.fine_stem(local)
        f80 = self.encode80(self.down80(f160))
        content40 = self.encode40(self.down40(f80))
        merged40 = self.merge40(
            content40, self._physical_token40(fine, coarse_k, support, context15)
        )
        f20 = self.encode20(self.down20(merged40))
        f10 = self.bottleneck10(self.down10(f20))
        d20_up = self.up20(F.interpolate(
            f10, size=f20.shape[-2:], mode="bilinear", align_corners=False
        ))
        d20 = self.decode20(f20, d20_up)
        d40_up = self.up40(F.interpolate(
            d20, size=merged40.shape[-2:], mode="bilinear", align_corners=False
        ))
        d40 = self.decode40(merged40, d40_up)
        d80_up = self.up80(F.interpolate(
            d40, size=f80.shape[-2:], mode="bilinear", align_corners=False
        ))
        d80 = self.decode80(f80, d80_up)

        middle = self.middle_head(d80)
        d160 = self._condition_high(f160, d80)
        high = self.high_head(d160)
        return middle, high

    def _condition_high(self, f160: Tensor, d80: Tensor) -> Tensor:
        """Return D160 with no computational bypass around decoded D80."""

        if f160.ndim != 4 or d80.ndim != 4:
            raise ValueError("high conditioner expects NCHW feature tensors")
        if f160.shape[0] != d80.shape[0] \
                or f160.shape[-2] != 2 * d80.shape[-2] \
                or f160.shape[-1] != 2 * d80.shape[-1]:
            raise ValueError("F160/D80 geometry must have an exact 2x relation")
        basis = self.high_basis(f160)
        gate = torch.tanh(F.interpolate(
            self.high_gate(d80),
            size=f160.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ))
        value = F.interpolate(
            self.high_value(d80),
            size=f160.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.decode160(torch.cat((value, basis * gate), dim=1))

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> IPMRQComponents:
        query_fine, query_coarse, query_support_bool, query_context = (
            self._validate_inputs(
                fine, coarse_k, support, context, temporal_available, query_index
            )
        )
        query_support = query_support_bool.to(dtype=query_fine.dtype)
        context15 = self._context15(query_context)
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            middle_coefficients, high_proposal = gradient_checkpoint(
                self._band_proposals,
                query_fine,
                query_coarse,
                query_support,
                context15,
                use_reentrant=False,
            )
        else:
            middle_coefficients, high_proposal = self._band_proposals(
                query_fine, query_coarse, query_support, context15
            )
        coarse_valid = torch.isfinite(query_coarse)
        middle_lift = lift_p80_coefficients(middle_coefficients, query_support_bool)
        middle_bands = orthogonal_q_bands(
            middle_lift.float(), query_support_bool, coarse_valid
        )
        high_bands = orthogonal_q_bands(
            high_proposal.float(), query_support_bool, coarse_valid
        )
        q_middle = middle_bands.q_middle
        q_high = high_bands.q_high
        q_preclosure = q_middle + q_high
        q_k = support_project(
            q_preclosure,
            torch.zeros_like(query_coarse, dtype=q_preclosure.dtype),
            query_support_bool,
            coarse_valid,
        ) * query_support_bool.to(dtype=q_preclosure.dtype)
        base_k = support_project(
            query_fine[:, :1], query_coarse, query_support_bool
        )
        prediction = base_k + q_k
        return IPMRQComponents(
            prediction,
            base_k,
            q_middle,
            q_high,
            q_k,
            q_preclosure,
            middle_coefficients,
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
