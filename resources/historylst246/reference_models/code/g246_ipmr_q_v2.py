"""Factorized Information-Preserving Multiresolution Q network for G246.

``IPMRQV2`` keeps the useful full-resolution route of IPMR-Q while making the
content/physics separation literal:

* the fine spatial encoder receives support-weighted, within-4x4-parent
  contrasts of Fine52 plus the support mask, while the corresponding
  target-free Fine52 parent means rejoin as content only on the 40 lattice;
* Context19 geolocation columns 5:9 are physically deleted;
* the remaining Context15 and target-free coarse/base state are reduced to a
  low-dimensional *non-spatial* token and injected exactly once by channel
  FiLM on the 40 lattice;
* the high branch has no F160 route that can bypass decoded D80 semantics; and
* the learned middle/high proposals are formed and finally closed with the
  registered support-aware operators from :mod:`g246_q_bands`.

The public inference signature deliberately contains no target, validity,
eligible, city, region, or coordinate argument.  ``forward_components`` has
the same field names as IPMR-Q v1 so the existing band-supervised loss needs
no scientific change when the trainer is later wired to this model.
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


__all__ = [
    "IPMRQV2",
    "IPMRQV2Components",
    "support_parent_contrast",
]


_SCALE = 4
_REGISTERED_WIDTHS = (48, 64, 96)
_S_CHANNELS = {
    "fine": 64,
    "half": 96,
    "parent": 144,
    "middle": 208,
    "low": 288,
}
_COARSE_STATE_DIM = 10


def _round_channels(value: float) -> int:
    return max(8, int(math.floor(value / 8.0 + 0.5)) * 8)


def _channels(width: int) -> dict[str, int]:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width not in _REGISTERED_WIDTHS:
        raise ValueError("registered IPMR-Q v2 width must be one of 48, 64, 96")
    if width == 48:
        return dict(_S_CHANNELS)
    factor = width / 48.0
    return {
        name: _round_channels(value * factor)
        for name, value in _S_CHANNELS.items()
    }


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


def support_parent_contrast(
    values: Tensor,
    support: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return supported Fine52 contrasts and weighted 4x4 parent means.

    This content transform intentionally does not define the Q observation
    operator; all Q projections remain in :mod:`g246_q_bands`.  Unsupported
    pixels and empty parents are exactly zero.
    """

    if not isinstance(values, Tensor) or not isinstance(support, Tensor):
        raise TypeError("values and support must be torch.Tensor instances")
    if values.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("parent contrast expects NCHW values and N1HW support")
    if values.shape[0] != support.shape[0] \
            or values.shape[-2:] != support.shape[-2:]:
        raise ValueError("parent contrast values/support geometry differs")
    if values.shape[-2] % _SCALE or values.shape[-1] % _SCALE:
        raise ValueError("parent contrast geometry must be divisible by four")
    support_bool = _binary(support, "support")
    means, _ = support_block_mean(values, support_bool, _SCALE)
    expanded = means.repeat_interleave(_SCALE, dim=-2).repeat_interleave(
        _SCALE, dim=-1
    )
    contrast = (values - expanded) * support_bool.to(dtype=values.dtype)
    return contrast, means


def _weighted_scene_stats(
    values: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Stable scalar mean/std for each NCHW channel, with empty -> zero."""

    if values.ndim != 4 or weights.ndim != 4 or weights.shape[1] != 1:
        raise ValueError("scene statistics require NCHW values and N1HW weights")
    if values.shape[0] != weights.shape[0] \
            or values.shape[-2:] != weights.shape[-2:]:
        raise ValueError("scene-statistic values/weights geometry differs")
    work_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    work = values.to(dtype=work_dtype)
    weight = weights.to(dtype=work_dtype)
    count = weight.sum(dim=(-2, -1))
    mean = (work * weight).sum(dim=(-2, -1)) / count.clamp_min(1.0)
    centered = work - mean[..., None, None]
    variance = (centered.square() * weight).sum(dim=(-2, -1)) \
        / count.clamp_min(1.0)
    active = count > 0
    mean = torch.where(active, mean, torch.zeros_like(mean))
    standard_deviation = torch.where(
        active,
        torch.sqrt(variance.clamp_min(0.0)),
        torch.zeros_like(variance),
    )
    return mean.to(dtype=values.dtype), standard_deviation.to(dtype=values.dtype)


class _ResidualDW(nn.Module):
    """Depthwise residual block with a fixed, open optimization route."""

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
    def __init__(
        self,
        skip_channels: int,
        decoded_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.project = _Project(skip_channels + decoded_channels, out_channels)
        self.blocks = nn.Sequential(
            _ResidualDW(out_channels),
            _ResidualDW(out_channels),
        )

    def forward(self, skip: Tensor, decoded: Tensor) -> Tensor:
        return self.blocks(self.project(torch.cat((skip, decoded), dim=1)))


class IPMRQV2Components(NamedTuple):
    """Trainer-compatible prediction and strict Q-band components."""

    prediction_k: Tensor
    base_k: Tensor
    q_middle_k: Tensor
    q_high_k: Tensor
    q_k: Tensor
    q_preclosure_k: Tensor
    middle_coefficients_k: Tensor


class IPMRQV2(nn.Module):
    """Fine52/Context15 factorized single-date multiresolution Q model."""

    widths = _REGISTERED_WIDTHS
    geolocation_context_indices = (5, 6, 7, 8)
    scale_path = "160->80->40->20->10->20->40->80->160"
    band_contract = "QM=P80-P40;QH=I-P80;Q=QM+QH=I-P40"
    content_contract = (
        "support_weighted_4x4_parent_contrast_Fine52_plus_support_at_160_80_40_"
        "and_target_free_Fine52_parent_means_at_40"
    )
    physical_contract = "Context15_plus_coarse_scalar_state_vector_FiLM_at_40_only"
    physical_merge_lattice = 40
    local_content_lattices = (160, 80, 40)
    coarse_state_dim = _COARSE_STATE_DIM

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
        physical_token_dim: int = 32,
        closure_tolerance_k: float = 5.0e-5,
    ) -> None:
        super().__init__()
        if fine_channels != 52 or context_dim != 19:
            raise ValueError(
                "IPMR-Q v2 requires the exact Fine52/Context19 stored input contract"
            )
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be bool")
        if isinstance(blocks_per_scale, bool) \
                or not isinstance(blocks_per_scale, int) \
                or blocks_per_scale <= 0:
            raise ValueError("blocks_per_scale must be positive")
        if channels is None:
            resolved = _channels(width)
            channel_tuple = tuple(
                resolved[name]
                for name in ("fine", "half", "parent", "middle", "low")
            )
        else:
            provided_channels = tuple(channels)
            if len(provided_channels) != 5 \
                    or any(int(value) <= 0 for value in provided_channels):
                raise ValueError("channels must provide five positive widths")
            channel_tuple = tuple(int(value) for value in provided_channels)
        fine_width, half_width, parent_width, middle_width, low_width = channel_tuple
        if selector_rank <= 0 or selector_rank > min(fine_width, half_width):
            raise ValueError(
                "selector_rank must be positive and no larger than fine/half widths"
            )
        if isinstance(physical_token_dim, bool) or physical_token_dim <= 0 \
                or physical_token_dim > parent_width:
            raise ValueError(
                "physical_token_dim must be positive and no larger than parent width"
            )
        if not math.isfinite(closure_tolerance_k) or closure_tolerance_k <= 0.0:
            raise ValueError("closure_tolerance_k must be finite and positive")

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
        self.physical_token_dim = int(physical_token_dim)
        self.closure_tolerance_k = float(closure_tolerance_k)

        # Every Fine52 channel is parent-centered before this fine stem.  Its
        # absolute parent state rejoins separately at C40 below.
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
        # Parent means are still local Fine52 content.  They must not be
        # confused with Context/coarse conditioning: a padding-free 1x1
        # projection restores their absolute land-cover/reflectance state at
        # C40, then an ordinary shared content fusion combines both routes.
        self.parent_mean40 = _Project(fine_channels + 1, parent_width)
        self.content_fuse40 = _Merge(
            parent_width, parent_width, parent_width
        )

        # This branch is vector -> vector only.  It cannot emit a map, and no
        # Conv2d ever observes a broadcast Context vector.  FiLM is injected
        # exactly once at 40, followed by shared spatial processing.
        descriptor_dim = self.effective_context_dim + self.coarse_state_dim
        self.physical_token_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, physical_token_dim),
            nn.LayerNorm(physical_token_dim),
            nn.SiLU(inplace=False),
            nn.Linear(physical_token_dim, physical_token_dim),
            nn.SiLU(inplace=False),
        )
        self.physical_film40 = nn.Linear(physical_token_dim, 2 * parent_width)
        self.post_film40 = nn.Sequential(
            *(_ResidualDW(parent_width) for _ in range(blocks_per_scale))
        )

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

        # Strict non-bypass conditioner: F160 can influence D160 only through
        # multiplication by a D80-derived bounded selector.  No raw F160 or
        # ``1 + gate`` route exists.
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
        nn.init.normal_(self.middle_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.middle_head[-1].bias)
        nn.init.normal_(self.high_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.high_head[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
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
            raise ValueError("IPMR-Q v2 requires a single query date (T=1)")
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
            fine.device,
            coarse_k.device,
            support.device,
            context.device,
            temporal_available.device,
            query_index.device,
        }) != 1:
            raise ValueError("all IPMR-Q v2 inputs must be on one device")
        if not fine.is_floating_point() or not coarse_k.is_floating_point() \
                or not context.is_floating_point():
            raise TypeError("fine, coarse_k and context must be floating tensors")
        if _contains_true(~torch.isfinite(fine)) \
                or _contains_true(~torch.isfinite(context)):
            raise ValueError("fine and context must be finite")
        if _contains_true(torch.isinf(coarse_k)):
            raise ValueError("coarse_k may contain NaN but not infinity")
        support_bool = _binary(support[:, 0], "support")
        available = _binary(temporal_available, "temporal_available")
        if _contains_true(~available):
            raise ValueError("the single query date must be available")
        if query_index.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ) or _contains_true(query_index != 0):
            raise ValueError("single-date query_index must be integer zero")
        return fine[:, 0], coarse_k[:, 0], support_bool, context[:, 0]

    @staticmethod
    def _context15(context19: Tensor) -> Tensor:
        if context19.shape[-1] != 19:
            raise ValueError("physical context surgery requires Context19")
        return torch.cat((context19[..., :5], context19[..., 9:]), dim=-1)

    def _coarse_state(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
    ) -> Tensor:
        """Return ten target-free scalar coarse/base/support descriptors."""

        support_fraction = F.avg_pool2d(
            support.to(dtype=fine.dtype), _SCALE, _SCALE
        )
        base_parent_k, active_parent = support_block_mean(
            fine[:, :1], support, _SCALE
        )
        active_weight = active_parent.to(dtype=fine.dtype)
        coarse_valid = torch.isfinite(coarse_k) & active_parent
        coarse_safe = torch.where(
            coarse_valid,
            coarse_k.to(dtype=fine.dtype),
            torch.full_like(coarse_k, 300.0, dtype=fine.dtype),
        )
        coarse_z = (coarse_safe - 300.0) / 20.0
        base_z = (base_parent_k - 300.0) / 20.0
        observed_weight = support_fraction * coarse_valid.to(dtype=fine.dtype)
        base_weight = support_fraction

        coarse_mean, coarse_std = _weighted_scene_stats(
            coarse_z, observed_weight
        )
        base_mean, base_std = _weighted_scene_stats(base_z, base_weight)
        delta_mean, delta_std = _weighted_scene_stats(
            base_z - coarse_z, observed_weight
        )
        support_mean, support_std = _weighted_scene_stats(
            support_fraction, active_weight
        )
        active_count = active_weight.sum(dim=(-2, -1)).clamp_min(1.0)
        valid_fraction = coarse_valid.to(dtype=fine.dtype).sum(
            dim=(-2, -1)
        ) / active_count
        coverage = support.to(dtype=fine.dtype).mean(dim=(-2, -1))
        state = torch.cat((
            coarse_mean,
            coarse_std,
            base_mean,
            base_std,
            delta_mean,
            delta_std,
            support_mean,
            support_std,
            valid_fraction,
            coverage,
        ), dim=1)
        if state.shape != (fine.shape[0], self.coarse_state_dim):
            raise AssertionError("coarse scalar-state shape changed")
        return state

    def _physical_token(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> Tensor:
        """Return a BxD token; this path never creates a spatial tensor."""

        descriptor = torch.cat(
            (context15, self._coarse_state(fine, coarse_k, support)), dim=1
        )
        return self.physical_token_mlp(descriptor.to(dtype=fine.dtype))

    def _apply_physical_film40(
        self,
        content40: Tensor,
        physical_token: Tensor,
    ) -> Tensor:
        if content40.ndim != 4 or physical_token.ndim != 2:
            raise ValueError("physical FiLM expects NCHW content and a BD token")
        if content40.shape[0] != physical_token.shape[0] \
                or physical_token.shape[1] != self.physical_token_dim:
            raise ValueError("physical token shape differs from the 40-lattice batch")
        scale, shift = self.physical_film40(physical_token).chunk(2, dim=1)
        scale = 0.25 * torch.tanh(scale)[..., None, None]
        shift = 0.25 * torch.tanh(shift)[..., None, None]
        return self.post_film40(content40 * (1.0 + scale) + shift)

    def _band_proposals(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> tuple[Tensor, Tensor]:
        f160, f80, content40 = self._encode_content(fine, support)
        token = self._physical_token(fine, coarse_k, support, context15)
        merged40 = self._apply_physical_film40(content40, token)

        f20 = self.encode20(self.down20(merged40))
        f10 = self.bottleneck10(self.down10(f20))
        d20_up = self.up20(F.interpolate(
            f10,
            size=f20.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ))
        d20 = self.decode20(f20, d20_up)
        d40_up = self.up40(F.interpolate(
            d20,
            size=merged40.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ))
        d40 = self.decode40(merged40, d40_up)
        d80_up = self.up80(F.interpolate(
            d40,
            size=f80.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ))
        d80 = self.decode80(f80, d80_up)

        middle = self.middle_head(d80)
        high = self.high_head(self._condition_high(f160, d80))
        return middle, high

    def _encode_content(
        self,
        fine: Tensor,
        support: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode explicit parent contrast and parent-mean Fine52 content."""

        values = torch.cat(
            (((fine[:, :1] - 300.0) / 20.0), fine[:, 1:]), dim=1
        )
        contrast, parent_means = support_parent_contrast(values, support)
        support_value = support.to(dtype=fine.dtype)
        local = torch.cat((contrast, support_value), dim=1)
        f160 = self.fine_stem(local)
        f80 = self.encode80(self.down80(f160))
        contrast40 = self.encode40(self.down40(f80))
        support_fraction = F.avg_pool2d(support_value, _SCALE, _SCALE)
        means40 = self.parent_mean40(torch.cat(
            (parent_means, support_fraction), dim=1
        ))
        content40 = self.content_fuse40(contrast40, means40)
        return f160, f80, content40

    def _condition_high(self, f160: Tensor, d80: Tensor) -> Tensor:
        """Return D160 with no computational F160 bypass around D80."""

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

    def _assert_preclosure(self, q_preclosure: Tensor, q_closed: Tensor) -> None:
        correction = (q_closed - q_preclosure).detach().abs().amax()
        condition = correction <= self.closure_tolerance_k
        message = (
            "IPMR-Q v2 preclosure left its declared QM+QH subspace; "
            "final Q repair exceeds tolerance"
        )
        if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(condition, message)
        elif not bool(condition.cpu().item()):
            raise RuntimeError(message)

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> IPMRQV2Components:
        query_fine, query_coarse, query_support_bool, query_context = (
            self._validate_inputs(
                fine,
                coarse_k,
                support,
                context,
                temporal_available,
                query_index,
            )
        )
        query_support = query_support_bool.to(dtype=query_fine.dtype)
        context15 = self._context15(query_context)
        if self.activation_checkpointing and self.training \
                and torch.is_grad_enabled():
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
                query_fine,
                query_coarse,
                query_support,
                context15,
            )

        coarse_valid = torch.isfinite(query_coarse)
        middle_lift = lift_p80_coefficients(
            middle_coefficients, query_support_bool
        )
        middle_bands = orthogonal_q_bands(
            middle_lift.float(), query_support_bool, coarse_valid
        )
        high_bands = orthogonal_q_bands(
            high_proposal.float(), query_support_bool, coarse_valid
        )
        q_middle = middle_bands.q_middle
        q_high = high_bands.q_high
        q_preclosure = q_middle + q_high

        # A final registered Q closure is retained as a fail-closed numerical
        # guard.  A material repair is an architecture error, not something
        # silently absorbed into the prediction.
        q_k = orthogonal_q_bands(
            q_preclosure, query_support_bool, coarse_valid
        ).q_total
        self._assert_preclosure(q_preclosure, q_k)

        base_k = support_project(
            query_fine[:, :1], query_coarse, query_support_bool
        )
        prediction = base_k + q_k
        return IPMRQV2Components(
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
            fine,
            coarse_k,
            support,
            context,
            temporal_available,
            query_index,
        ).prediction_k
