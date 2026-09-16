"""Scene-calibrated continuous-grid Q reconstructor for G246 R2.

This candidate addresses three concrete failure modes observed in the first
QParent screen: phase-dependent error, weak scene-specific calibration, and
insufficient low-frequency context.  It deliberately does *not* emit sixteen
independent sub-pixel phases.  Instead, temporal evidence is summarized on the
4x4 parent lattice, decoded through a 40 -> 20 -> 10 -> 40 context path, and
then interpolated into a shared fine-grid decoder.  Three continuous-grid
residual experts are modulated by target-free scene statistics before the
result is projected strictly into the query support's Q space.

The public inference surface matches ``MTStyleQParent`` and contains no target,
validity, or eligibility argument.  The only scientific mask consumed by the
model is physical fine-pixel support.
"""

from __future__ import annotations

import math
from math import gcd
from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from ocnir import support_project

try:
    from .g246_content_q_pyramid import ContentOnlyQPyramid
except ImportError:  # pragma: no cover - direct code-path import
    from g246_content_q_pyramid import ContentOnlyQPyramid

try:
    from .g246_parent_dct_q import ParentDCT15QDecoder
except ImportError:  # pragma: no cover - direct code-path import
    from g246_parent_dct_q import ParentDCT15QDecoder


__all__ = ["SceneCalibratedContinuousQ", "CalibratedContinuousQComponents"]

_SCALE = 4
_PHASES = _SCALE**2
_REGISTERED_BASE_WIDTHS = (48, 64, 96)
_SIZE_LABELS = {48: "S", 64: "M", 96: "L"}
_S_REFERENCE_CHANNELS = {
    "fine_width": 64,
    "parent_width": 144,
    "middle_width": 208,
    "low_width": 288,
}


def _round_channels(value: float) -> int:
    """Round a scaled channel count to the nearest hardware-friendly multiple of 8."""

    return max(8, int(math.floor(value / 8.0 + 0.5)) * 8)


def _registered_channels(width: int) -> dict[str, int]:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width not in _REGISTERED_BASE_WIDTHS:
        raise ValueError("registered calibrated width must be one of 48, 64, 96")
    factor = width / 48.0
    if width == 48:
        # Literal values guarantee byte-compatible S state dictionaries.
        return dict(_S_REFERENCE_CHANNELS)
    return {
        name: _round_channels(channels * factor)
        for name, channels in _S_REFERENCE_CHANNELS.items()
    }


def _groups(channels: int) -> int:
    return gcd(channels, min(8, channels))


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels), channels)


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
        raise TypeError(f"{name} must be bool or a numeric binary tensor")
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _support_parent_contrast(values: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    """Remove each supported 4x4 parent mean and return contrast plus mean."""

    if values.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("parent contrast expects NCHW values and N1HW support")
    if values.shape[0] != support.shape[0] or values.shape[-2:] != support.shape[-2:]:
        raise ValueError("parent contrast values/support geometry differs")
    height, width = values.shape[-2:]
    if height % _SCALE or width % _SCALE:
        raise ValueError("parent contrast geometry must be divisible by four")
    parent_height, parent_width = height // _SCALE, width // _SCALE
    blocks = values.reshape(
        values.shape[0], values.shape[1], parent_height, _SCALE,
        parent_width, _SCALE,
    )
    weights = support.to(dtype=values.dtype).reshape(
        values.shape[0], 1, parent_height, _SCALE, parent_width, _SCALE
    )
    counts = weights.sum(dim=(3, 5))
    means = (blocks * weights).sum(dim=(3, 5)) / counts.clamp_min(1.0)
    means = torch.where(counts > 0, means, torch.zeros_like(means))
    expanded = means.repeat_interleave(_SCALE, dim=-2).repeat_interleave(
        _SCALE, dim=-1
    )
    return (values - expanded) * support.to(dtype=values.dtype), means


def _support_stats(values: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    """Return support-weighted per-channel spatial mean and standard deviation."""

    weights = support.to(dtype=values.dtype)
    counts = weights.sum(dim=(-2, -1)).clamp_min(1.0)
    mean = (values * weights).sum(dim=(-2, -1)) / counts
    centered = (values - mean[..., None, None]) * weights
    variance = centered.square().sum(dim=(-2, -1)) / counts
    return mean, torch.sqrt(variance.clamp_min(1.0e-6))


class _ResidualDW(nn.Module):
    """Memory-light spatial block used on both fine and parent lattices."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        expansion: int = 3,
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
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1.0e-2))

    def forward(self, inputs: Tensor) -> Tensor:
        value = F.silu(self.norm(self.depthwise(inputs)))
        value = self.contract(F.silu(self.expand(value)))
        return inputs + self.scale * value


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
                in_channels, in_channels, 3, stride=2, padding=1,
                groups=in_channels, bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=False),
        )


class _HierarchicalPack(nn.Module):
    """Gradually encode fine features on H -> H/2 -> H/4 lattices.

    The legacy pack exposes all sixteen 4x4 phases as channels and compresses
    ``16 * fine_width`` directly to ``parent_width`` with one pointwise
    projection.  This opt-in alternative keeps the same parent-token output
    contract but performs spatial mixing at the otherwise absent H/2 scale
    before reaching H/4.  For registered S widths its parameter count is
    138,624 versus 147,744 for the legacy pack.
    """

    schema_version = "g246-hierarchical-pack-v1"
    scale_path = "H -> H/2 -> H/4 (160->80->40)"
    residual_blocks_at_half_scale = 2
    feature_lattice_receptive_field = 23

    def __init__(self, fine_width: int, parent_width: int) -> None:
        super().__init__()
        # Two thirds of the parent width gives registered S/M/L intermediate
        # widths 96/128/192 while remaining valid for lightweight tests.
        half_width = _round_channels((2.0 / 3.0) * parent_width)
        self.fine_width = fine_width
        self.half_width = half_width
        self.parent_width = parent_width
        self.down_half = _Downsample(fine_width, half_width)
        self.blocks_half = nn.Sequential(
            _ResidualDW(half_width),
            _ResidualDW(half_width),
        )
        self.down_parent = _Downsample(half_width, parent_width)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputs: Tensor) -> Tensor:
        half = self.blocks_half(self.down_half(inputs))
        return self.down_parent(half)


class _ContinuousDateEncoder(nn.Module):
    """Create a parent token while retaining a continuous query fine feature."""

    def __init__(
        self,
        fine_channels: int,
        context_dim: int,
        fine_width: int,
        parent_width: int,
        fine_blocks: int,
        parent_blocks: int,
        hierarchical_pack: bool = False,
    ) -> None:
        super().__init__()
        self.fine_channels = fine_channels
        self.context_dim = context_dim
        self.fine_width = fine_width
        self.parent_width = parent_width
        self.hierarchical_pack_enabled = hierarchical_pack

        # Kelvin/base contrast, all guidance contrasts, and physical support.
        self.fine_stem = nn.Sequential(
            nn.Conv2d(fine_channels + 1, fine_width, 3, padding=1, bias=False),
            _norm(fine_width),
            nn.SiLU(inplace=False),
            *(_ResidualDW(fine_width, expansion=2) for _ in range(fine_blocks)),
        )
        # Pixel-unshuffle is used only to construct a context token.  The final
        # residual is decoded on the fine lattice and has no phase-specific head.
        self.pack = (
            None
            if hierarchical_pack
            else _Project(_PHASES * fine_width, parent_width)
        )
        self.hierarchical_pack = (
            _HierarchicalPack(fine_width, parent_width)
            if hierarchical_pack else None
        )
        self.physical = nn.Sequential(
            nn.Conv2d(fine_channels + 4, parent_width, 3, padding=1, bias=False),
            _norm(parent_width),
            nn.SiLU(inplace=False),
        )
        descriptor_dim = 2 * fine_channels + context_dim + 6
        self.descriptor = nn.Sequential(
            nn.Linear(descriptor_dim, parent_width),
            nn.SiLU(inplace=False),
            nn.Linear(parent_width, parent_width),
            nn.SiLU(inplace=False),
        )
        self.parent_film = nn.Linear(parent_width, 2 * parent_width)
        self.merge = _Project(2 * parent_width, parent_width)
        self.parent_blocks = nn.Sequential(
            *(_ResidualDW(parent_width) for _ in range(parent_blocks))
        )

    def forward(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values = torch.cat(((fine[:, :1] - 300.0) / 20.0, fine[:, 1:]), dim=1)
        contrast, parent_means = _support_parent_contrast(values, support)
        fine_features = self.fine_stem(torch.cat((contrast, support), dim=1))
        if self.hierarchical_pack is None:
            if self.pack is None:
                raise AssertionError("legacy pack module is missing")
            packed = self.pack(F.pixel_unshuffle(fine_features, _SCALE))
        else:
            packed = self.hierarchical_pack(fine_features)

        packed_support = F.pixel_unshuffle(support, _SCALE)
        support_fraction = packed_support.mean(dim=1, keepdim=True)
        coarse_valid = torch.isfinite(coarse_k) & available[:, None, None, None]
        coarse_safe = torch.where(
            coarse_valid,
            coarse_k.to(dtype=fine.dtype),
            torch.full_like(coarse_k, 300.0, dtype=fine.dtype),
        )
        base_parent_k = 300.0 + 20.0 * parent_means[:, :1]
        physical = self.physical(
            torch.cat(
                (
                    parent_means,
                    (coarse_safe - 300.0) / 20.0,
                    coarse_valid.to(dtype=fine.dtype),
                    support_fraction,
                    (base_parent_k - coarse_safe) / 20.0,
                ),
                dim=1,
            )
        )

        mean, std = _support_stats(values, support)
        coarse_centered = (coarse_safe - 300.0) / 20.0
        coarse_weight = coarse_valid.to(dtype=fine.dtype)
        coarse_count = coarse_weight.sum(dim=(-2, -1)).clamp_min(1.0)
        coarse_mean = (coarse_centered * coarse_weight).sum(dim=(-2, -1))
        coarse_mean = coarse_mean / coarse_count
        coarse_variance = (
            (coarse_centered - coarse_mean[..., None, None]).square()
            * coarse_weight
        ).sum(dim=(-2, -1)) / coarse_count
        coarse_std = coarse_variance.clamp_min(1.0e-6).sqrt()
        delta_x = coarse_centered[..., :, 1:] - coarse_centered[..., :, :-1]
        delta_y = coarse_centered[..., 1:, :] - coarse_centered[..., :-1, :]
        valid_x = coarse_valid[..., :, 1:] & coarse_valid[..., :, :-1]
        valid_y = coarse_valid[..., 1:, :] & coarse_valid[..., :-1, :]
        weight_x = valid_x.to(dtype=fine.dtype)
        weight_y = valid_y.to(dtype=fine.dtype)
        grad_x = (
            (delta_x.square() * weight_x).sum(dim=(-2, -1))
            / weight_x.sum(dim=(-2, -1)).clamp_min(1.0)
        ).sqrt()
        grad_y = (
            (delta_y.square() * weight_y).sum(dim=(-2, -1))
            / weight_y.sum(dim=(-2, -1)).clamp_min(1.0)
        ).sqrt()
        coarse_coverage = coarse_weight.mean(dim=(-2, -1))
        support_scene = support.mean(dim=(-2, -1))
        descriptor = self.descriptor(
            torch.cat(
                (
                    mean,
                    std,
                    context.to(dtype=fine.dtype),
                    coarse_mean,
                    coarse_std,
                    grad_x,
                    grad_y,
                    coarse_coverage,
                    support_scene,
                ),
                dim=1,
            )
        )
        scale, shift = self.parent_film(descriptor).chunk(2, dim=1)
        packed = packed * (1.0 + 0.15 * torch.tanh(scale)[..., None, None])
        packed = packed + 0.15 * shift[..., None, None]
        token = self.parent_blocks(self.merge(torch.cat((packed, physical), dim=1)))

        mask = available[:, None].to(dtype=fine.dtype)
        return (
            fine_features * mask[..., None, None],
            token * mask[..., None, None],
            descriptor * mask,
        )


class _TemporalSetFusion(nn.Module):
    """Permutation-invariant query-relative temporal fusion."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("parent_width must be divisible by attention_heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
        )
        self.merge = _Project(2 * channels, channels)

    @staticmethod
    def gather_query(values: Tensor, query_index: Tensor) -> Tensor:
        shape = (values.shape[0], 1, *values.shape[2:])
        index = query_index.reshape(values.shape[0], 1, *([1] * (values.ndim - 2)))
        return torch.gather(values, 1, index.expand(shape)).squeeze(1)

    def forward(self, tokens: Tensor, available: Tensor, query_index: Tensor) -> Tensor:
        batch, times, channels, height, width = tokens.shape
        query_token = self.gather_query(tokens, query_index)
        flat = tokens.reshape(batch * times, channels, height, width)
        keys = self.key(flat).reshape(
            batch, times, self.heads, self.head_dim, height, width
        )
        differences = tokens - query_token[:, None]
        values = self.value(differences.reshape(batch * times, channels, height, width))
        values = values.reshape(batch, times, self.heads, self.head_dim, height, width)
        query = self.query(query_token).reshape(
            batch, self.heads, self.head_dim, height, width
        )
        scores = torch.einsum("bhdxy,bthdxy->bthxy", query, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~available[:, :, None, None, None], float("-inf"))
        weights = torch.softmax(scores.float(), dim=1).to(dtype=tokens.dtype)
        delta = torch.einsum("bthxy,bthdxy->bhdxy", weights, values)
        delta = self.output(delta.reshape(batch, channels, height, width))
        gate = torch.sigmoid(self.gate(torch.cat((query_token, delta), dim=1)))
        fused = query_token + gate * delta
        return self.merge(torch.cat((fused, query_token), dim=1))


class _QueryRelativeT3Fusion(nn.Module):
    """Zero-gated, permutation-invariant auxiliary-date residuals.

    The registered single-date path is computed before this module is called.
    This module therefore never replaces the query representation: it can only
    add three independently zero-gated residuals to its fine-grid feature,
    parent token and scene code.  Every date uses the same pointwise rule and
    the aggregation is a masked sum, so auxiliary slot order has no semantics.

    A shared descriptor/context similarity path controls how much each
    available auxiliary contributes.  Unavailable dates and the query slot are
    masked *before* normalization and reduction.  There is deliberately no
    phase index, temporal-position embedding, or unconditional temporal mean.
    """

    schema_version = "g246-query-relative-t3-fusion-v1"

    def __init__(
        self,
        fine_width: int,
        parent_width: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        similarity_width = max(8, min(32, parent_width // 4))
        # Five explicit, scale-free comparison statistics are used: encoded
        # descriptor cosine/distance, raw context cosine/distance, and the
        # circular DOY similarity carried by context channels 1:3.  The same
        # MLP is applied to every auxiliary date.
        self.seasonal_similarity = nn.Sequential(
            nn.Linear(5, similarity_width),
            nn.SiLU(inplace=False),
            nn.Linear(similarity_width, 1),
        )
        self.fine_residual = nn.Conv2d(
            fine_width, fine_width, 1, bias=False
        )
        self.parent_residual = nn.Conv2d(
            parent_width, parent_width, 1, bias=False
        )
        self.scene_residual = nn.Linear(
            parent_width + context_dim, parent_width, bias=False
        )
        self.fine_residual_gate = nn.Parameter(torch.zeros(()))
        self.parent_residual_gate = nn.Parameter(torch.zeros(()))
        self.scene_residual_gate = nn.Parameter(torch.zeros(()))
        self.apply(_initialize)

    @staticmethod
    def _cosine_and_distance(values: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        query_expanded = query[:, None].expand_as(values)
        cosine = F.cosine_similarity(values.float(), query_expanded.float(), dim=-1)
        scale = query_expanded.float().square().mean(dim=-1).clamp_min(
            1.0e-6
        ).sqrt()
        distance = (
            (
                (values.float() - query_expanded.float())
                .square().mean(dim=-1).clamp_min(1.0e-6).sqrt()
                - 1.0e-3
            )
            / scale
        )
        return cosine.to(dtype=values.dtype), distance.to(dtype=values.dtype)

    @staticmethod
    def _masked_average(values: Tensor, weights: Tensor) -> Tensor:
        expanded = weights.reshape(
            weights.shape[0], weights.shape[1],
            *([1] * (values.ndim - 2)),
        )
        return (values * expanded).sum(dim=1)

    def forward(
        self,
        query_fine: Tensor,
        query_parent_token: Tensor,
        baseline_parent: Tensor,
        query_descriptor: Tensor,
        query_context: Tensor,
        fine_features: Tensor,
        parent_tokens: Tensor,
        descriptors: Tensor,
        contexts: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
        baseline_scene_code: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _batch, times = temporal_available.shape
        slot = torch.arange(times, device=query_index.device)[None, :]
        auxiliary = temporal_available & (slot != query_index[:, None])

        descriptor_cosine, descriptor_distance = self._cosine_and_distance(
            descriptors, query_descriptor
        )
        context_cosine, context_distance = self._cosine_and_distance(
            contexts, query_context
        )
        seasonal_cosine = F.cosine_similarity(
            contexts[..., 1:3].float(),
            query_context[:, None, 1:3].float(),
            dim=-1,
        ).to(dtype=contexts.dtype)
        similarity_features = torch.stack(
            (
                descriptor_cosine,
                descriptor_distance,
                context_cosine,
                context_distance,
                seasonal_cosine,
            ),
            dim=-1,
        )
        affinity = torch.sigmoid(
            self.seasonal_similarity(similarity_features).squeeze(-1)
        )
        affinity = torch.where(auxiliary, affinity, torch.zeros_like(affinity))
        # Divide by the number of usable auxiliaries, not by total affinity.
        # The latter would force even a uniformly dissimilar set to contribute
        # at full strength and would make similarity irrelevant when only one
        # auxiliary is present.
        auxiliary_count = auxiliary.sum(dim=1, keepdim=True).to(affinity.dtype)
        weights = affinity / auxiliary_count.clamp_min(1.0)

        fine_delta = self._masked_average(
            fine_features - query_fine[:, None], weights
        )
        parent_delta = self._masked_average(
            parent_tokens - query_parent_token[:, None], weights
        )
        descriptor_delta = self._masked_average(
            descriptors - query_descriptor[:, None], weights
        )
        context_delta = self._masked_average(
            contexts - query_context[:, None], weights
        )

        fine = query_fine + torch.tanh(self.fine_residual_gate) * self.fine_residual(
            fine_delta
        )
        parent = baseline_parent + torch.tanh(
            self.parent_residual_gate
        ) * self.parent_residual(parent_delta)
        scene = baseline_scene_code + torch.tanh(
            self.scene_residual_gate
        ) * self.scene_residual(torch.cat((descriptor_delta, context_delta), dim=1))
        return fine, parent, scene


class _ParentContextUNet(nn.Module):
    """Low-frequency parent-lattice context: H/4 -> H/8 -> H/16 -> H/4."""

    def __init__(
        self,
        widths: tuple[int, int, int],
        blocks: tuple[int, int, int, int, int],
    ) -> None:
        super().__init__()
        high, middle, low = widths
        enc_high, enc_middle, enc_low, dec_middle, dec_high = blocks
        self.high = nn.Sequential(*(_ResidualDW(high) for _ in range(enc_high)))
        self.down_middle = _Downsample(high, middle)
        self.middle = nn.Sequential(*(_ResidualDW(middle) for _ in range(enc_middle)))
        self.down_low = _Downsample(middle, low)
        self.low = nn.Sequential(*(_ResidualDW(low) for _ in range(enc_low)))
        self.up_middle = _Project(low, middle)
        self.merge_middle = _Project(2 * middle, middle)
        self.decode_middle = nn.Sequential(
            *(_ResidualDW(middle) for _ in range(dec_middle))
        )
        self.up_high = _Project(middle, high)
        self.merge_high = _Project(2 * high, high)
        self.decode_high = nn.Sequential(*(_ResidualDW(high) for _ in range(dec_high)))

    def forward(self, inputs: Tensor) -> Tensor:
        high = self.high(inputs)
        middle = self.middle(self.down_middle(high))
        low = self.low(self.down_low(middle))
        decoded_middle = F.interpolate(
            low, size=middle.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_middle = self.up_middle(decoded_middle)
        decoded_middle = self.decode_middle(
            self.merge_middle(torch.cat((decoded_middle, middle), dim=1))
        )
        decoded_high = F.interpolate(
            decoded_middle, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_high = self.up_high(decoded_high)
        return self.decode_high(
            self.merge_high(torch.cat((decoded_high, high), dim=1))
        )


class _ContinuousMixtureHead(nn.Module):
    """Fine-grid experts with scene and spatial calibration, never phase heads."""

    def __init__(self, fine_width: int, parent_width: int, experts: int) -> None:
        super().__init__()
        self.experts = experts
        self.fine_projection = _Project(parent_width, fine_width)
        self.fuse = _Project(2 * fine_width, fine_width)
        self.scene_film = nn.Linear(parent_width, 2 * fine_width)
        self.blocks = nn.Sequential(
            _ResidualDW(fine_width, dilation=1),
            _ResidualDW(fine_width, dilation=2),
            _ResidualDW(fine_width, dilation=3),
            _ResidualDW(fine_width, dilation=1),
        )
        self.local_expert = nn.Conv2d(fine_width, 1, 3, padding=1)
        self.edge_expert = nn.Sequential(
            nn.Conv2d(
                fine_width, fine_width, 5, padding=2, groups=fine_width, bias=False
            ),
            _norm(fine_width),
            nn.SiLU(inplace=False),
            nn.Conv2d(fine_width, 1, 1),
        )
        self.context_expert = nn.Conv2d(fine_width, 1, 5, padding=2)
        if experts != 3:
            raise ValueError("continuous mixture currently defines exactly three experts")
        self.spatial_gate = nn.Conv2d(fine_width, experts, 3, padding=1)
        self.scene_gate = nn.Linear(parent_width, experts)
        self.scene_gain = nn.Linear(parent_width, experts)

    def forward(
        self, query_fine: Tensor, parent_context: Tensor, scene_code: Tensor
    ) -> tuple[Tensor, Tensor]:
        parent_up = F.interpolate(
            parent_context,
            size=query_fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        parent_up = self.fine_projection(parent_up)
        fused = self.fuse(torch.cat((query_fine, parent_up), dim=1))
        scale, shift = self.scene_film(scene_code).chunk(2, dim=1)
        fused = fused * (1.0 + 0.15 * torch.tanh(scale)[..., None, None])
        fused = fused + 0.15 * shift[..., None, None]
        fused = self.blocks(fused)

        expert_values = torch.cat(
            (
                self.local_expert(fused),
                self.edge_expert(fused),
                self.context_expert(parent_up),
            ),
            dim=1,
        )
        gains = 1.0 + 0.25 * torch.tanh(self.scene_gain(scene_code))
        expert_values = expert_values * gains[..., None, None]
        logits = self.spatial_gate(fused) + self.scene_gate(scene_code)[..., None, None]
        weights = torch.softmax(logits.float(), dim=1).to(dtype=fused.dtype)
        return (expert_values * weights).sum(dim=1, keepdim=True), weights


class _SupportAwareAllocationAdapter(nn.Module):
    """Shared-token, support-aware competition inside every 4x4 parent.

    The adapter deliberately has no phase-indexed parameters.  Each supported
    fine pixel is scored by the same pointwise token network, conditioned on a
    broadcast parent-context token.  The sixteen locations then compete only
    through permutation-symmetric reductions.  Consequently this module is D4
    equivariant (in fact, equivariant to every within-parent permutation) when
    its fine and parent feature maps are transformed together.

    The detached old continuous-Q residual supplies a parent-local, unit-RMS
    ranking score.  A shared pointwise network learns only a delta to that
    score.  Competition is centred and RMS-normalised over physical support in
    each parent.  Unsupported pixels and parents with fewer than two supported
    pixels are exactly zero.  The parent-local amplitude controls are
    zero-initialised, so migration is an exact identity while every usable
    parent can receive a local amplitude gradient on the first step.
    """

    schema_version = "g246-support-aware-allocation-v2"
    max_amplitude_k = 4.0
    min_temperature = 0.25
    max_temperature = 4.0
    initial_temperature = 1.0
    normalization_epsilon = 1.0e-6

    def __init__(self, fine_width: int, parent_width: int) -> None:
        super().__init__()
        token_width = max(8, min(32, fine_width))
        self.fine_token = nn.Conv2d(fine_width, token_width, 1, bias=False)
        self.parent_token = nn.Conv2d(parent_width, token_width, 1, bias=False)
        self.delta_score = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Conv2d(token_width, token_width, 1),
            nn.SiLU(inplace=False),
            nn.Conv2d(token_width, 1, 1),
        )
        # Channel zero predicts signed bounded amplitude; channel one predicts
        # a bounded positive softmax temperature on the parent lattice.
        self.parent_controls = nn.Conv2d(parent_width, 2, 1)
        self.apply(_initialize)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Restore the registered zero-impact v2 controls after outer init."""

        # Allocation v2 is an exact identity at initialization without a
        # global scalar bottleneck.  Amplitude is locally learnable from the
        # first backward pass.  Temperature starts at the safe 1.0 constant.
        nn.init.zeros_(self.delta_score[-1].weight)
        nn.init.zeros_(self.delta_score[-1].bias)
        with torch.no_grad():
            self.parent_controls.weight[0].zero_()
            self.parent_controls.bias[0].zero_()
            self.parent_controls.weight[1].zero_()
            temperature_fraction = (
                (self.initial_temperature - self.min_temperature)
                / (self.max_temperature - self.min_temperature)
            )
            self.parent_controls.bias[1].fill_(
                math.log(temperature_fraction / (1.0 - temperature_fraction))
            )

    @staticmethod
    def _phase_blocks(values: Tensor) -> Tensor:
        batch, channels, height, width = values.shape
        if height % _SCALE or width % _SCALE:
            raise ValueError("allocation geometry must be divisible by four")
        return (
            values.reshape(
                batch, channels, height // _SCALE, _SCALE,
                width // _SCALE, _SCALE,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, height // _SCALE, width // _SCALE, _PHASES)
        )

    @staticmethod
    def _fine_lattice(values: Tensor) -> Tensor:
        batch, channels, parent_height, parent_width, phases = values.shape
        if phases != _PHASES:
            raise ValueError("allocation blocks must contain sixteen phases")
        return (
            values.reshape(
                batch, channels, parent_height, parent_width, _SCALE, _SCALE
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(
                batch, channels, parent_height * _SCALE, parent_width * _SCALE
            )
        )

    def forward(
        self,
        query_fine_features: Tensor,
        parent_context: Tensor,
        query_support: Tensor,
        detached_raw_q: Tensor,
    ) -> Tensor:
        if query_fine_features.ndim != 4 or parent_context.ndim != 4:
            raise ValueError("allocation features must be NCHW tensors")
        if query_support.ndim != 4 or query_support.shape[1] != 1:
            raise ValueError("allocation support must have shape [B,1,H,W]")
        if detached_raw_q.ndim != 4 or detached_raw_q.shape[1] != 1:
            raise ValueError("allocation old raw_q must have shape [B,1,H,W]")
        if query_fine_features.shape[0] != parent_context.shape[0]:
            raise ValueError("allocation fine/parent batch sizes differ")
        if query_support.shape[0] != query_fine_features.shape[0] or \
                query_support.shape[-2:] != query_fine_features.shape[-2:]:
            raise ValueError("allocation fine/support geometry differs")
        if detached_raw_q.shape[0] != query_fine_features.shape[0] or \
                detached_raw_q.shape[-2:] != query_fine_features.shape[-2:]:
            raise ValueError("allocation fine/old raw_q geometry differs")
        if query_fine_features.shape[-2:] != (
            parent_context.shape[-2] * _SCALE,
            parent_context.shape[-1] * _SCALE,
        ):
            raise ValueError("allocation parent context is not on the H/4 lattice")

        support = _binary(query_support, "query_support").to(
            dtype=query_fine_features.dtype
        )
        parent_up = parent_context.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)
        delta_score = self.delta_score(
            self.fine_token(query_fine_features) + self.parent_token(parent_up)
        )
        controls = self.parent_controls(parent_context)
        amplitude = self.max_amplitude_k * torch.tanh(controls[:, :1])
        temperature = self.min_temperature + (
            self.max_temperature - self.min_temperature
        ) * torch.sigmoid(controls[:, 1:2])

        phase_support = self._phase_blocks(support).to(dtype=torch.bool)
        support_fp32 = phase_support.to(dtype=torch.float32)
        counts = phase_support.sum(dim=-1, keepdim=True)
        counts_fp32 = counts.to(dtype=torch.float32).clamp_min(1.0)
        usable = counts >= 2

        # The old head supplies a stable parent-local ranking in Kelvin, but
        # never receives adapter gradients.  RMS normalization removes the
        # parent offset and converts it to a suitable unitless softmax scale.
        old_blocks = self._phase_blocks(detached_raw_q.detach().float())
        old_mean = (old_blocks * support_fp32).sum(
            dim=-1, keepdim=True
        ) / counts_fp32
        old_centred = (old_blocks - old_mean) * support_fp32
        old_variance = old_centred.square().sum(
            dim=-1, keepdim=True
        ) / counts_fp32
        old_rms = torch.sqrt(old_variance.clamp_min(
            self.normalization_epsilon**2
        ))
        old_score = torch.where(
            usable, old_centred / old_rms, torch.zeros_like(old_centred)
        )
        phase_delta = self._phase_blocks(delta_score).float()
        phase_logits = old_score + phase_delta
        scaled_logits = phase_logits / temperature.float()[..., None]
        # Softmax is evaluated in fp32 for stable mixed-precision competition.
        # Multiplication by the mask makes the all-unsupported case finite.
        masked_logits = scaled_logits.float().masked_fill(
            ~phase_support, torch.finfo(torch.float32).min
        )
        probabilities = torch.softmax(masked_logits, dim=-1) * support_fp32
        means = probabilities.sum(dim=-1, keepdim=True) / counts_fp32
        centred = (probabilities - means) * support_fp32
        variance = centred.square().sum(dim=-1, keepdim=True) / counts_fp32
        rms = torch.sqrt(variance.clamp_min(self.normalization_epsilon**2))
        normalised = torch.where(
            usable, centred / rms, torch.zeros_like(centred)
        )
        allocation = self._fine_lattice(
            amplitude.float()[..., None] * normalised
        )
        return allocation.to(dtype=query_fine_features.dtype)


class _D4SymmetricDepthwiseConv(nn.Module):
    """Depthwise convolution whose effective kernel is exactly D4 invariant.

    A conventional learned depthwise kernel is translation equivariant but is
    not generally equivariant to rotations or reflections. Averaging the
    shared kernel over its eight dihedral transforms at every forward pass
    keeps the learnable representation compact while making the effective
    spatial rule independent of orientation. There are no phase or spatial
    position parameters.
    """

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("D4 depthwise kernel size must be positive and odd")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(self.channels, 1, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(self.channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    @staticmethod
    def symmetrize(weight: Tensor) -> Tensor:
        reflected = torch.flip(weight, dims=(-1,))
        transforms = tuple(torch.rot90(weight, k, dims=(-2, -1)) for k in range(4))
        transforms += tuple(
            torch.rot90(reflected, k, dims=(-2, -1)) for k in range(4)
        )
        return torch.stack(transforms, dim=0).mean(dim=0)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self.symmetrize(self.weight),
            self.bias,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )


class _D4DepthwiseResidual(nn.Module):
    """D4-equivariant depthwise residual block with shared pointwise mixing."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = int(channels) * int(expansion)
        self.depthwise = _D4SymmetricDepthwiseConv(channels, kernel_size=5)
        self.norm = _norm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.contract = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1.0e-2))
        self.expand.apply(_initialize)
        self.contract.apply(_initialize)

    def forward(self, inputs: Tensor) -> Tensor:
        value = F.silu(self.norm(self.depthwise(inputs)))
        value = self.contract(F.silu(self.expand(value)))
        return inputs + self.scale * value


class _QResidualRefiner(nn.Module):
    """Identity-initialized, D4-equivariant correction of an already learned Q.

    The refiner consumes the old continuous residual itself, its support-aware
    parent contrast and RMS (without dividing by that RMS), centred Fine52
    predictors, physical support/coarse availability, and the fine, parent and
    scene features that produced the old residual. A shared orientation-free
    branch predicts a signed bounded residual, while a parent-local bounded
    signed gain rescales the already learned Q. Both output heads are
    initialized to exact zero, so adding the module preserves every old
    prediction bit-for-bit at migration time without first opening a random
    spatial template.

    The returned tensor is only a residual proposal. The caller still applies
    the established support-aware Q projection, which remains the sole coarse
    closure mechanism.
    """

    schema_version = "g246-q-residual-refiner-v2"
    max_residual_k = 4.0
    rms_epsilon = 1.0e-6

    def __init__(
        self,
        fine_channels: int,
        fine_width: int,
        parent_width: int,
        *,
        hidden_width: int = 96,
        blocks: int = 3,
    ) -> None:
        super().__init__()
        for value, name in (
            (fine_channels, "fine_channels"),
            (fine_width, "fine_width"),
            (parent_width, "parent_width"),
            (hidden_width, "hidden_width"),
            (blocks, "blocks"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.fine_channels = fine_channels
        self.fine_width = fine_width
        self.parent_width = parent_width
        self.hidden_width = hidden_width

        self.fine_projection = nn.Conv2d(
            fine_width, hidden_width, 1, bias=False
        )
        self.predictor_projection = nn.Conv2d(
            fine_channels, hidden_width, 1, bias=False
        )
        # old Q, parent contrast, parent RMS, support and coarse availability
        self.raw_projection = nn.Conv2d(5, hidden_width, 1, bias=False)
        self.parent_projection = nn.Conv2d(
            parent_width, hidden_width, 1, bias=False
        )
        self.scene_projection = nn.Linear(
            parent_width, hidden_width, bias=False
        )
        self.input_norm = _norm(hidden_width)
        self.residual_branch = nn.Sequential(
            *(_D4DepthwiseResidual(hidden_width) for _ in range(blocks))
        )
        self.signed_residual = nn.Conv2d(hidden_width, 1, 1)

        gain_width = max(32, min(64, parent_width))
        # Parent context, invariant scene code, old-Q RMS, support fraction and
        # coarse availability jointly determine a local bounded signed gain.
        self.parent_gain = nn.Sequential(
            nn.Conv2d(2 * parent_width + 3, gain_width, 1),
            nn.SiLU(inplace=False),
            nn.Conv2d(gain_width, 1, 1),
        )
        for module in (
            self.fine_projection,
            self.predictor_projection,
            self.raw_projection,
            self.parent_projection,
            self.scene_projection,
            self.input_norm,
            self.signed_residual,
            self.parent_gain,
        ):
            module.apply(_initialize)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Make both additive paths exact zero with useful first gradients."""

        final_gain = self.parent_gain[-1]
        nn.init.zeros_(final_gain.weight)
        nn.init.zeros_(final_gain.bias)
        nn.init.zeros_(self.signed_residual.weight)
        nn.init.zeros_(self.signed_residual.bias)

    @staticmethod
    def _phase_blocks(values: Tensor) -> Tensor:
        return _SupportAwareAllocationAdapter._phase_blocks(values)

    @staticmethod
    def _fine_lattice(values: Tensor) -> Tensor:
        return _SupportAwareAllocationAdapter._fine_lattice(values)

    def _support_parent_statistics(
        self, raw_q: Tensor, support: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        phase_support = self._phase_blocks(support).to(dtype=torch.bool)
        support_fp32 = phase_support.to(dtype=torch.float32)
        counts = phase_support.sum(dim=-1, keepdim=True)
        counts_fp32 = counts.to(dtype=torch.float32).clamp_min(1.0)
        raw_blocks = self._phase_blocks(raw_q.float())
        means = (raw_blocks * support_fp32).sum(
            dim=-1, keepdim=True
        ) / counts_fp32
        centred = (raw_blocks - means) * support_fp32
        variance = centred.square().sum(dim=-1, keepdim=True) / counts_fp32
        rms = variance.clamp_min(self.rms_epsilon**2).sqrt()
        # A singleton has no identifiable within-parent contrast. Returning
        # exact zero after the safe sqrt keeps count-0/count-1 forward and
        # backward finite without turning epsilon into a learned signal.
        rms = torch.where(counts >= 2, rms, torch.zeros_like(rms))
        contrast = self._fine_lattice(centred)
        support_fraction = counts.to(dtype=torch.float32).squeeze(-1) / float(_PHASES)
        return contrast, rms.squeeze(-1), support_fraction

    def forward(
        self,
        raw_q: Tensor,
        query_predictor: Tensor,
        query_support: Tensor,
        coarse_valid: Tensor,
        query_fine_features: Tensor,
        parent_context: Tensor,
        scene_code: Tensor,
    ) -> Tensor:
        if raw_q.ndim != 4 or raw_q.shape[1] != 1:
            raise ValueError("refiner raw_q must have shape [B,1,H,W]")
        if query_predictor.ndim != 4 \
                or query_predictor.shape[1] != self.fine_channels:
            raise ValueError(
                f"refiner predictor must have shape [B,{self.fine_channels},H,W]"
            )
        if query_fine_features.ndim != 4 \
                or query_fine_features.shape[1] != self.fine_width:
            raise ValueError("refiner fine feature channel contract differs")
        if parent_context.ndim != 4 \
                or parent_context.shape[1] != self.parent_width:
            raise ValueError("refiner parent feature channel contract differs")
        if scene_code.shape != (raw_q.shape[0], self.parent_width):
            raise ValueError("refiner scene code contract differs")
        if query_support.shape != raw_q.shape:
            raise ValueError("refiner support must match raw_q")
        expected_parent = (
            raw_q.shape[0], 1, raw_q.shape[-2] // _SCALE,
            raw_q.shape[-1] // _SCALE,
        )
        if coarse_valid.shape != expected_parent:
            raise ValueError("refiner coarse-valid parent geometry differs")
        if query_predictor.shape[0] != raw_q.shape[0] \
                or query_predictor.shape[-2:] != raw_q.shape[-2:] \
                or query_fine_features.shape[0] != raw_q.shape[0] \
                or query_fine_features.shape[-2:] != raw_q.shape[-2:]:
            raise ValueError("refiner fine-lattice inputs differ")
        if parent_context.shape[0] != raw_q.shape[0] \
                or parent_context.shape[-2:] != coarse_valid.shape[-2:]:
            raise ValueError("refiner parent-lattice inputs differ")

        support = _binary(query_support, "query_support").to(
            dtype=query_fine_features.dtype
        )
        coarse_available = _binary(coarse_valid, "coarse_valid")
        contrast, parent_rms, support_fraction = (
            self._support_parent_statistics(raw_q, support)
        )
        parent_up = parent_context.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)
        rms_up = parent_rms.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)
        coarse_up = coarse_available.to(dtype=query_fine_features.dtype)
        coarse_up = coarse_up.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)

        # Only the Kelvin predictor needs physical centring. The remaining
        # Fine52 channels have already been normalized by their registered
        # sidecar contract and must not be shifted by a hard-coded temperature.
        centred_predictor = torch.cat(
            (
                (query_predictor[:, :1] - 300.0) / 20.0,
                query_predictor[:, 1:],
            ),
            dim=1,
        )
        # Match the old model's actually delivered Q: support-centred where a
        # coarse observation constrains the parent, and the supported raw head
        # where that observation is missing. Feeding the unprojected raw-head
        # gauge on a constrained parent through nonlinear convolutions would
        # let an arbitrary parent constant reappear as learned fine detail.
        q0 = torch.where(
            coarse_up.to(dtype=torch.bool),
            contrast.to(dtype=raw_q.dtype),
            raw_q * support.to(dtype=raw_q.dtype),
        )
        raw_inputs = torch.cat(
            (
                q0 / self.max_residual_k,
                contrast.to(dtype=raw_q.dtype) / self.max_residual_k,
                rms_up.to(dtype=raw_q.dtype) / self.max_residual_k,
                support.to(dtype=raw_q.dtype),
                coarse_up.to(dtype=raw_q.dtype),
            ),
            dim=1,
        )
        hidden = (
            self.fine_projection(query_fine_features)
            + self.predictor_projection(
                centred_predictor.to(dtype=query_fine_features.dtype)
            )
            + self.raw_projection(raw_inputs.to(dtype=query_fine_features.dtype))
            + self.parent_projection(parent_up)
            + self.scene_projection(scene_code)[..., None, None]
        )
        hidden = self.residual_branch(F.silu(self.input_norm(hidden)))
        signed = self.max_residual_k * torch.tanh(self.signed_residual(hidden))

        parent_height, parent_width = parent_context.shape[-2:]
        scene_parent = scene_code[..., None, None].expand(
            -1, -1, parent_height, parent_width
        )
        gain_inputs = torch.cat(
            (
                parent_context,
                scene_parent,
                parent_rms.to(dtype=parent_context.dtype),
                support_fraction.to(dtype=parent_context.dtype),
                coarse_available.to(dtype=parent_context.dtype),
            ),
            dim=1,
        )
        # Limit the inherited-Q amplitude to 0.5--1.5 after the caller adds
        # this proposal to raw_q. Unlike a zero gate on a random residual
        # template, gain * q0 provides a deterministic, meaningful first
        # gradient. The independently zero-initialized signed head can learn a
        # new spatial correction on that same first step.
        gain = 0.5 * torch.tanh(self.parent_gain(gain_inputs))
        gain_up = gain.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)
        proposal = gain_up.to(dtype=signed.dtype) * q0.to(dtype=signed.dtype)
        proposal = proposal + signed
        return proposal * support.to(dtype=signed.dtype)


class CalibratedContinuousQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_k: Tensor
    expert_weights: Tensor


class SceneCalibratedContinuousQ(nn.Module):
    """Target-free S/M/L scene-calibrated model with a continuous Q head.

    Registered base widths 48/64/96 scale the original S channels by
    ``width/48`` and round to the nearest multiple of eight.  Width 48 is a
    literal compatibility anchor: its parameter count and state-dict tensors
    are unchanged from the pre-preset implementation.  Explicit internal
    widths remain available only for lightweight engineering tests.
    """

    scale_path = "H/4 -> H/8 -> H/16 -> H/8 -> H/4 (40->20->10->20->40 for H=160)"
    output_lattice = "continuous fine grid"
    widths = _REGISTERED_BASE_WIDTHS
    size_labels = dict(_SIZE_LABELS)
    geolocation_context_indices = (5, 6, 7, 8)
    geolocation_context_names = (
        "latitude_sin", "latitude_cos", "longitude_sin", "longitude_cos",
    )

    @staticmethod
    def registered_channels(width: int) -> dict[str, int]:
        """Resolve an S/M/L base width into the four internal channel widths."""

        return _registered_channels(width)

    def __init__(
        self,
        fine_channels: int = 22,
        context_dim: int = 12,
        *,
        width: int = 48,
        fine_width: int | None = None,
        parent_width: int | None = None,
        middle_width: int | None = None,
        low_width: int | None = None,
        fine_blocks: int = 2,
        date_parent_blocks: int = 2,
        scale_blocks: Sequence[int] = (2, 2, 2, 2, 2),
        attention_heads: int = 6,
        experts: int = 3,
        activation_checkpointing: bool = True,
        allocation_adapter: bool = False,
        t3_fusion: bool = False,
        q_refiner: bool = False,
        content_q_pyramid: bool = False,
        parent_dct15: bool = False,
        no_geo_core: bool = False,
        hierarchical_pack: bool = False,
    ) -> None:
        super().__init__()
        registered = self.registered_channels(width)
        fine_width = registered["fine_width"] if fine_width is None else fine_width
        parent_width = (
            registered["parent_width"] if parent_width is None else parent_width
        )
        middle_width = (
            registered["middle_width"] if middle_width is None else middle_width
        )
        low_width = registered["low_width"] if low_width is None else low_width
        for value, name in (
            (fine_channels, "fine_channels"),
            (context_dim, "context_dim"),
            (fine_width, "fine_width"),
            (parent_width, "parent_width"),
            (middle_width, "middle_width"),
            (low_width, "low_width"),
            (fine_blocks, "fine_blocks"),
            (date_parent_blocks, "date_parent_blocks"),
            (attention_heads, "attention_heads"),
            (experts, "experts"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if fine_channels < 2:
            raise ValueError("fine_channels must include Kelvin and guidance")
        if parent_width % attention_heads:
            raise ValueError("parent_width must be divisible by attention_heads")
        block_tuple = tuple(int(value) for value in scale_blocks)
        if len(block_tuple) != 5 or any(value < 1 for value in block_tuple):
            raise ValueError("scale_blocks must contain five positive depths")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be bool")
        if not isinstance(allocation_adapter, bool):
            raise TypeError("allocation_adapter must be bool")
        if not isinstance(t3_fusion, bool):
            raise TypeError("t3_fusion must be bool")
        if not isinstance(q_refiner, bool):
            raise TypeError("q_refiner must be bool")
        if not isinstance(content_q_pyramid, bool):
            raise TypeError("content_q_pyramid must be bool")
        if not isinstance(parent_dct15, bool):
            raise TypeError("parent_dct15 must be bool")
        if not isinstance(no_geo_core, bool):
            raise TypeError("no_geo_core must be bool")
        if not isinstance(hierarchical_pack, bool):
            raise TypeError("hierarchical_pack must be bool")
        if t3_fusion and context_dim < 3:
            raise ValueError("T3 fusion requires context DOY sin/cos channels 1:3")
        if content_q_pyramid and (fine_channels != 52 or context_dim != 19):
            raise ValueError(
                "content_q_pyramid requires the exact Fine52/Context19 contract"
            )
        if parent_dct15 and (fine_channels != 52 or context_dim != 19):
            raise ValueError(
                "parent_dct15 requires the exact Fine52/Context19 contract"
            )
        if no_geo_core and (fine_channels != 52 or context_dim != 19):
            raise ValueError(
                "no_geo_core requires the exact Fine52/Context19 public contract"
            )
        if hierarchical_pack and not no_geo_core:
            raise ValueError(
                "hierarchical_pack requires the Fine52/Context15 no_geo_core"
            )
        if content_q_pyramid and t3_fusion:
            raise ValueError(
                "content_q_pyramid and T3 fusion are mutually exclusive"
            )
        if sum((allocation_adapter, q_refiner, content_q_pyramid)) > 1:
            raise ValueError(
                "allocation_adapter, q_refiner and content_q_pyramid are "
                "mutually exclusive"
            )
        if parent_dct15 and any(
            (allocation_adapter, t3_fusion, q_refiner, content_q_pyramid)
        ):
            raise ValueError(
                "parent_dct15 is mutually exclusive with allocation_adapter, "
                "T3 fusion, q_refiner and content_q_pyramid"
            )
        if no_geo_core and any(
            (
                allocation_adapter,
                t3_fusion,
                q_refiner,
                content_q_pyramid,
                parent_dct15,
            )
        ):
            raise ValueError(
                "no_geo_core is mutually exclusive with allocation_adapter, "
                "T3 fusion, q_refiner, content_q_pyramid and parent_dct15"
            )

        self.fine_channels = fine_channels
        self.context_dim = context_dim
        # The public data contract remains Context19.  The explicit no-geo
        # core is structurally narrower: the four coordinate channels are
        # removed before the first learned operation rather than replaced by
        # constants that still leave unused weights in the encoder.
        self.encoder_context_dim = context_dim - 4 if no_geo_core else context_dim
        self.width = width
        self.size_label = self.size_labels[width]
        self.fine_width = fine_width
        self.parent_width = parent_width
        self.middle_width = middle_width
        self.low_width = low_width
        self.fine_blocks = fine_blocks
        self.date_parent_blocks = date_parent_blocks
        self.scale_blocks = block_tuple
        self.attention_heads = attention_heads
        self.experts = experts
        self.activation_checkpointing = activation_checkpointing
        self.allocation_adapter_enabled = allocation_adapter
        self.t3_fusion_enabled = t3_fusion
        self.q_refiner_enabled = q_refiner
        self.content_q_pyramid_enabled = content_q_pyramid
        self.parent_dct15_enabled = parent_dct15
        self.no_geo_core_enabled = no_geo_core
        self.hierarchical_pack_enabled = hierarchical_pack
        # This flag is not merely descriptive.  ``forward_components`` uses it
        # to zero the four identity-like coordinate channels before *any* core
        # encoder, temporal fusion, scene code, or extension can observe them.
        self.geolocation_context_masked = (
            content_q_pyramid or parent_dct15 or no_geo_core
        )
        fine_stem_receptive_field = 3 + 4 * fine_blocks
        self.pack_receptive_field_fine_pixels = (
            fine_stem_receptive_field
            + (_HierarchicalPack.feature_lattice_receptive_field - 1)
            if hierarchical_pack
            else fine_stem_receptive_field + (_SCALE - 1)
        )
        # Every date-parent block has a 5x5 depthwise convolution on an H/4
        # lattice and therefore adds sixteen fine pixels to the receptive-field
        # side length.  These values document mechanism, not an empirical RF.
        self.parent_token_receptive_field_fine_pixels = (
            self.pack_receptive_field_fine_pixels
            + 16 * date_parent_blocks
        )

        self.date_encoder = _ContinuousDateEncoder(
            fine_channels,
            self.encoder_context_dim,
            fine_width,
            parent_width,
            fine_blocks,
            date_parent_blocks,
            hierarchical_pack,
        )
        self.temporal_fusion = _TemporalSetFusion(parent_width, attention_heads)
        self.scene_fusion = nn.Sequential(
            nn.Linear(3 * parent_width, parent_width),
            nn.SiLU(inplace=False),
            nn.Linear(parent_width, parent_width),
            nn.SiLU(inplace=False),
        )
        self.context_unet = _ParentContextUNet(
            (parent_width, middle_width, low_width), block_tuple
        )
        self.continuous_head = _ContinuousMixtureHead(
            fine_width, parent_width, experts
        )
        # None is intentional: with the default flag the module tree, state
        # dictionary, parameter count and numerical path remain unchanged.
        self.allocation_adapter = (
            _SupportAwareAllocationAdapter(fine_width, parent_width)
            if allocation_adapter else None
        )
        # The optional T3 module is constructed only after the complete legacy
        # module tree has been initialized.  Thus opting into it cannot perturb
        # any pre-existing parameter, even before a checkpoint is migrated.
        self.t3_fusion = None
        self.q_refiner = None
        self.content_q_pyramid = None
        self.parent_dct15 = None
        self.apply(_initialize)

        # Stay close to the physical interpolation at initialization without
        # killing gradients into gates, style, or context paths.
        for head in (
            self.continuous_head.local_expert,
            self.continuous_head.edge_expert[-1],
            self.continuous_head.context_expert,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1.0e-3)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        nn.init.zeros_(self.continuous_head.spatial_gate.weight)
        nn.init.zeros_(self.continuous_head.spatial_gate.bias)
        nn.init.zeros_(self.continuous_head.scene_gate.weight)
        nn.init.zeros_(self.continuous_head.scene_gate.bias)
        nn.init.zeros_(self.continuous_head.scene_gain.weight)
        nn.init.zeros_(self.continuous_head.scene_gain.bias)
        if self.allocation_adapter is not None:
            # ``self.apply`` above visits the nested adapter.  Reapply its
            # scientific identity initialization after that outer traversal.
            self.allocation_adapter.reset_identity_initialization()
        self.t3_fusion = (
            _QueryRelativeT3Fusion(fine_width, parent_width, context_dim)
            if t3_fusion else None
        )
        self.q_refiner = (
            _QResidualRefiner(fine_channels, fine_width, parent_width)
            if q_refiner else None
        )
        self.content_q_pyramid = (
            ContentOnlyQPyramid(fine_channels, fine_width, parent_width)
            if content_q_pyramid else None
        )
        # Like the other opt-in migration modules, construct Parent-DCT15 only
        # after the complete legacy tree has received its historical
        # initialization.  Enabling it therefore cannot perturb inherited
        # parameters, while the default module tree remains byte-compatible.
        self.parent_dct15 = ParentDCT15QDecoder() if parent_dct15 else None

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
    ) -> tuple[Tensor, Tensor, Tensor]:
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
            raise ValueError(f"fine must have shape [B,T,{self.fine_channels},H,W]")
        batch, times, _, height, width = fine.shape
        if self.no_geo_core_enabled and times != 1:
            raise ValueError("no_geo_core requires single-date inputs (T=1)")
        if height % 16 or width % 16:
            raise ValueError("fine geometry must be divisible by sixteen")
        if coarse_k.shape != (batch, times, 1, height // _SCALE, width // _SCALE):
            raise ValueError("coarse_k must be [B,T,1,H/4,W/4]")
        if support.shape != (batch, times, 1, height, width):
            raise ValueError("support must be [B,T,1,H,W]")
        if context.shape != (batch, times, self.context_dim):
            raise ValueError(f"context must be [B,T,{self.context_dim}]")
        if temporal_available.shape != (batch, times):
            raise ValueError("temporal_available must be [B,T]")
        if query_index.shape != (batch,):
            raise ValueError("query_index must be [B]")
        if not fine.is_floating_point() or not coarse_k.is_floating_point():
            raise TypeError("fine and coarse_k must have floating dtypes")
        if not context.is_floating_point():
            raise TypeError("context must have a floating dtype")
        if len(
            {
                fine.device,
                coarse_k.device,
                support.device,
                context.device,
                temporal_available.device,
                query_index.device,
            }
        ) != 1:
            raise ValueError("all model inputs must be on one device")
        if _contains_true(~torch.isfinite(fine)) or _contains_true(~torch.isfinite(context)):
            raise ValueError("fine and context must be finite")
        if _contains_true(torch.isinf(coarse_k)):
            raise ValueError("coarse_k may contain NaN but not infinity")
        support_bool = _binary(support, "support")
        available = _binary(temporal_available, "temporal_available")
        if query_index.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("query_index must have an integer dtype")
        query_long = query_index.to(dtype=torch.long)
        if _contains_true((query_long < 0) | (query_long >= times)):
            raise ValueError("query_index is outside the temporal dimension")
        query_available = torch.gather(available, 1, query_long[:, None]).squeeze(1)
        if _contains_true(~query_available):
            raise ValueError("the selected query date must be available")
        return support_bool, available, query_long

    @staticmethod
    def _gather_query(values: Tensor, query_index: Tensor) -> Tensor:
        return _TemporalSetFusion.gather_query(values, query_index)

    def _effective_context(self, context: Tensor) -> Tensor:
        """Apply the fail-closed no-coordinate policy for content-only runs.

        Q-Pyramid and Parent-DCT15 retain the registered Context19 encoder and
        replace latitude/longitude sine and cosine with constants.  The
        explicit no-geo core instead removes those channels physically, so its
        date encoder really receives Context15.  Concatenation avoids an
        in-place write into a caller-owned tensor and makes gradients for the
        four prohibited public-input channels exactly zero.
        """

        if not self.geolocation_context_masked:
            return context
        start, stop = self.geolocation_context_indices[0], (
            self.geolocation_context_indices[-1] + 1
        )
        if context.shape[-1] != 19 or (start, stop) != (5, 9):
            raise AssertionError("registered geolocation mask contract drifted")
        if self.no_geo_core_enabled:
            return torch.cat((context[..., :start], context[..., stop:]), dim=-1)
        return torch.cat(
            (
                context[..., :start],
                torch.zeros_like(context[..., start:stop]),
                context[..., stop:],
            ),
            dim=-1,
        )

    def _add_content_q_pyramid(
        self,
        raw_q: Tensor,
        query_predictor: Tensor,
        query_support: Tensor,
        query_coarse: Tensor,
        query_fine_features: Tensor,
        parent_context: Tensor,
    ) -> Tensor:
        """Add the content-only multiscale proposal to an inherited raw Q."""

        pyramid = self.content_q_pyramid
        if pyramid is None:
            return raw_q
        coarse_valid = torch.isfinite(query_coarse)
        delivered_q0 = support_project(
            raw_q,
            torch.zeros_like(query_coarse),
            query_support,
            coarse_valid,
        ) * query_support
        proposal = pyramid(
            delivered_q0,
            query_predictor,
            query_support,
            coarse_valid,
            query_fine_features,
            parent_context,
        )
        return raw_q + proposal.to(dtype=raw_q.dtype)

    def _raw_core(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, times, channels, height, width = fine.shape
        fine_features, tokens, descriptors = self.date_encoder(
            fine.reshape(batch * times, channels, height, width),
            coarse_k.reshape(batch * times, 1, height // _SCALE, width // _SCALE),
            support.reshape(batch * times, 1, height, width),
            context.reshape(batch * times, self.encoder_context_dim),
            temporal_available.reshape(batch * times),
        )
        fine_features = fine_features.reshape(
            batch, times, self.fine_width, height, width
        )
        tokens = tokens.reshape(
            batch, times, self.parent_width, height // _SCALE, width // _SCALE
        )
        descriptors = descriptors.reshape(batch, times, self.parent_width)

        query_fine = self._gather_query(fine_features, query_index)
        query_descriptor = self._gather_query(descriptors, query_index)
        available_float = temporal_available.to(dtype=descriptors.dtype)
        set_mean = (descriptors * available_float[..., None]).sum(dim=1)
        set_mean = set_mean / available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        scene_code = self.scene_fusion(
            torch.cat((query_descriptor, set_mean, query_descriptor - set_mean), dim=1)
        )
        parent = self.context_unet(
            self.temporal_fusion(tokens, temporal_available, query_index)
        )
        raw_q, expert_weights = self.continuous_head(
            query_fine, parent, scene_code
        )
        if self.allocation_adapter is not None:
            query_support = self._gather_query(support, query_index)
            raw_q = raw_q + self.allocation_adapter(
                query_fine, parent, query_support, raw_q.detach()
            )
        if self.q_refiner is not None:
            query_predictor = self._gather_query(fine, query_index)
            query_support = self._gather_query(support, query_index)
            query_coarse = self._gather_query(coarse_k, query_index)
            raw_q = raw_q + self.q_refiner(
                raw_q,
                query_predictor,
                query_support,
                torch.isfinite(query_coarse),
                query_fine,
                parent,
                scene_code,
            )
        if self.content_q_pyramid is not None:
            query_predictor = self._gather_query(fine, query_index)
            query_support = self._gather_query(support, query_index)
            query_coarse = self._gather_query(coarse_k, query_index)
            raw_q = self._add_content_q_pyramid(
                raw_q,
                query_predictor,
                query_support,
                query_coarse,
                query_fine,
                parent,
            )
        if self.parent_dct15 is not None:
            query_predictor = self._gather_query(fine, query_index)
            query_support = self._gather_query(support, query_index)
            query_coarse = self._gather_query(coarse_k, query_index)
            coarse_valid = torch.isfinite(query_coarse)
            delivered_q0 = support_project(
                raw_q,
                torch.zeros_like(query_coarse),
                query_support,
                coarse_valid,
            ) * query_support
            corrected_q = self.parent_dct15(
                query_predictor,
                delivered_q0,
                query_support,
                coarse_valid,
            )
            # The decoder returns a corrected Q field, not a residual.  Add
            # only its proposal relative to the exact field it consumed;
            # adding ``corrected_q`` itself would duplicate inherited q0.
            proposal = corrected_q - delivered_q0
            raw_q = raw_q + proposal.to(dtype=raw_q.dtype)
        return raw_q, expert_weights

    def _raw_core_t3(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the exact old query-only core, then add masked T3 residuals."""

        if self.t3_fusion is None:
            raise AssertionError("T3 core called without the explicit T3 module")
        batch, times, channels, height, width = fine.shape

        # Compute the registered single-date baseline on a literal T=1 query
        # batch.  In particular, do not derive it from a full-T batch and do
        # not use the old unconditional descriptor set mean.
        query_fine_input = self._gather_query(fine, query_index)
        query_coarse_input = self._gather_query(coarse_k, query_index)
        query_support = self._gather_query(support, query_index)
        query_context = self._gather_query(context, query_index)
        query_fine, query_token, query_descriptor = self.date_encoder(
            query_fine_input,
            query_coarse_input,
            query_support,
            query_context,
            torch.ones(batch, dtype=torch.bool, device=fine.device),
        )
        baseline_temporal_token = self.temporal_fusion(
            query_token[:, None],
            torch.ones(batch, 1, dtype=torch.bool, device=fine.device),
            torch.zeros(batch, dtype=torch.long, device=fine.device),
        )
        baseline_parent = self.context_unet(baseline_temporal_token)
        baseline_scene_code = self.scene_fusion(
            torch.cat(
                (
                    query_descriptor,
                    query_descriptor,
                    torch.zeros_like(query_descriptor),
                ),
                dim=1,
            )
        )

        # Encode dates independently with the shared legacy date encoder.
        # Replace unavailable payloads with the query *before* any convolution
        # or similarity calculation.  The encoder also applies availability,
        # and T3 aggregation masks it once more before reduction; this early
        # replacement makes even extreme finite sentinel payloads irrelevant
        # instead of relying on a late ``0 * value`` cancellation.
        available_grid = temporal_available[:, :, None, None, None]
        available_context = temporal_available[:, :, None]
        safe_fine = torch.where(
            available_grid, fine, query_fine_input[:, None]
        )
        safe_coarse = torch.where(
            available_grid, coarse_k, query_coarse_input[:, None]
        )
        safe_support = torch.where(
            available_grid, support, query_support[:, None]
        )
        safe_context = torch.where(
            available_context, context, query_context[:, None]
        )
        all_fine, all_tokens, all_descriptors = self.date_encoder(
            safe_fine.reshape(batch * times, channels, height, width),
            safe_coarse.reshape(
                batch * times, 1, height // _SCALE, width // _SCALE
            ),
            safe_support.reshape(batch * times, 1, height, width),
            safe_context.reshape(batch * times, self.encoder_context_dim),
            temporal_available.reshape(batch * times),
        )
        all_fine = all_fine.reshape(
            batch, times, self.fine_width, height, width
        )
        all_tokens = all_tokens.reshape(
            batch, times, self.parent_width, height // _SCALE, width // _SCALE
        )
        all_descriptors = all_descriptors.reshape(
            batch, times, self.parent_width
        )
        fused_fine, fused_parent, fused_scene_code = self.t3_fusion(
            query_fine,
            query_token,
            baseline_parent,
            query_descriptor,
            query_context,
            all_fine,
            all_tokens,
            all_descriptors,
            safe_context,
            temporal_available,
            query_index,
            baseline_scene_code,
        )
        raw_q, expert_weights = self.continuous_head(
            fused_fine, fused_parent, fused_scene_code
        )
        if self.allocation_adapter is not None:
            raw_q = raw_q + self.allocation_adapter(
                fused_fine, fused_parent, query_support, raw_q.detach()
            )
        if self.q_refiner is not None:
            raw_q = raw_q + self.q_refiner(
                raw_q,
                query_fine_input,
                query_support,
                torch.isfinite(query_coarse_input),
                fused_fine,
                fused_parent,
                fused_scene_code,
            )
        return raw_q, expert_weights

    def _run_raw_core(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raw_core = self._raw_core_t3 if self.t3_fusion is not None else self._raw_core
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return gradient_checkpoint(
                raw_core,
                fine,
                coarse_k,
                support,
                context,
                temporal_available,
                query_index,
                use_reentrant=False,
            )
        return raw_core(
            fine, coarse_k, support, context, temporal_available, query_index
        )

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> CalibratedContinuousQComponents:
        support_bool, available, query_long = self._validate_inputs(
            fine, coarse_k, support, context, temporal_available, query_index
        )
        effective_context = self._effective_context(context)
        support_float = support_bool.to(dtype=fine.dtype)
        raw_q, expert_weights = self._run_raw_core(
            fine,
            coarse_k,
            support_float,
            effective_context,
            available,
            query_long,
        )

        query_fine = self._gather_query(fine, query_long)
        query_coarse = self._gather_query(coarse_k, query_long)
        query_support_bool = self._gather_query(support_bool, query_long)
        query_support = query_support_bool.to(dtype=fine.dtype)
        coarse_valid = torch.isfinite(query_coarse)
        base_k = support_project(query_fine[:, :1], query_coarse, query_support)
        q_k = support_project(
            raw_q, torch.zeros_like(query_coarse), query_support, coarse_valid
        )
        # Invalid coarse parents are unconstrained, not invalid fine pixels.
        # Keep their learned Q residual and restrict only by physical support;
        # multiplying by a repeated coarse-valid mask would silently erase the
        # model exactly where no direct parent observation is available.
        q_k = q_k * query_support
        return CalibratedContinuousQComponents(
            prediction_k=base_k + q_k,
            base_k=base_k,
            q_k=q_k,
            expert_weights=expert_weights,
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
