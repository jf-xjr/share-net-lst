"""Target-free multi-temporal parent-lattice Q reconstructor for G246 R2.

``MTStyleQParent`` consumes only deployable inputs.  A shared per-date encoder
packs 4x4 fine-grid features onto the observed parent lattice, target-free
scene statistics condition that representation, and masked attention fuses an
unordered set of auxiliary dates around a caller-selected query date.  One
continuous head predicts the query-date fine-scale nullspace field.

The head is centred with the physical query support before it is added to a
support-consistent interpolation base.  Consequently every observed query
parent has the requested coarse mean, while neither targets nor scientific
scoring masks are part of the public interface.
"""

from __future__ import annotations

import math
from math import gcd
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ocnir import support_project


__all__ = ["MTStyleQParent", "QParentComponents"]

_SCALE = 4


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


class _ResidualDW(nn.Module):
    """Depthwise spatial mixing followed by channel mixing."""

    def __init__(self, channels: int, *, kernel_size: int, expansion: int) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.norm = _norm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.contract = nn.Conv2d(hidden, channels, 1)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.depthwise(inputs)
        value = self.activation(self.norm(value))
        value = self.activation(self.expand(value))
        return inputs + self.contract(value)


class _ParentBlock(nn.Module):
    """Stable ConvNeXt-like block on the 40x40 parent lattice."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.norm = _norm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.contract = nn.Conv2d(hidden, channels, 1)
        self.activation = nn.SiLU(inplace=False)
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1.0e-2))

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.depthwise(inputs)
        value = self.activation(self.norm(value))
        value = self.activation(self.expand(value))
        value = self.contract(value)
        return inputs + self.layer_scale * value


class _TemporalParentEncoder(nn.Module):
    """Shared target-free encoder used independently for every date."""

    def __init__(
        self,
        fine_channels: int,
        context_dim: int,
        fine_width: int,
        parent_width: int,
    ) -> None:
        super().__init__()
        self.fine_channels = fine_channels
        self.context_dim = context_dim
        self.fine_stem = nn.Sequential(
            nn.Conv2d(fine_channels + 1, fine_width, 3, padding=1, bias=False),
            _norm(fine_width),
            nn.SiLU(inplace=False),
            _ResidualDW(fine_width, kernel_size=3, expansion=2),
            _ResidualDW(fine_width, kernel_size=3, expansion=2),
        )
        self.pack = nn.Sequential(
            nn.Conv2d(fine_width * _SCALE**2, parent_width, 1, bias=False),
            _norm(parent_width),
            nn.SiLU(inplace=False),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(3, parent_width, 3, padding=1, bias=False),
            _norm(parent_width),
            nn.SiLU(inplace=False),
            _ResidualDW(parent_width, kernel_size=3, expansion=2),
        )
        # Context plus per-scene means/stds of all non-Kelvin fine channels.
        style_dim = context_dim + 2 * (fine_channels - 1)
        self.style = nn.Sequential(
            nn.Linear(style_dim, parent_width),
            nn.SiLU(inplace=False),
            nn.Linear(parent_width, 2 * parent_width),
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, parent_width),
            nn.SiLU(inplace=False),
            nn.Linear(parent_width, parent_width),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(3 * parent_width, parent_width, 1, bias=False),
            _norm(parent_width),
            nn.SiLU(inplace=False),
        )

    def forward(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        available: Tensor,
    ) -> Tensor:
        # The leading B*T dimension makes normalization independent across
        # dates, so masked auxiliary entries cannot affect available entries.
        base_scaled = (fine[:, :1] - 300.0) / 20.0
        fine_features = self.fine_stem(torch.cat((base_scaled, fine[:, 1:], support), dim=1))
        packed = self.pack(F.pixel_unshuffle(fine_features, _SCALE))

        coarse_valid = torch.isfinite(coarse_k) & available[:, None, None, None]
        coarse_safe = torch.where(coarse_valid, coarse_k, torch.full_like(coarse_k, 300.0))
        parent_height, parent_width = coarse_k.shape[-2:]
        support_fraction = support.reshape(
            support.shape[0], 1, parent_height, _SCALE, parent_width, _SCALE
        ).mean(dim=(3, 5))
        coarse_features = self.coarse(
            torch.cat(
                (
                    (coarse_safe - 300.0) / 20.0,
                    support_fraction,
                    coarse_valid.to(dtype=fine.dtype),
                ),
                dim=1,
            )
        )

        non_kelvin = fine[:, 1:]
        style_mean = non_kelvin.mean(dim=(-2, -1))
        style_std = torch.sqrt(
            torch.clamp(non_kelvin.var(dim=(-2, -1), unbiased=False), min=1.0e-6)
        )
        style = self.style(torch.cat((context, style_mean, style_std), dim=1))
        scale, shift = style.chunk(2, dim=1)
        packed = packed * (1.0 + 0.1 * torch.tanh(scale)[..., None, None])
        packed = packed + shift[..., None, None]
        context_features = self.context(context)[..., None, None].expand_as(packed)
        return self.merge(torch.cat((packed, coarse_features, context_features), dim=1))


class _MaskedTemporalAttention(nn.Module):
    """Query-to-set attention with no temporal-position parameter."""

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
        self.merge = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            _norm(channels),
            nn.SiLU(inplace=False),
        )

    @staticmethod
    def _gather_query(values: Tensor, query_index: Tensor) -> Tensor:
        # values: [B,T,...].  Gather is used instead of a Python loop so this
        # remains differentiable and device-local.
        shape = (values.shape[0], 1, *values.shape[2:])
        index = query_index.reshape(values.shape[0], 1, *([1] * (values.ndim - 2)))
        return torch.gather(values, 1, index.expand(shape)).squeeze(1)

    def forward(self, tokens: Tensor, available: Tensor, query_index: Tensor) -> Tensor:
        batch, times, channels, height, width = tokens.shape
        flat = tokens.reshape(batch * times, channels, height, width)
        keys = self.key(flat).reshape(batch, times, self.heads, self.head_dim, height, width)
        values = self.value(flat).reshape(
            batch, times, self.heads, self.head_dim, height, width
        )
        query_token = self._gather_query(tokens, query_index)
        query = self.query(query_token).reshape(
            batch, self.heads, self.head_dim, height, width
        )
        scores = torch.einsum("bhdxy,bthdxy->bthxy", query, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~available[:, :, None, None, None], float("-inf"))
        weights = torch.softmax(scores.float(), dim=1).to(dtype=tokens.dtype)
        attended = torch.einsum("bthxy,bthdxy->bhdxy", weights, values)
        attended = attended.reshape(batch, channels, height, width)
        attended = self.output(attended)
        return self.merge(torch.cat((query_token, attended), dim=1))


class QParentComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_k: Tensor


class MTStyleQParent(nn.Module):
    """Multi-temporal scene-style-conditioned parent-lattice Q model.

    The temporal dimension is a set: auxiliary order has no semantics.  Date,
    platform, and forcing information belongs in each date's ``context``.
    ``query_index`` selects the date whose coarse observation is reconstructed.

    Targets, validity masks, and eligibility masks are deliberately absent from
    both :meth:`forward` and :meth:`forward_components`.
    """

    def __init__(
        self,
        fine_channels: int = 11,
        context_dim: int = 5,
        *,
        fine_width: int = 64,
        parent_width: int = 160,
        parent_blocks: int = 10,
        attention_heads: int = 5,
    ) -> None:
        super().__init__()
        for value, name in (
            (fine_channels, "fine_channels"),
            (context_dim, "context_dim"),
            (fine_width, "fine_width"),
            (parent_width, "parent_width"),
            (parent_blocks, "parent_blocks"),
            (attention_heads, "attention_heads"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if fine_channels < 2:
            raise ValueError("fine_channels must include Kelvin and at least one guidance channel")
        if parent_width % attention_heads:
            raise ValueError("parent_width must be divisible by attention_heads")

        self.fine_channels = fine_channels
        self.context_dim = context_dim
        self.fine_width = fine_width
        self.parent_width = parent_width
        self.parent_blocks_count = parent_blocks
        self.attention_heads = attention_heads
        self.temporal_encoder = _TemporalParentEncoder(
            fine_channels, context_dim, fine_width, parent_width
        )
        self.temporal_attention = _MaskedTemporalAttention(parent_width, attention_heads)
        self.parent_blocks = nn.Sequential(
            *(_ParentBlock(parent_width) for _ in range(parent_blocks))
        )
        self.head = nn.Sequential(
            _norm(parent_width),
            nn.SiLU(inplace=False),
            nn.Conv2d(parent_width, _SCALE**2, 3, padding=1),
        )
        self.apply(_initialize)
        # The initial predictor is exactly the physical interpolation base.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _validate_inputs(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
            raise ValueError(
                f"fine must have shape [B,T,{self.fine_channels},H,W]"
            )
        batch, times, _, height, width = fine.shape
        if height % _SCALE or width % _SCALE:
            raise ValueError("fine geometry must be divisible by four")
        expected_coarse = (batch, times, 1, height // _SCALE, width // _SCALE)
        if coarse_k.shape != expected_coarse:
            raise ValueError(f"coarse_k must have shape {expected_coarse}")
        if support.shape != (batch, times, 1, height, width):
            raise ValueError("support must have shape [B,T,1,H,W]")
        if context.shape != (batch, times, self.context_dim):
            raise ValueError(f"context must have shape [B,T,{self.context_dim}]")
        if temporal_available.shape != (batch, times):
            raise ValueError("temporal_available must have shape [B,T]")
        if query_index.shape != (batch,):
            raise ValueError("query_index must have shape [B]")
        if not fine.is_floating_point() or not coarse_k.is_floating_point() or not context.is_floating_point():
            raise TypeError("fine, coarse_k, and context must have floating dtypes")
        devices = {
            fine.device,
            coarse_k.device,
            support.device,
            context.device,
            temporal_available.device,
            query_index.device,
        }
        if len(devices) != 1:
            raise ValueError("all model inputs must be on one device")
        if _contains_true(~torch.isfinite(fine)) or _contains_true(~torch.isfinite(context)):
            raise ValueError("fine and context must be finite")
        if _contains_true(torch.isinf(coarse_k)):
            raise ValueError("coarse_k may contain NaN for missing parents but not infinity")
        available = _binary(temporal_available, "temporal_available")
        support_bool = _binary(support, "support")
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
        return support_bool, available

    @staticmethod
    def _gather_query(values: Tensor, query_index: Tensor) -> Tensor:
        shape = (values.shape[0], 1, *values.shape[2:])
        index = query_index.to(dtype=torch.long).reshape(
            values.shape[0], 1, *([1] * (values.ndim - 2))
        )
        return torch.gather(values, 1, index.expand(shape)).squeeze(1)

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> QParentComponents:
        support_bool, available = self._validate_inputs(
            fine, coarse_k, support, context, temporal_available, query_index
        )
        batch, times, channels, height, width = fine.shape
        flat_fine = fine.reshape(batch * times, channels, height, width)
        flat_coarse = coarse_k.reshape(
            batch * times, 1, height // _SCALE, width // _SCALE
        )
        flat_support = support_bool.reshape(batch * times, 1, height, width).to(fine.dtype)
        flat_context = context.reshape(batch * times, self.context_dim)
        flat_available = available.reshape(batch * times)
        tokens = self.temporal_encoder(
            flat_fine, flat_coarse, flat_support, flat_context, flat_available
        ).reshape(batch, times, self.parent_width, height // _SCALE, width // _SCALE)

        fused = self.temporal_attention(tokens, available, query_index.to(torch.long))
        parent = self.parent_blocks(fused)
        raw_q = F.pixel_shuffle(self.head(parent), _SCALE)

        query_fine = self._gather_query(fine, query_index)
        query_coarse = self._gather_query(coarse_k, query_index)
        query_support_bool = self._gather_query(support_bool, query_index)
        query_support = query_support_bool.to(dtype=fine.dtype)
        coarse_valid = torch.isfinite(query_coarse)
        base_k = support_project(query_fine[:, :1], query_coarse, query_support)

        zero_coarse = torch.zeros_like(query_coarse)
        q_k = support_project(raw_q, zero_coarse, query_support, coarse_valid)
        # ``support_project`` is intentionally the identity on parents whose
        # coarse observation is missing.  Their supported fine cells still
        # carry learnable Q signal; only unsupported cells are forced to zero.
        q_k = q_k * query_support
        prediction_k = base_k + q_k
        return QParentComponents(prediction_k=prediction_k, base_k=base_k, q_k=q_k)

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
