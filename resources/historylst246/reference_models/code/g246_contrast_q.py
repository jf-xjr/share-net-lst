"""D4-equivariant contrastive parent-lattice Q reconstructor for G246 R2.

``ContrastQParent`` is the next deterministic field model after
``MTStyleQParent``.  It makes the missing high-frequency signal explicit:
every date is decomposed into support-weighted 4x4 parent means and
within-parent guidance/base contrasts before ``pixel_unshuffle`` creates a
parent-lattice token.  Query-to-set attention then fuses an unordered set of
auxiliary dates through *differences from the query*.  A 40 -> 20 -> 10 -> 40
parent-grid U-Net supplies spatial context for a learned mixture-of-experts
head that emits all sixteen sub-pixel phases.

The raw head is Reynolds-averaged over the eight D4 raster transforms.  This
makes the complete learned residual exactly D4 equivariant (up to floating
point reduction order), rather than merely exposing the model to D4 data
augmentation.  Finally, :func:`ocnir.support_project` projects that residual
onto the query-date Q space.  Targets and scientific scoring masks are absent
from every public inference method.
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


__all__ = ["ContrastQParent", "ContrastQComponents"]

_SCALE = 4
_PHASES = _SCALE**2
_D4_SIZE = 8


def _groups(channels: int) -> int:
    return gcd(channels, min(8, channels))


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


def _d4_spatial(value: Tensor, code: int) -> Tensor:
    """Apply the dataset's ``rot90 then horizontal flip`` D4 convention."""

    if not 0 <= int(code) < _D4_SIZE:
        raise ValueError("D4 code must be in [0, 7]")
    transformed = torch.rot90(value, k=int(code) & 3, dims=(-2, -1))
    if int(code) & 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed


def _d4_inverse_spatial(value: Tensor, code: int) -> Tensor:
    """Invert :func:`_d4_spatial` without assuming rotations commute with flips."""

    if not 0 <= int(code) < _D4_SIZE:
        raise ValueError("D4 code must be in [0, 7]")
    transformed = value
    if int(code) & 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return torch.rot90(transformed, k=-(int(code) & 3), dims=(-2, -1))


def _d4_context(context: Tensor, code: int) -> Tensor:
    """Rotate the optional east/north solar pair used by Core22 context.

    The strict R2 context has twelve fields and stores solar east/north at
    indices 10/11.  Shorter generic contexts contain no declared raster-frame
    vector and are therefore treated as D4 scalars.
    """

    if not 0 <= int(code) < _D4_SIZE:
        raise ValueError("D4 code must be in [0, 7]")
    if context.shape[-1] < 12:
        return context
    value = context.clone()
    east = context[..., 10]
    north = context[..., 11]
    rotation = int(code) & 3
    if rotation == 0:
        transformed_east, transformed_north = east, north
    elif rotation == 1:
        transformed_east, transformed_north = -north, east
    elif rotation == 2:
        transformed_east, transformed_north = -east, -north
    else:
        transformed_east, transformed_north = north, -east
    if int(code) & 4:
        transformed_east = -transformed_east
    value[..., 10] = transformed_east
    value[..., 11] = transformed_north
    return value


def _support_parent_contrast(values: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    """Return supported 4x4 contrasts and their support-weighted parent means.

    ``values`` is NCHW and ``support`` is N1HW.  Unsupported cells are zero in
    the contrast.  A parent with no support receives a zero mean; unavailable
    temporal entries are masked after their independent date encoder.
    """

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
    total = (blocks * weights).sum(dim=(3, 5))
    means = total / counts.clamp_min(1.0)
    means = torch.where(counts > 0, means, torch.zeros_like(means))
    expanded = means.repeat_interleave(_SCALE, dim=-2).repeat_interleave(
        _SCALE, dim=-1
    )
    contrast = (values - expanded) * support.to(dtype=values.dtype)
    return contrast, means


class _ParentBlock(nn.Module):
    """ConvNeXt-style parent-lattice block with a small residual scale."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = expansion * channels
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


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=2, padding=1,
            groups=in_channels, bias=False,
        )
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = _norm(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.silu(self.norm(self.project(self.depthwise(inputs))))


class _Project(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=False),
        )


class _ContrastDateEncoder(nn.Module):
    """Encode one date after explicit within-parent contrast removal."""

    def __init__(
        self,
        fine_channels: int,
        context_dim: int,
        width: int,
        blocks: int,
    ) -> None:
        super().__init__()
        # 16 contrast phases per fine channel, 16 support phases, absolute
        # parent means, and four physical coarse/support descriptors.
        token_channels = _PHASES * fine_channels + _PHASES + fine_channels + 4
        self.fine_channels = fine_channels
        self.context_dim = context_dim
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(token_channels, width, 1, bias=False),
            _norm(width),
            nn.SiLU(inplace=False),
        )
        self.blocks = nn.Sequential(*(_ParentBlock(width) for _ in range(blocks)))
        self.context = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.SiLU(inplace=False),
            nn.Linear(width, 2 * width),
        )

    def forward(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        available: Tensor,
    ) -> Tensor:
        values = torch.cat(((fine[:, :1] - 300.0) / 20.0, fine[:, 1:]), dim=1)
        contrast, parent_means = _support_parent_contrast(values, support)
        packed_contrast = F.pixel_unshuffle(contrast, _SCALE)
        packed_support = F.pixel_unshuffle(support.to(dtype=fine.dtype), _SCALE)

        coarse_valid = torch.isfinite(coarse_k) & available[:, None, None, None]
        coarse_safe = torch.where(
            coarse_valid, coarse_k.to(dtype=fine.dtype),
            torch.full_like(coarse_k, 300.0, dtype=fine.dtype),
        )
        support_fraction = packed_support.mean(dim=1, keepdim=True)
        base_parent_k = 300.0 + 20.0 * parent_means[:, :1]
        physical = torch.cat(
            (
                (coarse_safe - 300.0) / 20.0,
                coarse_valid.to(dtype=fine.dtype),
                support_fraction,
                (base_parent_k - coarse_safe) / 20.0,
            ),
            dim=1,
        )
        token = self.stem(
            torch.cat((packed_contrast, packed_support, parent_means, physical), dim=1)
        )
        scale, shift = self.context(context.to(dtype=fine.dtype)).chunk(2, dim=1)
        token = token * (1.0 + 0.1 * torch.tanh(scale)[..., None, None])
        token = token + shift[..., None, None]
        token = self.blocks(token)
        # An unavailable entry is exactly zero and therefore cannot influence
        # temporal attention, even if its caller-supplied predictors differ.
        return token * available[:, None, None, None].to(dtype=token.dtype)


class _TemporalContrastFusion(nn.Module):
    """Permutation-invariant query-to-set attention over query-relative tokens."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("width must be divisible by attention_heads")
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
        # Values are explicit auxiliary/query contrasts.  The query's own
        # value is zero, while its key remains in the normalized set.
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


class _ParentContextUNet(nn.Module):
    """Parent lattice context path: 40 -> 20 -> 10 -> 20 -> 40 at full size."""

    def __init__(
        self,
        widths: tuple[int, int, int],
        blocks: tuple[int, int, int, int, int],
    ) -> None:
        super().__init__()
        high, middle, low = widths
        enc_high, enc_middle, enc_low, dec_middle, dec_high = blocks
        self.high = nn.Sequential(*(_ParentBlock(high) for _ in range(enc_high)))
        self.down_middle = _Downsample(high, middle)
        self.middle = nn.Sequential(*(_ParentBlock(middle) for _ in range(enc_middle)))
        self.down_low = _Downsample(middle, low)
        self.low = nn.Sequential(*(_ParentBlock(low) for _ in range(enc_low)))
        self.up_middle = _Project(low, middle)
        self.merge_middle = _Project(2 * middle, middle)
        self.decode_middle = nn.Sequential(
            *(_ParentBlock(middle) for _ in range(dec_middle))
        )
        self.up_high = _Project(middle, high)
        self.merge_high = _Project(2 * high, high)
        self.decode_high = nn.Sequential(*(_ParentBlock(high) for _ in range(dec_high)))

    def forward(self, inputs: Tensor) -> Tensor:
        high = self.high(inputs)
        middle = self.middle(self.down_middle(high))
        low = self.low(self.down_low(middle))
        decoded_middle = F.interpolate(
            low, size=middle.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_middle = self.up_middle(decoded_middle)
        decoded_middle = self.merge_middle(torch.cat((decoded_middle, middle), dim=1))
        decoded_middle = self.decode_middle(decoded_middle)
        decoded_high = F.interpolate(
            decoded_middle, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_high = self.up_high(decoded_high)
        decoded_high = self.merge_high(torch.cat((decoded_high, high), dim=1))
        return self.decode_high(decoded_high)


class _PhaseMixtureHead(nn.Module):
    """Learn an independent expert mixture for each of the sixteen phases."""

    def __init__(self, channels: int, experts: int) -> None:
        super().__init__()
        self.experts = experts
        self.phase_count = _PHASES
        self.pre = nn.Sequential(_norm(channels), nn.SiLU(inplace=False))
        self.expert_values = nn.Conv2d(channels, experts * _PHASES, 3, padding=1)
        self.expert_gates = nn.Conv2d(channels, experts * _PHASES, 3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.pre(inputs)
        shape = (inputs.shape[0], self.experts, _PHASES, *inputs.shape[-2:])
        experts = self.expert_values(value).reshape(shape)
        gates = self.expert_gates(value).reshape(shape).softmax(dim=1)
        phases = (experts * gates).sum(dim=1)
        return F.pixel_shuffle(phases, _SCALE)


class ContrastQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_k: Tensor


class ContrastQParent(nn.Module):
    """D4-equivariant multi-temporal contrast-to-Q field model.

    The default Core22 configuration has roughly five million trainable
    parameters.  All heavy activations live on the H/4 parent lattice, and
    optional per-D4-element checkpointing keeps batch-32 96x96 training within
    a 16 GiB consumer GPU budget.

    ``fine``, ``coarse_k``, ``support``, ``context``, ``temporal_available``
    and ``query_index`` have exactly the same meanings and ranks as in
    ``MTStyleQParent``.  No target, validity, or eligibility argument exists.
    """

    scale_path = "H/4 -> H/8 -> H/16 -> H/8 -> H/4 (40->20->10->20->40 for H=160)"

    def __init__(
        self,
        fine_channels: int = 22,
        context_dim: int = 12,
        *,
        width: int = 144,
        middle_width: int = 224,
        low_width: int = 320,
        date_blocks: int = 2,
        scale_blocks: Sequence[int] = (2, 2, 2, 2, 2),
        attention_heads: int = 6,
        phase_experts: int = 4,
        d4_average: bool = True,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        for value, name in (
            (fine_channels, "fine_channels"), (context_dim, "context_dim"),
            (width, "width"), (middle_width, "middle_width"),
            (low_width, "low_width"), (date_blocks, "date_blocks"),
            (attention_heads, "attention_heads"), (phase_experts, "phase_experts"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if fine_channels < 2:
            raise ValueError("fine_channels must include Kelvin and guidance")
        if width % attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if len(tuple(scale_blocks)) != 5:
            raise ValueError("scale_blocks must contain five positive depths")
        block_tuple = tuple(int(value) for value in scale_blocks)
        if any(value < 1 for value in block_tuple):
            raise ValueError("scale_blocks depths must be positive")
        if not isinstance(d4_average, bool) or not isinstance(activation_checkpointing, bool):
            raise TypeError("d4_average and activation_checkpointing must be bool")

        self.fine_channels = fine_channels
        self.context_dim = context_dim
        self.width = width
        self.middle_width = middle_width
        self.low_width = low_width
        self.date_blocks = date_blocks
        self.scale_blocks = block_tuple
        self.attention_heads = attention_heads
        self.phase_experts = phase_experts
        self.d4_average = d4_average
        self.activation_checkpointing = activation_checkpointing

        self.date_encoder = _ContrastDateEncoder(
            fine_channels, context_dim, width, date_blocks
        )
        self.temporal_fusion = _TemporalContrastFusion(width, attention_heads)
        self.context_unet = _ParentContextUNet(
            (width, middle_width, low_width), block_tuple
        )
        self.phase_head = _PhaseMixtureHead(width, phase_experts)
        self.apply(_initialize)
        # Small, diverse experts begin near the physical interpolation while
        # retaining a nonzero learning signal for the mixture gates.
        nn.init.normal_(self.phase_head.expert_values.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.phase_head.expert_values.bias)
        nn.init.zeros_(self.phase_head.expert_gates.weight)
        nn.init.zeros_(self.phase_head.expert_gates.bias)

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
            (fine, "fine"), (coarse_k, "coarse_k"), (support, "support"),
            (context, "context"), (temporal_available, "temporal_available"),
            (query_index, "query_index"),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if fine.ndim != 5 or fine.shape[2] != self.fine_channels:
            raise ValueError(f"fine must have shape [B,T,{self.fine_channels},H,W]")
        batch, times, _, height, width = fine.shape
        if height % 16 or width % 16:
            raise ValueError("fine geometry must be divisible by sixteen for two parent scales")
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
        if not fine.is_floating_point() or not coarse_k.is_floating_point():
            raise TypeError("fine and coarse_k must have floating dtypes")
        if not context.is_floating_point():
            raise TypeError("context must have a floating dtype")
        if len({
            fine.device, coarse_k.device, support.device, context.device,
            temporal_available.device, query_index.device,
        }) != 1:
            raise ValueError("all model inputs must be on one device")
        if _contains_true(~torch.isfinite(fine)) or _contains_true(~torch.isfinite(context)):
            raise ValueError("fine and context must be finite")
        if _contains_true(torch.isinf(coarse_k)):
            raise ValueError("coarse_k may contain NaN but not infinity")
        support_bool = _binary(support, "support")
        available = _binary(temporal_available, "temporal_available")
        if query_index.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
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
        return _TemporalContrastFusion.gather_query(values, query_index)

    def _raw_core(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> Tensor:
        batch, times, channels, height, width = fine.shape
        tokens = self.date_encoder(
            fine.reshape(batch * times, channels, height, width),
            coarse_k.reshape(batch * times, 1, height // _SCALE, width // _SCALE),
            support.reshape(batch * times, 1, height, width),
            context.reshape(batch * times, self.context_dim),
            temporal_available.reshape(batch * times),
        ).reshape(batch, times, self.width, height // _SCALE, width // _SCALE)
        fused = self.temporal_fusion(tokens, temporal_available, query_index)
        return self.phase_head(self.context_unet(fused))

    def _run_raw_core(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return gradient_checkpoint(
                self._raw_core, fine, coarse_k, support, context,
                temporal_available, query_index, use_reentrant=False,
            )
        return self._raw_core(
            fine, coarse_k, support, context, temporal_available, query_index
        )

    def _d4_raw_q(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> Tensor:
        if not self.d4_average:
            return self._run_raw_core(
                fine, coarse_k, support, context, temporal_available, query_index
            )
        total: Tensor | None = None
        for code in range(_D4_SIZE):
            oriented = self._run_raw_core(
                _d4_spatial(fine, code),
                _d4_spatial(coarse_k, code),
                _d4_spatial(support, code),
                _d4_context(context, code),
                temporal_available,
                query_index,
            )
            canonical = _d4_inverse_spatial(oriented, code)
            total = canonical if total is None else total + canonical
        if total is None:  # pragma: no cover - fixed nonempty D4 group
            raise AssertionError("D4 group unexpectedly empty")
        return total / float(_D4_SIZE)

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> ContrastQComponents:
        support_bool, available, query_long = self._validate_inputs(
            fine, coarse_k, support, context, temporal_available, query_index
        )
        support_float = support_bool.to(dtype=fine.dtype)
        raw_q = self._d4_raw_q(
            fine, coarse_k, support_float, context, available, query_long
        )

        query_fine = self._gather_query(fine, query_long)
        query_coarse = self._gather_query(coarse_k, query_long)
        query_support_bool = self._gather_query(support_bool, query_long)
        query_support = query_support_bool.to(dtype=fine.dtype)
        coarse_valid = torch.isfinite(query_coarse)
        base_k = support_project(query_fine[:, :1], query_coarse, query_support)
        q_k = support_project(raw_q, torch.zeros_like(query_coarse), query_support, coarse_valid)
        # Missing coarse parents have no closure constraint: retain the
        # learned residual on supported fine cells instead of erasing it.
        q_k = q_k * query_support
        return ContrastQComponents(
            prediction_k=base_k + q_k,
            base_k=base_k,
            q_k=q_k,
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
