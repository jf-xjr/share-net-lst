"""Ordered residual temperature reconstruction (ORQ).

This module is intentionally independent of the historical G246 model family.
It implements the public, target-free inference surface used by the ORQ data
adapter: static Core22/TEMPO fields, a direct coarse observation, and one
causal 48-hour forcing sequence.  No coordinates, dates, city identifiers, or
target-derived values are accepted by the model.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ocnir import support_project


__all__ = [
    "OrderedResidualQ", "OrderedResidualComponents", "MaskedDepthwiseConv2d",
    "parent_support_center", "parent_support_mean",
]


SCALE = 4
CORE22_CHANNELS = 22
TEMPO_CHANNELS = 3
SOLAR_CHANNELS = 3
FORCING_STEPS = 48
FORCING_CHANNELS = 10
STAGES = 8
WIDTH = 48
TIME_WIDTH = 64


def _contains_true(value: Tensor) -> bool:
    return bool(torch.any(value).detach().cpu().item())


def _binary(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if not value.is_floating_point() and value.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise TypeError(f"{name} must be boolean or numeric binary")
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


class PixelChannelNorm(nn.Module):
    """LayerNorm over channels only for an NCHW feature map."""

    def __init__(self, channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("PixelChannelNorm expects NCHW input")
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MaskedDepthwiseConv2d(nn.Module):
    """Fixed-support depthwise convolution with valid-neighbour renormalization.

    The input physical mask is never updated.  At an image boundary the full
    in-bounds neighbourhood is the reference support, so all-valid input is
    exactly equivalent to the ordinary depthwise convolution.
    """

    def __init__(self, channels: int, kernel_size: int = 5, *, bias: bool = True) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("masked convolution kernel must be positive and odd")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, padding=self.padding,
            groups=channels, bias=bias,
        )
        self.register_buffer("_ones_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, value: Tensor, support: Tensor) -> Tensor:
        if value.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
            raise ValueError("masked convolution expects [B,C,H,W] and [B,1,H,W]")
        if value.shape[0] != support.shape[0] or value.shape[-2:] != support.shape[-2:]:
            raise ValueError("masked convolution value/support geometry differs")
        mask = _binary(support, "support")
        if not value.is_floating_point() or _contains_true(~torch.isfinite(value)):
            raise ValueError("masked convolution values must be finite floating tensors")
        weight = mask.to(dtype=value.dtype)
        ones = self._ones_kernel.to(dtype=value.dtype)
        observed = F.conv2d(weight, ones, padding=self.padding)
        reference = F.conv2d(torch.ones_like(weight), ones, padding=self.padding)
        scale = reference / observed.clamp_min(1.0)
        result = self.conv(value * weight) * scale
        return result * weight


def parent_support_mean(value: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    """Actual-support parent mean and support count on the H/4 lattice."""

    if value.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("parent support mean expects NCHW value/support tensors")
    if value.shape[0] != support.shape[0] or value.shape[-2:] != support.shape[-2:]:
        raise ValueError("parent support mean value/support geometry differs")
    height, width = value.shape[-2:]
    if height % SCALE or width % SCALE:
        raise ValueError("parent support geometry must be divisible by four")
    mask = _binary(support, "support").to(dtype=value.dtype)
    parent_height, parent_width = height // SCALE, width // SCALE
    block_value = value.reshape(value.shape[0], value.shape[1], parent_height, SCALE, parent_width, SCALE)
    block_mask = mask.reshape(value.shape[0], 1, parent_height, SCALE, parent_width, SCALE)
    count = block_mask.sum(dim=(3, 5))
    total = (block_value * block_mask).sum(dim=(3, 5))
    return total / count.clamp_min(1.0), count


def parent_support_center(value: Tensor, support: Tensor) -> Tensor:
    """Apply Q_S: remove each actual-support parent mean without mask growth."""

    mean, _count = parent_support_mean(value, support)
    expanded = mean.repeat_interleave(SCALE, dim=-2).repeat_interleave(SCALE, dim=-1)
    return (value - expanded) * _binary(support, "support").to(dtype=value.dtype)


class _MaskedResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = PixelChannelNorm(channels)
        self.depthwise = MaskedDepthwiseConv2d(channels, 5, bias=True)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.contract = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, value: Tensor, support: Tensor) -> Tensor:
        residual = self.depthwise(self.norm(value), support)
        residual = self.contract(F.gelu(self.expand(residual)))
        return (value + residual) * _binary(support, "support").to(dtype=value.dtype)


class _ParentCoarseContext(nn.Module):
    """One support-normalized 3x3 parent-lattice coarse context layer."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, channels, 3, padding=1)
        self.channels = int(channels)
        self.register_buffer("_ones_kernel", torch.ones(1, 1, 3, 3))

    def forward(self, coarse_k: Tensor, observed: Tensor, active: Tensor) -> Tensor:
        valid = observed.to(dtype=coarse_k.dtype)
        coarse_safe = torch.where(observed, coarse_k, torch.zeros_like(coarse_k))
        inputs = torch.cat(((coarse_safe - 300.0) / 20.0 * valid, valid), dim=1)
        numerator = self.conv(inputs * valid)
        full = F.conv2d(torch.ones_like(valid), self._ones_kernel.to(dtype=valid.dtype), padding=1)
        count = F.conv2d(valid, self._ones_kernel.to(dtype=valid.dtype), padding=1)
        context = numerator * (full / count.clamp_min(1.0))
        return context * (count > 0).to(dtype=context.dtype) * active.to(dtype=context.dtype)


class _ConditionedUpdate(nn.Module):
    """The single parameter-shared spatial update used at all temporal stages."""

    def __init__(self, channels: int, time_width: int) -> None:
        super().__init__()
        self.norm = PixelChannelNorm(2 * channels)
        self.film = nn.Linear(time_width, 4 * channels)
        self.depthwise = MaskedDepthwiseConv2d(2 * channels, 5, bias=True)
        self.expand = nn.Conv2d(2 * channels, 4 * channels, 1)
        self.contract = nn.Conv2d(4 * channels, channels, 1)
        self.gate = nn.Linear(time_width, channels)
        nn.init.zeros_(self.contract.weight)
        nn.init.zeros_(self.contract.bias)

    def forward(self, state: Tensor, static: Tensor, temporal: Tensor, support: Tensor) -> Tensor:
        fused = self.norm(torch.cat((state, static), dim=1))
        gamma, beta = self.film(temporal).chunk(2, dim=1)
        conditioned = gamma[:, :, None, None] * fused + beta[:, :, None, None]
        mixed = self.depthwise(conditioned, support)
        delta = self.contract(F.gelu(self.expand(mixed)))
        gate = torch.sigmoid(self.gate(temporal))[:, :, None, None]
        return (state + 0.1 * gate * delta) * _binary(support, "support").to(dtype=state.dtype)


class OrderedResidualComponents(NamedTuple):
    prediction_k: Tensor
    detail_k: Tensor
    u_mean_k: Tensor
    observed_parent: Tensor
    unknown_parent: Tensor
    zero_parent: Tensor
    history_used: Tensor


class OrderedResidualQ(nn.Module):
    """Support-aware ordered-residual field model with strict coarse closure."""

    schema_version = "g246-ordered-residual-q-v1"
    fine_channels = CORE22_CHANNELS
    tempo_channels = TEMPO_CHANNELS
    forcing_channels = FORCING_CHANNELS
    width = WIDTH
    time_width = TIME_WIDTH

    def __init__(self) -> None:
        super().__init__()
        self.support_embedding = nn.Embedding(3, 4)
        self.coarse_context = _ParentCoarseContext(16)
        # Core22 + TEMPO3 + Solar3 + M_TEMPO + M_phys + support semantic6 + g_c16.
        self.stem = MaskedDepthwiseConv2d(52, 5, bias=True)
        self.stem_project = nn.Conv2d(52, WIDTH, 1)
        self.static_blocks = nn.ModuleList(_MaskedResidualBlock(WIDTH) for _ in range(3))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TIME_WIDTH, nhead=4, dim_feedforward=128, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.time_input = nn.Sequential(nn.Linear(FORCING_CHANNELS, TIME_WIDTH), nn.GELU())
        self.time_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.time_norm = nn.LayerNorm(TIME_WIDTH)
        self.initial = nn.Conv2d(WIDTH, WIDTH, 1)
        self.update = _ConditionedUpdate(WIDTH, TIME_WIDTH)
        self.detail_head = nn.Conv2d(WIDTH, 1, 1)
        self.u_head = nn.Sequential(
            nn.Linear(2 * WIDTH + TIME_WIDTH + 16, 128), nn.GELU(), nn.Linear(128, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate(
        self, fine22: Tensor, tempo: Tensor, tempo_valid: Tensor, solar3: Tensor,
        support: Tensor, coarse_k: Tensor, forcing: Tensor, forcing_ready: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tensors = {
            "fine22": fine22, "tempo": tempo, "tempo_valid": tempo_valid,
            "solar3": solar3, "support": support, "coarse_k": coarse_k,
            "forcing": forcing, "forcing_ready": forcing_ready,
        }
        if any(not isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("all ORQ inputs must be torch tensors")
        if fine22.ndim != 4 or fine22.shape[1] != CORE22_CHANNELS:
            raise ValueError("fine22 must have shape [B,22,H,W]")
        batch, _, height, width = fine22.shape
        if height % SCALE or width % SCALE:
            raise ValueError("ORQ fine geometry must be divisible by four")
        if tempo.shape != (batch, TEMPO_CHANNELS, height, width):
            raise ValueError("tempo must have shape [B,3,H,W]")
        if tempo_valid.shape != (batch, 1, height, width):
            raise ValueError("tempo_valid must have shape [B,1,H,W]")
        if support.shape != (batch, 1, height, width):
            raise ValueError("support must have shape [B,1,H,W]")
        if solar3.shape != (batch, SOLAR_CHANNELS):
            raise ValueError("solar3 must have shape [B,3]")
        if coarse_k.shape != (batch, 1, height // SCALE, width // SCALE):
            raise ValueError("coarse_k must have shape [B,1,H/4,W/4]")
        if forcing.shape != (batch, FORCING_STEPS, FORCING_CHANNELS):
            raise ValueError("forcing must have shape [B,48,10]")
        if forcing_ready.shape not in {(batch,), (batch, 1)}:
            raise ValueError("forcing_ready must have shape [B] or [B,1]")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("all ORQ tensors must be on one device")
        for name in ("fine22", "tempo", "solar3", "forcing"):
            value = tensors[name]
            if not value.is_floating_point() or _contains_true(~torch.isfinite(value)):
                raise ValueError(f"{name} must be finite floating data")
        if _contains_true(torch.isinf(coarse_k)):
            raise ValueError("coarse_k may contain NaN but not infinity")
        support_bool = _binary(support, "support")
        tempo_bool = _binary(tempo_valid, "tempo_valid")
        ready = _binary(forcing_ready.reshape(batch), "forcing_ready")
        if _contains_true(tempo * (~tempo_bool).to(dtype=tempo.dtype)):
            raise ValueError("TEMPO values must be exact zero outside tempo_valid")
        return support_bool, tempo_bool, ready

    @staticmethod
    def _states(support: Tensor, coarse_k: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _mean, count = parent_support_mean(torch.zeros_like(support), support)
        active = count > 0
        observed = torch.isfinite(coarse_k)
        if _contains_true(observed & ~active):
            raise ValueError("an observed coarse parent has zero physical support")
        unknown = ~observed & active
        zero = ~active
        state = torch.where(observed, torch.zeros_like(count, dtype=torch.long),
                            torch.where(unknown, torch.ones_like(count, dtype=torch.long),
                                        torch.full_like(count, 2, dtype=torch.long)))
        return state, observed, unknown, zero

    def _time_states(self, forcing: Tensor, ready: Tensor) -> Tensor:
        # Earliest to latest.  Each checkpoint at 5,11,...,47 sees no future token.
        encoded = self.time_input(forcing)
        causal = torch.full((FORCING_STEPS, FORCING_STEPS), float("-inf"), device=forcing.device)
        causal = torch.triu(causal, diagonal=1)
        sequence = self.time_encoder(encoded, mask=causal)
        checkpoints = self.time_norm(sequence[:, 5::6])
        return checkpoints * ready[:, None, None].to(dtype=checkpoints.dtype)

    def forward_components(
        self, fine22: Tensor, tempo: Tensor, tempo_valid: Tensor, solar3: Tensor,
        support: Tensor, coarse_k: Tensor, forcing: Tensor, forcing_ready: Tensor,
    ) -> OrderedResidualComponents:
        physical, tempo_mask, ready = self._validate(
            fine22, tempo, tempo_valid, solar3, support, coarse_k, forcing, forcing_ready,
        )
        state_parent, observed, unknown, zero = self._states(physical, coarse_k)
        _unused, count = parent_support_mean(torch.zeros_like(physical), physical)
        support_fraction = count / float(SCALE * SCALE)
        singleton = (count == 1).to(dtype=fine22.dtype)
        state_embedding = self.support_embedding(state_parent[:, 0]).permute(0, 3, 1, 2)
        semantic = torch.cat((state_embedding,
                              support_fraction.to(dtype=fine22.dtype), singleton), dim=1)
        semantic_fine = semantic.repeat_interleave(SCALE, -2).repeat_interleave(SCALE, -1)
        parent_context = self.coarse_context(coarse_k, observed, ~zero)
        context_fine = parent_context.repeat_interleave(SCALE, -2).repeat_interleave(SCALE, -1)
        solar_fine = solar3[:, :, None, None].expand(-1, -1, fine22.shape[-2], fine22.shape[-1])
        mask_float = physical.to(dtype=fine22.dtype)
        static_input = torch.cat((
            fine22, tempo * tempo_mask.to(dtype=tempo.dtype), solar_fine,
            tempo_mask.to(dtype=fine22.dtype), mask_float, semantic_fine, context_fine,
        ), dim=1)
        static = self.stem_project(self.stem(static_input, physical)) * mask_float
        for block in self.static_blocks:
            static = block(static, physical)
        time_states = self._time_states(forcing, ready)
        hidden = self.initial(static) * mask_float
        for index in range(STAGES):
            hidden = self.update(hidden, static, time_states[:, index], physical)
        detail = parent_support_center(self.detail_head(hidden), physical)
        pooled_hidden, _ = parent_support_mean(hidden, physical)
        pooled_static, _ = parent_support_mean(static, physical)
        parent_temporal = time_states[:, -1, :, None, None].expand(
            -1, -1, pooled_hidden.shape[-2], pooled_hidden.shape[-1]
        )
        u_features = torch.cat((pooled_hidden, pooled_static, parent_temporal, parent_context), dim=1)
        u_mean = self.u_head(u_features.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)
        u_mean = u_mean * unknown.to(dtype=u_mean.dtype)
        u_fine = u_mean.repeat_interleave(SCALE, -2).repeat_interleave(SCALE, -1)
        candidate = (detail + u_fine) * mask_float
        # Projection is deliberately outside autocast and is repaired once in float64.
        with torch.autocast(device_type=fine22.device.type, enabled=False):
            projection = support_project(candidate.double(), coarse_k.double(), physical, observed)
            projection = support_project(projection, coarse_k.double(), physical, observed)
        prediction = projection * physical.to(dtype=projection.dtype)
        return OrderedResidualComponents(
            prediction, detail, u_mean, observed, unknown, zero, ready,
        )

    def forward(
        self, fine22: Tensor, tempo: Tensor, tempo_valid: Tensor, solar3: Tensor,
        support: Tensor, coarse_k: Tensor, forcing: Tensor, forcing_ready: Tensor,
    ) -> Tensor:
        return self.forward_components(
            fine22, tempo, tempo_valid, solar3, support, coarse_k, forcing, forcing_ready,
        ).prediction_k
