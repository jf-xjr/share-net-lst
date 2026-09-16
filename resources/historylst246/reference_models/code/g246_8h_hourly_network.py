"""Full-trainable Fine52 backbone conditioned on existing 48-hour POWER data.

The sequence contains Tair/RH/wind/shortwave and solar state; it contains no
new rainfall or soil-wetness observations. Only feature/tangent heads start at
zero, so initialization reproduces a warm-start backbone exactly.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import support_project


def transform_hourly_d4(value, code: int):
    """D4-transform hourly solar east/north (slots 5/6), preserving time order.

    Accepts numpy or torch arrays [...,48,10]. The convention matches
    g246_8h_augment: counterclockwise rot90(code&3), then horizontal reflection.
    """
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 8:
        raise ValueError("D4 code must be an integer in [0,7]")
    if not isinstance(value, (Tensor, np.ndarray)) or value.shape[-2:] != (48, 10):
        raise ValueError("hourly values must have shape [...,48,10]")
    result = value.clone() if isinstance(value, Tensor) else value.copy()
    east, north = value[..., 5], value[..., 6]
    rotation = code & 3
    if rotation == 0:
        new_east, new_north = east, north
    elif rotation == 1:
        new_east, new_north = -north, east
    elif rotation == 2:
        new_east, new_north = -east, -north
    else:
        new_east, new_north = north, -east
    if code & 4:
        new_east = -new_east
    result[..., 5], result[..., 6] = new_east, new_north
    return result


def _norm(width: int) -> nn.GroupNorm:
    return nn.GroupNorm(gcd(width, 8), width)


class _SpatialBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 5, padding=2, groups=width),
            _norm(width), nn.SiLU(), nn.Conv2d(width, 2 * width, 1), nn.SiLU(),
            nn.Conv2d(2 * width, width, 1))

    def forward(self, value: Tensor) -> Tensor:
        return value + 0.5 * self.body(value)


class HourlyConditionedNet(nn.Module):
    """forward(fine52, coarse, support, context15, hourly[B,48,10]) -> Kelvin.

    A small Transformer learns order-sensitive forcing features. Its pooled
    state conditions fine-resolution spatial features and their parent-scale
    neighborhoods, then changes all 51 normalized Fine52 coordinates and
    predicts an unrestricted support-feasible dense correction. Complete
    modality dropout and all-zero missing tokens use the current backbone
    exactly, including after training.
    """
    def __init__(self, backbone: nn.Module, *, width: int = 32, time_width: int = 48,
                 modality_dropout: float = 0.10):
        super().__init__()
        if width < 8 or width % 8 or time_width < 8 or time_width % 4:
            raise ValueError("width must be a multiple of 8; time_width a multiple of 4")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must be in [0,1]")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.width, self.time_width = int(width), int(time_width)
        self.modality_dropout = float(modality_dropout)
        self.time_input = nn.Sequential(nn.Linear(10, time_width), nn.LayerNorm(time_width), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=time_width, nhead=4,
            dim_feedforward=2 * time_width, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.time_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.time_norm = nn.LayerNorm(time_width)
        self.time_attention = nn.Linear(time_width, 1)
        self.state = nn.Sequential(nn.Linear(3 * time_width + 15, 3 * width), nn.SiLU(),
                                   nn.Linear(3 * width, 3 * width))
        # Fine52 (Kelvin channel centered) + support + observed coarse level.
        self.stem = nn.Sequential(nn.Conv2d(54, width, 1, bias=False), _norm(width),
                                  nn.SiLU(), _SpatialBlock(width))
        self.parent = nn.Sequential(nn.Conv2d(width, width, 4, stride=4, bias=False),
                                    _norm(width), nn.SiLU(), _SpatialBlock(width))
        self.merge = nn.Sequential(nn.Conv2d(3 * width, width, 1, bias=False),
                                   _norm(width), nn.SiLU(), _SpatialBlock(width),
                                   _SpatialBlock(width))
        self.fine_injection = nn.Conv2d(width, 51, 1)
        self.dense_proposal = nn.Conv2d(width, 1, 3, padding=1)
        for head in (self.fine_injection, self.dense_proposal):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.extra_parameter_count >= 500_000:
            raise ValueError("hourly conditioning must add fewer than 0.5M parameters")
        if self.parameter_count >= 20_000_000:
            raise ValueError("combined model must contain fewer than 20M parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def extra_parameter_count(self) -> int:
        return self.parameter_count - sum(parameter.numel() for parameter in self.backbone.parameters())

    @property
    def hourly_config(self) -> dict[str, object]:
        return {"schema_version": "g246-8h-hourly-conditioned-network-v1", "width": self.width,
                "time_width": self.time_width, "modality_dropout": self.modality_dropout,
                "extra_parameter_count": self.extra_parameter_count, "parameter_count": self.parameter_count,
                "hourly_shape": [48, 10], "backbone_trainable": True,
                "fine_injection_channels": [1, 52], "kelvin_channel_modified_by_injection": False,
                "proposal_amplitude_cap": None, "hourly_rain_or_soil_wetness_present": False}

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, hourly: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must be [B,52,H,W]")
        batch, _, height, width = fine.shape
        if hourly.shape != (batch, 48, 10) or context.shape != (batch, 15):
            raise ValueError("hourly/context shape differs from the query")
        if height % 4 or width % 4 or coarse.shape != (batch, 1, height // 4, width // 4) \
                or support.shape != (batch, 1, height, width):
            raise ValueError("support/coarse must be aligned to the x4 fine grid")
        tokens = hourly.float()
        available = (tokens.abs().amax(dim=(1, 2), keepdim=True) > 0).float().reshape(batch, 1, 1, 1)
        if self.training and self.modality_dropout:
            available = available * (torch.rand((batch, 1, 1, 1), device=fine.device)
                                       >= self.modality_dropout).float()
        tokens = tokens * available.reshape(batch, 1, 1)
        encoded = self.time_norm(self.time_encoder(self.time_input(tokens)))
        attention = self.time_attention(encoded).softmax(dim=1)
        summary = torch.cat((encoded.mean(dim=1), encoded[:, -1],
                             (encoded * attention).sum(dim=1), context.float()), dim=1)
        scale, shift, state = self.state(summary).chunk(3, dim=1)
        mask = support.float()
        scaled_fine = torch.cat(((fine[:, :1].float() - 300.0) / 20.0,
                                fine[:, 1:].float()), dim=1) * mask
        coarse_scaled = torch.where(torch.isfinite(coarse), (coarse.float() - 300.0) / 20.0,
                                    torch.zeros_like(coarse, dtype=torch.float32))
        coarse_up = F.interpolate(coarse_scaled, size=(height, width), mode="nearest")
        value = self.stem(torch.cat((scaled_fine, mask, coarse_up), dim=1))
        value = value * (1.0 + scale.tanh()[:, :, None, None]) + shift[:, :, None, None]
        parent = F.interpolate(self.parent(value), size=(height, width), mode="bilinear", align_corners=False)
        state = state[:, :, None, None].expand(-1, -1, height, width)
        value = self.merge(torch.cat((value, parent, state), dim=1))
        gate = mask * available
        injection = self.fine_injection(value).float() * gate
        augmented_fine = torch.cat((fine[:, :1].float(), fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented_fine, coarse, support, context)
        proposal = self.dense_proposal(value).float() * gate
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse)
        tangent = support_project(proposal, zero_coarse.float(), support)
        return prediction.float() + tangent


__all__ = ["HourlyConditionedNet", "transform_hourly_d4"]
