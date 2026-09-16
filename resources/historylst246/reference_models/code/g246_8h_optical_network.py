"""Trainable optical-detail fusion around a registered four-input backbone.

The optional 30m detail is encoded with the query's registered content and
physical context. It can change all 51 non-Kelvin Fine52 coordinates seen by
the full trainable backbone and propose an unrestricted dense tangent field.
Only the two injection heads start at zero. Thus initialization is exactly the
backbone and the detail encoder receives gradients after the first head step.
"""
from __future__ import annotations

from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import support_project


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(gcd(channels, 8), channels)


class _DetailBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False)
        self.norm = _norm(width)
        self.expand = nn.Conv2d(width, 2 * width, 1)
        self.contract = nn.Conv2d(2 * width, width, 1)

    def forward(self, value: Tensor) -> Tensor:
        return value + 0.5 * self.contract(F.silu(self.expand(F.silu(self.norm(self.spatial(value))))))


class OpticalDetailNet(nn.Module):
    """Forward(fine52, coarse, support, context15, detail128) -> Kelvin field.

    Missing detail and stochastic modality dropout both execute the current
    backbone exactly. The backbone is never frozen. Identity metadata and
    scoring masks are absent from this module's interface.
    """
    def __init__(self, backbone: nn.Module, *, width: int = 48,
                 modality_dropout: float = 0.25) -> None:
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("detail width must be a positive multiple of eight")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must lie in [0,1]")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.width = int(width)
        self.modality_dropout = float(modality_dropout)
        # 128 details + normalized Fine52 + support + observed coarse level.
        self.stem = nn.Sequential(nn.Conv2d(182, width, 1, bias=False),
                                  _norm(width), nn.SiLU(), _DetailBlock(width))
        self.parent = nn.Sequential(nn.Conv2d(width, 2 * width, 4, stride=4, bias=False),
                                    _norm(2 * width), nn.SiLU(),
                                    _DetailBlock(2 * width), _DetailBlock(2 * width))
        self.context = nn.Linear(15, 4 * width)
        nn.init.normal_(self.context.weight, std=0.01)
        nn.init.zeros_(self.context.bias)
        self.merge = nn.Sequential(nn.Conv2d(3 * width, width, 1, bias=False),
                                   _norm(width), nn.SiLU(),
                                   _DetailBlock(width), _DetailBlock(width))
        # Channel zero is an absolute Kelvin base; it is not a latent feature.
        self.fine_injection = nn.Conv2d(width, 51, 1)
        self.dense_proposal = nn.Conv2d(width, 1, 3, padding=1)
        for head in (self.fine_injection, self.dense_proposal):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.extra_parameter_count >= 1_000_000:
            raise ValueError("optical detail adds one million or more parameters")
        if self.parameter_count >= 20_000_000:
            raise ValueError("combined model must contain fewer than twenty million parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def extra_parameter_count(self) -> int:
        return self.parameter_count - sum(parameter.numel() for parameter in self.backbone.parameters())

    @property
    def detail_config(self) -> dict[str, object]:
        """Root trainer owns model_spec/family and checkpoint reconstruction."""
        return {"schema_version": "g246-8h-optical-detail-network-v1",
                "width": self.width, "modality_dropout": self.modality_dropout,
                "extra_parameter_count": self.extra_parameter_count,
                "parameter_count": self.parameter_count,
                "detail_channels": 128, "fine_injection_channels": [1, 52],
                "kelvin_channel_modified_by_injection": False,
                "backbone_trainable": True, "proposal_amplitude_cap": None,
                "output": "feasible_backbone_plus_actual_support_dense_tangent"}

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, detail128: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must be [B,52,H,W]")
        batch, _, height, width = fine.shape
        if detail128.shape != (batch, 128, height, width):
            raise ValueError("detail128 must be [B,128,H,W] in the base-cache order")
        if context.shape != (batch, 15) or support.shape != (batch, 1, height, width):
            raise ValueError("context/support shapes differ from the query")
        if coarse.shape != (batch, 1, height // 4, width // 4) or height % 4 or width % 4:
            raise ValueError("coarse must be the aligned x4 parent grid")
        details = detail128.float()
        available = (details[:, 127:128].amax(dim=(-2, -1), keepdim=True) > 0).float()
        if self.training and self.modality_dropout:
            available = available * (torch.rand((batch, 1, 1, 1), device=fine.device)
                                       >= self.modality_dropout).float()
        details = details * available
        mask = support.float()
        scaled_fine = torch.cat(((fine[:, :1].float() - 300.0) / 20.0,
                                fine[:, 1:].float()), dim=1) * mask
        coarse_scaled = torch.where(torch.isfinite(coarse), (coarse.float() - 300.0) / 20.0,
                                    torch.zeros_like(coarse, dtype=torch.float32))
        coarse_up = F.interpolate(coarse_scaled, size=(height, width), mode="nearest")
        value = self.stem(torch.cat((details, scaled_fine, mask, coarse_up), dim=1))
        parent = self.parent(value)
        scale, shift = self.context(context.float()).chunk(2, dim=1)
        parent = parent * (1 + 0.25 * scale.tanh()[:, :, None, None]) \
            + 0.25 * shift[:, :, None, None]
        value = self.merge(torch.cat((value, F.interpolate(parent, size=(height, width),
                                                          mode="bilinear", align_corners=False)), dim=1))
        gate = available * mask
        injection = self.fine_injection(value).float() * gate
        augmented_fine = torch.cat((fine[:, :1].float(), fine[:, 1:].float() + injection), dim=1)
        base_prediction = self.backbone(augmented_fine, coarse, support, context)
        proposal = self.dense_proposal(value).float() * gate
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse)
        tangent = support_project(proposal, zero_coarse.float(), support)
        # Reprojecting the already feasible base again would introduce a second
        # float32 repair at initialization. A projected tangent preserves both
        # the u0 equality and actual-support closure up to normal float32 error.
        return base_prediction.float() + tangent


__all__ = ["OpticalDetailNet"]
