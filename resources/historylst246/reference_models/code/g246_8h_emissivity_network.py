"""Emissivity guidance with a Fit-screened physical initialization.

The four input channels must already obey the independently audited physical
validity mask and aggregation contract.  This module consumes neither source
quality arrays nor target/evaluation masks.  No validation statistics enter its
fixed physical scaling or the initial gain of 0.5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import _Block, _norm, support_project


class EmissivityNet(nn.Module):
    """Four-channel physical guidance around a fully trainable backbone."""

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 modality_dropout: float = 0.25) -> None:
        super().__init__()
        if isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError("emissivity width must be a positive multiple of eight")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must be in [0,1]")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.width = int(width)
        self.modality_dropout = float(modality_dropout)
        self.physical_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Four physical channels, Fine52, support, coarse value/validity/defect.
        self.stem = nn.Sequential(
            nn.Conv2d(60, width, 1, bias=False), _norm(width), nn.SiLU(), _Block(width),
        )
        self.parent = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=4, bias=False),
            _norm(2 * width), nn.SiLU(), _Block(2 * width), _Block(2 * width),
        )
        self.context = nn.Linear(15, 4 * width)
        nn.init.normal_(self.context.weight, std=0.01)
        nn.init.zeros_(self.context.bias)
        self.merge = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width), _Block(width),
        )
        self.fine_injection = nn.Conv2d(width, 51, 1)
        self.dense_proposal = nn.Conv2d(width, 1, 3, padding=1)
        for head in (self.fine_injection, self.dense_proposal):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.extra_parameter_count >= 300_000:
            raise ValueError("emissivity guidance must add fewer than 300,000 parameters")
        if self.parameter_count >= 20_000_000:
            raise ValueError("combined deployment model must have fewer than twenty million parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def extra_parameter_count(self) -> int:
        return self.parameter_count - sum(p.numel() for p in self.backbone.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-emissivity-network-v1",
            "class_name": "EmissivityNet", "width": self.width,
            "modality_dropout": self.modality_dropout,
            "extra_parameter_count": self.extra_parameter_count,
            "parameter_count": self.parameter_count,
            "input_channels": ["(meanE_filled-.98)/.01", "stdE120/.01",
                               "log1p(meanEMSD/.01)", "availability_fraction120"],
            "normalization": "fixed physical constants; no validation-fitted statistics",
            "input_mask": "externally audited joint physical E/EMSD mask before 4x4 aggregation",
            "physical_prior": "trainable gain * Q[-0.8 * emissivity_channel_0]",
            "physical_prior_equivalent": "gain * Q[80 * (.98 - meanE_filled)]",
            "initial_gain": 0.5,
            "initial_gain_basis": "Fit12 conditional correction screen; not whole-stack OOF",
            "u0": "backbone + 0.5 * actual-support projected physical prior",
            "cnn_heads_zero_initialized": True, "backbone_trainable": True,
            "missing_modality": "exact current-backbone fallback",
            "teacher_dependency": False, "target_or_scoring_mask_inputs": False,
            "geographic_inputs": False, "residual_amplitude_cap": None,
        }

    @property
    def emissivity_config(self) -> dict[str, object]:
        return self.model_config

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, emissivity: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must have shape [B,52,H,W]")
        b, _, h, w = fine.shape
        if emissivity.shape != (b, 4, h, w):
            raise ValueError("emissivity must have shape [B,4,H,W]")
        if support.shape != (b, 1, h, w) or context.shape != (b, 15):
            raise ValueError("support or Context15 geometry differs from the query")
        if coarse.shape != (b, 1, h // 4, w // 4) or h % 4 or w % 4:
            raise ValueError("coarse must be the query's exact x4 parent grid")

        pixel_available = emissivity[:, 3:4] > 0
        scene_available = pixel_available.flatten(1).any(dim=1).float()[:, None, None, None]
        if self.training and self.modality_dropout:
            keep = torch.rand((b, 1, 1, 1), device=fine.device) >= self.modality_dropout
            scene_available = scene_available * keep.float()
        # Joint-mask nearest filling is itself QA-path invariant. Preserve those
        # physical estimates and expose uncertainty through the coverage channel.
        # An entirely unavailable or dropped modality still falls back exactly.
        physical = torch.where(scene_available > 0, emissivity.float(),
                               torch.zeros_like(emissivity, dtype=torch.float32))
        mask = support.float()
        base = fine[:, :1].float()
        scaled_fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        parent_base = F.avg_pool2d(base * mask, 4, 4) / fraction.clamp_min(1 / 16)
        valid = torch.isfinite(coarse) & (fraction > 0)
        safe_coarse = torch.where(valid, coarse.float(), parent_base)
        coarse_maps = F.interpolate(torch.cat(
            ((safe_coarse - 300) / 20, valid.float(), (safe_coarse - parent_base) / 20), dim=1),
            size=(h, w), mode="nearest")
        value = self.stem(torch.cat((physical, scaled_fine, mask, coarse_maps), dim=1))
        parent = self.parent(value)
        scale, shift = self.context(context.float()).tanh().chunk(2, dim=1)
        parent = parent * (1 + 0.25 * scale[:, :, None, None]) + 0.25 * shift[:, :, None, None]
        value = self.merge(torch.cat((value, F.interpolate(
            parent, size=(h, w), mode="bilinear", align_corners=False)), dim=1))

        gate = scene_available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse)
        prior = support_project(-0.8 * physical[:, :1] * mask, zero_coarse.float(), support)
        learned = support_project(self.dense_proposal(value).float() * gate,
                                  zero_coarse.float(), support)
        return prediction + self.physical_gain.float() * prior + learned


__all__ = ["EmissivityNet"]
