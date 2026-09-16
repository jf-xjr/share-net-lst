"""Shared native-30m optical filtering before temperature-scale fusion.

The existing detail cache stores band-major 4x4 phase residuals relative to
each 120m optical mean.  Adding the registered global-z mean and pixel-shuffling
reconstructs normalized optical samples on their 30m lattice.  This preserves
cache quantization; it does not claim recovery of the original float32 values.
Only optical QA defines native validity.  Fine targets and scoring masks are
absent from reconstruction and forward.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import _Block, _Context, _Down, _Fuse, _norm, support_project


def reconstruct_native_optical(fine: Tensor, detail: Tensor) -> tuple[Tensor, Tensor]:
    """Return normalized six-band optical and QA on the exact x4 native grid.

    Phase order is ``band, row_in_4, column_in_4``, matching pixel_shuffle.
    Invalid optical pixels become zero; they are explicitly represented in QA.
    The cache availability flag is handled separately by the fusion module.
    """
    if fine.ndim != 4 or fine.shape[1] != 52:
        raise ValueError("fine must have shape [B,52,H,W]")
    b, _, h, w = fine.shape
    if detail.shape != (b, 128, h, w):
        raise ValueError("detail must have shape [B,128,H,W] on the same grid")
    phases = detail[:, :96].float().reshape(b, 6, 16, h, w)
    means = fine[:, 2:8].float().unsqueeze(2)
    qa_phases = detail[:, 111:127].float()
    normalized = (phases + means) * qa_phases.unsqueeze(1)
    optical30 = F.pixel_shuffle(normalized.reshape(b, 96, h, w), 4)
    qa30 = F.pixel_shuffle(qa_phases, 4)
    return optical30, qa30


class NativeOpticalNet(nn.Module):
    """Add a trainable native-grid branch to a complete four-input backbone."""

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 modality_dropout: float = 0.25) -> None:
        super().__init__()
        if isinstance(width, bool) or width < 16 or width % 8:
            raise ValueError("width must be a multiple of eight and at least sixteen")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must be in [0,1]")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.width = int(width)
        self.modality_dropout = float(modality_dropout)
        half_width, parent_width, low_width = width * 3 // 2, width * 2, width * 3
        # Even 4x4/stride-two kernels put the resulting 120m sample at the
        # centre of its four native phases (native coordinate 4*i + 1.5).
        self.native_stem = nn.Sequential(
            nn.Conv2d(7, 16, 4, stride=2, padding=1, bias=False),
            _norm(16), nn.SiLU(), _Block(16, expansion=2),
            nn.Conv2d(16, width, 4, stride=2, padding=1, bias=False),
            _norm(width), nn.SiLU(), _Block(width, expansion=2),
        )
        # Native features, all normalized Fine52, support, and three coarse maps.
        self.fine_merge = nn.Sequential(
            nn.Conv2d(width + 56, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        self.down80 = nn.Sequential(_Down(width, half_width), _Block(half_width))
        self.down40 = nn.Sequential(_Down(half_width, parent_width), _Block(parent_width))
        self.down20 = nn.Sequential(_Down(parent_width, low_width), _Block(low_width))
        self.context40 = _Context(parent_width)
        self.context20 = _Context(low_width)
        self.decode40 = _Fuse(parent_width, low_width, 1)
        self.decode80 = _Fuse(half_width, parent_width, 1)
        self.decode160 = _Fuse(width, half_width, 1)
        self.fine_injection = nn.Conv2d(width, 51, 1)
        self.dense_proposal = nn.Conv2d(width, 1, 3, padding=1)
        for head in (self.fine_injection, self.dense_proposal):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.extra_parameter_count > 500_000:
            raise ValueError("native optical branch must add at most 500,000 parameters")
        if self.parameter_count >= 20_000_000:
            raise ValueError("combined deployment model must have fewer than twenty million parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def extra_parameter_count(self) -> int:
        return self.parameter_count - sum(p.numel() for p in self.backbone.parameters())

    @property
    def detail_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-native-optical-v1",
            "width": self.width, "modality_dropout": self.modality_dropout,
            "extra_parameter_count": self.extra_parameter_count,
            "parameter_count": self.parameter_count,
            "native_inputs": "phase96 + Fine52 global-z optical means; phase16 optical QA",
            "native_grid": "30m 640x640; shared 7->16->width stride-two convolutions",
            "context_grid_path": "160-80-40-20-40-80-160",
            "fine_injection_channels": [1, 52],
            "kelvin_base_modified_by_injection": False,
            "backbone_trainable": True, "heads_zero_initialized": True,
            "proposal_amplitude_cap": None,
            "covariance_channels_consumed": False,
            "teacher_dependency": False, "auxiliary_target_inputs": False,
        }

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, detail: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must be [B,52,H,W]")
        b, _, h, w = fine.shape
        if h % 8 or w % 8:
            raise ValueError("fine geometry must be divisible by eight")
        if detail.shape != (b, 128, h, w):
            raise ValueError("detail must be [B,128,H,W]")
        if support.shape != (b, 1, h, w) or context.shape != (b, 15):
            raise ValueError("support or Context15 geometry differs from the query")
        if coarse.shape != (b, 1, h // 4, w // 4):
            raise ValueError("coarse must be the query's exact x4 grid")
        available = (detail[:, 127:128].amax(dim=(-2, -1), keepdim=True) > 0).float()
        if self.training and self.modality_dropout:
            keep = torch.rand((b, 1, 1, 1), device=fine.device) >= self.modality_dropout
            available = available * keep.float()
        active_detail = detail.float() * available
        optical30, qa30 = reconstruct_native_optical(fine, active_detail)
        native120 = self.native_stem(torch.cat((optical30, qa30), dim=1))

        mask = support.float()
        base = fine[:, :1].float()
        normalized_fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        parent_base = F.avg_pool2d(base * mask, 4, 4) / fraction.clamp_min(1 / 16)
        valid = torch.isfinite(coarse) & (fraction > 0)
        safe = torch.where(valid, coarse.float(), parent_base)
        coarse_features = F.interpolate(torch.cat(
            ((safe - 300) / 20, valid.float(), (safe - parent_base) / 20), dim=1),
            size=(h, w), mode="nearest")
        fine160 = self.fine_merge(torch.cat((native120, normalized_fine, mask, coarse_features), dim=1))
        half80 = self.down80(fine160)
        parent40 = self.context40(self.down40(half80), context)
        low20 = self.context20(self.down20(parent40), context)
        value = self.decode160(fine160, self.decode80(
            half80, self.decode40(parent40, low20)))

        gate = available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context).float()
        proposal = self.dense_proposal(value).float() * gate
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse)
        return prediction + support_project(proposal, zero_coarse.float(), support)


__all__ = ["NativeOpticalNet", "reconstruct_native_optical"]
