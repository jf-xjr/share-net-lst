"""Static historical thermal guidance around one trainable query backbone.

The source cache, not this module, registers and validates the three historical
2018/2019/2020 observations. They are independent earlier thermal observations,
not short-time initial conditions. Neither query labels nor evaluation masks
are inputs. Construction reads no assets, and forward stores no input tensors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import _Block, _norm, support_project


HISTORY_CHANNELS = (
    "(nearest_filled_mean_temperature_k-300)/20",
    "(filled_temperature_k-own_4x4_parent_mean_k)/5",
    "historical_clear_count/16",
    "nearest_filled_mean_historical_st_qa/3",
    "(joint_masked_mean_emissivity-.98)/.01",
    "historical_joint_emissivity_coverage",
    "sin(2*pi*historical_doy/365.25)",
    "cos(2*pi*historical_doy/365.25)",
    "positive_query_minus_history_age_days/3652.5",
)


class HistoricalGuideNet(nn.Module):
    """Forward(fine52, coarse, support, context15, history[B,3,9,H,W])."""

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 initial_gain: float = 0.0, modality_dropout: float = 0.25) -> None:
        super().__init__()
        if isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError("historical width must be a positive multiple of eight")
        if not math.isfinite(initial_gain):
            raise ValueError("initial historical prior gain must be finite")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must be in [0,1]")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.width = int(width)
        self.initial_gain = float(initial_gain)
        self.modality_dropout = float(modality_dropout)
        self.physical_gain = nn.Parameter(torch.tensor(initial_gain, dtype=torch.float32))
        # Shared weights can compare dates by observed content and physical age;
        # no learned city identity or slot-specific lookup table is introduced.
        self.source_encoder = nn.Sequential(
            nn.Conv2d(9, width, 3, padding=1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        # Full Fine52, support and coarse level/validity/base discrepancy.
        self.query_encoder = nn.Sequential(
            nn.Conv2d(56, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        self.query_context = nn.Linear(15, 2 * width)
        self.date_gate = nn.Sequential(
            nn.Conv2d(2 * width, width, 1), nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        self.parent = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=4, bias=False),
            _norm(2 * width), nn.SiLU(), _Block(2 * width), _Block(2 * width),
        )
        self.parent_context = nn.Linear(15, 4 * width)
        for layer in (self.query_context, self.parent_context):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
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
            raise ValueError("historical guidance must add fewer than 300,000 parameters")
        if self.parameter_count >= 20_000_000:
            raise ValueError("combined deployment must contain fewer than twenty million parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def extra_parameter_count(self) -> int:
        return self.parameter_count - sum(p.numel() for p in self.backbone.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-historical-guide-network-v1",
            "class_name": "HistoricalGuideNet", "width": self.width,
            "initial_gain": self.initial_gain, "modality_dropout": self.modality_dropout,
            "parameter_count": self.parameter_count,
            "extra_parameter_count": self.extra_parameter_count,
            "history_shape": ["B", 3, 9, "H", "W"],
            "history_date_slots": [2018, 2019, 2020],
            "history_channels": list(HISTORY_CHANNELS),
            "history_interpretation": "static earlier thermal observations, not short-time initial conditions",
            "normalization": "fixed physical constants; no validation-fitted statistics",
            "missing_date": "all nine channels zero; no date-encoder or attention contribution",
            "filled_pixels": "CNN reads filled fields with coverage; physical prior uses actual clear coverage",
            "physical_prior": "gain * query_actual_support_Q[sum_t(5*history[t,1]*history[t,2])/sum_t(history[t,2])]",
            "physical_prior_empty_pixels": "zero before query actual-support Q",
            "initial_gain_basis": "zero until an explicitly recorded Fit-only pilot provides a replacement",
            "cnn_heads_zero_initialized": True, "backbone_trainable": True,
            "u0": "backbone plus initial_gain times query-support historical prior",
            "missing_modality": "exact current-backbone fallback",
            "dropout": "independent scene-wise whole-history modality dropout",
            "teacher_dependency": False, "target_or_scoring_mask_inputs": False,
            "geographic_inputs": False, "residual_amplitude_cap": None,
        }

    @property
    def historical_config(self) -> dict[str, object]:
        return self.model_config

    @staticmethod
    def _condition(value: Tensor, affine: Tensor) -> Tensor:
        scale, shift = affine.float().tanh().chunk(2, dim=1)
        return value * (1 + 0.25 * scale[:, :, None, None]) + 0.25 * shift[:, :, None, None]

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, history: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must have shape [B,52,H,W]")
        b, _, h, w = fine.shape
        if history.shape != (b, 3, 9, h, w):
            raise ValueError("history must have three ordered dates with nine scalar fields each")
        if support.shape != (b, 1, h, w) or context.shape != (b, 15):
            raise ValueError("support or Context15 geometry differs from the query")
        if h % 4 or w % 4 or coarse.shape != (b, 1, h // 4, w // 4):
            raise ValueError("coarse must be the query's exact x4 parent grid")

        historical = history.float()
        # Availability comes exclusively from historical clear coverage, never
        # from query fine-temperature validity or the formal evaluation mask.
        date_available = (historical[:, :, 2:3] > 0).flatten(3).any(dim=3)[..., None, None]
        scene_available = date_available.any(dim=1).float()
        if self.training and self.modality_dropout:
            keep = torch.rand((b, 1, 1, 1), device=fine.device) >= self.modality_dropout
            scene_available = scene_available * keep.float()
        active_dates = date_available & (scene_available[:, None] > 0)
        historical = torch.where(active_dates, historical, torch.zeros_like(historical))

        mask = support.float()
        base = fine[:, :1].float()
        scaled_fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        parent_base = F.avg_pool2d(base * mask, 4, 4) / fraction.clamp_min(1 / 16)
        observed = torch.isfinite(coarse) & (fraction > 0)
        safe_coarse = torch.where(observed, coarse.float(), parent_base)
        coarse_maps = F.interpolate(torch.cat(
            ((safe_coarse - 300) / 20, observed.float(), (safe_coarse - parent_base) / 20), dim=1),
            size=(h, w), mode="nearest")
        query = self.query_encoder(torch.cat((scaled_fine, mask, coarse_maps), dim=1))
        query = self._condition(query, self.query_context(context.float()))
        source = self.source_encoder(historical.reshape(3 * b, 9, h, w))
        source = source.reshape(b, 3, self.width, h, w)
        source = torch.where(active_dates, source, torch.zeros_like(source))
        query_dates = query[:, None].expand(-1, 3, -1, -1, -1)
        logits = self.date_gate(torch.cat((source, query_dates), dim=2).reshape(
            3 * b, 2 * self.width, h, w)).float().reshape(b, 3, 1, h, w)
        # The normalization remains finite when all three dates are unavailable.
        weights = torch.softmax(logits.masked_fill(~active_dates, -10000.0), dim=1)
        weights = weights * active_dates.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused_source = (source.float() * weights).sum(dim=1)
        value = self.fuse(torch.cat((query, fused_source), dim=1))
        parent = self._condition(self.parent(value), self.parent_context(context.float()))
        value = self.merge(torch.cat((value, F.interpolate(
            parent, size=(h, w), mode="bilinear", align_corners=False)), dim=1))

        gate = scene_available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse).float()
        coverage = historical[:, :, 2:3]
        coverage_sum = coverage.sum(dim=1)
        # Clear coverage is count/16. The small denominator floor changes no
        # physically valid nonempty pixel and avoids NaNs for empty pixels.
        prior_field = (5 * historical[:, :, 1:2] * coverage).sum(dim=1) / coverage_sum.clamp_min(1 / 16)
        prior_field = torch.where(coverage_sum > 0, prior_field, torch.zeros_like(prior_field))
        prior = support_project(prior_field * mask, zero_coarse, support)
        learned = support_project(self.dense_proposal(value).float() * gate, zero_coarse, support)
        return prediction + self.physical_gain.float() * prior + learned


__all__ = ["HistoricalGuideNet", "HISTORY_CHANNELS"]
