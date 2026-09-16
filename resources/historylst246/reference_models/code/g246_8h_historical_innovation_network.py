"""Replace part of a query prediction's spatial pattern with historical data.

This is a separate candidate from the audited additive HistoricalGuideNet.
The shared encoder and fusion layers are inherited without modifying that file.
Only earlier observations and query predictors enter forward. Optional backbone
inputs, such as emissivity, are passed directly and never cached on the module.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_historical_network import HistoricalGuideNet
from g246_8h_network import support_project


class HistoricalInnovationNet(HistoricalGuideNet):
    """One backbone plus learned historical addition and pattern replacement.

    ``backbone_extra_inputs`` are supplied to the backbone in their original
    order, for example one emissivity tensor for an EmissivityModel backbone.
    The default gains reflect an exploratory same-Fit12 conditional screen,
    not independent validation or whole-stack leave-city-out training.
    """

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 initial_gain: float = 0.28,
                 initial_replacement_gain: float = 0.22,
                 modality_dropout: float = 0.25) -> None:
        if not math.isfinite(initial_replacement_gain):
            raise ValueError("initial replacement gain must be finite")
        super().__init__(backbone, width=width, initial_gain=initial_gain,
                         modality_dropout=modality_dropout)
        self.initial_replacement_gain = float(initial_replacement_gain)
        self.replacement_gain = nn.Parameter(torch.tensor(initial_replacement_gain, dtype=torch.float32))
        if self.extra_parameter_count >= 300_000 or self.parameter_count >= 20_000_000:
            raise ValueError("historical innovation exceeds its parameter budget")

    @property
    def model_config(self) -> dict[str, object]:
        return {
            **super().model_config,
            "schema_version": "g246-8h-historical-innovation-network-v1",
            "class_name": "HistoricalInnovationNet",
            "initial_replacement_gain": self.initial_replacement_gain,
            "initial_gain_basis": "default .28/.22 round the same exploratory Fit12 two-coefficient conditional screen; not independent confirmation",
            "replacement_prior": "Q[any_history_clear_pixel * center_all_actual_support_parents(backbone_prediction)]",
            "parent_centering": "all nonempty O and U parents as a feature; final Q constrains observed O only",
            "output": "backbone_prediction + physical_gain*Qhistory - replacement_gain*Qpattern + learned_Q",
            "u0": "backbone plus initial_gain*Qhistory minus initial_replacement_gain*Qpattern; unsupported Z coordinates zero",
            "missing_modality": "bitwise current-backbone fallback on actual support; unsupported output coordinates zero",
            "backbone_extra_inputs": "forwarded in order without caching or detached inference",
            "parameter_count": self.parameter_count,
            "extra_parameter_count": self.extra_parameter_count,
            "target_or_scoring_mask_inputs": False,
        }

    @staticmethod
    def parent_center_all(prediction: Tensor, support: Tensor) -> Tensor:
        """Center actual support inside every nonempty parent, including U."""
        mask = support.float()
        safe = torch.where(mask > 0, prediction.float(), torch.zeros_like(prediction, dtype=torch.float32))
        fraction = F.avg_pool2d(mask, 4, 4)
        mean = F.avg_pool2d(safe, 4, 4) / fraction.clamp_min(1 / 16)
        lifted = F.interpolate(mean, scale_factor=4, mode="nearest")
        return torch.where(mask > 0, safe - lifted, torch.zeros_like(safe))

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, history: Tensor, *backbone_extra_inputs: Tensor) -> Tensor:
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
        # No stored request state, closures over inputs, detached teacher, or
        # second backbone evaluation: the optional inputs stay local to forward.
        prediction = self.backbone(augmented, coarse, support, context, *backbone_extra_inputs).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse).float()
        coverage = historical[:, :, 2:3]
        coverage_sum = coverage.sum(dim=1)
        history_field = (5 * historical[:, :, 1:2] * coverage).sum(dim=1) / coverage_sum.clamp_min(1 / 16)
        history_field = torch.where(coverage_sum > 0, history_field, torch.zeros_like(history_field))
        h_prior = support_project(history_field * mask, zero_coarse, support)
        # Centering a U parent here only defines the replacement feature. The
        # support_project call preserves that U feature; it observes no U mean.
        any_history_clear = coverage_sum > 0
        replacement_field = any_history_clear * self.parent_center_all(prediction, support)
        p_prior = support_project(replacement_field, zero_coarse, support)
        learned = support_project(self.dense_proposal(value).float() * gate, zero_coarse, support)
        output = prediction + self.physical_gain.float() * h_prior - self.replacement_gain.float() * p_prior + learned
        return torch.where(mask > 0, output, torch.zeros_like(output))


__all__ = ["HistoricalInnovationNet"]
