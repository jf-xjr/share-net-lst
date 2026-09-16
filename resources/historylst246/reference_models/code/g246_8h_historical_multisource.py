"""Shared historical innovation layers with an explicitly registered source count.

This prepares a six-observation candidate without changing the running three-
observation implementation. Source selection and chronological admissibility
remain responsibilities of the separately bound input cache.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_historical_innovation_network import HistoricalInnovationNet
from g246_8h_network import support_project


class HistoricalMultiSourceNet(HistoricalInnovationNet):
    def __init__(self, backbone: nn.Module, *, width: int = 32, source_count: int = 6,
                 initial_gain: float = .28, initial_replacement_gain: float = .22,
                 modality_dropout: float = .25):
        if isinstance(source_count, bool) or source_count not in (3, 6):
            raise ValueError('historical source count must be explicitly three or six')
        super().__init__(backbone, width=width, initial_gain=initial_gain,
                         initial_replacement_gain=initial_replacement_gain,
                         modality_dropout=modality_dropout)
        self.source_count = int(source_count)

    @property
    def model_config(self):
        return {**super().model_config,
                'schema_version': 'g246-8h-historical-multisource-network-v1',
                'class_name': 'HistoricalMultiSourceNet', 'source_count': self.source_count,
                'history_shape': ['B', self.source_count, 9, 'H', 'W'],
                'history_date_slots': [2018, 2019, 2020] * (self.source_count // 3),
                'source_selection': 'registered cache contract; no query label or learned identity lookup',
                'parameter_sharing': 'identical per-observation encoder and query-conditioned gate',
                'three_source_reference': 'same operations and parameter names as HistoricalInnovationNet'}

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, history: Tensor, *backbone_extra_inputs: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError('fine must have shape [B,52,H,W]')
        b, _, h, w = fine.shape
        n = self.source_count
        if history.shape != (b, n, 9, h, w):
            raise ValueError('history differs from its registered source count or nine scalar fields')
        if support.shape != (b, 1, h, w) or context.shape != (b, 15):
            raise ValueError('support or Context15 geometry differs from the query')
        if h % 4 or w % 4 or coarse.shape != (b, 1, h // 4, w // 4):
            raise ValueError('coarse must be the query exact x4 parent grid')
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
            size=(h, w), mode='nearest')
        query = self.query_encoder(torch.cat((scaled_fine, mask, coarse_maps), dim=1))
        query = self._condition(query, self.query_context(context.float()))
        source = self.source_encoder(historical.reshape(n * b, 9, h, w))
        source = source.reshape(b, n, self.width, h, w)
        source = torch.where(active_dates, source, torch.zeros_like(source))
        query_dates = query[:, None].expand(-1, n, -1, -1, -1)
        logits = self.date_gate(torch.cat((source, query_dates), dim=2).reshape(
            n * b, 2 * self.width, h, w)).float().reshape(b, n, 1, h, w)
        weights = torch.softmax(logits.masked_fill(~active_dates, -10000.0), dim=1)
        weights = weights * active_dates.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused_source = (source.float() * weights).sum(dim=1)
        value = self.fuse(torch.cat((query, fused_source), dim=1))
        parent = self._condition(self.parent(value), self.parent_context(context.float()))
        value = self.merge(torch.cat((value, F.interpolate(
            parent, size=(h, w), mode='bilinear', align_corners=False)), dim=1))
        gate = scene_available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context, *backbone_extra_inputs).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse).float()
        coverage = historical[:, :, 2:3]
        coverage_sum = coverage.sum(dim=1)
        history_field = (5 * historical[:, :, 1:2] * coverage).sum(dim=1) / coverage_sum.clamp_min(1 / 16)
        history_field = torch.where(coverage_sum > 0, history_field, torch.zeros_like(history_field))
        h_prior = support_project(history_field * mask, zero_coarse, support)
        any_history_clear = coverage_sum > 0
        replacement_field = any_history_clear * self.parent_center_all(prediction, support)
        p_prior = support_project(replacement_field, zero_coarse, support)
        learned = support_project(self.dense_proposal(value).float() * gate, zero_coarse, support)
        output = prediction + self.physical_gain.float() * h_prior - self.replacement_gain.float() * p_prior + learned
        return torch.where(mask > 0, output, torch.zeros_like(output))


__all__ = ['HistoricalMultiSourceNet']
