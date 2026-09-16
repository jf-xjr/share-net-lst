"""Attention on physical historical fusion with one new trainable scalar.

All other encoder, feature injection, replacement, and output layers retain
HistoricalMultiSourceNet parameter names and operations. No constructor reads
assets; full historical deploy loading is explicit and validates every key.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_historical_multisource import HistoricalMultiSourceNet
from g246_8h_network import support_project


class HistoricalAttendedInnovationNet(HistoricalMultiSourceNet):
    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 source_count: int = 3, initial_gain: float = .28,
                 initial_replacement_gain: float = .22,
                 modality_dropout: float = .25,
                 initial_attention_gain: float = 0.0) -> None:
        if not math.isfinite(initial_attention_gain):
            raise ValueError('initial physical attention gain must be finite')
        super().__init__(backbone, width=width, source_count=source_count,
                         initial_gain=initial_gain,
                         initial_replacement_gain=initial_replacement_gain,
                         modality_dropout=modality_dropout)
        self.initial_attention_gain = float(initial_attention_gain)
        self.physical_attention_gain = nn.Parameter(torch.tensor(initial_attention_gain, dtype=torch.float32))
        if self.extra_parameter_count >= 300_000 or self.parameter_count >= 20_000_000:
            raise ValueError('attended historical model exceeds its parameter budget')

    @property
    def model_config(self):
        return {**super().model_config,
                'schema_version': 'g246-8h-historical-attended-innovation-network-v1',
                'class_name': 'HistoricalAttendedInnovationNet',
                'initial_attention_gain': self.initial_attention_gain,
                'physical_attention': 'coverage * exp(clamp(gain * (logit - common_date_center), -20, 20))',
                'physical_prior': 'physical_gain * actual-support Q[sum_t(5*history_channel1*attention_weight)/sum_t(attention_weight)]',
                'physical_logit_center': 'per-pixel mean of logits over dates with positive historical clear coverage; missing dates excluded',
                'physical_weight_normalization': 'divide weighted sum by actual weight sum clamped only at 1e-12; empty pixels zero',
                'zero_attention_gain': 'original coverage-weighted historical prior; no change to learned CNN fusion or replacement term',
                'three_source_reference': 'all existing layers and state names unchanged; only physical-prior date weights are attended',
                'warmstart_u0': 'zero new attention gain reproduces the complete trained historical innovation prediction',
                'additional_parameters_vs_multisource': 1,
                'warmstart': 'complete historical wrapper state; only new physical_attention_gain may be absent',
                'parameter_count': self.parameter_count,
                'extra_parameter_count': self.extra_parameter_count}

    def load_historical_deploy(self, checkpoint: Mapping) -> dict[str, object]:
        """Strictly import a full historical wrapper, adding only one zero scalar.

        Accept an already loaded deployment dictionary. The caller owns the
        immutable byte snapshot and provenance hash. Forward inputs and tensors
        from the checkpoint are not retained as temporary module attributes.
        """
        if checkpoint.get('schema') != 'g246-8h-deploy-v1' or checkpoint.get('locked_test_opened') is not False:
            raise ValueError('a complete locked-test-closed historical deployment is required')
        spec = checkpoint.get('model_spec', {})
        if not isinstance(spec, Mapping) or not str(spec.get('family', '')).startswith('historical_'):
            raise ValueError('warmstart source must be a complete historical model')
        state = checkpoint.get('state_dict')
        if not isinstance(state, Mapping) or not state or not all(str(k).startswith('net.') for k in state):
            raise ValueError('historical deployment must use complete wrapper state with net.* keys')
        incoming = {str(k)[4:]: value for k, value in state.items()}
        expected = self.state_dict()
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        if missing != ['physical_attention_gain'] or unexpected:
            raise ValueError(f'only physical_attention_gain may be new; missing={missing}, unexpected={unexpected}')
        for name, value in incoming.items():
            if not isinstance(value, Tensor) or value.shape != expected[name].shape:
                raise ValueError(f'historical warmstart tensor shape differs: {name}')
        # A transfer always begins at the exact old physical weighting. A full
        # attended cold reconstruction instead loads its complete state directly.
        incoming['physical_attention_gain'] = expected['physical_attention_gain'].new_zeros(())
        self.load_state_dict(incoming, strict=True)
        self.initial_attention_gain = 0.0
        return {'source_family': spec['family'],
                'source_selected_update': checkpoint.get('selected_update'),
                'source_selected_weights': checkpoint.get('selected_weights'),
                'source_parameter_count': checkpoint.get('parameter_count'),
                'parameter_count': self.parameter_count,
                'source_count': self.source_count,
                'wrapper_prefix_removed': 'net.',
                'only_added_state_key': 'physical_attention_gain',
                'physical_attention_gain_after_load': 0.0,
                'all_remaining_keys_strict': True}

    def attended_physical_template(self, historical: Tensor, logits: Tensor,
                                   active_dates: Tensor) -> Tensor:
        """Differentiable physical fusion; no targets or query masks are needed."""
        coverage = historical[:, :, 2:3].float()
        eligible = (coverage > 0) & active_dates
        safe_logits = torch.where(eligible, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
        center = safe_logits.sum(dim=1, keepdim=True) / eligible.float().sum(dim=1, keepdim=True).clamp_min(1)
        centered = torch.where(eligible, safe_logits - center, torch.zeros_like(safe_logits))
        exponent = (self.physical_attention_gain.float() * centered).clamp(-20, 20)
        physical_weights = coverage * exponent.exp()
        total = physical_weights.sum(dim=1)
        value = (5 * historical[:, :, 1:2].float() * physical_weights).sum(dim=1) / total.clamp_min(1e-12)
        return torch.where(total > 0, value, torch.zeros_like(value))

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
        history_field = self.attended_physical_template(historical, logits, active_dates)
        h_prior = support_project(history_field * mask, zero_coarse, support)
        any_history_clear = coverage_sum > 0
        replacement_field = any_history_clear * self.parent_center_all(prediction, support)
        p_prior = support_project(replacement_field, zero_coarse, support)
        learned = support_project(self.dense_proposal(value).float() * gate, zero_coarse, support)
        output = prediction + self.physical_gain.float() * h_prior - self.replacement_gain.float() * p_prior + learned
        return torch.where(mask > 0, output, torch.zeros_like(output))


__all__ = ['HistoricalAttendedInnovationNet']
