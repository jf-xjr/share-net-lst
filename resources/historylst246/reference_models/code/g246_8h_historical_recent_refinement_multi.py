"""Three separately encoded recent observations refine one complete old six-source model.

Every learned parameter and state name is shared with the seven-slot refinement
network. The extension adds only coverage-normalized fusion, not parameters.
Input and chronological registration require the separate nine-slot contract.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_historical_innovation_network import HistoricalInnovationNet
from g246_8h_historical_recent_refinement import HistoricalRecentRefinementNet
from g246_8h_network import support_project


class HistoricalRecentMultiRefinementNet(HistoricalRecentRefinementNet):
    """Forward(base4, history9, emissivity); first six slots retain their own backbone."""

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 modality_dropout: float = 0.0) -> None:
        super().__init__(backbone, width=width, modality_dropout=modality_dropout)
        self.source_count = 9

    @property
    def model_config(self) -> dict[str, object]:
        config = dict(super().model_config)
        config.pop('recent_history_slot', None)
        return {
            **config,
            'schema_version': 'g246-8h-historical-recent-multi-refinement-network-v1',
            'class_name': 'HistoricalRecentMultiRefinementNet',
            'source_count': 9, 'history_shape': ['B', 9, 9, 'H', 'W'],
            'recent_history_slots': [6, 7, 8],
            'recent_spatial_encoder': 'same learned encoder applied independently to each of three recent observations before explicit normalized fusion',
            'recent_CNN_fusion': 'per-pixel normalized positive clear coverage; where all three local coverages are zero, uniformly combine filled features from scene-available recent dates',
            'single_available_recent': 'direct encoder output and direct 5*channel1 clear-masked physical field preserve the original seven-slot operations exactly',
            'physical_prior': 'Q[sum_i(normalized_clear_coverage_i * 5*recent_i_channel1)]; local all-clear-missing field is zero before Q',
            'replacement_prior': 'Q[any_of_three_recent_clear_pixel * center_all_actual_support_parents(current_six_source_prediction)]',
            'missing_recent': 'all three dates missing or dropped: exact current six-source fallback on support; unsupported pixels zero',
            'additional_parameters_vs_seven_refinement': 0,
            'date_gate': None,
            'warmstart_six': 'inherited load_six_deploy strictly imports the whole six-source wrapper and zeros all new output heads/gains',
            'warmstart_seven': 'load_seven_deploy strictly preserves all seven-refinement parameters; two appended empty/NaN dates reproduce the complete seven-slot function',
            'u0': 'from old six: zero refinement outputs preserve old six for any nine-slot input; from trained seven: empty appended slots preserve old seven',
            'parameter_count': self.parameter_count,
            'extra_parameter_count': self.extra_parameter_count,
        }

    def load_seven_deploy(self, checkpoint: Mapping) -> dict[str, object]:
        """Strictly preserve the complete seven-slot refinement, including learned outputs."""
        if checkpoint.get('schema') != 'g246-8h-deploy-v1' or checkpoint.get('locked_test_opened') is not False:
            raise ValueError('a complete locked-test-closed seven-refinement deployment is required')
        spec = checkpoint.get('model_spec', {})
        if not isinstance(spec, Mapping) or spec.get('family') != 'historical_recent_refinement_emissivity_r6a_seven':
            raise ValueError('warmstart requires the registered seven-slot separate recent refinement family')
        state = checkpoint.get('state_dict')
        if not isinstance(state, Mapping) or not state or not all(str(key).startswith('net.') for key in state):
            raise ValueError('warmstart requires complete net.* seven-refinement wrapper state')
        incoming = {str(key)[4:]: value for key, value in state.items()}
        expected = self.state_dict()
        if set(incoming) != set(expected):
            raise ValueError('all seven-refinement state keys must match; nine-slot extension adds no parameters')
        for name, value in incoming.items():
            if not isinstance(value, Tensor) or value.shape != expected[name].shape:
                raise ValueError(f'seven-refinement tensor shape differs: {name}')
        self.load_state_dict(incoming, strict=True)
        return {
            'source_family': spec['family'], 'source_selected_update': checkpoint.get('selected_update'),
            'source_selected_weights': checkpoint.get('selected_weights'),
            'source_parameter_count': checkpoint.get('parameter_count'),
            'parameter_count': self.parameter_count, 'extra_parameter_count': self.extra_parameter_count,
            'mapping': 'remove exactly outer net. prefix; all remaining state keys strict',
            'additional_parameters': 0, 'all_seven_parameters_preserved': True,
            'existing_refinement_heads_and_gains_not_reset': True,
        }

    def _fuse_recent(self, recent: Tensor, active: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return CNN feature, physical field and any-clear mask without stored request state."""
        coverage = recent[:, :, 2:3]
        total = coverage.sum(dim=1, keepdim=True)
        # No denominator floor can perturb a positive singleton coverage value.
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        physical_weights = coverage / denominator
        count = active.float().sum(dim=1)
        scene_weights = active.float() / count[:, None].clamp_min(1)
        cnn_weights = torch.where(total > 0, physical_weights, scene_weights)
        encoded = []
        for index in range(3):
            value = recent[:, index]
            cnn = torch.cat((value[:, :8], value[:, 8:9] * (3652.5 / 64)), dim=1)
            # Retaining the original B-image encoder call preserves its exact
            # computation when two appended recent observations are empty.
            features = self.recent_encoder(cnn)
            encoded.append(torch.where(active[:, index], features, torch.zeros_like(features)))
        stacked = torch.stack(encoded, dim=1)
        fused = torch.where(cnn_weights > 0, stacked * cnn_weights, torch.zeros_like(stacked)).sum(dim=1)
        temperature_pattern = 5 * recent[:, :, 1:2]
        field = torch.where(physical_weights > 0, temperature_pattern * physical_weights,
                            torch.zeros_like(temperature_pattern)).sum(dim=1)
        for index in range(3):
            only = active[:, index] & (count == 1)
            fused = torch.where(only, encoded[index], fused)
            original_field = torch.where(coverage[:, index] > 0, 5 * recent[:, index, 1:2],
                                         torch.zeros_like(recent[:, index, 1:2]))
            field = torch.where(only, original_field, field)
        return fused, field, total[:, 0] > 0

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, history: Tensor, emissivity: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError('fine must have shape [B,52,H,W]')
        b, _, h, w = fine.shape
        if history.shape != (b, 9, 9, h, w):
            raise ValueError('history must contain six original slots and exactly three registered recent slots')
        if support.shape != (b, 1, h, w) or context.shape != (b, 15) or emissivity.shape != (b, 4, h, w):
            raise ValueError('query support, Context15 or emissivity geometry differs')
        if h % 4 or w % 4 or coarse.shape != (b, 1, h // 4, w // 4):
            raise ValueError('coarse must use the query exact x4 parent grid')
        recent = history[:, 6:9].float()
        date_available = (recent[:, :, 2:3] > 0).flatten(3).any(dim=3)[..., None, None]
        available = date_available.any(dim=1).float()
        if self.training and self.modality_dropout:
            keep = torch.rand((b, 1, 1, 1), device=fine.device) >= self.modality_dropout
            available = available * keep.float()
        active = date_available & (available[:, None] > 0)
        recent = torch.where(active, recent, torch.zeros_like(recent))
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
        query = self.query_encoder(torch.cat((scaled_fine, mask, coarse_maps, emissivity.float()), dim=1))
        query = HistoricalInnovationNet._condition(query, self.query_context(context.float()))
        recent_features, recent_field, recent_clear = self._fuse_recent(recent, active)
        value = self.fuse(torch.cat((query, recent_features), dim=1))
        parent = HistoricalInnovationNet._condition(self.parent(value), self.parent_context(context.float()))
        value = self.merge(torch.cat((value, F.interpolate(parent, size=(h, w), mode='bilinear',
                                                          align_corners=False)), dim=1))
        gate = available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context, emissivity, history[:, :6]).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse).float()
        h_prior = support_project(recent_field * mask, zero_coarse, support)
        replacement_field = recent_clear * HistoricalInnovationNet.parent_center_all(prediction, support)
        p_prior = support_project(replacement_field, zero_coarse, support)
        learned = support_project(self.dense_proposal(value).float() * gate, zero_coarse, support)
        result = prediction + self.physical_gain.float() * h_prior - self.replacement_gain.float() * p_prior + learned
        return torch.where(mask > 0, result, torch.zeros_like(result))


__all__ = ['HistoricalRecentMultiRefinementNet']
