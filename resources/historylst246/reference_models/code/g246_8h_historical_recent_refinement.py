"""A separate recent-observation innovation stream around a complete six-source model.

The first six historical slots go only to the existing six-source wrapper. The
seventh slot is encoded separately, so its physical contribution is not diluted
by averaging it with six older slots. All new outputs begin at exactly zero.
No constructor reads files, and no forward input is cached on the module.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_historical_innovation_network import HistoricalInnovationNet
from g246_8h_historical_network import HISTORY_CHANNELS
from g246_8h_network import _Block, _norm, support_project


class HistoricalRecentRefinementNet(nn.Module):
    """Forward(base4, history7, emissivity), using a complete six-source wrapper.

    ``backbone`` must expose the public HistoricalModel six-source signature:
    ``backbone(fine, coarse, support, context, emissivity, history6)``.
    It is evaluated once. Default recent dropout is zero and draws no new RNG;
    the existing backbone's own dropout calls and trainable parameters remain.
    """

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 modality_dropout: float = 0.0) -> None:
        super().__init__()
        if isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError('recent refinement width must be a positive multiple of eight')
        if not 0 <= modality_dropout <= 1:
            raise ValueError('recent modality dropout must be in [0,1]')
        if getattr(backbone, 'source_count', None) != 6 or getattr(backbone, 'emissivity', None) is not True:
            raise ValueError('backbone must be a complete six-source emissivity HistoricalModel wrapper')
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.width = int(width)
        self.source_count = 7
        self.modality_dropout = float(modality_dropout)
        self.initial_gain = 0.0
        self.initial_replacement_gain = 0.0
        self.physical_gain = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.replacement_gain = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.recent_encoder = nn.Sequential(
            nn.Conv2d(9, width, 3, padding=1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        # Full Fine52, support, three coarse maps, and the existing emissivity4.
        self.query_encoder = nn.Sequential(
            nn.Conv2d(60, width, 1, bias=False), _norm(width), nn.SiLU(), _Block(width),
        )
        self.query_context = nn.Linear(15, 2 * width)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, bias=False), _norm(width), nn.SiLU(), _Block(width),
        )
        self.parent = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=4, bias=False),
            _norm(2 * width), nn.SiLU(), _Block(2 * width), _Block(2 * width),
        )
        self.parent_context = nn.Linear(15, 4 * width)
        for layer in (self.query_context, self.parent_context):
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)
        self.merge = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, bias=False), _norm(width), nn.SiLU(),
            _Block(width), _Block(width),
        )
        self.fine_injection = nn.Conv2d(width, 51, 1)
        self.dense_proposal = nn.Conv2d(width, 1, 3, padding=1)
        self._reset_new_outputs()
        if self.extra_parameter_count >= 300_000 or self.parameter_count >= 20_000_000:
            raise ValueError('recent refinement exceeds its new-branch or complete deployment parameter budget')

    def _reset_new_outputs(self) -> None:
        nn.init.zeros_(self.physical_gain)
        nn.init.zeros_(self.replacement_gain)
        for head in (self.fine_injection, self.dense_proposal):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def extra_parameter_count(self) -> int:
        return self.parameter_count - sum(parameter.numel() for parameter in self.backbone.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            'schema_version': 'g246-8h-historical-recent-refinement-network-v1',
            'class_name': 'HistoricalRecentRefinementNet', 'width': self.width,
            'source_count': 7, 'history_shape': ['B', 7, 9, 'H', 'W'],
            'history_channels': list(HISTORY_CHANNELS),
            'backbone_history_slots': list(range(6)), 'recent_history_slot': 6,
            'backbone_forward': 'base4,emissivity,history[:,:6]; complete six-source wrapper evaluated once',
            'recent_spatial_encoder': 'single-observation local and parent spatial fusion; no unused single-date softmax/gate',
            'recent_age_input_scaling': 'CNN only: history channel8 * (3652.5/64), equivalently positive age_days/64; fixed physical scale',
            'initial_gain': self.initial_gain, 'initial_replacement_gain': self.initial_replacement_gain,
            'modality_dropout': self.modality_dropout,
            'default_recent_dropout': 'zero: no additional RNG draw; existing complete six-source stochastic operations preserved',
            'physical_prior': 'Q[5*recent_channel1 at pixels with positive recent historical clear coverage, else zero]',
            'replacement_prior': 'Q[any_recent_clear_pixel * center_all_actual_support_parents(current_six_source_prediction)]',
            'parent_centering': 'O and U for feature construction; only observed O constrained by Q',
            'output': 'six_source_prediction + physical_gain*Qrecent - replacement_gain*Qpattern + learned_Q',
            'u0': 'exact complete six-source function even with a nonzero recent slot; all new output heads and gains are zero',
            'missing_recent': 'exact current six-source fallback on support; unsupported pixels zero',
            'initial_gain_basis': 'zero, no new Fit or Validation coefficient fit',
            'recent_interpretation': 'earlier thermal spatial observation, not an hours-scale dynamical initial state',
            'source_admissibility': 'separately registered recent cache contract; acquisition chronology and alias exclusions are outside this network',
            'backbone_trainable': True, 'residual_amplitude_cap': None,
            'target_or_scoring_mask_inputs': False, 'teacher_dependency': False,
            'geographic_inputs': False, 'parameter_count': self.parameter_count,
            'extra_parameter_count': self.extra_parameter_count,
            'backbone_model_config': self.backbone.model_config,
        }

    def load_six_deploy(self, checkpoint: Mapping) -> dict[str, object]:
        """Strictly load the full old wrapper; reset only the new stream's outputs."""
        if checkpoint.get('schema') != 'g246-8h-deploy-v1' or checkpoint.get('locked_test_opened') is not False:
            raise ValueError('a complete locked-test-closed six-source deployment is required')
        spec = checkpoint.get('model_spec', {})
        if not isinstance(spec, Mapping) or spec.get('family') not in (
            'historical_innovation_emissivity_r6a_six',
            'historical_multiscale_innovation_emissivity_r6a_six',
        ):
            raise ValueError('warmstart requires a registered six-source emissivity deployment')
        state = checkpoint.get('state_dict')
        if not isinstance(state, Mapping) or not state or not all(str(key).startswith('net.') for key in state):
            raise ValueError('warmstart requires complete six-source net.* wrapper state')
        expected = self.backbone.state_dict()
        if set(state) != set(expected):
            raise ValueError('six-source backbone keys must match completely and strictly')
        for name, value in state.items():
            if not isinstance(value, Tensor) or value.shape != expected[name].shape:
                raise ValueError(f'six-source backbone tensor shape differs: {name}')
        self.backbone.load_state_dict(state, strict=True)
        self._reset_new_outputs()
        return {
            'source_family': spec['family'], 'source_selected_update': checkpoint.get('selected_update'),
            'source_selected_weights': checkpoint.get('selected_weights'),
            'source_parameter_count': checkpoint.get('parameter_count'),
            'parameter_count': self.parameter_count, 'extra_parameter_count': self.extra_parameter_count,
            'mapping': 'complete source state_dict -> backbone state_dict with no prefix stripping',
            'all_backbone_keys_strict': True,
            'new_physical_gain': 0.0, 'new_replacement_gain': 0.0,
            'new_fine_injection_and_dense_heads_zero': True,
        }

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, history: Tensor, emissivity: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError('fine must have shape [B,52,H,W]')
        b, _, h, w = fine.shape
        if history.shape != (b, 7, 9, h, w):
            raise ValueError('history must contain six registered original slots plus one recent slot')
        if support.shape != (b, 1, h, w) or context.shape != (b, 15) or emissivity.shape != (b, 4, h, w):
            raise ValueError('query support, Context15 or emissivity geometry differs')
        if h % 4 or w % 4 or coarse.shape != (b, 1, h // 4, w // 4):
            raise ValueError('coarse must use the query exact x4 parent grid')
        recent = history[:, 6].float()
        available = (recent[:, 2:3] > 0).flatten(1).any(dim=1).float()[:, None, None, None]
        if self.training and self.modality_dropout:
            keep = torch.rand((b, 1, 1, 1), device=fine.device) >= self.modality_dropout
            available = available * keep.float()
        recent = torch.where(available > 0, recent, torch.zeros_like(recent))
        recent_cnn = torch.cat((recent[:, :8], recent[:, 8:9] * (3652.5 / 64)), dim=1)
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
        recent_features = self.recent_encoder(recent_cnn)
        recent_features = torch.where(available > 0, recent_features, torch.zeros_like(recent_features))
        value = self.fuse(torch.cat((query, recent_features), dim=1))
        parent = HistoricalInnovationNet._condition(self.parent(value), self.parent_context(context.float()))
        value = self.merge(torch.cat((value, F.interpolate(parent, size=(h, w), mode='bilinear',
                                                          align_corners=False)), dim=1))
        gate = available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context, emissivity, history[:, :6]).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse).float()
        recent_clear = recent[:, 2:3] > 0
        recent_field = torch.where(recent_clear, 5 * recent[:, 1:2], torch.zeros_like(recent[:, 1:2]))
        h_prior = support_project(recent_field * mask, zero_coarse, support)
        replacement_field = recent_clear * HistoricalInnovationNet.parent_center_all(prediction, support)
        p_prior = support_project(replacement_field, zero_coarse, support)
        learned = support_project(self.dense_proposal(value).float() * gate, zero_coarse, support)
        result = prediction + self.physical_gain.float() * h_prior - self.replacement_gain.float() * p_prior + learned
        return torch.where(mask > 0, result, torch.zeros_like(result))


__all__ = ['HistoricalRecentRefinementNet']
