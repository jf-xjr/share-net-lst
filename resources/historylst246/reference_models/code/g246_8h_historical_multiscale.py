"""Identity-initialized multiscale encoding of each historical observation.

The original source_encoder layers, date fusion, physical priors, backbone and
projection retain their exact names and operations. Only source_encoder.4.* is
new. All dates share the spatial pyramid; no files or request tensors are kept
on construction or forward. Source chronology remains the cache's contract.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_historical_multisource import HistoricalMultiSourceNet
from g246_8h_network import _Block, _norm


class _Down(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
            _norm(outgoing), nn.SiLU(), _Block(outgoing), _Block(outgoing),
        )


class _Merge(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__(
            nn.Conv2d(incoming, outgoing, 1, bias=False),
            _norm(outgoing), nn.SiLU(), _Block(outgoing),
        )


class _SpatialPyramidResidual(nn.Module):
    """Shared 160->80->40->20 pyramid, ending in an exactly zero 1x1 head."""

    def __init__(self, width: int) -> None:
        super().__init__()
        c0, c1, c2, c3 = width, 2 * width, 3 * width, 4 * width
        self.down1 = _Down(c0, c1)
        self.down2 = _Down(c1, c2)
        self.down3 = _Down(c2, c3)
        self.merge2 = _Merge(c2 + c3, c2)
        self.merge1 = _Merge(c1 + c2, c1)
        self.merge0 = _Merge(c0 + c1, c0)
        self.output = nn.Conv2d(c0, c0, 1)
        self.reset_identity_output()

    def reset_identity_output(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _up(value: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, value: Tensor) -> Tensor:
        skip1 = self.down1(value)
        skip2 = self.down2(skip1)
        deep = self.down3(skip2)
        merged2 = self.merge2(torch.cat((skip2, self._up(deep, skip2)), dim=1))
        merged1 = self.merge1(torch.cat((skip1, self._up(merged2, skip1)), dim=1))
        merged0 = self.merge0(torch.cat((value, self._up(merged1, value)), dim=1))
        return value + self.output(merged0)


class HistoricalMultiscaleInnovationNet(HistoricalMultiSourceNet):
    """Forward(fine, coarse, support, context, history, *backbone_extra_inputs)."""

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 source_count: int = 6, initial_gain: float = .28,
                 initial_replacement_gain: float = .22,
                 modality_dropout: float = .25) -> None:
        super().__init__(backbone, width=width, source_count=source_count,
                         initial_gain=initial_gain,
                         initial_replacement_gain=initial_replacement_gain,
                         modality_dropout=modality_dropout)
        if len(self.source_encoder) != 4:
            raise ValueError('historical source encoder contract changed; original four layers required')
        self.source_encoder.add_module('4', _SpatialPyramidResidual(self.width))
        if self.parameter_count >= 20_000_000:
            raise ValueError('multiscale historical deployment exceeds twenty million parameters')

    @property
    def multiscale_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.source_encoder[4].parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            **super().model_config,
            'schema_version': 'g246-8h-historical-multiscale-innovation-network-v1',
            'class_name': 'HistoricalMultiscaleInnovationNet',
            'source_encoder_extension': 'shared identity-initialized spatial pyramid after all original per-date encoder layers and before date fusion',
            'multiscale_channels': [self.width, 2 * self.width, 3 * self.width, 4 * self.width],
            'multiscale_downsample_factors': [1, 2, 4, 8],
            'multiscale_down_blocks_per_scale': 2,
            'multiscale_decoder_blocks_per_scale': 1,
            'multiscale_output': 'unbounded residual with zero-initialized 1x1 weight and bias',
            'multiscale_parameter_count': self.multiscale_parameter_count,
            'parameter_count': self.parameter_count,
            'extra_parameter_count': self.extra_parameter_count,
            'only_added_state_prefix': 'source_encoder.4.',
            'original_source_encoder_layers_unchanged': True,
            'three_source_reference': 'complete original function at zero new residual; shared extension also supports explicitly registered six sources',
            'warmstart_u0': 'complete original three/six-source historical innovation function; prior and trained CNN heads retained',
            'warmstart': 'all existing full-wrapper weights strict; only source_encoder.4.* may be new',
            'physical_prior_and_replacement_unchanged': True,
            'new_observations': False,
        }

    def load_historical_deploy(self, checkpoint: Mapping) -> dict[str, object]:
        """Migrate a complete original three/six-source deployment strictly.

        The caller owns immutable checkpoint bytes and their hash. A transfer
        resets only the new pyramid's final head to zero; an already-trained
        multiscale deployment is reconstructed by direct strict state loading.
        Source count is an explicit input contract and does not alter weights.
        """
        if checkpoint.get('schema') != 'g246-8h-deploy-v1' or checkpoint.get('locked_test_opened') is not False:
            raise ValueError('a complete locked-test-closed historical deployment is required')
        spec = checkpoint.get('model_spec', {})
        if not isinstance(spec, Mapping) or spec.get('family') not in (
            'historical_innovation_r6a', 'historical_innovation_emissivity_r6a',
            'historical_innovation_emissivity_r6a_six',
        ):
            raise ValueError('warmstart requires an original three/six-source historical innovation deployment')
        state = checkpoint.get('state_dict')
        if not isinstance(state, Mapping) or not state or not all(str(key).startswith('net.') for key in state):
            raise ValueError('warmstart requires complete net.* wrapper state')
        incoming = {str(key)[4:]: value for key, value in state.items()}
        expected = self.state_dict()
        added = sorted(key for key in expected if key.startswith('source_encoder.4.'))
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        if not added or missing != added or unexpected:
            raise ValueError(f'only source_encoder.4.* may be new; missing={missing}, unexpected={unexpected}')
        for name, value in incoming.items():
            if not isinstance(value, Tensor) or value.shape != expected[name].shape:
                raise ValueError(f'historical warmstart tensor shape differs: {name}')
        self.source_encoder[4].reset_identity_output()
        expected = self.state_dict()
        incoming.update({name: expected[name] for name in added})
        self.load_state_dict(incoming, strict=True)
        return {
            'source_family': spec['family'],
            'source_selected_update': checkpoint.get('selected_update'),
            'source_selected_weights': checkpoint.get('selected_weights'),
            'source_parameter_count': checkpoint.get('parameter_count'),
            'source_count': self.source_count,
            'parameter_count': self.parameter_count,
            'multiscale_parameter_count': self.multiscale_parameter_count,
            'wrapper_prefix_removed': 'net.',
            'only_added_state_prefix': 'source_encoder.4.',
            'added_state_keys': added,
            'new_output_head_exact_zero': True,
            'all_remaining_keys_strict': True,
        }


__all__ = ['HistoricalMultiscaleInnovationNet']
