"""Four identity-initialized fine-scale NAF blocks before both output heads.

NAFBlock is imported unchanged from the project's original implementation.
Upstream: megvii-research/NAFNet, commit 2b4af71ebe098a92a75910c233a3965a3e93ede4.
Copyright (c) 2022 megvii-model; original MIT notice and bundled BasicSR notices
remain in ../../sub04_20260911/naf_history/upstream/LICENSE.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

OLD = Path(__file__).resolve().parents[2] / 'sub04_20260911'
sys.path.insert(0, str(OLD))
from naf_history.model import HistoryNAFReconstructor, NAFBlock, project


class RefinedHistoryNAF(HistoryNAFReconstructor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refinement = nn.Sequential(*(NAFBlock(48) for _ in range(4)))

    def load_parent_state(self, state):
        """Load every original state key; only zero-initialized new keys may miss."""
        own = self.state_dict()
        extra = {name for name in own if name.startswith('refinement.')}
        expected = set(own) - extra
        if set(state) != expected:
            raise ValueError(f'Original state mismatch: missing={sorted(expected-set(state))}, unexpected={sorted(set(state)-expected)}')
        # Validate all tensors before mutating even one original parameter.
        for name in sorted(expected):
            if not isinstance(state[name], torch.Tensor) or state[name].shape != own[name].shape or state[name].dtype != own[name].dtype:
                raise ValueError(f'Original tensor shape/dtype mismatch: {name}')
        for block in self.refinement:
            if torch.count_nonzero(block.beta).item() or torch.count_nonzero(block.gamma).item():
                raise ValueError('Parent initialization requires untouched zero beta/gamma')
        incompatible = self.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != extra or incompatible.unexpected_keys:
            raise RuntimeError('Only the four refinement blocks may be newly initialized')
        return tuple(sorted(extra))

    def forward(self, fine, coarse, support, context, emissivity, history):
        b, _, h, w = fine.shape
        if fine.shape[1] != 52 or context.shape != (b, 15) or emissivity.shape != (b, 4, h, w):
            raise ValueError('Expected original Fine52, Context15 and Emissivity4 inputs')
        if history.ndim != 5 or history.shape[0] != b or history.shape[2:] != (9, h, w) or history.shape[1] < 1:
            raise ValueError('Expected at least one source with nine registered history fields')
        if support.shape != (b, 1, h, w) or coarse.shape != (b, 1, h // 4, w // 4) or h % 8 or w % 8:
            raise ValueError('Expected x4 coarse observations and spatial dimensions divisible by eight')
        count = history.shape[1]
        base = fine[:, :1].float()
        scaled_fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1)
        cm = F.interpolate(torch.cat((torch.nan_to_num((coarse.float() - 300) / 20),
                                      torch.isfinite(coarse).float()), dim=1), size=(h, w), mode='nearest')
        hist, em = history.float().clone(), emissivity.float()
        if self.training:
            # Always draw both masks, including p=0, to align controlled RNGs.
            keep_history = torch.rand(b, 1, 1, 1, 1, device=hist.device) >= self.history_dropout
            keep_emissivity = torch.rand(b, 1, 1, 1, device=em.device) >= self.emissivity_dropout
            hist = hist * keep_history
            em = em * keep_emissivity
        thermal, emis = hist[:, :, 2:3] > 0, hist[:, :, 5:6] > 0
        hist[:, :, :2] = torch.where(thermal, hist[:, :, :2], 0.)
        hist[:, :, 3:4] = torch.where(thermal, hist[:, :, 3:4], 0.)
        hist[:, :, 4:5] = torch.where(emis, hist[:, :, 4:5], 0.)
        current = self.current(torch.cat((scaled_fine, em, support.float(), cm,
                                          context.float()[:, :, None, None].expand(-1, -1, h, w)), dim=1))
        history_features = self.historical(hist.reshape(b * count, 9, h, w))
        skips, fine_summary = [], None
        for level in range(4):
            size = current.shape[-2:]
            metadata = self.metadata(hist, size)
            visible = (metadata[:, :, 0:1] > 0).reshape(b * count, 1, *size)
            history_features = self.history_blocks[level](history_features) * visible
            sequence = history_features.reshape(b, count, -1, *size)
            current, summary = self.fusion[level](current, sequence, metadata)
            if level == 0:
                fine_summary = summary
            if level < 3:
                current = self.encoders[level](current)
                skips.append(current)
                current = self.downs[level](current)
                history_features = self.history_downs[level](history_features)
        current = self.middle(current)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            current = decoder(up(current) + skip)
        current = current + self.detail_skip(fine_summary)
        current = self.refinement(current)
        # The anomaly is already masked and uses exactly the same post-dropout
        # sources as the neural path. No second backbone or fixed ensemble.
        gain = self.thermal_gain(torch.cat((current, fine_summary), dim=1)).float()
        residual = self.ending(current).float() + gain * (5 * fine_summary[:, :1])
        return project(base + residual, coarse, support)


def build_model(**kwargs):
    return RefinedHistoryNAF(**kwargs)
