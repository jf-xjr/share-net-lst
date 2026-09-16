"""Inference adapters for the sealed candidate classes; no parameter changes."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / 'resources/historylst246'
CANDIDATES = HERE.parent / 'candidates'
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(CANDIDATES))

from historylst.model import HistoryUTAE, project
from flexible_utae import FlexibleUTAE
from current_query import HistoryCrossAttention
from network_review import HistoryWideUTAE


class FlexibleWideUTAE(HistoryWideUTAE):
    """Original wide forward with only the history/time lengths made dynamic.

    Its state_dict is identical to HistoryWideUTAE. The original temporal GN
    remains unchanged and therefore still couples time tokens: cropping is a
    complete usage strategy, not normalization-invariant inference optimization.
    """

    def forward(self, fine, coarse, support, context, emissivity, history):
        batch, _, height, width = fine.shape
        count = history.shape[1]
        base = fine[:, :1].float()
        fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), 1)
        cm = F.interpolate(torch.cat((torch.nan_to_num((coarse.float() - 300) / 20),
                                      torch.isfinite(coarse).float()), 1),
                           size=(height, width), mode='nearest')
        hist = history.float().clone()
        em = emissivity.float()
        if self.training:
            hist = hist * (torch.rand(batch, 1, 1, 1, 1, device=hist.device) >= .25)
            em = em * (torch.rand(batch, 1, 1, 1, device=em.device) >= .25)
        thermal = hist[:, :, 2:3] > 0
        emis = hist[:, :, 5:6] > 0
        hist[:, :, :2] = hist[:, :, :2] * thermal
        hist[:, :, 3:4] = hist[:, :, 3:4] * thermal
        hist[:, :, 4:5] = hist[:, :, 4:5] * emis
        curr = self.current(torch.cat((fine, em, support.float(), cm,
                                      context.float()[:, :, None, None].expand(-1, -1, height, width)), 1))
        past = self.historical(hist.reshape(batch * count, 9, height, width))
        past = past.reshape(batch, count, 48, height, width)
        tokens = torch.cat((curr[:, None], past), 1)
        visibility = torch.cat((torch.ones(batch, 1, height, width, device=tokens.device),
                                thermal.squeeze(2).float()), 1)
        positions = torch.cat((hist.new_zeros(batch, 1),
                               -hist[:, :, 8].flatten(2).amax(2) * 3652.5), 1)
        maps = [self.core.in_conv.smart_forward(tokens)]
        for block in self.core.down_blocks:
            maps.append(block.smart_forward(maps[-1]))
        lowmask = F.adaptive_max_pool2d(visibility, maps[-1].shape[-2:]) == 0
        out, attention = self.core.temporal_encoder(maps[-1], batch_positions=positions, pad_mask=lowmask)
        heads = attention.shape[0]
        for i, up in enumerate(self.core.up_blocks):
            features = maps[-i - 2]
            shape = features.shape[-2:]
            weights = F.interpolate(attention.reshape(heads * batch, count + 1, *attention.shape[-2:]),
                                    size=shape, mode='bilinear', align_corners=False)
            weights = weights.reshape(heads, batch, count + 1, *shape)
            visible = F.adaptive_max_pool2d(visibility, shape)
            weights = weights * visible[None]
            weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-6)
            grouped = torch.stack(features.chunk(heads, dim=2))
            skip = (weights[:, :, :, None] * grouped).sum(2)
            skip = torch.cat(list(skip), dim=1)
            out = up(out, skip)
        return project(base + self.core.out_conv(out).float(), coarse, support)


def model_classes(architecture):
    """Return (dynamic-length inference class, original trained class)."""
    return {
        'baseline': (FlexibleUTAE, HistoryUTAE),
        'current_query': (HistoryCrossAttention, HistoryCrossAttention),
        'wide': (FlexibleWideUTAE, HistoryWideUTAE),
    }[architecture]
