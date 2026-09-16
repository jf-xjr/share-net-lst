"""U-TAE controls with explicit input dropout; unchanged parameter names."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'resources/historylst246'))
sys.path.insert(0, str(ROOT / 'research/strong_history_20260911'))
from historylst.model import HistoryUTAE, project
from candidates.network_review import HistoryWideUTAE


def _forward(self, fine, coarse, support, context, emissivity, history):
    b, _, h, w = fine.shape
    count = history.shape[1]
    base = fine[:, :1].float()
    fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), 1)
    valid_coarse = torch.isfinite(coarse)
    cm = F.interpolate(torch.cat((torch.nan_to_num((coarse.float() - 300) / 20),
                                  valid_coarse.float()), 1), size=(h, w), mode='nearest')
    hist, em = history.float().clone(), emissivity.float()
    if self.training:
        # Draw in both arms, including p=0, to keep later random streams paired.
        hist = hist * (torch.rand(b, 1, 1, 1, 1, device=hist.device) >= self.history_dropout)
        em = em * (torch.rand(b, 1, 1, 1, device=em.device) >= self.emissivity_dropout)
    thermal, emis = hist[:, :, 2:3] > 0, hist[:, :, 5:6] > 0
    hist[:, :, :2] = hist[:, :, :2] * thermal
    hist[:, :, 3:4] = hist[:, :, 3:4] * thermal
    hist[:, :, 4:5] = hist[:, :, 4:5] * emis
    curr = self.current(torch.cat((fine, em, support.float(), cm,
                                  context.float()[:, :, None, None].expand(-1, -1, h, w)), 1))
    channels = curr.shape[1]
    past = self.historical(hist.reshape(b * count, 9, h, w)).reshape(b, count, channels, h, w)
    x = torch.cat((curr[:, None], past), 1)
    visibility = torch.cat((torch.ones(b, 1, h, w, device=x.device),
                            thermal.squeeze(2).float()), 1)
    positions = torch.cat((hist.new_zeros(b, 1),
                           -hist[:, :, 8].flatten(2).amax(2) * 3652.5), 1)
    maps = [self.core.in_conv.smart_forward(x)]
    for block in self.core.down_blocks:
        maps.append(block.smart_forward(maps[-1]))
    lowmask = F.adaptive_max_pool2d(visibility, maps[-1].shape[-2:]) == 0
    out, att = self.core.temporal_encoder(maps[-1], batch_positions=positions, pad_mask=lowmask)
    heads = att.shape[0]
    for i, up in enumerate(self.core.up_blocks):
        features = maps[-i - 2]
        shape = features.shape[-2:]
        a = F.interpolate(att.reshape(heads * b, count + 1, *att.shape[-2:]),
                          size=shape, mode='bilinear', align_corners=False)
        a = a.reshape(heads, b, count + 1, *shape)
        a = a * F.adaptive_max_pool2d(visibility, shape)[None]
        a = a / a.sum(2, keepdim=True).clamp_min(1e-6)
        grouped = torch.stack(features.chunk(heads, dim=2))
        skip = (a[:, :, :, None] * grouped).sum(2)
        out = up(out, torch.cat(list(skip), dim=1))
    return project(base + self.core.out_conv(out).float(), coarse, support)


class DropoutUTAE(HistoryUTAE):
    def __init__(self, history_dropout=.25, emissivity_dropout=.25):
        super().__init__()
        self.history_dropout, self.emissivity_dropout = history_dropout, emissivity_dropout

    forward = _forward


class DropoutWideUTAE(HistoryWideUTAE):
    def __init__(self, history_dropout=.25, emissivity_dropout=.25):
        super().__init__()
        self.history_dropout, self.emissivity_dropout = history_dropout, emissivity_dropout

    forward = _forward
