"""A single 1.5x-width U-TAE-LST capacity control.

Mechanisms, input handling, dropout, normalization, attention and support
projection match resources/historylst246/historylst/model.py. Only the spatial
widths and temporal embedding width increase: 48/96/192/384, d_model=384.
This is a widened U-TAE-LST baseline, not a new named architecture.

Imports require resources/historylst246 on sys.path. No original resource file
is modified. The forward signature is identical to HistoryUTAE.
"""
import torch
from torch import nn
from torch.nn import functional as F
from historylst.model import project
from historylst.third_party.utae.utae import UTAE


def _stem(channels):
    return nn.Sequential(nn.Conv2d(channels, 48, 3, padding=1),
                         nn.GroupNorm(4, 48), nn.ReLU())


class HistoryWideUTAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.current = _stem(74)
        self.historical = _stem(9)
        self.core = UTAE(input_dim=48,
                         encoder_widths=[48, 96, 192, 384],
                         decoder_widths=[48, 96, 192, 384],
                         out_conv=[48, 1], n_head=8, d_model=384,
                         d_k=4, pad_value=None)
        self.core.out_conv = nn.Conv2d(48, 1, 1)
        nn.init.zeros_(self.core.out_conv.weight)
        nn.init.zeros_(self.core.out_conv.bias)

    def forward(self, fine, coarse, support, context, emissivity, history):
        batch, _, height, width = fine.shape
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
        past = self.historical(hist.reshape(batch * 9, 9, height, width))
        past = past.reshape(batch, 9, 48, height, width)
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
            weights = F.interpolate(attention.reshape(heads * batch, 10, *attention.shape[-2:]),
                                    size=shape, mode='bilinear', align_corners=False)
            weights = weights.reshape(heads, batch, 10, *shape)
            visible = F.adaptive_max_pool2d(visibility, shape)
            weights = weights * visible[None]
            weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-6)
            grouped = torch.stack(features.chunk(heads, dim=2))
            skip = (weights[:, :, :, None] * grouped).sum(2)
            skip = torch.cat(list(skip), dim=1)
            out = up(out, skip)
        return project(base + self.core.out_conv(out).float(), coarse, support)


def build_model():
    return HistoryWideUTAE()
