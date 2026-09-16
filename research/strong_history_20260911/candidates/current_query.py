"""Current-conditioned, multiscale temporal cross-attention for HistoryLST246.

This is a task adaptation of standard query/key/value attention, not a claimed
reproduction of a published LST architecture. U-TAE spatial blocks are reused
from the pinned MIT source already bundled with the resource. Add that resource
root to sys.path, then instantiate HistoryCrossAttention() or build_model().
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from historylst.model import project, stem
from historylst.third_party.utae.utae import UTAE


class CurrentQueryFusion(nn.Module):
    """At each pixel, the current feature queries locally visible time tokens.

    Current is also a visible key/value, so an empty history has a well-defined
    fallback. Dates affect keys, while values retain the image representation.
    Attention is recomputed at each spatial scale rather than interpolated from
    the coarsest feature map. The residual path preserves current predictors.
    """

    def __init__(self, channels, heads=8, key_dim=8, dropout=.1):
        super().__init__()
        if channels % heads:
            raise ValueError('Channels must be divisible by attention heads')
        self.channels = channels
        self.heads = heads
        self.key_dim = key_dim
        self.norm = nn.GroupNorm(4, channels)
        self.query = nn.Conv2d(channels, heads * key_dim, 1)
        self.key = nn.Conv2d(channels, heads * key_dim, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout2d(dropout)
        self.ffn = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.Conv2d(channels, 2 * channels, 1), nn.GELU(),
            nn.Dropout2d(dropout), nn.Conv2d(2 * channels, channels, 1),
        )
        # The relative acquisition day is available at inference for every token.
        denom = torch.pow(1000., 2 * (torch.arange(key_dim).float() // 2) / key_dim)
        self.register_buffer('date_denom', denom, persistent=False)
        self.date_scale = nn.Parameter(torch.ones(heads, 1))

    def forward(self, sequence, positions, visible):
        b, t, c, h, w = sequence.shape
        n, k = self.heads, self.key_dim
        norm = self.norm(sequence.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)
        query = self.query(norm[:, 0]).reshape(b, n, k, h, w)
        key = self.key(norm.reshape(b * t, c, h, w)).reshape(b, t, n, k, h, w)
        date = positions[..., None] / self.date_denom
        date = torch.stack((date[..., 0::2].sin(), date[..., 1::2].cos()), -1).flatten(-2)
        key = key + (date[:, :, None] * self.date_scale[None, None])[:, :, :, :, None, None]
        logits = torch.einsum('bnkhw,btnkhw->bnthw', query, key) / math.sqrt(k)
        logits = logits.float().masked_fill(~visible[:, None].bool(), -1e4)
        weights = self.attention_dropout(logits.softmax(dim=2)).to(sequence.dtype)
        value = self.value(norm.reshape(b * t, c, h, w)).reshape(b, t, n, c // n, h, w)
        history_context = torch.einsum('bnthw,btnchw->bnchw', weights, value).reshape(b, c, h, w)
        out = sequence[:, 0] + self.output_dropout(self.output(history_context))
        return out + self.ffn(out)


class HistoryCrossAttention(nn.Module):
    """One network with the same six inputs and exact support projection as U-TAE-LST."""

    def __init__(self):
        super().__init__()
        self.current = stem(74)
        self.historical = stem(9)
        widths = [32, 64, 128, 256]
        self.core = UTAE(
            input_dim=32, encoder_widths=widths, decoder_widths=widths,
            out_conv=[32, 1], n_head=8, d_model=256, d_k=4, pad_value=None,
        )
        # Remove the replaced bottleneck L-TAE from the parameter/state count.
        self.core.temporal_encoder = nn.Identity()
        self.fusion = nn.ModuleList(
            CurrentQueryFusion(width, heads=8, key_dim=key)
            for width, key in zip(widths, [4, 8, 16, 16])
        )
        self.core.out_conv = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.core.out_conv.weight)
        nn.init.zeros_(self.core.out_conv.bias)

    def forward(self, fine, coarse, support, context, emissivity, history):
        b, _, h, w = fine.shape
        count = history.shape[1]
        base = fine[:, :1].float()
        fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), 1)
        cvalid = torch.isfinite(coarse)
        cm = F.interpolate(torch.cat((torch.nan_to_num((coarse.float() - 300) / 20), cvalid.float()), 1),
                           size=(h, w), mode='nearest')
        hist = history.float().clone()
        em = emissivity.float()
        if self.training:
            hist = hist * (torch.rand(b, 1, 1, 1, 1, device=hist.device) >= .25)
            em = em * (torch.rand(b, 1, 1, 1, device=em.device) >= .25)
        thermal = hist[:, :, 2:3] > 0
        emis = hist[:, :, 5:6] > 0
        hist[:, :, :2] = torch.where(thermal, hist[:, :, :2], 0.)
        hist[:, :, 3:4] = torch.where(thermal, hist[:, :, 3:4], 0.)
        hist[:, :, 4:5] = torch.where(emis, hist[:, :, 4:5], 0.)
        curr = self.current(torch.cat((fine, em, support.float(), cm,
                                      context.float()[:, :, None, None].expand(-1, -1, h, w)), 1))
        past = self.historical(hist.reshape(b * count, 9, h, w)).reshape(b, count, 32, h, w)
        sequence = torch.cat((curr[:, None], past), 1)
        visibility = torch.cat((torch.ones(b, 1, h, w, device=sequence.device),
                                thermal.squeeze(2).float()), 1)
        positions = torch.cat((hist.new_zeros(b, 1), -hist[:, :, 8].flatten(2).amax(2) * 3652.5), 1)
        maps = [self.core.in_conv.smart_forward(sequence)]
        for block in self.core.down_blocks:
            maps.append(block.smart_forward(maps[-1]))
        fused = [block(features, positions, F.adaptive_max_pool2d(visibility, features.shape[-2:]) > 0)
                 for block, features in zip(self.fusion, maps)]
        out = fused[-1]
        for i, up in enumerate(self.core.up_blocks):
            out = up(out, fused[-i - 2])
        residual = self.core.out_conv(out).float()
        return project(base + residual, coarse, support)


def build_model():
    return HistoryCrossAttention()
