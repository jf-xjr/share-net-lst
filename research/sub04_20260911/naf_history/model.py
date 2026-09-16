"""Spatial history reconstruction using NAFNet blocks and local source fusion.

NAFBlock adapts the official ECCV 2022 NAFNet implementation, copyright (c)
2022 megvii-model. The complete upstream license/source/commit is in upstream/.
The history pyramid, local coverage-weighted fusion and thermal skip are task
adaptations, not claims that the original NAFNet handles thermal time series.
"""
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

PACKAGE = Path(__file__).resolve().parents[3] / 'resources/historylst246'
sys.path.insert(0, str(PACKAGE))
from historylst.model import project


class LayerNorm2d(nn.Module):
    """Channel-only LayerNorm; FP32 reductions also under mixed precision."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Channels-last LayerNorm has the same axes as upstream LayerNorm2d.
        value = F.layer_norm(x.float().permute(0, 2, 3, 1), (x.shape[1],),
                            self.weight.float(), self.bias.float(), self.eps)
        return value.permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """Official NAFBlock equations; dropout is zero as in default NAFNet."""
    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        return y + self.conv5(self.sg(self.conv4(self.norm2(y)))) * self.gamma


class LocalHistoryFusion(nn.Module):
    """Per-pixel scalar temporal weights without normalizing history values.

    The source prior is actual thermal coverage. A learned local convolution
    modifies its log-weight using current/history features and QA/date fields.
    Every value, moment and feature injection is zero for an empty source pool.
    """
    def __init__(self, query_width, history_width):
        super().__init__()
        self.score = nn.Sequential(nn.Conv2d(query_width + history_width + 4, 32, 1),
                                   nn.SiLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.inject = nn.Conv2d(history_width + 3, query_width, 1)

    def forward(self, current, history_features, metadata):
        b, count, width, h, w = history_features.shape
        coverage, anomaly, quality, age = metadata.unbind(dim=2)
        visible = coverage > 0
        features = history_features * visible[:, :, None]
        query = current[:, None].expand(-1, count, -1, -1, -1)
        # Metadata are normalized to the input packet's scales.
        logits = self.score(torch.cat((query, features, metadata), dim=2).reshape(
            b * count, -1, h, w)).reshape(b, count, h, w).float()
        logits = (logits + coverage.clamp_min(1e-8).log()).masked_fill(~visible, -1e4)
        weights = logits.softmax(dim=1) * visible
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (features * weights[:, :, None]).sum(dim=1)
        mean = (anomaly * weights).sum(dim=1, keepdim=True)
        second = (anomaly.square() * weights).sum(dim=1, keepdim=True)
        dispersion = (second - mean.square()).clamp_min(1e-8).sqrt()
        available = visible.any(dim=1, keepdim=True)
        summary = torch.cat((mean, dispersion, coverage.sum(dim=1, keepdim=True) / count), dim=1)
        summary = summary * available
        injected = self.inject(torch.cat((pooled, summary), dim=1)) * available
        return current + injected, summary


class HistoryNAFReconstructor(nn.Module):
    """Same six inputs and support projection; one spatial reconstruction net.

    Unlike HistoryCrossAttention, there is no shared current/history U-TAE
    encoder, QK dot-product attention or temporal normalization. A small
    separate history pyramid supplies unnormalized values to every NAF scale.
    Fine-resolution thermal moments also bypass the downsampling path.
    """
    def __init__(self, history_dropout=.25, emissivity_dropout=.25):
        super().__init__()
        for name, value in (('history_dropout', history_dropout), ('emissivity_dropout', emissivity_dropout)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        self.history_dropout = float(history_dropout)
        self.emissivity_dropout = float(emissivity_dropout)
        widths = (48, 96, 192, 384)
        history_widths = (16, 32, 64, 128)
        self.current = nn.Conv2d(74, widths[0], 3, padding=1)
        self.historical = nn.Conv2d(9, history_widths[0], 3, padding=1)
        self.history_blocks = nn.ModuleList(NAFBlock(width) for width in history_widths)
        self.history_downs = nn.ModuleList(nn.Conv2d(a, b, 2, stride=2)
                                          for a, b in zip(history_widths, history_widths[1:]))
        self.fusion = nn.ModuleList(LocalHistoryFusion(c, h) for c, h in zip(widths, history_widths))
        self.encoders = nn.ModuleList(nn.Sequential(*(NAFBlock(width) for _ in range(count)))
                                     for width, count in zip(widths, (2, 2, 4)))
        self.downs = nn.ModuleList(nn.Conv2d(a, b, 2, stride=2) for a, b in zip(widths, widths[1:]))
        self.middle = nn.Sequential(*(NAFBlock(widths[-1]) for _ in range(6)))
        self.ups = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width * 2, 1, bias=False), nn.PixelShuffle(2))
                                for width in reversed(widths[1:]))
        self.decoders = nn.ModuleList(nn.Sequential(NAFBlock(width), NAFBlock(width))
                                     for width in reversed(widths[:-1]))
        self.detail_skip = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.ending = nn.Conv2d(widths[0], 1, 3, padding=1)
        self.thermal_gain = nn.Conv2d(widths[0] + 3, 1, 1)
        for layer in (self.ending, self.thermal_gain):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def metadata(history, size):
        """Pooled within-source fields: coverage/anomaly/QA/age, no new source."""
        b, count, _, height, width = history.shape
        coverage = history[:, :, 2:3]
        raw = torch.cat((history[:, :, 1:2], history[:, :, 3:4], history[:, :, 8:9]), dim=2)
        cov = F.adaptive_avg_pool2d(coverage.reshape(b * count, 1, height, width), size)
        fields = F.adaptive_avg_pool2d((raw * coverage).reshape(b * count, 3, height, width), size)
        fields = torch.where(cov > 0, fields / cov.clamp_min(1e-8), 0.)
        return torch.cat((cov, fields), dim=1).reshape(b, count, 4, *size)

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
        # The anomaly is already masked and uses exactly the same post-dropout
        # sources as the neural path. No second backbone or fixed ensemble.
        gain = self.thermal_gain(torch.cat((current, fine_summary), dim=1)).float()
        residual = self.ending(current).float() + gain * (5 * fine_summary[:, :1])
        return project(base + residual, coarse, support)


def build_model(**kwargs):
    return HistoryNAFReconstructor(**kwargs)
