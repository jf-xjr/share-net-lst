"""Controlled source-weight ablation of the actual final 5.98 M architecture."""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/sub04_20260913'))
from compact_query_product.model import QueryProductCompactHistoryNAF


class CoverageFusion(nn.Module):
    def __init__(self, source):
        super().__init__()
        # Preserve identical tensor schemas and initialization. These score
        # tensors are explicitly inactive, not counted as trainable capacity.
        self.score = source.score.requires_grad_(False)
        self.query_product = source.query_product.requires_grad_(False)
        self.inject = source.inject

    def forward(self, current, history_features, metadata):
        coverage, anomaly, quality, age = metadata.unbind(dim=2)
        visible = coverage > 0
        weights = coverage.float()/coverage.float().sum(1,keepdim=True).clamp_min(1e-8)
        features = history_features * visible[:,:,None]
        pooled = (features * weights[:,:,None]).sum(1)
        mean = (anomaly*weights).sum(1,keepdim=True)
        second = (anomaly.square()*weights).sum(1,keepdim=True)
        dispersion = (second-mean.square()).clamp_min(1e-8).sqrt()
        available = visible.any(1,keepdim=True)
        summary = torch.cat((mean,dispersion,coverage.sum(1,keepdim=True)/coverage.shape[1]),1)*available
        return current+self.inject(torch.cat((pooled,summary),1))*available, summary


def construct(variant, history_dropout=0., emissivity_dropout=0.):
    model = QueryProductCompactHistoryNAF(history_dropout=history_dropout,
                                        emissivity_dropout=emissivity_dropout)
    if variant == 'coverage':
        model.fusion = nn.ModuleList(CoverageFusion(x) for x in model.fusion)
    elif variant != 'learned':
        raise ValueError(variant)
    return model
