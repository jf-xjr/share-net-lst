"""Zero-initialized query/history dot products augment the original four logits."""
from collections.abc import Mapping
from pathlib import Path
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F

NEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW))
from compact_recovery.model import CompactFourHeadHistoryNAF
from multihead_fusion.model import FourHeadLocalHistoryFusion, HEADS

PARENT_PARAMETERS = 5_911_525
ADDED_PARAMETERS = 65_520
PARAMETERS = 5_977_045
ADDED_KEYS = tuple(f'fusion.{level}.query_product.{kind}' for level in range(4) for kind in ('weight','bias'))


class QueryProductFusion(FourHeadLocalHistoryFusion):
    def __init__(self, original, query_width, history_width):
        nn.Module.__init__(self)
        if not isinstance(original, FourHeadLocalHistoryFusion) or history_width % HEADS:
            raise TypeError('An original four-head fusion with four channel groups is required')
        self.score, self.inject = original.score, original.inject
        self.query_product = nn.Conv2d(query_width, history_width, 1)
        nn.init.zeros_(self.query_product.weight)
        nn.init.zeros_(self.query_product.bias)

    def forward(self, current, history_features, metadata):
        b, count, width, h, w = history_features.shape
        if width % HEADS or metadata.shape != (b,count,4,h,w):
            raise ValueError('Original metadata and four channel groups are required')
        coverage, anomaly, quality, age = metadata.unbind(dim=2)
        visible = coverage > 0
        features = history_features * visible[:, :, None]
        query = current[:, None].expand(-1,count,-1,-1,-1)
        shared = self.score[1](self.score[0](torch.cat((query,features,metadata),dim=2).reshape(b*count,-1,h,w)))
        head_width = width // HEADS
        projected = self.query_product(current).float().reshape(b,HEADS,head_width,h,w)
        keys = features.float().reshape(b,count,HEADS,head_width,h,w)
        keys = keys / (keys.square().sum(dim=3,keepdim=True)+1e-6).sqrt()
        interaction = (keys*projected[:,None]).sum(dim=3)/math.sqrt(head_width)
        weights = []
        for head in range(HEADS):
            # Preserve each original score row, coverage prior and source softmax.
            logits = F.conv2d(shared,self.score[-1].weight[head:head+1],self.score[-1].bias[head:head+1])
            logits = logits.reshape(b,count,h,w).float() + interaction[:,:,head]
            logits = (logits+coverage.clamp_min(1e-8).log()).masked_fill(~visible,-1e4)
            distribution = logits.softmax(dim=1)*visible
            weights.append(distribution/distribution.sum(dim=1,keepdim=True).clamp_min(1e-8))
        heads = torch.stack(weights,dim=2)
        channel_weights = heads.repeat_interleave(width//HEADS,dim=2)
        pooled = (features*channel_weights).sum(dim=1)
        average = ((weights[0]+weights[1])+(weights[2]+weights[3]))*.25
        mean = (anomaly*average).sum(dim=1,keepdim=True)
        second = (anomaly.square()*average).sum(dim=1,keepdim=True)
        dispersion = (second-mean.square()).clamp_min(1e-8).sqrt()
        available = visible.any(dim=1,keepdim=True)
        summary = torch.cat((mean,dispersion,coverage.sum(dim=1,keepdim=True)/count),dim=1)*available
        injected = self.inject(torch.cat((pooled,summary),dim=1))*available
        return current+injected,summary


class QueryProductCompactHistoryNAF(CompactFourHeadHistoryNAF):
    def __init__(self, history_dropout=.25, emissivity_dropout=.25):
        super().__init__(history_dropout=history_dropout,emissivity_dropout=emissivity_dropout)
        self.fusion = nn.ModuleList(QueryProductFusion(layer,q,h) for layer,q,h in
            zip(self.fusion,(48,96,192,384),(16,32,64,128)))
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError('The registered below-six-million architecture changed')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False or not isinstance(state_dict,Mapping):
            raise ValueError('Selected checkpoints require complete strict tensor loading')
        expected = self.state_dict()
        if set(state_dict) != set(expected): raise ValueError('Selected checkpoint key set differs')
        for key,value in state_dict.items():
            if not isinstance(value,torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError('Selected checkpoint tensor schema differs: '+key)
        return super().load_state_dict(state_dict,strict=True,assign=False)


def load_compact_parent_state(model,state):
    if not isinstance(model,QueryProductCompactHistoryNAF) or not isinstance(state,Mapping):
        raise TypeError('Expected query-product model and actual compact parent state')
    expected = model.state_dict(); extra = set(ADDED_KEYS)
    if set(state) != set(expected)-extra: raise ValueError('Only the eight new query tensors may be absent')
    for key,value in state.items():
        if not isinstance(value,torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError('Original compact tensor schema differs: '+key)
        if value.is_floating_point() and not torch.isfinite(value).all(): raise ValueError('Nonfinite parent tensor: '+key)
    if any(torch.count_nonzero(expected[key]).item() for key in extra):
        raise ValueError('Parent initialization requires zero query-product parameters')
    loaded = nn.Module.load_state_dict(model,state,strict=False)
    if set(loaded.missing_keys) != extra or loaded.unexpected_keys: raise ValueError('Unexpected parent-loading difference')
    if any(not torch.equal(model.state_dict()[key],value) for key,value in state.items()):
        raise ValueError('Parent tensor changed during strict initialization')
    return dict(parent_parameters=PARENT_PARAMETERS,parameters=PARAMETERS,added_parameters=ADDED_PARAMETERS,
        initialized_keys=sorted(extra),retained_parent_tensors_bitwise_equal=True,initial_function_identity=True,
        heads=4,raw_sources=9,history_scales=4,inference_views=1,added_norm_parameters=0)


def build_model(**kwargs): return QueryProductCompactHistoryNAF(**kwargs)
