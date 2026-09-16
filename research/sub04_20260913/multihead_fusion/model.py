"""Four source-attention heads over fixed-width history channel groups.

Only fusion.*.score.2 expands 1 -> 4 outputs. Every original history source,
coverage prior, query, metadata field, availability test and support projection
is retained. This is an untrained structural alternative, not a measured gain.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/sub04_20260911'))
from naf_history.model import HistoryNAFReconstructor, LocalHistoryFusion

HEADS = 4
EXPANDED_KEYS = tuple(f'fusion.{level}.score.2.{kind}' for level in range(4) for kind in ('weight','bias'))


class FourHeadLocalHistoryFusion(LocalHistoryFusion):
    """Each channel quarter learns its own spatial source distribution."""
    def __init__(self, query_width, history_width):
        if history_width % HEADS:
            raise ValueError('History width must split into four unchanged channel groups')
        super().__init__(query_width, history_width)
        self.score[-1] = nn.Conv2d(32, HEADS, 1)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, current, history_features, metadata):
        b, count, width, h, w = history_features.shape
        if width % HEADS or metadata.shape != (b,count,4,h,w):
            raise ValueError('Original metadata and four equal feature channel groups required')
        coverage, anomaly, quality, age = metadata.unbind(dim=2)
        visible = coverage > 0
        features = history_features * visible[:, :, None]
        query = current[:, None].expand(-1,count,-1,-1,-1)
        shared = self.score[1](self.score[0](torch.cat((query,features,metadata),dim=2).reshape(b*count,-1,h,w)))
        weights = []
        for head in range(HEADS):
            # Four independent rows use the original 32->1 operation. Keeping
            # this arithmetic layout also preserves copied initialization exactly
            # instead of silently changing a convolution reduction kernel.
            logits = F.conv2d(shared,self.score[-1].weight[head:head+1],self.score[-1].bias[head:head+1])
            logits = logits.reshape(b,count,h,w).float()
            logits = (logits+coverage.clamp_min(1e-8).log()).masked_fill(~visible,-1e4)
            distribution = logits.softmax(dim=1)*visible
            weights.append(distribution/distribution.sum(dim=1,keepdim=True).clamp_min(1e-8))
        heads = torch.stack(weights,dim=2)
        # This is grouped pooling followed by concatenation, with the original
        # source reduction layout retained for exact initialization equivalence.
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


class FourHeadHistoryNAF(HistoryNAFReconstructor):
    def __init__(self, history_dropout=.25, emissivity_dropout=.25):
        super().__init__(history_dropout=history_dropout,emissivity_dropout=emissivity_dropout)
        self.fusion = nn.ModuleList(FourHeadLocalHistoryFusion(q,h) for q,h in
            zip((48,96,192,384),(16,32,64,128)))


def expand_single_head_state_dict(state, target):
    """Strict conversion: only the eight named tensors may change shape/value."""
    expected = target.state_dict()
    if set(state) != set(expected):
        raise ValueError('State keys differ beyond the explicitly allowed last-score expansion')
    result = OrderedDict()
    for name, value in state.items():
        if name in EXPANDED_KEYS:
            wanted = (1,32,1,1) if name.endswith('weight') else (1,)
            if tuple(value.shape) != wanted or value.dtype != expected[name].dtype:
                raise ValueError('Only an original single-head last-score tensor may be expanded: '+name)
            changed = value.repeat((HEADS,)+(1,)*(value.ndim-1))
            if changed.shape != expected[name].shape:
                raise ValueError('Unexpected four-head last-score shape: '+name)
            for head in range(HEADS):
                if not torch.equal(changed[head:head+1],value):
                    raise ValueError('Score head was not an exact copy')
            result[name] = changed
        else:
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError('Unapproved state conversion: '+name)
            result[name] = value.clone()
    target.load_state_dict(result,strict=True)
    loaded = target.state_dict()
    if any(not torch.equal(loaded[k],v) for k,v in result.items()):
        raise ValueError('Strictly loaded state differs from declared conversion')
    return result


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def initialize_from_single_checkpoint(path, expected_sha256, *, history_dropout=.25, emissivity_dropout=.25):
    if sha(path) != expected_sha256:
        raise ValueError('Initialization must match the explicitly specified single-head checkpoint')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['config']['architecture']!='naf_history' or checkpoint.get('smoke_only',False):
        raise ValueError('Actual NAF history source checkpoint required')
    model=FourHeadHistoryNAF(history_dropout=history_dropout,emissivity_dropout=emissivity_dropout)
    state=expand_single_head_state_dict(checkpoint['state_dict'],model)
    return model,dict(source_checkpoint=str(Path(path).resolve()),source_checkpoint_sha256=expected_sha256,
        allowed_expanded_keys=list(EXPANDED_KEYS),heads=HEADS,history_widths=[16,32,64,128],
        source_count=9,source_weights=checkpoint.get('weights'),source_step=checkpoint.get('step'),
        all_other_state_tensors_exactly_preserved=True,strict_state_load=True,
        inference_networks=1,initialization_only=True,trained=False,scientific_goal_complete=False),state


def build_model(**kwargs): return FourHeadHistoryNAF(**kwargs)
