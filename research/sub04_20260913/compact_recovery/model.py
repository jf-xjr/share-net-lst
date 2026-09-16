"""Fixed block removal from the completed four-head NAF; 5,911,525 parameters.

All feature widths, nine historical sources, four spatial scales, readouts and
support projection are preserved. Removing trained blocks changes the function;
the model requires measured recovery, not an identity/equivalence claim.
"""
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import hashlib
import sys
import torch
from torch import nn

NEW = Path(__file__).resolve().parents[1]
if str(NEW) not in sys.path:
    sys.path.insert(0, str(NEW))
from multihead_fusion.model import FourHeadHistoryNAF

MIDDLE_KEEP = (0, 2, 4)
ENCODER2_KEEP = (0, 1, 3)
PARENT_PARAMETERS = 9_310_501
PARAMETERS = 5_911_525
VARIANT = 'compact_fourhead_middle024_encoder2_013'


class CompactFourHeadHistoryNAF(FourHeadHistoryNAF):
    def __init__(self, history_dropout=.25, emissivity_dropout=.25):
        super().__init__(history_dropout=history_dropout,
                         emissivity_dropout=emissivity_dropout)
        self.middle = nn.Sequential(*(self.middle[i] for i in MIDDLE_KEEP))
        self.encoders[2] = nn.Sequential(*(self.encoders[2][i] for i in ENCODER2_KEEP))
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError('The fixed compact architecture parameter count changed')


def parent_key(compact_key):
    """Explicit destination -> full-parent mapping for the two reindexed stacks."""
    parts = compact_key.split('.')
    if parts[0] == 'middle':
        index = int(parts[1])
        if not 0 <= index < len(MIDDLE_KEEP):
            raise ValueError('Unexpected compact middle key: ' + compact_key)
        parts[1] = str(MIDDLE_KEEP[index])
    elif parts[:2] == ['encoders', '2']:
        index = int(parts[2])
        if not 0 <= index < len(ENCODER2_KEEP):
            raise ValueError('Unexpected compact encoder key: ' + compact_key)
        parts[2] = str(ENCODER2_KEEP[index])
    return '.'.join(parts)


def load_parent_state(model, parent_state):
    """Load one complete four-head parent state via exact fixed key extraction.

    Returns a receipt. Accepts a state_dict, not a selected checkpoint envelope.
    A complete original key/shape/dtype schema is checked before any extraction.
    Compact selected checkpoints use ordinary load_state_dict(strict=True).
    """
    if not isinstance(model, CompactFourHeadHistoryNAF) or not isinstance(parent_state, Mapping):
        raise TypeError('Expected the fixed compact model and a full parent state_dict')
    # Preserve the caller's training RNG: constructing a schema must not consume it.
    with torch.random.fork_rng(devices=[]):
        original = FourHeadHistoryNAF(history_dropout=model.history_dropout,
                                      emissivity_dropout=model.emissivity_dropout)
    schema = original.state_dict()
    if sum(p.numel() for p in original.parameters()) != PARENT_PARAMETERS:
        raise ValueError('Original four-head architecture changed')
    if set(parent_state) != set(schema):
        missing = sorted(set(schema) - set(parent_state))
        extra = sorted(set(parent_state) - set(schema))
        raise ValueError(f'Full four-head parent key set differs: missing={missing}, extra={extra}')
    for name, value in parent_state.items():
        expected = schema[name]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('Parent tensor schema differs: ' + name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite parent tensor: ' + name)
    target = model.state_dict()
    mapping = {name: parent_key(name) for name in target}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Parent mapping is not injective')
    deleted = sorted(set(parent_state) - set(mapping.values()))
    allowed_prefixes = ('middle.1.', 'middle.3.', 'middle.5.', 'encoders.2.2.')
    expected_deleted = sorted(name for name in parent_state if name.startswith(allowed_prefixes))
    if deleted != expected_deleted or not deleted:
        raise ValueError('Deleted tensors differ from the four specified residual blocks')
    converted = OrderedDict()
    for name, expected in target.items():
        value = parent_state[mapping[name]]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('Retained tensor is incompatible: ' + name)
        converted[name] = value.detach().clone()
    model.load_state_dict(converted, strict=True)
    if any(not torch.equal(model.state_dict()[name], value) for name, value in converted.items()):
        raise ValueError('Strict compact load changed retained tensor values')
    return dict(variant=VARIANT,parameters=PARAMETERS,parent_parameters=PARENT_PARAMETERS,
                removed_parameters=PARENT_PARAMETERS-PARAMETERS,
                middle_keep=list(MIDDLE_KEEP),encoder2_keep=list(ENCODER2_KEEP),
                key_mapping=mapping,deleted_parent_keys=deleted,
                all_retained_tensors_bitwise_equal=True,strict_parent_schema=True,
                strict_compact_load=True,source_count=9,history_scales=4,
                inference_views=1,function_identity_claim=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize_from_parent_checkpoint(path, expected_sha256, *, history_dropout=.25,
                                      emissivity_dropout=.25):
    """Hash-bound optional convenience loader; returns (model, receipt)."""
    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise ValueError('Exact completed four-head parent checkpoint required')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('config', {}).get('architecture') != 'naf_history' or saved.get('smoke_only', False):
        raise ValueError('An actual NAF history checkpoint is required')
    model = CompactFourHeadHistoryNAF(history_dropout=history_dropout,
                                    emissivity_dropout=emissivity_dropout)
    receipt = load_parent_state(model, saved['state_dict'])
    receipt.update(parent_checkpoint=dict(path=str(path),sha256=expected_sha256),
                   parent_step=saved.get('step'),parent_weights=saved.get('weights'))
    return model, receipt


def build_model(**kwargs):
    return CompactFourHeadHistoryNAF(**kwargs)
