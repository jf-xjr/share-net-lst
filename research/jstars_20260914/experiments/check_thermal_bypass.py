"""Verify the exact intervention on a real training scene before training."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import json
import sys
import torch
from thermal_bypass_model import WithoutThermalBypass, QueryProductCompactHistoryNAF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'resources/historylst246'))
from historylst.data import Dataset, INPUTS

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
initial = []
for seed in (20260914, 20260915):
    torch.manual_seed(seed)
    full = QueryProductCompactHistoryNAF(history_dropout=0., emissivity_dropout=0.)
    torch.manual_seed(seed)
    cut = WithoutThermalBypass()
    assert all(torch.equal(v, cut.state_dict()[k]) for k, v in full.state_dict().items())
    initial.append({'seed': seed, 'all_initial_tensors_identical': True})

checkpoint = ROOT / 'research/near_neighbor_attribution_20260914/attribution/learned_20260914/best.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
full.load_state_dict(saved)
cut.load_state_dict(saved)
full = full.cuda().eval()
cut = cut.cuda().eval()
data = Dataset(ROOT / 'resources/historylst246', 'fit', labels=True)
batch = {k: torch.from_numpy(v).cuda() for k, v in data.batch([0]).items()}
inputs = {k: batch[k] for k in INPUTS}
features = {}
handles = []
for name, model in [('full', full), ('cut', cut)]:
    handles.append(model.fusion[0].register_forward_hook(
        lambda m, i, o, name=name: features.update({name: tuple(v.detach().clone() for v in o)})))
with torch.no_grad():
    p_full, p_cut = full(**inputs), cut(**inputs)
assert torch.equal(features['full'][0], features['cut'][0])
assert torch.count_nonzero(features['cut'][1]).item() == 0
assert torch.count_nonzero(features['full'][1]).item() > 0
assert not torch.equal(p_full, p_cut)
for handle in handles:
    handle.remove()
loss = ((p_cut - batch['target']) ** 2)[batch['formal'].bool()].mean() if p_cut.requires_grad else None
cut.zero_grad(set_to_none=True)
p_cut = cut(**inputs)
loss = ((p_cut - batch['target']) ** 2)[batch['formal'].bool()].mean()
loss.backward()
assert all(p.grad is None for p in cut.detail_skip.parameters())
assert all(p.grad is None for p in cut.thermal_gain.parameters())
active = {name: bool(any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                        for p in module.parameters()))
          for name, module in [('historical', cut.historical),
                               *[(f'fusion_{i}', m) for i, m in enumerate(cut.fusion)]]}
assert all(active.values())
with torch.no_grad():
    inputs['history'] = torch.zeros_like(inputs['history'])
    assert torch.equal(full(**inputs), cut(**inputs))
out = HERE / 'thermal_bypass'
out.mkdir(exist_ok=True)
record = {'initialization_checks': initial,
          'first_scale_historical_feature_and_moment_injection_unchanged': True,
          'returned_full_resolution_summary_zero': True,
          'removed_connections': ['summary_to_decoder_detail_skip', 'summary_to_gated_temperature_residual'],
          'all_scales_keep_feature_fusion_and_moment_injection': True,
          'history_encoder_and_all_fusion_gradients_active': active,
          'no_history_forward_equivalence': True,
          'inactive_parameters': sum(p.numel() for p in cut.parameters() if not p.requires_grad),
          'nonzero_bypass_effect_on_training_scene_K': float((p_full - p_cut).detach().abs().max()),
          'split': 'fit', 'scene_index': 0}
(out / 'intervention_check.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
