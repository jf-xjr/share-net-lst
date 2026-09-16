"""Select on validation, predict both fixed seeds, and compare paired cities."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / 'resources/historylst246'
OLD = ROOT / 'research/near_neighbor_attribution_20260914'
sys.path[:0] = [str(HERE), str(PACKAGE), str(OLD)]
from thermal_bypass_model import construct
from evaluate_comparisons import digest, dump, evaluate, mean_scores, bootstrap
spec = importlib.util.spec_from_file_location('bypass_evaluation_runner', PACKAGE / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

out = HERE / 'thermal_bypass/analysis'
out.mkdir(exist_ok=True)
assert not (out / 'complete.json').exists(), 'Completed analysis already exists'
selected = []
for seed in (20260914, 20260915):
    folder = HERE / 'thermal_bypass/runs' / str(seed)
    complete = json.loads((folder / 'complete.json').read_text())
    assert complete['status'] == 'complete' and complete['updates'] == 12000
    rows = json.loads((folder / 'validation.json').read_text())
    assert len(rows) == 26
    best = min(rows, key=lambda item: item['rmse'])
    checkpoint = folder / 'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert (saved['step'], saved['weights'], saved['validation_rmse']) == (
        best['step'], best['weights'], best['rmse'])
    selected.append({'seed': seed, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                     'step': saved['step'], 'weights': saved['weights'],
                     'validation_rmse': saved['validation_rmse']})
dump(out / 'selection.json', {'criterion': 'minimum validation regional macro RMSE', 'models': selected})
r.setup('cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = r.Dataset(PACKAGE, 'test', labels=False)
for selected_model in selected:
    model = construct().cuda()
    saved = torch.load(selected_model['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    prediction = r.inference(model, data, 'cuda', 1, amp=False)
    path = out / f"without_bypass_{selected_model['seed']}.npy"
    assert not path.exists()
    np.save(path, prediction)
    selected_model.update(prediction=str(path), prediction_sha256=digest(path))
    del model
dump(out / 'predictions_complete.json', {'models': selected, 'labels_opened': False,
                                        'scene_ids': [q['scene_id'] for q in data.records]})

data = r.Dataset(PACKAGE, 'test', labels=True)
old_receipt = json.loads((OLD / 'attribution_predictions/predictions_complete.json').read_text())
groups = {'without_bypass': [], 'with_bypass': []}
members = []
for selected_model in selected:
    seed = selected_model['seed']
    new_score = evaluate(np.load(selected_model['prediction'], mmap_mode='r'), data)
    original = next(e for e in old_receipt['entries'] if e['seed'] == seed and e['variant'] == 'learned')
    assert original['scene_order'] == [q['scene_id'] for q in data.records]
    assert digest(Path(original['prediction'])) == original['prediction_sha256']
    old_score = evaluate(np.load(original['prediction'], mmap_mode='r'), data)
    groups['without_bypass'].append(new_score)
    groups['with_bypass'].append(old_score)
    members.append({'seed': seed, 'without_bypass': new_score, 'with_bypass': old_score,
                    'paired_rmse': bootstrap(new_score, old_score),
                    'selection': selected_model,
                    'control_prediction': original['prediction'],
                    'control_prediction_sha256': original['prediction_sha256']})
combined = {key: mean_scores(values) for key, values in groups.items()}
assert abs(combined['with_bypass']['macro']['rmse'] - 0.45776602848061604) < 1e-9
paired = {metric: bootstrap(combined['without_bypass'], combined['with_bypass'], metric)
          for metric in ['rmse', 'mae', 'hotspot_iou', 'hotspot_mae']}
dump(out / 'scores.json', combined)
dump(out / 'paired.json', paired)
dump(out / 'members.json', members)
dump(out / 'complete.json', {'status': 'complete', 'seeds': [20260914, 20260915],
                            'new_training_runs': 2, 'updates_per_run': 12000,
                            'new_test_predictions': 180,
                            'source_sha256': {str(p): digest(p) for p in
                                             [Path(__file__), HERE / 'thermal_bypass_model.py',
                                              HERE / 'train_thermal_bypass.py']}})
print(json.dumps({'scores': {key: value['macro'] for key, value in combined.items()},
                  'paired_rmse': {k: v for k, v in paired['rmse'].items() if k != 'city_deltas'}}, indent=2))
