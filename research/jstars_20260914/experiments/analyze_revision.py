"""Reuse fixed predictions for U-TAE comparison and naturally sparse history."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/near_neighbor_attribution_20260914'
sys.path[:0] = [str(OLD), str(ROOT / 'resources/historylst246')]
from evaluate_comparisons import Dataset, digest, dump, bootstrap, stratified, evaluate, mean_scores

out = HERE / 'revision_evidence'
out.mkdir(exist_ok=True)
data = Dataset(ROOT / 'resources/historylst246', 'test', labels=True)
folder = ROOT / 'research/sub04_20260913/compact_query_product/final_evaluation/test'
receipt_path = folder / 'predictions_complete.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['scene_ids'] == [r['scene_id'] for r in data.records]
arrays = {'final_0.426': [], 'original_UTAE': []}
bindings = {}
for entry in receipt['entries']:
    key = 'final_0.426' if entry['architecture'] == 'naf_history' else 'original_UTAE'
    path = folder / entry['prediction']
    assert digest(path) == entry['prediction_sha256']
    arrays[key].append(np.load(path, mmap_mode='r'))
    bindings[str(path.relative_to(ROOT))] = digest(path)
assert all(len(v) == 3 for v in arrays.values())
scores = {key: mean_scores([evaluate(p, data) for p in values]) for key, values in arrays.items()}
original = json.loads((OLD / 'analysis_augmented/scores.json').read_text())
for key in scores:
    for metric in scores[key]['macro']:
        assert abs(scores[key]['macro'][metric] - original[key]['macro'][metric]) < 1e-12
paired = {metric: bootstrap(scores['original_UTAE'], scores['final_0.426'], metric)
          for metric in ['rmse', 'mae', 'hotspot_iou', 'hotspot_mae']}
strata = stratified(arrays, data)
dump(out / 'scores.json', scores)
dump(out / 'paired_utae.json', paired)
dump(out / 'sparse_history.json', strata)
dump(out / 'source_records.json', {
    'existing_predictions_only': True,
    'prediction_receipt_sha256': digest(receipt_path),
    'predictions': bindings,
    'sources': {str(p.relative_to(ROOT)): digest(p) for p in
                [Path(__file__), OLD / 'evaluate_comparisons.py']},
    'minimum_scored_pixels_per_scene_and_stratum': 32,
    'coverage_definition': 'Mean accepted thermal fraction across all nine historical slots, including empty slots',
})
print(json.dumps({'paired_rmse': {k: v for k, v in paired['rmse'].items() if k != 'city_deltas'},
                  'sparse_history': {k: v for k, v in strata.items()
                                     if k.startswith('history_') or k == 'coverage_low'}}, indent=2))
