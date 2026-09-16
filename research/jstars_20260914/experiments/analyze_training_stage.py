"""Rescore the existing stage-matched parent/U-TAE comparison without training."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import json
import sys
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/near_neighbor_attribution_20260914'
sys.path[:0] = [str(OLD), str(ROOT / 'resources/historylst246')]
from evaluate_comparisons import Dataset, digest, dump, evaluate, mean_scores, bootstrap

out = HERE / 'revision_evidence'
folder = ROOT / 'research/sub04_20260913/final_confirmation/test'
receipt = json.loads((folder / 'predictions_complete.json').read_text())
data = Dataset(ROOT / 'resources/historylst246', 'test', labels=True)
assert receipt['scene_ids'] == [q['scene_id'] for q in data.records]
repo = HERE.parent / 'code_repository'
if not repo.is_dir():
    repo = ROOT
lineages = json.loads((repo / 'docs/training_lineage.json').read_text())
groups = {'full_parent': [], 'utae_matched_stage': []}
checks = []
for entry in receipt['entries']:
    role = 'full_parent' if entry['architecture'] == 'naf_history' else 'utae_matched_stage'
    checkpoint = Path(entry['checkpoint'])
    assert digest(checkpoint) == entry['checkpoint_sha256']
    run = json.loads((checkpoint.parent / 'run.json').read_text())
    assert run['config']['updates'] == 3000
    assert run['config']['teacher_weight'] == .9
    if role == 'full_parent':
        lineage = next(x for x in lineages['lineages'] if x['initialization_seed'] == entry['seed'])
        assert str(checkpoint.relative_to(ROOT)) == lineage['stages'][2]['checkpoint']
    path = folder / entry['prediction']
    assert digest(path) == entry['prediction_sha256']
    groups[role].append(evaluate(np.load(path, mmap_mode='r'), data))
    checks.append({'role': role, 'seed': entry['seed'], 'parameters': run['parameters'],
                   'prediction': str(path.relative_to(ROOT)), 'prediction_sha256': digest(path),
                   'checkpoint': str(checkpoint.relative_to(ROOT)),
                   'checkpoint_sha256': entry['checkpoint_sha256'],
                   'configuration': run['config'], 'teacher': run.get('teacher'),
                   'schedule_sha256': run.get('schedule_sha256')})
assert all(len(x) == 3 for x in groups.values())
scores = {key: mean_scores(values) for key, values in groups.items()}
final = json.loads((out / 'scores.json').read_text())['final_0.426']
matched = bootstrap(scores['utae_matched_stage'], scores['full_parent'])
parent_to_final = bootstrap(scores['full_parent'], final)
matched_stages = []
for seed in [20260905, 20260912, 20260913]:
    chains = []
    for role in ['full_parent', 'utae_matched_stage']:
        record = next(x for x in checks if x['seed'] == seed and x['role'] == role)
        directories = [ROOT / Path(record['checkpoint']).parent]
        for _ in range(2):
            initializer = json.loads((directories[0] / 'run.json').read_text())['initialization']
            directories.insert(0, Path(initializer['checkpoint'] if isinstance(initializer, dict) else initializer).parent)
        chains.append(directories)
    for stage, (parent_dir, utae_dir), updates in zip(
            ['reference', 'first_teacher', 'second_teacher'], zip(*chains), [18000, 1000, 3000]):
        runs = [json.loads((p / 'run.json').read_text()) for p in [parent_dir, utae_dir]]
        completions = [json.loads((p / 'complete.json').read_text()) for p in [parent_dir, utae_dir]]
        assert all(x['updates'] == updates for x in completions)
        keys = ['seed', 'updates', 'batch_size', 'lr', 'warmup', 'weight_decay', 'validation_interval']
        if stage != 'reference':
            keys += ['history_dropout', 'emissivity_dropout']
            keys += ['ground_truth_loss_weight', 'teacher_loss_weight'] if stage == 'first_teacher' else ['teacher_weight']
        shared = {key: runs[0]['config'][key] for key in keys}
        assert all(runs[1]['config'][k] == v for k, v in shared.items())
        assert runs[0].get('teacher') == runs[1].get('teacher')
        schedule_check = None
        if stage != 'reference':
            with np.load(parent_dir / 'schedule.npz') as a, np.load(utae_dir / 'schedule.npz') as b:
                schedule_check = {k: bool(np.array_equal(a[k], b[k])) for k in ['fit_ids', 'd4']}
                assert all(schedule_check.values())
        matched_stages.append({'seed': seed, 'stage': stage, 'completed_updates_per_model': updates,
                               'shared_configuration': shared, 'same_teacher': True,
                               'stored_schedule_equal': schedule_check,
                               'parent_run': str((parent_dir / 'run.json').relative_to(ROOT)),
                               'utae_run': str((utae_dir / 'run.json').relative_to(ROOT))})
        for directory in [parent_dir, utae_dir]:
            for basename in ['run.json', 'complete.json', 'validation.json', 'config.json', 'schedule.npz']:
                source = directory / basename
                if source.is_file():
                    target = repo / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.resolve() != target.resolve():
                        shutil.copyfile(source, target)
report = {'scores': scores, 'utae_minus_parent': matched,
          'parent_minus_final': parent_to_final,
          'parent_parameters': 9310501, 'final_parameters': 5977045,
          'parameter_reduction_percent': 100 * (1 - 5977045 / 9310501),
          'records': checks,
          'stage_matching': matched_stages,
          'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), OLD / 'evaluate_comparisons.py']},
          'interpretation': 'The parent and U-TAE have the same completed supervised and two teacher-refinement update budgets; architecture and parameter count differ. Compact recovery/query refinement follow this parent.'}
dump(out / 'training_stage.json', report)
print(json.dumps({'metrics': {k: v['macro'] for k, v in scores.items()},
                  'utae_minus_parent': {k: v for k, v in matched.items() if k != 'city_deltas'},
                  'parent_minus_final': {k: v for k, v in parent_to_final.items() if k != 'city_deltas'},
                  'parameter_reduction_percent': report['parameter_reduction_percent']}, indent=2))
