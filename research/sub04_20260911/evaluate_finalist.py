"""Thin fixed-finalist adapter for the original three-seed paired evaluation.

Plan: candidate, pipeline (original/fixed_bn), manifest_sha256, runs[6]. Each
run: architecture (baseline/candidate), seed, run, files mapping run.json,
complete.json, validation.json and best.pt to SHA256. Paths use the plan folder.
For fixed_bn every run has override={path,sha256,receipt:{path,sha256},
design:{path,sha256}}. Plan calibration={sample_sha256,protocol_source:{path,
sha256}} binds the common fixed calibration. Even a no-BN model needs an
explicit unchanged-checkpoint/no-op receipt; this program never calibrates.
"""
from pathlib import Path
import argparse
import itertools
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_screens as screens
import paired_evaluation as paired
import torch

NAMES = dict(baseline='baseline905', wide='wide905', naf_history='naf905', baseline_history0='baseline_history0')
FILES = ('run.json', 'complete.json', 'validation.json', 'best.pt')


def bound(record, folder):
    path = (folder / record['path']).resolve()
    if paired.sha(path) != record['sha256']:
        raise ValueError(f'Frozen file changed: {path}')
    return path


def load_model(entry, device):
    return screens.load_model(dict(entry, name=NAMES[entry['architecture']]), device)


def checkpoint_override(entry, spec, plan, folder, original):
    path = bound(spec, folder)
    receipt = json.loads(bound(spec['receipt'], folder).read_text())
    design_path = bound(spec['design'], folder)
    design = json.loads(design_path.read_text())
    protocol = bound(plan['calibration']['protocol_source'], folder)
    sample_sha = plan['calibration']['sample_sha256']
    if design['sample_sha256'] != sample_sha or design['source_sha256'].get(str(protocol)) != paired.sha(protocol):
        raise ValueError('All six overrides must use the same frozen calibration source and sample')
    checks = ('original_checkpoint_preserved', 'all_parameters_exactly_unchanged', 'all_non_bn_buffers_exactly_unchanged')
    if any(receipt.get(key) is not True for key in checks) or receipt.get('validation_opened') is not False or receipt.get('test_opened') is not False:
        raise ValueError('Calibration receipt does not certify the required invariants')
    if receipt['original_checkpoint_sha256'] != entry['checkpoint_sha256'] or receipt['checkpoint_sha256'] != spec['sha256']:
        raise ValueError('Calibration receipt does not bind the original selected best and override')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    meta = saved['bn_recalibration']
    if meta['source_checkpoint_sha256'] != entry['checkpoint_sha256'] or meta['sample_sha256'] != sample_sha or meta['design_sha256'] != paired.sha(design_path):
        raise ValueError('Calibrated checkpoint provenance differs from the bound recipe')
    if meta.get('original_selection_retained') is not True or meta.get('calibrated_validation_scored') is not False:
        raise ValueError('Calibration must retain original selection without calibrated-score reselection')
    if any(saved[key] != original[key] for key in ('config', 'step', 'weights', 'validation_rmse')):
        raise ValueError('Calibration changed the originally selected weight identity')
    model = load_model(entry, 'cpu')
    mutable = {f'{name}.{key}' for name, module in model.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for key in ('running_mean', 'running_var', 'num_batches_tracked')}
    before, after = original['state_dict'], saved['state_dict']
    if before.keys() != after.keys() or any(not torch.equal(value, after[key]) for key, value in before.items() if key not in mutable):
        raise ValueError('An override changed a learned parameter or non-BN state')
    model.load_state_dict(after, strict=True)
    entry.update(original_checkpoint=entry['checkpoint'], original_checkpoint_sha256=entry['checkpoint_sha256'],
                 checkpoint=str(path), checkpoint_sha256=spec['sha256'], calibration_override=spec)


def plan_entries(path):
    plan = json.loads(path.read_text()); folder = path.parent
    candidate = plan['candidate']
    if candidate not in ('naf_history', 'baseline_history0', 'wide') or plan['pipeline'] not in ('original', 'fixed_bn'):
        raise ValueError('Explicit supported finalist and complete pipeline are required')
    expected = list(itertools.product(paired.SEEDS, ('baseline', candidate)))
    rows = {(row['seed'], row['architecture']): row for row in plan['runs']}
    if len(plan['runs']) != 6 or set(rows) != set(expected):
        raise ValueError('Plan must bind baseline/finalist at seeds 20260905/20260912/20260913')
    if paired.sha(paired.PACKAGE / 'manifest.json') != plan['manifest_sha256']:
        raise ValueError('The common data manifest changed')
    entries = []
    for seed, role in expected:
        row = rows[(seed, role)]; run = (folder / row['run']).resolve()
        if set(row['files']) != set(FILES):
            raise ValueError('Bind exactly the four original training/selection files')
        for name, digest in row['files'].items():
            bound(dict(path=str(run / name), sha256=digest), folder)
        complete = paired.verify_training(run)
        metadata = json.loads((run / 'run.json').read_text()); cfg = metadata['config']
        wanted = dict(screens.EXPECTED_BUDGET, seed=seed)
        if {key: cfg[key] for key in screens.FAIR_KEYS} != wanted or metadata['manifest_sha256'] != plan['manifest_sha256']:
            raise ValueError('Common manifest, 18k budget and 50 selection opportunities are required')
        architecture = 'baseline' if role == 'baseline_history0' else role
        if cfg.get('architecture', 'baseline') != architecture or cfg.get('history_dropout', .25) != (0. if role == 'baseline_history0' else .25) or cfg.get('emissivity_dropout', .25) != .25:
            raise ValueError('Model architecture/dropout differs from its fixed role')
        checkpoint = run / 'best.pt'; original = torch.load(checkpoint, map_location='cpu', weights_only=False)
        best = min(json.loads((run / 'validation.json').read_text()), key=lambda row: row['rmse'])
        if original.get('smoke_only') or original['config'] != cfg or (original['step'], original['weights'], original['validation_rmse']) != (best['step'], best['weights'], best['rmse']):
            raise ValueError('Use original selected best.pt, including the original earliest tie rule')
        entry = dict(architecture=role, seed=seed, run=str(run), checkpoint=str(checkpoint),
                     checkpoint_sha256=row['files']['best.pt'], config=cfg, training=complete, bound_run_files=row['files'])
        if ('override' in row) != (plan['pipeline'] == 'fixed_bn'):
            raise ValueError('Original forbids all overrides; fixed_bn requires all six, without seed-wise choice')
        if 'override' in row:
            checkpoint_override(entry, row['override'], plan, folder, original)
        entries.append(entry)
    return plan, entries


def source_seal():
    paths = [Path(__file__), Path(screens.__file__), Path(paired.__file__),
             HERE / 'dropout_models.py', HERE / 'naf_history/model.py',
             screens.STRONG / 'experiment.py', screens.STRONG / 'run_queue.py',
             screens.STRONG / 'candidates/network_review.py', paired.PACKAGE / 'run.py']
    paths += list((paired.PACKAGE / 'historylst').rglob('*.py'))
    return {str(path.resolve()): paired.sha(path) for path in paths}


def execute(args):
    seal = source_seal(); plan_sha = paired.sha(args.plan)
    if args.command == 'predict':
        plan, entries = plan_entries(args.plan)
        if args.split == 'test':
            if args.test_freeze is None:
                raise ValueError('Consumed Test30 requires a separate frozen effective-checkpoint list')
            freeze = json.loads(args.test_freeze.read_text())
            if freeze.get('split') != 'test' or freeze.get('model_selection_complete') is not True or [(r['architecture'], r['seed'], r['checkpoint_sha256']) for r in freeze['checkpoints']] != [(r['architecture'], r['seed'], r['checkpoint_sha256']) for r in entries]:
                raise ValueError('Separate Test freeze does not match this complete pipeline')
        paired.setup(args.device)
    else:
        plan = json.loads(args.plan.read_text())
        receipt = json.loads((args.output / 'predictions_complete.json').read_text())
        if receipt.get('finalist_plan_sha256') != plan_sha or receipt.get('pipeline') != plan['pipeline'] or receipt.get('evaluator_source_sha256') != seal:
            raise ValueError('Plan/pipeline/evaluator changed after all predictions were sealed')
        entries = receipt['entries']
    original_pairs, original_load, original_dump = paired.pairs, paired.load_model, paired.dump
    def bound_dump(path, value):
        if Path(path).name in ('prediction_started.json', 'predictions_complete.json', 'results.json'):
            value = dict(value, finalist_plan_sha256=plan_sha, pipeline=plan['pipeline'],
                         evaluator_source_sha256=seal, weights_reselected_by_fp32=False,
                         calibration_selected_per_seed=False)
        original_dump(path, value)
    paired.pairs, paired.load_model, paired.dump = lambda unused: entries, load_model, bound_dump
    args.queue = args.plan
    try:
        (paired.predict if args.command == 'predict' else paired.evaluate)(args)
    finally:
        paired.pairs, paired.load_model, paired.dump = original_pairs, original_load, original_dump
    if source_seal() != seal or paired.sha(args.plan) != plan_sha:
        raise RuntimeError('Evaluator or plan changed during execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'predict', 'score'))
    parser.add_argument('--plan', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--test-freeze', type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(dict(status='import_check_pass', reused=['predict', 'evaluate', 'compare_scores'],
                              data_opened=False, checkpoints_opened=False, gpu_used=False, winner_selected=False)))
        return
    if args.plan is None or args.output is None:
        parser.error('--plan and --output are required')
    args.plan, args.output = args.plan.resolve(), args.output.resolve()
    execute(args)


if __name__ == '__main__':
    main()
