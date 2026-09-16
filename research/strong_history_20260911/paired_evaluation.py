"""Sealed, paired-seed evaluation for completed training runs.

Validation is the default. Test requires a separately frozen checkpoint list;
this program does not choose architectures or tune a history policy on Test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'resources/historylst246'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PACKAGE))
from experiment import factory, runner
from run_queue import verify_training
import numpy as np
import torch
from historylst.data import Dataset
from historylst.metrics import score
from historylst.hotspots import add_hotspot_metrics

SEEDS = (20260905, 20260912, 20260913)
METRICS = ('rmse', 'mae', 'hotspot_iou', 'hotspot_mae')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    runner.dump(path, value)


def pairs(queue_path):
    queue = json.loads(queue_path.read_text())
    if queue['status'] != 'training_complete_needs_validation_review':
        raise RuntimeError('The complete paired training queue is not yet ready')
    plan_path = HERE / 'queue_plan.json'
    assert sha(plan_path) == queue['plan_sha256']
    for relative, expected in json.loads(plan_path.read_text())['source_sha256'].items():
        assert sha(ROOT / relative) == expected, f'Training source changed: {relative}'
    candidate = queue['selection']['architecture']
    entries = []
    fair_keys = ('seed', 'updates', 'batch_size', 'lr', 'warmup', 'weight_decay',
                 'validation_interval', 'evaluation_batch')
    for seed in SEEDS:
        settings = []
        for architecture in ('baseline', candidate):
            run = (ROOT / 'research/route_a_20260911/utae' if architecture == 'baseline' and seed == SEEDS[0]
                   else HERE / 'runs' / f'{architecture}_{seed}')
            complete = verify_training(run)
            metadata = json.loads((run / 'run.json').read_text())
            config = metadata['config']
            assert config['seed'] == seed
            settings.append({key: config[key] for key in fair_keys})
            checkpoint = run / 'best.pt'
            entries.append(dict(architecture=architecture, seed=seed,
                                checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                                run=str(run.relative_to(ROOT)), training=complete, config=config))
        assert settings[0] == settings[1], 'Paired training opportunities do not match'
    return entries


def setup(device):
    runner.setup(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(entry, device):
    checkpoint = ROOT / entry['checkpoint']
    if sha(checkpoint) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    model = factory(entry['architecture'])().to(device).eval()
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(stored['state_dict'], strict=True)
    assert stored['config'] == entry['config']
    return model


def predict(args):
    entries = pairs(args.queue)
    if args.split == 'test':
        if args.test_freeze is None:
            raise ValueError('Test prediction requires the separately frozen checkpoint list')
        freeze = json.loads(args.test_freeze.read_text())
        wanted = [(e['architecture'], e['seed'], e['checkpoint_sha256']) for e in entries]
        observed = [(e['architecture'], e['seed'], e['checkpoint_sha256']) for e in freeze['checkpoints']]
        assert freeze['split'] == 'test' and freeze['model_selection_complete'] is True
        assert observed == wanted, 'The frozen Test checkpoint list differs'
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    # Record complete identities before any prediction; labels remain unopened.
    receipt = dict(stage='predicting', split=args.split, device=args.device, dtype='float32',
                   tf32=False, labels_opened=False, queue_sha256=sha(args.queue),
                   code_sha256=sha(__file__), entries=entries,
                   test_freeze_sha256=sha(args.test_freeze) if args.test_freeze else None,
                   cohort_identity='Consumed 30-city follow-up' if args.split == 'test' else 'Val45 development')
    dump(args.output / 'prediction_started.json', receipt)
    data = Dataset(PACKAGE, args.split, labels=False)
    receipt['scene_ids'] = [r['scene_id'] for r in data.records]
    for entry in entries:
        model = load_model(entry, args.device)
        started = time.perf_counter()
        prediction = runner.inference(model, data, args.device, 2, amp=False)
        filename = f"{entry['architecture']}_{entry['seed']}.npy"
        path = args.output / filename
        np.save(path, prediction, allow_pickle=False)
        entry.update(prediction=filename, prediction_sha256=sha(path),
                     inference_seconds_including_io=time.perf_counter() - started,
                     timing_is_benchmark=False)
        print(json.dumps(dict(event='prediction_sealed', architecture=entry['architecture'],
                              seed=entry['seed'], split=args.split)), flush=True)
        del model, prediction
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    receipt['stage'] = 'all_predictions_sealed'
    dump(args.output / 'predictions_complete.json', receipt)


def compare_scores(scores, candidate, draws=4000):
    """Pair cities and training seeds; intervals are descriptive with 3 seeds."""
    cities = scores[('baseline', SEEDS[0])]['cities']
    keys = [(city['region'], city['city']) for city in cities]
    arrays = {}
    for architecture in ('baseline', candidate):
        for seed in SEEDS:
            rows = scores[(architecture, seed)]['cities']
            assert [(city['region'], city['city']) for city in rows] == keys
        arrays[architecture] = np.array([[[city[k] for k in METRICS]
                                          for city in scores[(architecture, seed)]['cities']]
                                         for seed in SEEDS], dtype=float)
    groups = [np.array([i for i, key in enumerate(keys) if key[0] == region])
              for region in sorted({key[0] for key in keys})]
    deltas = arrays['baseline'] - arrays[candidate]
    seed_effect = np.mean([deltas[:, group].mean(1) for group in groups], axis=0)
    estimate = seed_effect.mean(0)
    rng = np.random.default_rng(20260911)
    city_only = np.empty((draws, len(METRICS)))
    seed_and_city = np.empty_like(city_only)
    for i in range(draws):
        indices = [rng.choice(group, len(group), replace=True) for group in groups]
        aggregate = np.mean([deltas[:, index].mean(1) for index in indices], axis=0)
        city_only[i] = aggregate.mean(0)
        seed_and_city[i] = aggregate[rng.choice(len(SEEDS), len(SEEDS), replace=True)].mean(0)
    means = {architecture: {metric: float(np.mean([scores[(architecture, seed)]['macro'][metric]
                                                 for seed in SEEDS])) for metric in METRICS}
             for architecture in arrays}
    for j, metric in enumerate(METRICS):
        assert abs(means['baseline'][metric] - means[candidate][metric] - estimate[j]) < 1e-12
    effects = {metric: dict(baseline_minus_candidate=float(estimate[j]),
                            paired_seed_differences=[float(x) for x in seed_effect[:, j]],
                            city_95_interval=np.quantile(city_only[:, j], [.025, .975]).tolist(),
                            seed_city_95_interval=np.quantile(seed_and_city[:, j], [.025, .975]).tolist())
               for j, metric in enumerate(METRICS)}
    # IoU is higher-is-better, unlike the other reported metrics.
    effects['hotspot_iou']['improvement_direction'] = 'negative baseline-minus-candidate'
    criteria = dict(minimum_mean_rmse_reduction_k=.01,
                    observed_mean_rmse_reduction_k=float(estimate[0]),
                    all_three_seed_macro_improvements=bool((seed_effect[:, 0] > 0).all()))
    criteria['network_numeric_criteria_pass'] = bool(estimate[0] >= .01 and criteria['all_three_seed_macro_improvements'])
    return dict(seed_count=3, seeds=list(SEEDS), mean_metrics=means, effects=effects,
                criteria=criteria, ensemble_used=False,
                city_mean_rmse_differences=[dict(region=key[0], city=key[1],
                    baseline_minus_candidate=float(deltas[:, i, 0].mean())) for i, key in enumerate(keys)],
                interval_scope='Paired sampled cities and three observed training seeds only; descriptive, not full training-population uncertainty')


def evaluate(args):
    receipt_path = args.output / 'predictions_complete.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['stage'] == 'all_predictions_sealed' and receipt['labels_opened'] is False
    assert receipt['split'] == args.split
    if (args.output / 'results.json').exists():
        raise FileExistsError('Paired scores already exist')
    for entry in receipt['entries']:
        assert sha(args.output / entry['prediction']) == entry['prediction_sha256']
    data = Dataset(PACKAGE, args.split, labels=True)
    assert receipt['scene_ids'] == [r['scene_id'] for r in data.records]
    scores = {}
    for entry in receipt['entries']:
        prediction = np.load(args.output / entry['prediction'], mmap_mode='r', allow_pickle=False)
        result = score(prediction, data.arrays['target'], data.arrays['formal'], data.records)
        add_hotspot_metrics(result, prediction, data.arrays['target'], data.arrays['formal'])
        key = (entry['architecture'], entry['seed'])
        scores[key] = result
        dump(args.output / f'{key[0]}_{key[1]}_scores.json', result)
    names = {entry['architecture'] for entry in receipt['entries']} - {'baseline'}
    assert len(names) == 1
    comparison = compare_scores(scores, names.pop())
    comparison.update(split=args.split, cohort_identity=receipt['cohort_identity'],
                      prediction_receipt_sha256=sha(receipt_path),
                      full_goal_complete=False,
                      remaining_requirements='Usage-rule evidence, actual compute benchmark and final independent review remain separate; Val criteria are developmental only')
    dump(args.output / 'results.json', comparison)
    print(json.dumps(comparison['criteria'], indent=2), flush=True)


def check():
    """Small exact city/region fixture; no data arrays or model files."""
    scores = {}
    for architecture in ('baseline', 'candidate'):
        for seed in SEEDS:
            rows = [dict(region='A', city='a', rmse=1., mae=.8, hotspot_iou=.7, hotspot_mae=1.),
                    dict(region='B', city='b', rmse=2., mae=1.8, hotspot_iou=.5, hotspot_mae=2.),
                    dict(region='B', city='c', rmse=3., mae=2.8, hotspot_iou=.3, hotspot_mae=3.)]
            if architecture == 'candidate':
                for row in rows:
                    row['rmse'] -= .02
                    row['hotspot_iou'] += .01
            macro = {metric: (rows[0][metric] + (rows[1][metric] + rows[2][metric]) / 2) / 2
                     for metric in METRICS}
            scores[(architecture, seed)] = dict(cities=rows, macro=macro)
    result = compare_scores(scores, 'candidate', draws=20)
    assert abs(result['effects']['rmse']['baseline_minus_candidate'] - .02) < 1e-12
    assert abs(result['effects']['hotspot_iou']['baseline_minus_candidate'] + .01) < 1e-12
    assert result['criteria']['network_numeric_criteria_pass']
    for row in scores[('candidate', SEEDS[-1])]['cities']:
        row['rmse'] += .03
    scores[('candidate', SEEDS[-1])]['macro']['rmse'] += .03
    result = compare_scores(scores, 'candidate', draws=20)
    assert not result['criteria']['network_numeric_criteria_pass']
    return dict(status='passed', synthetic_only=True, test_opened=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('predict', 'score', 'check'))
    parser.add_argument('--queue', type=Path, default=HERE / 'queue_runtime.json')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--test-freeze', type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check()))
    else:
        if args.output is None:
            parser.error('--output is required')
        if args.command == 'predict':
            setup(args.device)
            predict(args)
        else:
            evaluate(args)
