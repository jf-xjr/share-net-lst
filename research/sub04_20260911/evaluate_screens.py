"""Fixed-checkpoint, single-seed Val45 screening and matched GPU inference cost.

Predict all four models before opening labels. Training selection remains the
original 50 raw/EMA opportunities; FP32 evaluation never selects new weights.
Uses the existing inference/FP64 repair, core metrics and hotspot definitions.
"""
from pathlib import Path
import argparse
import itertools
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
STRONG = HERE.parent / 'strong_history_20260911'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STRONG))
from paired_evaluation import PACKAGE, ROOT, METRICS, runner, setup, sha, dump, verify_training
from historylst.data import Dataset
from historylst.metrics import score, repair
from historylst.hotspots import add_hotspot_metrics
from historylst.model import HistoryUTAE
from candidates.network_review import HistoryWideUTAE
from dropout_models import DropoutUTAE
from naf_history.model import HistoryNAFReconstructor
import numpy as np
import torch

NAMES = ('baseline905', 'wide905', 'naf905', 'baseline_history0')
FAIR_KEYS = ('seed', 'updates', 'batch_size', 'lr', 'warmup', 'weight_decay',
             'validation_interval', 'evaluation_batch')
EXPECTED_BUDGET = dict(seed=20260905, updates=18000, batch_size=4, lr=.001,
                       warmup=300, weight_decay=.0001, validation_interval=750,
                       evaluation_batch=2)


def guard(*, labels=False, fit_only=False):
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        if 'data' not in path.parts:
            return
        if 'test' in path.parts or (fit_only and 'validation' in path.parts):
            raise RuntimeError('Test is unavailable; the benchmark only opens Fit inputs')
        if not labels and 'labels' in path.parts:
            raise RuntimeError('Labels cannot be opened in this stage')
    sys.addaudithook(audit)


def entries(queue_path):
    queue = json.loads(queue_path.read_text())
    if queue['status'] != 'screens_complete_need_validation_review':
        raise RuntimeError('Both new 18k training screens must finish before evaluation')
    for path, expected in queue['source_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Screen source changed: {path}')
    jobs = [job for job in queue['jobs'] if job['mode'] == 'train']
    if len(jobs) != 2 or {job['name'] for job in jobs} != {'naf_history', 'baseline_history0'}:
        raise ValueError('Expected exactly the two registered training screens')
    if any(job['status'] != 'complete' or job['exit_code'] != 0 for job in jobs):
        raise ValueError('A screen did not complete successfully')
    runs = {'baseline905': ROOT / 'research/route_a_20260911/utae',
            'wide905': STRONG / 'runs/wide_20260905',
            'naf905': Path(next(job['output'] for job in jobs if job['name'] == 'naf_history')),
            'baseline_history0': Path(next(job['output'] for job in jobs if job['name'] == 'baseline_history0'))}
    expected_architecture = dict(baseline905='baseline', wide905='wide',
                                 naf905='naf_history', baseline_history0='baseline')
    result = []
    for name in NAMES:
        run = runs[name].resolve()
        completion = verify_training(run)
        metadata = json.loads((run / 'run.json').read_text())
        cfg = metadata['config']
        if metadata['manifest_sha256'] != sha(PACKAGE / 'manifest.json'):
            raise ValueError(f'{name}: training manifest differs from the current evaluation cohort')
        if {key: cfg[key] for key in FAIR_KEYS} != EXPECTED_BUDGET:
            raise ValueError(f'{name}: unequal training or selection opportunities')
        architecture = cfg.get('architecture', 'baseline')
        if architecture != expected_architecture[name]:
            raise ValueError(f'{name}: architecture does not match its registered role')
        dropout = cfg.get('history_dropout', .25)
        if dropout != (0. if name == 'baseline_history0' else .25) or cfg.get('emissivity_dropout', .25) != .25:
            raise ValueError(f'{name}: unexpected input-dropout recipe')
        checkpoint = run / 'best.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        rows = json.loads((run / 'validation.json').read_text())
        best = min(rows, key=lambda row: row['rmse'])  # Stable: first exact tie wins.
        if ck['config'] != cfg or ck['step'] != best['step'] or ck['weights'] != best['weights'] or ck['validation_rmse'] != best['rmse']:
            raise ValueError(f'{name}: best.pt is not the originally selected weight state')
        if ck.get('smoke_only'):
            raise ValueError('Smoke weights cannot enter scientific screening')
        result.append(dict(name=name, architecture=architecture, seed=cfg['seed'],
                           run=str(run), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                           config=cfg, training=completion, selected_step=ck['step'],
                           selected_weights=ck['weights'], training_selection_rmse=ck['validation_rmse'],
                           selection='Original minimum Val45 macro RMSE among 50 raw/EMA candidates; earliest exact tie',
                           training_record_sha256={file: sha(run / file) for file in
                                                   ('run.json', 'complete.json', 'validation.json')}))
    return result


def load_model(entry, device):
    if sha(entry['checkpoint']) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after prediction/selection sealing')
    classes = dict(baseline905=HistoryUTAE, wide905=HistoryWideUTAE,
                   naf905=HistoryNAFReconstructor, baseline_history0=DropoutUTAE)
    cfg = entry['config']
    kwargs = {} if entry['name'] in ('baseline905', 'wide905') else dict(
        history_dropout=cfg['history_dropout'], emissivity_dropout=cfg['emissivity_dropout'])
    model = classes[entry['name']](**kwargs).float().to(device).eval()
    ck = torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)
    if ck['config'] != cfg:
        raise ValueError('Checkpoint configuration changed')
    model.load_state_dict(ck['state_dict'], strict=True)
    return model


def prediction_receipt(output):
    receipt = json.loads((output / 'predictions_complete.json').read_text())
    if receipt['stage'] != 'all_predictions_sealed' or receipt['split'] != 'validation' or receipt['labels_opened']:
        raise ValueError('All four input-only Val predictions must be sealed first')
    if tuple(entry['name'] for entry in receipt['entries']) != NAMES:
        raise ValueError('The sealed model list changed')
    for path, expected in receipt['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen evaluation/training source changed: {path}')
    for entry in receipt['entries']:
        if sha(output / entry['prediction']) != entry['prediction_sha256']:
            raise ValueError('Prediction array changed after sealing')
        if sha(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed after sealing')
    return receipt


def predict(args):
    guard()
    selected = entries(args.queue)
    setup(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    queue = json.loads(args.queue.read_text())
    sources = dict(queue['source_sha256'])
    sources.update({str(path): sha(path) for path in
                    (Path(__file__), STRONG / 'paired_evaluation.py', STRONG / 'benchmark_models.py',
                     STRONG / 'experiment.py', STRONG / 'run_queue.py')})
    receipt = dict(stage='predicting', split='validation', cohort='Val45 development; single training seed',
                   device=args.device, dtype='float32', tf32=False, amp=False, evaluation_batch=2,
                   labels_opened=False, test_opened=False, queue_sha256=sha(args.queue),
                   source_sha256=sources, entries=selected,
                   weights_reselected_by_fp32=False, full_goal_complete=False)
    dump(args.output / 'prediction_started.json', receipt)
    data = Dataset(PACKAGE, 'validation', labels=False)
    if len(data) != 45 or len({row['city'] for row in data.records}) != 15:
        raise ValueError('Expected the complete original Val45/15-city cohort')
    receipt['scene_ids'] = [row['scene_id'] for row in data.records]
    for entry in selected:
        model = load_model(entry, args.device)
        prediction = runner.inference(model, data, args.device, 2, amp=False)
        path = args.output / f"{entry['name']}.npy"
        np.save(path, prediction, allow_pickle=False)
        entry.update(prediction=path.name, prediction_sha256=sha(path),
                     parameters=sum(p.numel() for p in model.parameters()))
        print(json.dumps(dict(event='prediction_sealed', name=entry['name'])), flush=True)
        del model, prediction
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    for path, expected in sources.items():
        if sha(path) != expected:
            raise RuntimeError(f'Source changed during prediction: {path}')
    receipt['stage'] = 'all_predictions_sealed'
    dump(args.output / 'predictions_complete.json', receipt)


def differences(scores, reference, candidate):
    ref, cand = scores[reference], scores[candidate]
    keys = [(row['region'], row['city']) for row in ref['cities']]
    if keys != [(row['region'], row['city']) for row in cand['cities']]:
        raise ValueError('City pairing changed')
    delta = lambda a, b: {metric: float(b[metric] - a[metric]) for metric in METRICS}
    return dict(reference=reference, candidate=candidate, sign='candidate minus reference',
                lower_is_better=['rmse', 'mae', 'hotspot_mae'], higher_is_better=['hotspot_iou'],
                macro=delta(ref['macro'], cand['macro']),
                regions={region: delta(ref['regions'][region], cand['regions'][region]) for region in ref['regions']},
                cities=[dict(region=key[0], city=key[1], **delta(a, b))
                        for key, a, b in zip(keys, ref['cities'], cand['cities'])])


def evaluate(args):
    guard(labels=True)
    receipt = prediction_receipt(args.output)
    score_dir = args.output / 'scores'
    score_dir.mkdir(exist_ok=False)
    data = Dataset(PACKAGE, 'validation', labels=True)
    if receipt['scene_ids'] != [row['scene_id'] for row in data.records]:
        raise ValueError('Prediction and label scene order differ')
    scores = {}
    for entry in receipt['entries']:
        prediction = np.load(args.output / entry['prediction'], mmap_mode='r', allow_pickle=False)
        result = score(prediction, data.arrays['target'], data.arrays['formal'], data.records)
        add_hotspot_metrics(result, prediction, data.arrays['target'], data.arrays['formal'])
        scores[entry['name']] = result
        dump(score_dir / f"{entry['name']}.json", result)
    comparisons = [differences(scores, a, b) for a, b in itertools.combinations(NAMES, 2)]
    result = dict(status='complete', split='validation', scenes=45, cities=15, training_seeds=[20260905],
                  macro_metrics={name: scores[name]['macro'] for name in NAMES}, comparisons=comparisons,
                  hotspot_definition='Existing scene-relative top decile over the unchanged formal support, with stable ties',
                  prediction_receipt_sha256=sha(args.output / 'predictions_complete.json'),
                  selected_checkpoints=receipt['entries'], weights_reselected_by_fp32=False,
                  ensemble_used=False, test_opened=False, full_goal_complete=False,
                  interpretation='Single-seed development screening only. Matched independent training repeats are required before claiming reproducible superiority; no success gate is introduced here.')
    dump(args.output / 'results.json', result)
    print(json.dumps(result['macro_metrics'], indent=2), flush=True)


def benchmark(args):
    """The existing benchmark_models protocol generalized to four fixed models."""
    if args.device != 'cuda':
        raise ValueError('This entry point reports actual GPU inference cost; use --device cuda')
    guard(fit_only=True)
    receipt = prediction_receipt(args.output)
    setup('cuda')
    started_file = args.output / 'benchmark_started.json'
    if started_file.exists() or (args.output / 'benchmark.json').exists():
        raise FileExistsError('A benchmark attempt already exists; no automatic overwrite or retry')
    dump(started_file, dict(repeats=args.repeats, device='cuda', dtype='float32', tf32=False,
                           prediction_receipt_sha256=sha(args.output / 'predictions_complete.json')))
    models = {entry['name']: load_model(entry, 'cuda') for entry in receipt['entries']}
    data = Dataset(PACKAGE, 'fit', labels=False)
    samples = [data.batch([i]) for i in (0, 1, 2)]
    def execute(name, sample):
        torch.cuda.synchronize()
        begun = time.perf_counter()
        batch = {key: torch.from_numpy(value).to('cuda') for key, value in sample.items()}
        torch.cuda.synchronize()
        forward_begun = time.perf_counter()
        with torch.inference_mode():
            output = models[name](**batch)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_begun
        prediction = repair(output.float().cpu().numpy(), sample['coarse'], sample['support'])
        torch.cuda.synchronize()
        total_seconds = time.perf_counter() - begun
        if not np.isfinite(prediction[sample['support'].astype(bool)]).all():
            raise RuntimeError('Nonfinite benchmark prediction')
        return forward_seconds, total_seconds
    for sample in samples:
        for name in NAMES:
            execute(name, sample)
    rows = []
    rng = np.random.default_rng(20260911)
    for repeat in range(args.repeats):
        index = repeat % 3
        for name in rng.permutation(NAMES):
            forward, total = execute(str(name), samples[index])
            rows.append(dict(name=str(name), repeat=repeat, fit_index=index,
                             forward_seconds=forward, total_seconds=total))
    summaries = {}
    for entry in receipt['entries']:
        name = entry['name']
        summaries[name] = dict(parameters=entry['parameters'], checkpoint_sha256=entry['checkpoint_sha256'])
        for field in ('forward_seconds', 'total_seconds'):
            values = np.array([row[field] for row in rows if row['name'] == name])
            summaries[name][field] = dict(median=float(np.median(values)), mean=float(values.mean()),
                                          p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))
    result = dict(status='complete', device='cuda', gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                  dtype='float32', tf32=False, amp=False, batch_size=1, cpu_threads=2,
                  fit_indices=[0, 1, 2], fit_cohort='Akron, three dates', crop_shape=[160, 160],
                  fit_scene_ids=[data.records[i]['scene_id'] for i in (0, 1, 2)],
                  labels_opened=False, test_opened=False,
                  warmups_per_model=3, repeats_per_model=args.repeats, summaries=summaries,
                  paired_order_observations=rows,
                  total_scope='Input transfer, forward, output transfer and common FP64 repair; excludes disk/weight loading',
                  caveat='Akron three-date Fit inputs, fixed 160x160 and batch1; not representative multi-city cost, training cost, or a full deployment workload.',
                  source_protocol=str(STRONG / 'benchmark_models.py'),
                  prediction_receipt_sha256=sha(args.output / 'predictions_complete.json'))
    dump(args.output / 'benchmark.json', result)
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('predict', 'benchmark', 'score', 'check'))
    parser.add_argument('--queue', type=Path, default=HERE / 'screen_runtime.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(dict(status='import_and_cli_check_pass', names=NAMES,
                              data_opened=False, checkpoints_opened=False, gpu_used=False,
                              expected_training_budget=EXPECTED_BUDGET)))
        return
    if args.output is None:
        parser.error('--output is required')
    if args.repeats < 10:
        parser.error('--repeats must be at least ten')
    args.output, args.queue = args.output.resolve(), args.queue.resolve()
    {'predict': predict, 'benchmark': benchmark, 'score': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
