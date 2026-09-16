"""Matched Val45 GPU timing of fixed source strategies and the primary rule.

Resident NumPy inputs -> online selection -> transfer -> FP32 model -> FP64
repair. Every method sees every city/date in interleaved order. No query labels,
Test, disk/weight-loading cost, download savings or inferred latency is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / 'resources/historylst246'
USAGE = ROOT / 'research/strong_history_20260911/usage'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(USAGE))
sys.path.insert(0, str(PACKAGE))
import evaluate as evaluation
from policy_inputs import choose_sources
from predict_candidate_sources import CONSTANTS, MODES, specification, guard_inputs_only
from historylst.data import Dataset
from historylst.metrics import repair
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def select(strategy, sample, record):
    if strategy in CONSTANTS:
        return CONSTANTS[strategy], {}
    if strategy in evaluation.SIMPLE_RULES:
        config = evaluation.SIMPLE_RULES[strategy]
    elif strategy in ('greedy3', 'cover_greedy_0'):
        config = dict(mode='greedy3', conditions=[['greedy_uncovered_reliable', '<=',
                                                  0. if strategy == 'cover_greedy_0' else 1.]])
    else:
        raise ValueError(f'Unregistered source strategy: {strategy}')
    return choose_sources({key: value[0] for key, value in sample.items()}, record, config)


@torch.inference_mode()
def execute(model, strategy, sample, record):
    torch.cuda.synchronize()
    started = time.perf_counter()
    indices, features = select(strategy, sample, record)
    selected_at = time.perf_counter()
    inputs = sample if len(indices) == 9 else evaluation.crop(sample, np.asarray([indices]))
    tensors = {key: torch.from_numpy(value).to('cuda') for key, value in inputs.items()}
    torch.cuda.synchronize()
    transferred_at = time.perf_counter()
    with torch.autocast('cuda', enabled=False):
        output = model(**tensors)
    torch.cuda.synchronize()
    forward_at = time.perf_counter()
    raw = output.float().cpu().numpy()
    prediction = repair(raw, sample['coarse'], sample['support'])
    finished = time.perf_counter()
    supported = sample['support'].astype(bool)
    if output.dtype != torch.float32 or prediction.dtype != np.float64:
        raise RuntimeError('Expected FP32 forward and original FP64 repair')
    if not np.isfinite(prediction[supported]).all() or not np.isnan(prediction[~supported]).all():
        raise RuntimeError('Invalid repaired output on original support')
    return dict(source_count=len(indices), selected_slots=list(indices), selector_features=features,
                selection_seconds=selected_at-started, slicing_and_transfer_seconds=transferred_at-selected_at,
                forward_seconds=forward_at-transferred_at, output_and_repair_seconds=finished-forward_at,
                seconds=finished-started)


def run(args):
    guard_inputs_only()
    evaluation.verify(args.plan, args.predictions_receipt.parent)
    plan, entries = evaluation.read_plan(args.plan)
    predictions = json.loads(args.predictions_receipt.read_text())
    # Bind the exact seven effective checkpoints, even when a receipt stores
    # additional per-weight prediction metadata. No best checkpoint is chosen.
    expected = {row['name']: row['checkpoint_sha256'] for row in entries}
    observed = {row['name']: row['checkpoint_sha256'] for row in predictions['entries']}
    if expected != observed:
        raise ValueError('Benchmark and sealed source predictions use different weights')
    paths = [Path(__file__), Path(evaluation.__file__), HERE / 'train.py',
             USAGE / 'policy_inputs.py', USAGE / 'predict_candidate_sources.py',
             USAGE / 'candidate_source_models.py', HERE.parent / 'dropout_models.py',
             HERE.parent / 'naf_history/model.py', PACKAGE / 'run.py']
    paths += list((PACKAGE / 'historylst').rglob('*.py'))
    seal = {str(path.resolve()): sha(path) for path in paths}
    plan_sha = sha(args.plan)
    prediction_sha = sha(args.predictions_receipt)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Actual GPU timing requires CUDA')
    models = {}
    for entry in entries:
        model, original = evaluation.load_models(entry['architecture'], entry['checkpoint'])
        models[entry['name']] = model.to('cuda').eval()
        del original
    data = Dataset(PACKAGE, 'validation', labels=False)
    if len(data) != 45 or len({row['city'] for row in data.records}) != 15:
        raise ValueError('Expected all original Val45 scenes and 15 cities')
    spec = specification(USAGE, 'validation', data)
    methods = [(entry['name'], strategy) for entry in entries for strategy in MODES]
    methods.append(('mixed__joint', 'cover_greedy_0'))
    methods += [('mixed__joint', name) for name in evaluation.SIMPLE_RULES]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(measured=True, includes_selection=True, device='cuda', dtype='float32',
                    batch_size=1, cost_scope='Resident NumPy input through online selection, history slicing, input transfer, FP32 forward, output transfer and FP64 repair; disk/weight loading and downloads excluded',
                    gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), tf32=False,
                    amp=False, cpu_threads=2, repetitions=args.repeats, split='validation',
                    common_queries=True, interleaved_order=True, warmups_per_method=3)
    write(args.output / 'started.json', dict(plan_sha256=plan_sha, predictions_receipt_sha256=prediction_sha,
          source_sha256=seal, timing_protocol=protocol, labels_opened=False, test_opened=False))
    warm = data.batch([0])
    for _ in range(3):
        for weight, strategy in methods:
            execute(models[weight], strategy, warm, data.records[0])
    rng = np.random.default_rng(20260911)
    observations = []
    for scene_index, record in enumerate(data.records):
        sample = data.batch([scene_index])
        greedy, features = select('greedy3', sample, record)
        cover, _ = select('cover_greedy_0', sample, record)
        if greedy != spec['strategies']['greedy3'][scene_index] or cover != (greedy if features['greedy_uncovered_reliable'] <= 0 else list(range(9))):
            raise ValueError('Online source selection differs from frozen prediction indices')
        for repeat in range(args.repeats):
            for method_index in rng.permutation(len(methods)):
                weight, strategy = methods[method_index]
                measured = execute(models[weight], strategy, sample, record)
                observations.append(dict(method=f'{weight}__{strategy}', scene_id=record['scene_id'],
                                         city=record['city'], region=record['region'], repeat=repeat, **measured))
        print(json.dumps(dict(event='scene_timing_complete', completed=scene_index+1, total=len(data))), flush=True)
    timings, actions = {}, {}
    for weight, strategy in methods:
        name = f'{weight}__{strategy}'
        timings[name], actions[name] = [], []
        for record in data.records:
            rows = [row for row in observations if row['method'] == name and row['scene_id'] == record['scene_id']]
            if len(rows) != args.repeats or len({tuple(row['selected_slots']) for row in rows}) != 1:
                raise RuntimeError('Repeated query identities or input-only actions differ')
            identity = {key: record[key] for key in ('scene_id', 'city', 'region')}
            timings[name].append(dict(**identity, seconds=float(np.mean([row['seconds'] for row in rows]))))
            actions[name].append(dict(**identity, source_count=rows[0]['source_count']))
    if sha(args.plan) != plan_sha or sha(args.predictions_receipt) != prediction_sha:
        raise RuntimeError('Bound plan or predictions changed during timing')
    if any(sha(path) != digest for path, digest in seal.items()) or any(sha(row['checkpoint']) != row['checkpoint_sha256'] for row in entries):
        raise RuntimeError('Model or timing code changed during measurement')
    write(args.output / 'benchmark.json', dict(status='complete', split='validation', scenes=45, cities=15,
          plan_sha256=plan_sha, predictions_receipt_sha256=prediction_sha, source_sha256=seal,
          timing_protocol=protocol, timings=timings, actions=actions, observations=observations,
          labels_opened=False, test_opened=False, cost_branch_pass_not_assessed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--predictions-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least three measured repeats are required')
    args.plan, args.predictions_receipt, args.output = args.plan.resolve(), args.predictions_receipt.resolve(), args.output.resolve()
    run(args)


if __name__ == '__main__':
    main()
