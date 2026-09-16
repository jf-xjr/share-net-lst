"""Actual GPU cost for a separately frozen consumed-Test30 confirmation."""
from pathlib import Path
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark as measured
import evaluate_test as confirmation
import numpy as np
import torch


def run(args):
    def input_only(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            parts = Path(os.fsdecode(values[0])).parts
            if 'labels' in parts or ('data' in parts and {'fit', 'validation'} & set(parts)):
                raise RuntimeError('This frozen timing stage opens Test inputs and sealed predictions only')
    sys.addaudithook(input_only)
    freeze, experiments = confirmation.read_freeze(args.freeze)
    confirmation.verify_predictions(args.freeze, args.predictions)
    receipt_path = args.predictions / 'test_predictions_complete.json'
    freeze_sha, predictions_sha = measured.sha(args.freeze), measured.sha(receipt_path)
    paths = [Path(__file__), Path(measured.__file__), Path(confirmation.__file__),
             Path(measured.evaluation.__file__), HERE / 'train.py',
             measured.USAGE / 'policy_inputs.py', measured.USAGE / 'predict_candidate_sources.py',
             measured.USAGE / 'candidate_source_models.py', measured.USAGE / 'fit_policy.py',
             measured.USAGE / 'policy_acceptance.py', HERE.parent / 'dropout_models.py',
             HERE.parent / 'naf_history/model.py', measured.PACKAGE / 'run.py']
    paths += list((measured.PACKAGE / 'historylst').rglob('*.py'))
    paths += list((measured.USAGE.parent / 'candidates').glob('*.py'))
    seal = {str(path.resolve()): measured.sha(path) for path in paths}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Actual confirmation cost requires CUDA')
    models, methods = {}, []
    for experiment in experiments:
        seed = str(experiment['seed'])
        for entry in experiment['entries']:
            model, original = measured.evaluation.load_models(entry['architecture'], entry['checkpoint'])
            models[(seed, entry['name'])] = model.to('cuda').eval()
            del original
            methods += [(seed, entry['name'], strategy) for strategy in measured.MODES]
        methods += [(seed, 'mixed__joint', strategy) for strategy in ('cover_greedy_0', *measured.evaluation.SIMPLE_RULES)]
    data = measured.Dataset(measured.PACKAGE, 'test', labels=False)
    if len(data) != 90 or len({record['city'] for record in data.records}) != 30:
        raise ValueError('Expected all 90 original scenes and 30 consumed Test cities')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(measured=True, includes_selection=True, device='cuda', dtype='float32',
                    batch_size=1, cost_scope='Resident NumPy input through online selection, slicing, input transfer, FP32 forward, output transfer and original FP64 repair; excludes disk/weight loading and downloads',
                    gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), tf32=False,
                    amp=False, cpu_threads=2, repetitions=args.repeats, split='test',
                    common_queries=True, interleaved_order=True, warmups_per_method=3,
                    cohort_identity='Consumed 30-city follow-up')
    measured.write(args.output / 'started.json', dict(freeze_sha256=freeze_sha,
                   predictions_receipt_sha256=predictions_sha, source_sha256=seal,
                   timing_protocol=protocol, labels_opened=False, test_opened=True))
    warm = data.batch([0])
    for _ in range(3):
        for seed, weight, strategy in methods:
            measured.execute(models[(seed, weight)], strategy, warm, data.records[0])
    rng = np.random.default_rng(20260911)
    observations = []
    for scene_index, record in enumerate(data.records):
        sample = data.batch([scene_index])
        # Check the actual selector implementation shared with frozen prediction.
        for strategy in ('greedy3', 'cover_greedy_0', *measured.evaluation.SIMPLE_RULES):
            if measured.select(strategy, sample, record)[0] != confirmation.select(strategy, sample, record)[0]:
                raise ValueError('Timed source selection differs from the frozen prediction selector')
        for repeat in range(args.repeats):
            for method_index in rng.permutation(len(methods)):
                seed, weight, strategy = methods[method_index]
                value = measured.execute(models[(seed, weight)], strategy, sample, record)
                observations.append(dict(seed=seed, method=f'{weight}__{strategy}',
                    scene_id=record['scene_id'], city=record['city'], region=record['region'], repeat=repeat, **value))
        print(json.dumps(dict(event='consumed_test_scene_timing_complete', completed=scene_index+1, total=len(data))), flush=True)
    output = {str(experiment['seed']): dict(timings={}, actions={}) for experiment in experiments}
    grouped = {}
    for row in observations:
        grouped.setdefault((row['seed'], row['method'], row['scene_id']), []).append(row)
    for seed, weight, strategy in methods:
        method = f'{weight}__{strategy}'
        rows = output[seed]
        rows['timings'][method], rows['actions'][method] = [], []
        for record in data.records:
            values = grouped[(seed, method, record['scene_id'])]
            if len(values) != args.repeats or len({tuple(row['selected_slots']) for row in values}) != 1:
                raise RuntimeError('Repetitions differ in query identity or selected slots')
            identity = {key: record[key] for key in ('scene_id', 'city', 'region')}
            rows['timings'][method].append(dict(**identity, seconds=float(np.mean([row['seconds'] for row in values]))))
            rows['actions'][method].append(dict(**identity, source_count=values[0]['source_count']))
    if measured.sha(args.freeze) != freeze_sha or measured.sha(receipt_path) != predictions_sha or any(measured.sha(path) != digest for path, digest in seal.items()):
        raise RuntimeError('Freeze, predictions or executed code changed during measurement')
    if any(measured.sha(entry['checkpoint']) != entry['checkpoint_sha256'] for experiment in experiments for entry in experiment['entries']):
        raise RuntimeError('An effective frozen checkpoint changed during measurement')
    measured.write(args.output / 'benchmark.json', dict(status='complete', split='test', scenes=90, cities=30,
        freeze_sha256=freeze_sha, predictions_receipt_sha256=predictions_sha, source_sha256=seal,
        timing_protocol=protocol, experiments=output, observations=observations,
        cohort_identity='Consumed 30-city follow-up', labels_opened=False, test_opened=True,
        cost_branch_pass_not_assessed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least three measured repetitions are required')
    args.freeze, args.predictions, args.output = args.freeze.resolve(), args.predictions.resolve(), args.output.resolve()
    run(args)


if __name__ == '__main__':
    main()
