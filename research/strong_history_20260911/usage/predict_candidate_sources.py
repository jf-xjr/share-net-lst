"""Input-only five-strategy Fit/Val prediction for explicit candidate weights.

Creates an exclusive directory; writes a completion seal only after all ten
arrays finish. Scoring is intentionally absent. The predictions/ layout and
copied input feature tables are compatible with the unchanged fit_policy.py.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / 'resources/historylst246'
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HERE))
from candidate_source_models import model_classes
from historylst.data import Dataset
from historylst.metrics import repair

MODES = ('full9', 'recent3', 'old6', 'nearest1', 'greedy3')
CONSTANTS = dict(full9=list(range(9)), recent3=[6, 7, 8], old6=list(range(6)), nearest1=[6])
COUNTS = dict(full9=9, recent3=3, old6=6, nearest1=1, greedy3=3)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def guard_inputs_only(fit_only=False):
    opened = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        parts = path.parts
        if 'labels' in parts or ('data' in parts and 'test' in parts):
            raise RuntimeError('Source prediction may not open query labels or Test data')
        if fit_only and 'data' in parts and 'validation' in parts:
            raise RuntimeError('Mechanical checks may only open Fit data')
        if 'data' in parts and path.suffix == '.npy':
            opened.add(str(path))
    sys.addaudithook(audit)
    return opened


def setup(device):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')


def specification(folder, split, data):
    feature = folder / f'{split}_features.csv'
    source = folder / f'{split}_subsets.json'
    spec = json.loads(source.read_text())
    scene_ids = [row['scene_id'] for row in data.records]
    if spec['scene_ids'] != scene_ids or spec['n_queries'] != len(data):
        raise ValueError('Source specification and manifest scene order differ')
    if spec['split'] != split or spec['labels_opened'] is not False:
        raise ValueError('Expected an input-only source specification')
    if digest(feature) != spec['features_sha256']:
        raise ValueError('Feature table hash mismatch')
    with feature.open() as file:
        if [row['scene_id'] for row in csv.DictReader(file)] != scene_ids:
            raise ValueError('Feature table and manifest scene order differ')
    if set(spec['strategies']) != set(MODES):
        raise ValueError('Only the five frozen source strategies are supported')
    for mode in MODES:
        indices = np.asarray(spec['strategies'][mode])
        if indices.shape != (len(data), COUNTS[mode]) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f'Invalid source index geometry for {mode}')
        if np.any(indices < 0) or np.any(indices > 8):
            raise ValueError('Source indices must address original slots 0..8')
        if any(len(set(row)) != len(row) for row in indices):
            raise ValueError('Duplicate source slots are not allowed')
        if mode in CONSTANTS and not np.all(indices == np.asarray(CONSTANTS[mode])):
            raise ValueError(f'The fixed {mode} definition was changed')
    return spec


def crop(inputs, indices):
    result = dict(inputs)
    result['history'] = inputs['history'][np.arange(len(indices))[:, None], indices]
    return result


@torch.inference_mode()
def forward(model, inputs, device='cpu'):
    tensors = {key: torch.from_numpy(value).to(device) for key, value in inputs.items()}
    # Prediction is always FP32; no AMP or TF32 shortcut is used.
    with torch.autocast(device, enabled=False):
        raw = model(**tensors).float().cpu().numpy()
    if not np.isfinite(raw).all():
        raise RuntimeError('Nonfinite raw source prediction')
    return raw


def load_models(architecture, checkpoint):
    variable_cls, original_cls = model_classes(architecture)
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved_architecture = checkpoint_data.get('config', {}).get('architecture')
    if saved_architecture is not None and saved_architecture != architecture:
        raise ValueError('Checkpoint architecture does not match the requested model')
    model = variable_cls().float().eval()
    original = original_cls().float().eval()
    model.load_state_dict(checkpoint_data['state_dict'], strict=True)
    original.load_state_dict(checkpoint_data['state_dict'], strict=True)
    return model, original


def predict(args):
    opened = guard_inputs_only()
    setup(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, original = load_models(args.architecture, args.checkpoint)
    sample = Dataset(args.root, 'fit', labels=False).batch([0])
    original_cpu = forward(original, sample)
    candidate_cpu = forward(model, sample)
    cpu_difference = float(np.max(np.abs(original_cpu - candidate_cpu)))
    if not np.array_equal(original_cpu, candidate_cpu):
        raise RuntimeError(f'Full9 CPU adapter is not exact: {cpu_difference} K')
    model = model.to(args.device)
    original = original.to(args.device)
    device_difference = float(np.max(np.abs(forward(model, sample, args.device) - forward(original, sample, args.device))))
    if device_difference > (0. if args.device == 'cpu' else 1e-4):
        raise RuntimeError(f'Full9 device adapter mismatch: {device_difference} K')
    del original
    code_files = {Path(__file__), HERE / 'candidate_source_models.py'}
    for cls in model_classes(args.architecture):
        code_files.add(Path(sys.modules[cls.__module__].__file__))
    code_files.update((PACKAGE / 'historylst').rglob('*.py'))
    receipt = dict(
        stage='predictions_only', architecture=args.architecture,
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        manifest_sha256=digest(args.root / 'manifest.json'),
        full9_cpu_max_difference_k=cpu_difference,
        full9_device_max_difference_k=device_difference,
        labels_opened=False, test_opened=False, device=args.device,
        forward_dtype='float32', output_dtype='float64', tf32=False,
        repair='FP64 projection on original query observation support',
        parameters=sum(p.numel() for p in model.parameters()),
        evaluation_batch=args.batch_size, splits={},
        code_sha256={str(path): digest(path) for path in sorted(code_files)},
    )
    dump(args.output / 'prediction_run.json', receipt)
    for split in ('fit', 'validation'):
        data = Dataset(args.root, split, labels=False)
        spec = specification(args.source_specs, split, data)
        for suffix in ('_features.csv', '_subsets.json'):
            shutil.copy2(args.source_specs / (split + suffix), args.output / (split + suffix))
        outdir = args.output / 'predictions' / split
        outdir.mkdir(parents=True, exist_ok=False)
        items = {}
        for mode in MODES:
            path = outdir / (mode + '.npy')
            temporary = outdir / (mode + '.partial.npy')
            result = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64',
                                               shape=(len(data), 1, 160, 160))
            began = time.perf_counter()
            indices = np.asarray(spec['strategies'][mode], dtype=np.int64)
            for start in range(0, len(data), args.batch_size):
                ids = np.arange(start, min(start + args.batch_size, len(data)))
                inputs = crop(data.batch(ids), indices[ids])
                raw = forward(model, inputs, args.device)
                result[start:start + len(ids)] = repair(raw, inputs['coarse'], inputs['support'])
            result.flush()
            del result
            temporary.replace(path)
            items[mode] = dict(path=str(path.relative_to(args.output)),
                               seconds=time.perf_counter() - began, sha256=digest(path),
                               historical_tokens=COUNTS[mode], queries=len(data),
                               shape=[len(data), 1, 160, 160])
            print(json.dumps(dict(split=split, strategy=mode, **items[mode])), flush=True)
        receipt['splits'][split] = items
    receipt['input_table_sha256'] = {split + suffix: digest(args.output / (split + suffix))
        for split in ('fit', 'validation') for suffix in ('_features.csv', '_subsets.json')}
    receipt['opened_data_files'] = sorted(opened)
    receipt['total_seconds'] = time.perf_counter() - started
    verify_payload(args.output, receipt)
    dump(args.output / 'source_predictions_complete.json', receipt)
    print('All five Fit/Val strategies sealed. No labels opened or scores computed.', flush=True)


def verify(output):
    receipt = json.loads((output / 'source_predictions_complete.json').read_text())
    verify_payload(output, receipt)
    return receipt


def verify_payload(output, receipt):
    if receipt['stage'] != 'predictions_only' or receipt['labels_opened'] or receipt['test_opened']:
        raise ValueError('Expected an input-only prediction seal')
    if set(receipt['splits']) != {'fit', 'validation'}:
        raise ValueError('Incomplete split seal')
    for split, items in receipt['splits'].items():
        if set(items) != set(MODES):
            raise ValueError('Incomplete strategy seal')
        for mode, item in items.items():
            path = output / 'predictions' / split / (mode + '.npy')
            if digest(path) != item['sha256']:
                raise ValueError('Prediction hash mismatch')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != item['shape'] or array.dtype != np.float64:
                raise ValueError('Prediction geometry or dtype mismatch')
    expected_tables = {split + suffix for split in ('fit', 'validation')
                       for suffix in ('_features.csv', '_subsets.json')}
    if set(receipt['input_table_sha256']) != expected_tables:
        raise ValueError('Incomplete input-table seal')
    for name, expected in receipt['input_table_sha256'].items():
        if digest(output / name) != expected:
            raise ValueError('Input table changed after prediction')


def mechanical_check(args):
    opened = guard_inputs_only(fit_only=True)
    setup('cpu')
    args.output.mkdir(parents=True, exist_ok=False)
    data = Dataset(args.root, 'fit', labels=False)
    inputs = data.batch([0])
    spec = specification(args.source_specs, 'fit', data)
    results = {}
    for architecture in ('baseline', 'current_query', 'wide'):
        torch.manual_seed(7713)
        variable_cls, original_cls = model_classes(architecture)
        original = original_cls().float().eval()
        torch.nn.init.normal_(original.core.out_conv.weight, std=.05)
        model = variable_cls().float().eval()
        model.load_state_dict(original.state_dict(), strict=True)
        raw_original = forward(original, inputs)
        raw_variable = forward(model, inputs)
        assert np.array_equal(raw_original, raw_variable), architecture
        counts = {}
        for mode in MODES:
            selected = crop(inputs, np.asarray(spec['strategies'][mode][:1], dtype=np.int64))
            seen = []
            hook = model.historical.register_forward_pre_hook(lambda module, arguments: seen.append(arguments[0].shape[0]))
            raw = forward(model, selected)
            hook.remove()
            assert seen == [COUNTS[mode]], (architecture, mode, seen)
            fixed = repair(raw, selected['coarse'], selected['support'])
            support = selected['support'].astype(bool)
            assert np.isfinite(fixed[support]).all() and np.isnan(fixed[~support]).all()
            sums = np.where(support, fixed, 0.).reshape(1, 1, 40, 4, 40, 4).sum((3, 5))
            count = support.reshape(1, 1, 40, 4, 40, 4).sum((3, 5))
            mask = np.isfinite(selected['coarse']) & (count > 0)
            error = float(np.max(np.abs((sums / np.maximum(count, 1))[mask] - selected['coarse'][mask])))
            assert error < 1e-10
            counts[mode] = dict(actual_encoded_history_tokens=seen[0], support_error_k=error)
        empty = dict(inputs, history=np.zeros_like(inputs['history']))
        assert np.isfinite(forward(model, empty)).all()
        results[architecture] = dict(full9_cpu_max_difference_k=0.,
            nonzero_regression_head=True, strict_state_load=True,
            parameters=sum(p.numel() for p in model.parameters()),
            strategies=counts, all_history_missing_finite=True)
        del model, original
    result = dict(status='pass', random_weights_only=True, device='cpu',
                  fit_indices=[0], labels_opened=False, validation_or_test_opened=False,
                  models=results, opened_data_files=sorted(opened))
    dump(args.output / 'mechanical_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('predict', 'check'):
        part = sub.add_parser(name)
        part.add_argument('--root', type=Path, default=PACKAGE)
        part.add_argument('--source-specs', type=Path, default=HERE)
        part.add_argument('--output', type=Path, required=True)
        if name == 'predict':
            part.add_argument('--architecture', choices=('baseline', 'current_query', 'wide'), required=True)
            part.add_argument('--checkpoint', type=Path, required=True)
            part.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
            part.add_argument('--batch-size', type=int, default=2)
    part = sub.add_parser('verify')
    part.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == 'verify':
        verify(args.output)
        print('Prediction seal verified; no scoring performed.')
    else:
        args.root = args.root.resolve()
        args.source_specs = args.source_specs.resolve()
        if args.mode == 'check':
            mechanical_check(args)
        else:
            args.checkpoint = args.checkpoint.resolve()
            if args.batch_size < 1:
                parser.error('--batch-size must be positive')
            predict(args)
