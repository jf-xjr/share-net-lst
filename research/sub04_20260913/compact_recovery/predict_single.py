"""One frozen compact/full NAF or matched U-TAE; six original inputs, no ensemble.

Supply either --inputs-npz with exactly fine/coarse/support/context/emissivity/history
or the original input-only --split and --scene-index interface. No deadline is
applied to deployment. The selected family and checkpoint come from the freeze.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('single_recovery_reader', HERE / 'final_reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
original = reader.p.runner
INPUTS = tuple(original.INPUTS)


def validate_inputs(sample):
    if set(sample) != set(INPUTS):
        raise ValueError('Exactly six original fields required; labels and extra arrays are rejected')
    fine = sample['fine']
    if fine.ndim != 4 or fine.shape[0] != 1 or fine.shape[1] != 52:
        raise ValueError('fine must have shape [1,52,H,W]; inference is batch one')
    h, w = fine.shape[-2:]
    if min(h, w) < 16 or h % 8 or w % 8:
        raise ValueError('Spatial dimensions must be at least16 and divisible by8')
    shapes = dict(fine=(1,52,h,w), coarse=(1,1,h//4,w//4),
                  support=(1,1,h,w), context=(1,15),
                  emissivity=(1,4,h,w), history=(1,9,9,h,w))
    for name, shape in shapes.items():
        value = sample[name]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError('Original input shape differs: ' + name)
        expected = np.dtype(bool if name == 'support' else np.float32)
        if value.dtype != expected:
            raise ValueError('Original input dtype differs: ' + name + '; expected ' + str(expected))
        if name == 'coarse':
            if np.isinf(value).any():
                raise ValueError('coarse may contain missing NaN observations, never infinity')
        elif name != 'support' and not np.isfinite(value).all():
            raise ValueError('Nonfinite encoded input: ' + name)
    return {name: np.ascontiguousarray(sample[name]) for name in INPUTS}


def infer(model, sample, device):
    sample = validate_inputs(sample)
    model = model.float().to(device).eval()
    with torch.inference_mode():
        batch = {k: torch.from_numpy(v).to(device) for k, v in sample.items()}
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            output = original.forward(model, batch)
        if output.dtype != torch.float32 or tuple(output.shape) != sample['support'].shape:
            raise ValueError('Expected one FP32 prediction at original spatial support')
        prediction = original.repair(output.cpu().numpy(), sample['coarse'], sample['support'])
    mask = sample['support']
    if prediction.dtype != np.float64 or not np.isfinite(prediction[mask]).all() or not np.isnan(prediction[~mask]).all():
        raise ValueError('Original repaired finite-support / missing-support convention failed')
    return prediction


def source_binding():
    paths = [Path(__file__), HERE / 'final_loader.py', HERE / 'final_reader.py',
             HERE / 'train.py', HERE / 'model.py', Path(original.__file__),
             reader.PACKAGE / 'historylst/data.py', reader.PACKAGE / 'historylst/metrics.py',
             reader.PACKAGE / 'historylst/model.py', reader.NEW / 'multihead_fusion/model.py',
             reader.OLD / 'naf_history/model.py', reader.OLD / 'dropout_models.py']
    return {str(p.resolve()): reader.sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--architecture', choices=('naf_history', 'baseline'), required=True)
    parser.add_argument('--seed', type=int, choices=reader.p.SEEDS, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--inputs-npz', type=Path)
    inputs.add_argument('--split', choices=('fit', 'validation', 'test'))
    parser.add_argument('--scene-index', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.npy':
        parser.error('--output must end in .npy; receipt is the adjacent .json')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Existing predictions are preserved; use a new output')
    if (args.split is None) != (args.scene_index is None):
        parser.error('--scene-index is required with --split and forbidden with --inputs-npz')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    frozen = reader.read_freeze(args.freeze)
    freeze_binding = reader.binding(args.freeze)
    entry, = [e for e in frozen['checkpoints'] if (e['architecture'], e['seed']) == (args.architecture, args.seed)]
    sources = source_binding()
    if args.inputs_npz:
        input_binding = reader.binding(args.inputs_npz)
        with np.load(args.inputs_npz, allow_pickle=False) as data:
            sample = {k: np.array(data[k], copy=True) for k in data.files}
        identity = dict(inputs_npz=input_binding, split=None, scene_id=None)
    else:
        data = reader.p.Dataset(reader.PACKAGE, args.split, labels=False)
        if not 0 <= args.scene_index < len(data):
            raise IndexError('Scene index outside the chosen input split')
        sample = data.batch([args.scene_index])
        identity = dict(split=args.split, scene_index=args.scene_index,
                        scene_id=data.records[args.scene_index]['scene_id'],
                        manifest=reader.binding(reader.PACKAGE / 'manifest.json'))
    sample = validate_inputs(sample)
    tensor_bindings = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                       sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for k, v in sample.items()}
    model = reader.loader.load_model(entry, args.device)
    began = time.perf_counter()
    prediction = infer(model, sample, args.device)
    seconds = time.perf_counter() - began
    if (sources != source_binding() or reader.sha(entry['checkpoint']) != entry['checkpoint_sha256']
            or reader.binding(args.freeze) != freeze_binding):
        raise ValueError('Bound source or selected checkpoint changed during inference')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.save(stream, prediction, allow_pickle=False)
    receipt = dict(status='one_frozen_single_model_prediction_complete',
        freeze=freeze_binding, architecture=args.architecture, seed=args.seed,
        selected_family=frozen['selected_family'], checkpoint=reader.binding(entry['checkpoint']),
        parameters=sum(p.numel() for p in model.parameters()), effective_method=reader.METHOD,
        device=args.device, network_calls=1, image_forwards=1, inference_views=1,
        teacher_used_at_inference=False, ensemble_used=False, labels_opened=False,
        dtype='float32', tf32=False, autocast=False, repair_dtype='float64',
        output=reader.binding(args.output), output_shape=list(prediction.shape),
        input_fields=tensor_bindings, input_identity=identity, source_sha256=sources,
        elapsed_seconds_including_input_transfer_and_repair=seconds,
        timing_is_benchmark=False, metrics_computed=False)
    reader.write(args.output.with_suffix('.json'), receipt)
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
