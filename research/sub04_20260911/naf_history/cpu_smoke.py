"""Bounded real Fit4 CPU checks; no validation/Test access or GPU calls."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import resource
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / 'resources/historylst246'
sys.path.insert(0, str(HERE))
from model import HistoryNAFReconstructor, NAFBlock, project


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard():
    opened = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        if 'data' in path.parts and any(split in path.parts for split in ('validation', 'test')):
            raise RuntimeError('This smoke may only read Fit arrays')
        if 'data' in path.parts and path.suffix == '.npy':
            opened.add(str(path.resolve()))
    sys.addaudithook(audit)
    return opened


def upstream_check():
    # Execute only the inspected upstream block/norm classes, avoiding BasicSR
    # imports and all unrelated modules or file access in the full repository.
    namespace = {'torch': torch, 'nn': torch.nn}
    for file, names in (('arch_util.py', {'LayerNormFunction', 'LayerNorm2d'}),
                        ('NAFNet_arch.py', {'SimpleGate', 'NAFBlock'})):
        tree = ast.parse((HERE / 'upstream' / file).read_text())
        selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names], type_ignores=[])
        exec(compile(selected, str(HERE / 'upstream' / file), 'exec'), namespace)
    local = NAFBlock(48).eval()
    with torch.no_grad():
        local.beta.fill_(.3)
        local.gamma.fill_(.2)
    official = namespace['NAFBlock'](48).eval()
    official.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(2, 48, 16, 16)
    with torch.inference_mode():
        a, b = local(x), official(x)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    return float((a - b).abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'cpu_smoke')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    opened = guard()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260923)
    begun = time.perf_counter()
    spec = importlib.util.spec_from_file_location('sub04_original_runner', PACKAGE / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    upstream_difference = upstream_check()
    data = runner.Dataset(PACKAGE, 'fit', labels=True)
    batch = runner.batch(data, np.arange(4), 'cpu')
    inputs = {key: value for key, value in batch.items() if key in runner.INPUTS}
    single = {key: value[:1] for key, value in inputs.items()}
    model = HistoryNAFReconstructor().eval()
    with torch.inference_mode():
        initial = model(**single)
        expected = project(single['fine'][:, :1], single['coarse'], single['support'])
    assert torch.equal(initial, expected)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    rows = []
    parameter_groups = {'history_stem': model.historical, 'local_score': model.fusion[0].score,
                        'naf_middle': model.middle, 'thermal_gain': model.thermal_gain,
                        'current_stem': model.current}
    gradients = {}
    saved_bytes = []
    encoded_counts = []
    handle = model.historical.register_forward_pre_hook(lambda module, values: encoded_counts.append(values[0].shape[0]))
    for step in range(1, 3):
        torch.manual_seed(20260923 + 1000003 * step)
        b = runner.augment(batch, (step - 1) * 3)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        storages = {}
        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in parameter_storage:
                storages[storage.data_ptr()] = storage.nbytes()
            return value
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            prediction = runner.forward(model, b)
            loss = runner.loss_fn(prediction, b)
        saved_bytes.append(sum(storages.values()))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite Fit loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        gradients = {name: sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                     for name, module in parameter_groups.items()}
        optimizer.step()
        rows.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                         gradient_l1=gradients, seconds=time.perf_counter() - begun))
        print(json.dumps(rows[-1]), flush=True)
    handle.remove()
    assert encoded_counts == [36, 36]
    if not all(value > 0 for value in gradients.values()):
        raise RuntimeError('A required neural or thermal path has no gradient after the second update')
    model.eval()
    with torch.inference_mode():
        prediction = model(**single)
        corrupt = dict(single)
        history = single['history'].clone()
        missing = history[:, :, 2:3] == 0
        history[:, :, :2] = torch.where(missing, 1e6, history[:, :, :2])
        history[:, :, 3:4] = torch.where(missing, 1e6, history[:, :, 3:4])
        corrupt['history'] = history
        invalid_value_difference = float((prediction - model(**corrupt)).abs().max())
        assert invalid_value_difference == 0.
        empty = dict(single, history=torch.zeros_like(single['history']))
        empty_prediction = model(**empty)
        empty_but_nonthermal = dict(empty)
        empty_but_nonthermal['history'] = torch.randn_like(single['history'])
        empty_but_nonthermal['history'][:, :, 2] = 0
        empty_but_nonthermal['history'][:, :, 5] = 0
        assert torch.equal(empty_prediction, model(**empty_but_nonthermal))
        empty_fusion = []
        def inspect_empty(module, values, result):
            empty_fusion.append(bool(torch.equal(result[0], values[0]) and torch.count_nonzero(result[1]) == 0))
        handles = [fusion.register_forward_hook(inspect_empty) for fusion in model.fusion]
        model(**empty)
        for item in handles:
            item.remove()
        assert empty_fusion == [True] * 4
        model.train()
        model.emissivity_dropout = 0.
        rng_states = []
        for rate in (0., .25, 1.):
            model.history_dropout = rate
            torch.manual_seed(73129)
            dropped = model(**single)
            rng_states.append(torch.get_rng_state())
        assert all(torch.equal(rng_states[0], item) for item in rng_states)
        assert torch.equal(dropped, empty_prediction)
        model.history_dropout = .25
        model.emissivity_dropout = .25
        model.eval()
    zero_outside = bool(torch.count_nonzero(prediction[~single['support'].bool()]) == 0)
    repaired = runner.repair(prediction.numpy(), single['coarse'].numpy(), single['support'].numpy())
    p = repaired.reshape(1, 1, 40, 4, 40, 4)
    m = single['support'].numpy().reshape(1, 1, 40, 4, 40, 4)
    means = np.where(m, p, 0.).sum((3, 5)) / m.sum((3, 5)).clip(1)
    observed = np.isfinite(single['coarse'].numpy()) & (m.sum((3, 5)) > 0)
    closure = float(np.max(np.abs(means[observed] - single['coarse'].numpy()[observed])))
    assert closure < 1e-10 and zero_outside and torch.isfinite(prediction).all()
    checkpoint = dict(state_dict=runner.state(model), optimizer=optimizer.state_dict(), updates=2,
                      history_dropout=.25, emissivity_dropout=.25, smoke_only=True)
    runner.save(args.output / 'smoke_checkpoint.pt', checkpoint)
    restored = HistoryNAFReconstructor().eval()
    loaded = torch.load(args.output / 'smoke_checkpoint.pt', map_location='cpu', weights_only=False)
    restored.load_state_dict(loaded['state_dict'], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=1e-4)
    restored_optimizer.load_state_dict(loaded['optimizer'])
    with torch.inference_mode():
        restored_difference = float((prediction - restored(**single)).abs().max())
    assert restored_difference == 0.
    result = dict(status='pass', device='cpu', fit_only=True, fit_scene_indices=[0, 1, 2, 3],
                  validation_opened=False, test_opened=False, gpu_used=False,
                  parameters=sum(p.numel() for p in model.parameters()),
                  architecture='HistoryNAFReconstructor', batch_size=4, successful_updates=2,
                  actual_encoded_history_counts=encoded_counts, upstream_nafblock_max_abs_difference=upstream_difference,
                  zero_initialized_output_equals_projected_base=True, steps=rows,
                  finite_loss_gradients=True, required_path_gradient_l1_after_step2=gradients,
                  invalid_thermal_value_max_abs_difference_k=invalid_value_difference,
                  empty_source_injections_exact_zero=empty_fusion,
                  dropout_rates_zero_quarter_one_consume_identical_rng=True,
                  whole_history_dropout_one_equals_empty_history=True,
                  output_zero_outside_support=zero_outside, fp64_repair_max_observed_mean_error_k=closure,
                  checkpoint_prediction_max_abs_difference_k=restored_difference,
                  saved_nonparameter_storage_bytes_fp32_forward=saved_bytes,
                  memory_measurement_caveat='CPU autograd saved storage, not CUDA allocation, peak residency, or an AMP cost estimate.',
                  process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  opened_data_files=sorted(opened), seconds=time.perf_counter() - begun,
                  model_sha256=sha(HERE / 'model.py'), smoke_sha256=sha(Path(__file__)),
                  upstream=json.loads((HERE / 'upstream/UPSTREAM.json').read_text()))
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
