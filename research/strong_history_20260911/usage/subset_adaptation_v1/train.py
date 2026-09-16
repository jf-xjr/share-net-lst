"""Matched Fit-only source exposure continuation; validation never opens Test.

All training math is the original runner's math. The only arm difference is
actual K=3 history cropping on half the mixed arm's predetermined batches.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
USAGE = HERE.parent
EXPERIMENT = USAGE.parent
ROOT = EXPERIMENT.parents[1]
PACKAGE = ROOT / 'resources/historylst246'
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(USAGE))
from experiment import runner
from candidate_source_models import model_classes
from predict_candidate_sources import specification

ACTIONS = ('full9', 'greedy3')
SELECTIONS = {'joint': ('best.pt', 'mean_full9_greedy3_macro_rmse'),
              'full9': ('best_full9.pt', 'full9_macro_rmse'),
              'greedy3': ('best_greedy3.pt', 'greedy3_macro_rmse')}


def config(args):
    return dict(architecture=args.architecture, arm=args.arm, seed=args.seed,
                updates=3000, batch_size=4, lr=1e-4, warmup=100,
                weight_decay=1e-4, validation_interval=250, evaluation_batch=2,
                history_dropout=.25, emissivity_dropout=.25,
                selection_metric=SELECTIONS['joint'][1],
                initialization_checkpoint=str(args.checkpoint),
                initialization_sha256=runner.digest(args.checkpoint))


def guard(fit_only):
    opened = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        if 'data' not in path.parts:
            return
        if 'test' in path.parts or (fit_only and 'validation' in path.parts):
            raise RuntimeError('This entry point cannot open Test; smoke is Fit-only')
        if path.suffix == '.npy':
            opened.add(str(path.resolve()))
    sys.addaudithook(audit)
    return opened


def schedule(data, cfg):
    rng = np.random.default_rng(cfg['seed'])
    probability = data.sampling_probabilities()
    ids, codes = [], []
    for _ in range(cfg['updates']):
        ids.append(rng.choice(len(data), cfg['batch_size'], p=probability))
        codes.append(int(rng.integers(8)))
    # This RNG never advances the original Fit-sampling/D4 random stream.
    source_rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 73129]))
    partial = np.repeat(np.array([False, True]), cfg['updates'] // 2)
    source_rng.shuffle(partial)
    return np.asarray(ids, dtype=np.int64), np.asarray(codes, dtype=np.int64), partial


def schedule_receipt(plan):
    ids, codes, partial = plan
    return dict(fit_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                d4_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),
                independent_source_schedule_sha256=hashlib.sha256(partial.tobytes()).hexdigest(),
                source_schedule_counts=dict(full9=int((~partial).sum()), greedy3=int(partial.sum())),
                first_five=[dict(ids=i.tolist(), d4=int(c), scheduled_action='greedy3' if p else 'full9')
                            for i, c, p in zip(ids[:5], codes[:5], partial[:5])])


def crop_batch(batch, indices, action):
    if action == 'full9':
        return batch
    out = dict(batch)
    chosen = torch.as_tensor(indices, dtype=torch.long, device=batch['history'].device)
    out['history'] = batch['history'][torch.arange(len(chosen), device=chosen.device)[:, None], chosen].contiguous()
    if out['history'].shape[1] != 3:
        raise RuntimeError('greedy3 must physically contain exactly three history tokens')
    return out


def source_seal(args):
    paths = {Path(__file__).resolve(), HERE / 'design.json', EXPERIMENT / 'experiment.py',
             USAGE / 'candidate_source_models.py', USAGE / 'predict_candidate_sources.py',
             PACKAGE / 'run.py', PACKAGE / 'manifest.json'}
    paths.update((PACKAGE / 'historylst').rglob('*.py'))
    paths.update((PACKAGE / 'third_party').rglob('*.py'))
    paths.add(args.source_completion or args.checkpoint.parent / 'complete.json')
    paths.update(EXPERIMENT / 'candidates' / name for name in
                 ('flexible_utae.py', 'current_query.py', 'network_review.py'))
    for split in ('fit',) if args.mode == 'smoke' else ('fit', 'validation'):
        paths.update(USAGE / f'{split}_{name}' for name in ('features.csv', 'subsets.json'))
    return {str(p): runner.digest(p) for p in sorted(paths)}


def verify_seal(seal, checkpoint, checkpoint_sha):
    for path, expected in seal.items():
        if runner.digest(path) != expected:
            raise RuntimeError(f'Source changed after run initialization: {path}')
    if runner.digest(checkpoint) != checkpoint_sha:
        raise RuntimeError('Initialization checkpoint changed after run initialization')


def initialization(args, data):
    completion_path = args.source_completion or args.checkpoint.parent / 'complete.json'
    completion = json.loads(completion_path.read_text())
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('smoke_only') or completion.get('status') != 'complete':
        raise ValueError('Initialization requires an explicitly completed, non-smoke training run')
    if completion.get('updates') != ck.get('config', {}).get('updates') or not 0 < ck['step'] <= completion['updates']:
        raise ValueError('Initialization checkpoint step/config does not match the completed source run')
    saved_architecture = ck.get('config', {}).get('architecture', 'baseline')
    if saved_architecture != args.architecture:
        raise ValueError('Initialization architecture does not match --architecture')
    dynamic_cls, original_cls = model_classes(args.architecture)
    model = dynamic_cls().float().eval()
    original = original_cls().float().eval()
    model.load_state_dict(ck['state_dict'], strict=True)
    original.load_state_dict(ck['state_dict'], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, ck['state_dict'][name]):
            raise RuntimeError('Initialization state mismatch')
    # Use Fit inputs only, before any optimization and always on CPU.
    inputs = {k: torch.from_numpy(np.array(data.arrays[k][0:1], copy=True)) for k in runner.INPUTS}
    with torch.inference_mode():
        reference = runner.forward(original, inputs)
        actual = runner.forward(model, inputs)
    if not torch.isfinite(actual).all() or not torch.equal(reference, actual):
        raise RuntimeError('Dynamic full9 does not exactly equal the original class on CPU')
    receipt = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=runner.digest(args.checkpoint),
                   source_completion=str(completion_path.resolve()), source_completion_sha256=runner.digest(completion_path),
                   source_step=ck['step'], source_weights=ck.get('weights'),
                   full9_original_cpu_max_abs_difference_k=0., strict_state_equality=True,
                   fresh_optimizer_ema_scaler=True)
    return model, receipt


@torch.inference_mode()
def inference(model, data, spec, action, device, batch_size):
    model.eval()
    parts = []
    for start in range(0, len(data), batch_size):
        ids = np.arange(start, min(start + batch_size, len(data)))
        b = crop_batch(runner.batch(data, ids, device), spec[ids], action)
        with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
            p = runner.forward(model, b)
        parts.append(p.float().cpu().numpy())
    return runner.repair(np.concatenate(parts), data.arrays['coarse'], data.arrays['support'])


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        if not torch.equal(left.cpu(), right.cpu()):
            raise RuntimeError('Checkpoint tensor did not restore exactly')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError('Checkpoint keys changed during restore')
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError('Checkpoint sequence changed during restore')
        for x, y in zip(left, right):
            assert_tree_equal(x, y)
    elif left != right:
        raise RuntimeError('Checkpoint scalar changed during restore')


def restore_last(ck, model, ema, optimizer, scaler, cfg, seal, plan_receipt, smoke):
    if ck['config'] != cfg or ck['source_sha256'] != seal or ck['schedule'] != plan_receipt or ck['smoke_only'] != smoke:
        raise ValueError('Resume config, source seal, schedule, or smoke/formal mode mismatch')
    model.load_state_dict(ck['state_dict'], strict=True)
    ema.load_state_dict(ck['ema'], strict=True)
    optimizer.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler'])
    for expected, restored in ((ck['state_dict'], model.state_dict()), (ck['ema'], ema.state_dict()),
                               (ck['optimizer'], optimizer.state_dict()), (ck['scaler'], scaler.state_dict())):
        assert_tree_equal(expected, restored)


def run(args):
    smoke = args.mode == 'smoke'
    if smoke and (args.device != 'cpu' or args.resume):
        raise ValueError('Smoke is CPU-only and must use a fresh output directory')
    opened = guard(smoke)
    runner.setup(args.device)
    cfg = config(args)
    seal = source_seal(args)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    data = runner.Dataset(PACKAGE, 'fit', labels=True)
    fit_spec = np.asarray(specification(USAGE, 'fit', data)['strategies']['greedy3'])
    plan = schedule(data, cfg)
    repeated_plan = schedule(data, cfg)
    if not all(np.array_equal(a, b) for a, b in zip(plan, repeated_plan)):
        raise RuntimeError('Sampling/D4/source schedule is not reproducible')
    plan_receipt = schedule_receipt(plan)
    if smoke and args.arm == 'mixed' and len(set(plan[2][:5].tolist())) != 2:
        raise ValueError('This smoke seed does not exercise both K=3/9 in five scheduled updates')
    model, initialized = initialization(args, data)
    model = model.to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    val = None if smoke else runner.Dataset(PACKAGE, 'validation', labels=True)
    val_spec = None if smoke else np.asarray(specification(USAGE, 'validation', val)['strategies']['greedy3'])
    best = dict.fromkeys(SELECTIONS, None)
    rows, start_step, previous_seconds = [], 0, 0.
    if args.resume:
        if (args.output / 'complete.json').exists() or (args.output / 'smoke.json').exists():
            raise FileExistsError('Completed runs cannot resume')
        runtime = json.loads((args.output / 'run.json').read_text())
        if runtime['device'] != args.device or runtime['initialization'] != initialized:
            raise ValueError('Resume device or initialization receipt changed')
        ck = torch.load(args.output / 'last.pt', map_location=args.device, weights_only=False)
        restore_last(ck, model, ema, optimizer, scaler, cfg, seal, plan_receipt, smoke)
        rows, best, start_step = ck['validation'], ck['best'], ck['step']
        previous_seconds = ck['elapsed_seconds']
        # A crash during validation must not silently create extra selections.
        if json.loads((args.output / 'validation.json').read_text()) != rows:
            raise ValueError('Validation and last checkpoint differ; manual review is required')
        for key, (filename, metric) in SELECTIONS.items():
            selected = torch.load(args.output / filename, map_location='cpu', weights_only=False)
            if selected['selection_metric'] != metric or selected['selection_score'] != best[key]:
                raise ValueError('Selected checkpoint and last checkpoint differ; manual review is required')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        runtime = dict(config=cfg, device=args.device, torch=str(torch.__version__),
                       initialization=initialized, source_sha256=seal, schedule=plan_receipt,
                       parameters=sum(p.numel() for p in model.parameters()), fit_queries=len(data),
                       validation_queries=0 if smoke else len(val), test_opened=False, smoke_only=smoke,
                       planned_weight_candidates=0 if smoke else 26,
                       planned_action_scores=0 if smoke else 52)
        runner.dump(args.output / 'run.json', runtime)
        np.savez(args.output / 'schedule.npz', fit_ids=plan[0], d4=plan[1], scheduled_greedy3=plan[2])
    began = time.perf_counter()
    def elapsed():
        return previous_seconds + time.perf_counter() - began
    def selected_checkpoint(net, step, weights, values, metric, value):
        return dict(state_dict=runner.state(net), step=step, weights=weights, config=cfg,
                    validation_macro_rmse=values, selection_metric=metric, selection_score=value,
                    source_sha256=seal, initialization=initialized, smoke_only=smoke)
    def save_last(step):
        runner.save(args.output / 'last.pt', dict(state_dict=runner.state(model), ema=runner.state(ema),
                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step,
                    best=best, validation=rows, config=cfg, source_sha256=seal, schedule=plan_receipt,
                    elapsed_seconds=elapsed(), smoke_only=smoke))
    def validate(step):
        verify_seal(seal, args.checkpoint, cfg['initialization_sha256'])
        for name, net in (('raw', model), ('ema', ema)):
            values = {}
            for action in ACTIONS:
                prediction = inference(net, val, val_spec, action, args.device, cfg['evaluation_batch'])
                values[action] = runner.score(prediction, val.arrays['target'], val.arrays['formal'], val.records)['macro']['rmse']
            scores = dict(values, joint=float(np.mean(list(values.values()))))
            row = dict(step=step, weights=name, validation_macro_rmse=values,
                       selection_metric=SELECTIONS['joint'][1], selection_score=scores['joint'], seconds=elapsed())
            rows.append(row)
            for key, (filename, metric) in SELECTIONS.items():
                if best[key] is None or scores[key] < best[key]:
                    best[key] = scores[key]
                    runner.save(args.output / filename, selected_checkpoint(net, step, name, values, metric, scores[key]))
            print(json.dumps(dict(validation=row, best_scores=best)), flush=True)
        runner.dump(args.output / 'validation.json', rows)
        save_last(step)
    if val is not None and not args.resume:
        validate(0)
    ids, codes, partial = plan
    smoke_rows = []
    max_steps = 5 if smoke else cfg['updates']
    # A mechanical D4 check does not consume either training random stream.
    if smoke:
        raw_batch = runner.batch(data, ids[0], 'cpu')
        a, b = runner.augment(raw_batch, int(codes[0])), runner.augment(raw_batch, int(codes[0]))
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
        del raw_batch, a, b
    for step in range(start_step + 1, max_steps + 1):
        action = 'greedy3' if args.arm == 'mixed' and partial[step - 1] else 'full9'
        for attempt in range(8):
            torch.manual_seed(cfg['seed'] + 1000003 * step)
            if args.device == 'cuda':
                torch.cuda.manual_seed_all(cfg['seed'] + 1000003 * step)
            b = runner.augment(runner.batch(data, ids[step - 1], args.device), int(codes[step - 1]))
            b = crop_batch(b, fit_spec[ids[step - 1]], action)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            factor = step / cfg['warmup'] if step <= cfg['warmup'] else .05 + .95 * .5 * (1 + math.cos(math.pi * (step - cfg['warmup']) / (cfg['updates'] - cfg['warmup'])))
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * factor
            with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == 'cuda'):
                prediction = runner.forward(model, b)
                loss = runner.loss_fn(prediction, b)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                if not scaler.is_enabled():
                    raise RuntimeError('Nonfinite gradient')
                scaler.update(new_scale=scaler.get_scale() * .5)
                continue
            scaler.step(optimizer)
            scaler.update()
            break
        else:
            raise RuntimeError('Eight AMP backoffs exhausted')
        decay = min(.995, (1 + step) / (10 + step))
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1 - decay)
            for eb, buf in zip(ema.buffers(), model.buffers()):
                eb.copy_(buf)
        if step % 50 == 0 or smoke:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), action=action,
                       actual_history_k=b['history'].shape[1], lr=optimizer.param_groups[0]['lr'],
                       amp_attempts=attempt + 1, seconds=elapsed())
            print(json.dumps(row), flush=True)
            with (args.output / 'progress.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')
            if smoke:
                smoke_rows.append(row)
        if val is not None and step % cfg['validation_interval'] == 0:
            validate(step)
    verify_seal(seal, args.checkpoint, cfg['initialization_sha256'])
    if smoke:
        # Serialization fixtures are explicitly not selected by a Val score.
        runner.save(args.output / 'best.pt', selected_checkpoint(model, 5, 'raw', None, 'not_selected_fit_only_smoke', None))
        save_last(5)
        dynamic_cls, _ = model_classes(args.architecture)
        restored = dynamic_cls().float().eval()
        restored_ema = copy.deepcopy(restored).requires_grad_(False)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        restored_scaler = torch.amp.GradScaler('cuda', enabled=False)
        ck = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        restore_last(ck, restored, restored_ema, restored_optimizer, restored_scaler, cfg, seal, plan_receipt, True)
        best_ck = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
        restored_best = dynamic_cls().float().eval()
        restored_best.load_state_dict(best_ck['state_dict'], strict=True)
        model.eval()
        with torch.inference_mode():
            for action in ACTIONS:
                b = crop_batch(runner.batch(data, np.array([0]), 'cpu'), fit_spec[:1], action)
                reference = runner.forward(model, b)
                if not torch.equal(reference, runner.forward(restored, b)) or not torch.equal(reference, runner.forward(restored_best, b)):
                    raise RuntimeError('Best/last checkpoint predictions did not restore exactly')
        report = dict(status='smoke_complete', arm=args.arm, updates=5, device='cpu', fit_only=True,
                      test_opened=False, validation_opened=False, finite_loss_and_gradients=True,
                      actual_history_counts=sorted({r['actual_history_k'] for r in smoke_rows}),
                      sampling_d4_source_schedule_reproducible=True, d4_tensor_check_exact=True,
                      best_last_model_ema_optimizer_scaler_restored=True,
                      restored_full9_greedy3_predictions_max_abs_difference_k=0.,
                      best_is_nonselected_serialization_fixture=True, initialization=initialized,
                      opened_data_files=sorted(opened), seconds=elapsed())
        runner.dump(args.output / 'smoke.json', report)
    else:
        expected = [(step, weight) for step in range(0, 3001, 250) for weight in ('raw', 'ema')]
        if [(row['step'], row['weights']) for row in rows] != expected:
            raise RuntimeError('Unexpected validation weight candidates')
        report = dict(status='complete', updates=3000, arm=args.arm, seconds=elapsed(),
                      parameters=runtime['parameters'], validation_weight_candidates=len(rows),
                      validation_action_scores=2 * len(rows), best_scores=best,
                      best_selection_metric=SELECTIONS['joint'][1],
                      test_opened=False, no_automatic_usage_scoring=True)
        runner.dump(args.output / 'complete.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'smoke'))
    parser.add_argument('--architecture', required=True, choices=('baseline', 'current_query', 'wide'))
    parser.add_argument('--arm', required=True, choices=('full_control', 'mixed'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-completion', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.source_completion:
        args.source_completion = args.source_completion.resolve()
    run(args)


if __name__ == '__main__':
    main()
