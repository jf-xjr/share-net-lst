"""Explicit single fixed-D4-teacher self-distillation pilot, never a queue. check uses synthetic CPU tensors only."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import math
import subprocess
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB04 = HERE.parent
sys.path.insert(0, str(HERE))


def load_bridge():
    spec = importlib.util.spec_from_file_location('d4_self_distillation_model_bridge', SUB04 / 'subset_adaptation_bridge/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_option(argv, name):
    values=[]
    for index,item in enumerate(argv):
        if item==name:
            if index+1>=len(argv):raise ValueError('Missing companion CLI value')
            values.append(argv[index+1])
        elif item.startswith(name+'='):values.append(item[len(name)+1:])
    if len(values)!=1:raise ValueError('Companion CLI option missing or duplicated: '+name)
    return values[0]


def validate_companion_command(argv, cwd, args, entry):
    # Accept Python -u/-B launch prefixes, but require the actual script slot.
    index=1
    while index<len(argv) and argv[index] in ('-u','-B'):index+=1
    def absolute(value):
        path=Path(value)
        return (cwd/path).resolve() if not path.is_absolute() else path.resolve()
    if index>=len(argv) or absolute(argv[index])!=entry.resolve() or argv[index+1:index+2]!=['train']:
        raise RuntimeError('GPU process is not the exact shared D4 train entry')
    opts=argv[index+2:]
    expected_arch='baseline' if args.architecture=='naf_history' else 'naf_history'
    if (absolute(command_option(opts,'--output'))!=args.companion_output
            or absolute(command_option(opts,'--companion-output'))!=args.output
            or command_option(opts,'--architecture')!=expected_arch
            or int(command_option(opts,'--original-seed'))!=args.original_seed
            or command_option(opts,'--device')!='cuda'):
        raise RuntimeError('GPU process is not the exact reciprocal planned same-seed companion')
    return True


def admit_shared_gpu(args):
    plan_path=HERE/'planned_runs.json';plan=json.loads(plan_path.read_text())
    items={(r['architecture'],r['seed']):r for r in plan['runs']}
    other='baseline' if args.architecture=='naf_history' else 'naf_history'
    if (args.mode!='train' or args.device!='cuda' or args.original_seed not in (20260912,20260913)
            or args.output!=Path(items[(args.architecture,args.original_seed)]['run']).resolve()
            or args.companion_output!=Path(items[(other,args.original_seed)]['run']).resolve()):
        raise ValueError('Only the exact planned same-seed 912/913 architecture pair may share this GPU')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    admitted=[]
    for value in sorted(set(active.splitlines())):
        value=value.strip()
        if not value:continue
        if not value.isdigit():raise RuntimeError('Unrecognized active GPU PID')
        pid=int(value)
        if pid==os.getpid():continue
        try:
            proc=Path('/proc')/str(pid)
            argv=[part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
            cwd=(proc/'cwd').resolve(strict=True)
            validate_companion_command(argv,cwd,args,Path(__file__))
        except (OSError,ValueError,RuntimeError) as exc:
            raise RuntimeError('Refusing unverified GPU occupant PID '+value) from exc
        admitted.append(dict(pid=pid,output=str(args.companion_output),architecture=other,seed=args.original_seed))
    if len(admitted)>1:raise RuntimeError('More than one companion process occupies the GPU')
    return dict(mode='same_gpu_planned_architecture_pair',companion_output=str(args.companion_output),
                verified_active_companions_at_admission=admitted,process_pid=os.getpid(),
                same_seed=args.original_seed,per_run_limit_seconds=540,
                scientific_recipe_unchanged=True,isolated_timing=False,
                overlap_scope='Admission snapshot only; both jobs independently retain RNG and are externally launched')


def run(args):
    execution_start_unix=time.time()
    started = time.perf_counter()
    execution_context=admit_shared_gpu(args)
    design = json.loads((HERE / 'design.json').read_text())
    deadline = design['hard_deadline_unix']
    from teacher import teacher_receipt, mixed_loss
    package = SUB04.parents[1] / 'resources/historylst246'
    teacher_info = teacher_receipt(args.teacher_receipt, package)
    feasibility=json.loads(Path(design['teacher_feasibility']['path']).read_text())
    if (not feasibility['teacher_feasibility_gate_pass'] or not feasibility['macro']['d4mean']['rmse']<.42
            or feasibility['macro']['code0']['rmse']-feasibility['macro']['d4mean']['rmse']<.003):
        raise ValueError('Actual teacher feasibility must pass first')
    freeze_path = SUB04 / 'final_delivery_20260912/network_freeze.json'
    import hashlib
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(freeze_path) != design['network_freeze_sha256']:
        raise ValueError('Final alpha25 reference freeze changed')
    frozen = json.loads(freeze_path.read_text())
    score_paths = [Path(p) for p in frozen['evidence_sha256'] if p.endswith('/alpha0.25/naf_history_20260905_scores.json')]
    if len(score_paths) != 1 or digest(score_paths[0]) != frozen['evidence_sha256'][str(score_paths[0])]:
        raise ValueError('Exact frozen alpha25_905 Val score required')
    reference_score = json.loads(score_paths[0].read_text())
    reference_rmse = reference_score['macro']['rmse']
    if len(reference_score['scenes']) != 45:
        raise ValueError('Frozen reference must cover all Val45')
    source = next(x for x in design['original_sources'] if (x['architecture'], x['seed']) == (args.architecture,args.original_seed))
    if (str(args.checkpoint) != source['checkpoint'] or str(args.source_completion) != source['completion']
            or digest(args.checkpoint) != source['checkpoint_sha256'] or digest(args.source_completion) != source['completion_sha256']):
        raise ValueError('Only each architecture/seed exact original best and real completion accepted')
    is_pilot = (args.architecture,args.original_seed) == ('naf_history',20260905)
    def check_deadline():
        if time.time() >= deadline or time.perf_counter()-started >= 540.:
            raise TimeoutError('13:00 UTC hard cutoff or 540-second shared run budget reached')
    check_deadline()
    import hashlib
    for path, expected_digest in design['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f'Declared unchanged model/training source changed: {path}')
    pilot_binding = None
    if (args.architecture, args.original_seed) != ('naf_history', 20260905):
        if args.pilot_completion is None:
            raise ValueError('Baseline/other seeds require the actually successful original NAF905 D4-teacher self-distillation pilot')
        pilot = json.loads(args.pilot_completion.read_text())
        pilot_run = json.loads((args.pilot_completion.parent / 'run.json').read_text())
        if (pilot.get('status') != 'complete' or pilot.get('updates') != 1000
                or pilot.get('validation_weight_candidates') != 10
                or pilot.get('continuation_gate_pass') is not True
                or pilot_run['config'].get('architecture') != 'naf_history'
                or pilot_run['config'].get('original_seed') != 20260905
                or pilot_run['config'].get('protocol') != 'd4_self_distillation_20260912_v1'
                or not pilot['selected_fp32']['macro']['rmse'] < .42
                or reference_rmse - pilot['selected_fp32']['macro']['rmse'] < .003
                or pilot.get('locked_alpha25_reference_rmse_k') != reference_rmse
                or digest(args.pilot_completion.parent / 'best.pt') != pilot['selected_checkpoint_sha256']
                or pilot_run['source_sha256'].get(str(HERE / 'design.json')) != digest(HERE / 'design.json')
                or pilot_run['teacher_receipt_sha256'] != digest(args.teacher_receipt)
                or pilot_run['teacher']['teacher'] != teacher_info['teacher']):
            raise ValueError('Actual complete NAF905 pilot has not passed the fixed continuation gate')
        pilot_binding = args.pilot_completion
    smoke = args.mode == 'smoke'
    bridge = load_bridge()
    v1 = bridge.load_v1()
    opened = v1.guard(smoke)
    runner = v1.runner
    runner.setup(args.device)
    args.arm = 'full_control'
    cfg = v1.config(args)
    cfg.update(updates=1000, algorithm='AdamW_fused_cuda', optimizer_fused=(args.device=='cuda'), original_seed=args.original_seed,
               selection_metric='full9_macro_rmse', training_forward_passes_per_finite_update=1,
               history_dropout=.25, emissivity_dropout=.25, spatial_training='all_full160',
               hard_deadline_unix=deadline, per_run_limit_seconds=540, execution_mode='same_gpu_planned_architecture_pair', companion_output=str(args.companion_output), ground_truth_loss_weight=.5, teacher_loss_weight=.5, teacher_cache_sha256=teacher_info['teacher']['sha256'], protocol='d4_self_distillation_20260912_v1')
    if any(cfg.get(key) != value for key, value in design['recipe'].items()):
        raise ValueError('Actual continuation configuration differs from the declared single recipe')
    completion = json.loads(args.source_completion.read_text())
    original_ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if (completion.get('status') != 'complete' or completion.get('updates') != 18000
            or completion.get('validation_candidates') != 50
            or original_ck.get('smoke_only')
            or original_ck['config'].get('seed') != args.original_seed
            or original_ck['config'].get('architecture', 'baseline') != args.architecture
            or original_ck['config'].get('updates') != 18000):
        raise ValueError('Use this architecture/seed original completed 18k/50 best checkpoint')
    for key in ('history_dropout', 'emissivity_dropout'):
        if original_ck['config'].get(key, .25) != .25:
            raise ValueError('Initialization must have the original .25/.25 dropout')
    seal = v1.source_seal(args)
    seal.update(design['source_sha256'])
    for path in (Path(__file__).resolve(), HERE / 'design.json',
                 SUB04 / 'four_hour_deadline_20260912.json', SUB04 / 'dropout_models.py',
                 SUB04 / 'naf_history/model.py', Path(bridge.__file__), HERE / 'teacher.py', args.teacher_receipt,
                 HERE / 'synthetic_check.py', HERE/'shared_execution_design.json', HERE/'planned_runs.json', freeze_path, score_paths[0],
                 v1.PACKAGE / 'provenance/optical_normalization.json',
                 v1.PACKAGE / 'provenance/weather_normalization.json'):
        seal[str(path)] = runner.digest(path)
    seal.update(teacher_info['source_sha256'])
    if pilot_binding is not None:
        seal[str(pilot_binding)] = runner.digest(pilot_binding)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    fit = runner.Dataset(v1.PACKAGE, 'fit', labels=True)
    plan = v1.schedule(fit, cfg)
    teacher = np.load(teacher_info['teacher']['path'],mmap_mode='r',allow_pickle=False)
    if teacher.shape != (603,1,160,160) or teacher.dtype != np.float64:
        raise ValueError('Expected exact FP64 Fit603 teacher array')
    for prediction, support in zip(teacher,fit.arrays['support']):
        mask=np.asarray(support,bool)
        if not mask.any() or not np.isfinite(prediction[mask]).all() or not np.isnan(prediction[~mask]).all():
            raise ValueError('Teacher support validity mismatch')
    base_schedule = v1.schedule_receipt(plan)
    schedule = dict(fit_ids_sha256=base_schedule['fit_ids_sha256'], d4_sha256=base_schedule['d4_sha256'],
                    history_source_count=9, full160_updates=cfg['updates'], dropout_policy='history0.25_emissivity0.25',
                    teacher_cache_sha256=teacher_info['teacher']['sha256'],
                    teacher_augmentation='Same stored Fit IDs and original D4, with original solar-context transform')
    factories = bridge.model_factories(v1, args.architecture, dict(history_dropout=.25, emissivity_dropout=.25))
    v1.model_classes = lambda architecture: factories
    model, initialized = v1.initialization(args, fit)
    if model.history_dropout != .25 or model.emissivity_dropout != .25:
        raise ValueError('Actual model dropout does not match the fixed continuation recipe')
    model = model.to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], fused=(args.device=='cuda'))
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    val = None if smoke else runner.Dataset(v1.PACKAGE, 'validation', labels=True)
    if not smoke and (len(val)!=45 or len({r['city'] for r in val.records})!=15 or len({r['region'] for r in val.records})!=3):
        raise ValueError('All original Val45/15 cities/3 regions required')
    check_deadline()
    args.output.mkdir(parents=True, exist_ok=False)
    counters = dict(training_forward_calls=0, training_backward_calls=0,
                    initialization_cpu_forward_calls=2, validation_forward_calls=0,
                    final_fp32_forward_calls=0, optimizer_updates=0, amp_attempts=0, amp_backoffs=0,
                    skipped_optimizer_updates=0, finite_loss_and_gradient_updates=0, full160_updates=0,
                    successful_update_fine_pixels=0, attempted_forward_fine_pixels=0,
                    attempted_history_frame_pixels=0, empty_formal_training_samples=0)
    runtime = dict(config=cfg, design=design, device=args.device, initialization=initialized,
                   execution_context=execution_context,execution_start_unix=execution_start_unix,isolated_training_timing=False,
                   source_sha256=seal, schedule=schedule, parameters=sum(p.numel() for p in model.parameters()),
                   fit_queries=len(fit), validation_queries=0 if smoke else len(val),
                   smoke_only=smoke, test_opened=False, started_unix=time.time(),
                   planned_weight_candidates=0 if smoke else 10, locked_alpha25_reference_rmse_k=reference_rmse,
                   teacher=teacher_info, teacher_receipt_sha256=runner.digest(args.teacher_receipt),
                   teacher_cache_shared_for_both_architectures=True,teacher_updated=False,
                   optimizer_numeric_path='Both architectures CUDA fused AdamW; CPU synthetic ordinary AdamW; no bitwise equivalence to original optimizer claimed',
                   cost_scope='Wall time includes setup, original initialization, training, validation and final FP32 gate')
    runner.dump(args.output / 'run.json', runtime)
    np.savez(args.output / 'schedule.npz', fit_ids=plan[0], d4=plan[1])
    rows, best, best_step, best_weights = [], None, None, None
    def elapsed():
        return time.perf_counter() - started
    def verify():
        v1.verify_seal(seal, args.checkpoint, cfg['initialization_sha256'])
    @torch.inference_mode()
    def infer(net, amp, final=False):
        net.eval()
        parts = []
        for start in range(0, len(val), cfg['evaluation_batch']):
            check_deadline()
            b = runner.batch(val, np.arange(start, min(start + cfg['evaluation_batch'], len(val))), args.device)
            if b['fine'].shape[-2:] != (160, 160) or b['coarse'].shape[-2:] != (40, 40):
                raise ValueError('All validation must use unchanged full160 scenes')
            with torch.autocast(args.device, dtype=torch.float16, enabled=amp and args.device == 'cuda'):
                p = runner.forward(net, b)
            counters['final_fp32_forward_calls' if final else 'validation_forward_calls'] += 1
            parts.append(p.float().cpu().numpy())
        p = runner.repair(np.concatenate(parts), val.arrays['coarse'], val.arrays['support'])
        result = runner.score(p, val.arrays['target'], val.arrays['formal'], val.records)
        if final:
            from historylst.hotspots import add_hotspot_metrics
            add_hotspot_metrics(result, p, val.arrays['target'], val.arrays['formal'])
        if not math.isfinite(result['macro']['rmse']):
            raise RuntimeError('Nonfinite full-cohort Val RMSE')
        return result
    def checkpoint(net, step, weights, result):
        return dict(state_dict=runner.state(net), step=step, weights=weights, config=cfg,
                    validation_macro_rmse=None if result is None else result['macro']['rmse'],
                    selection_metric='not_selected_fit_only_smoke' if smoke else 'full9_macro_rmse',
                    selection_score=None if result is None else result['macro']['rmse'],
                    source_sha256=seal, initialization=initialized, smoke_only=smoke)
    def save_last(step):
        runner.save(args.output / 'last.pt', dict(state_dict=runner.state(model), ema=runner.state(ema),
                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), config=cfg,
                    step=step, best=best, validation=rows, source_sha256=seal, schedule=schedule,
                    elapsed_seconds=elapsed(), counters=counters, smoke_only=smoke))
    def validate(step):
        nonlocal best, best_step, best_weights
        verify()
        for weights, net in (('raw', model), ('ema', ema)):
            result = infer(net, amp=True)
            value = result['macro']['rmse']
            row = dict(step=step, weights=weights, validation_macro_rmse=value, seconds=elapsed())
            rows.append(row)
            if best is None or value < best:
                best, best_step, best_weights = value, step, weights
                runner.save(args.output / 'best.pt', checkpoint(net, step, weights, result))
            print(json.dumps(dict(validation=row, best=best)), flush=True)
        runner.dump(args.output / 'validation.json', rows)
        save_last(step)
    try:
        if val is not None:
            validate(0)
        for step in range(1, 6 if smoke else cfg['updates'] + 1):
            check_deadline()
            torch.manual_seed(cfg['seed'] + 1000003 * step)
            if args.device == 'cuda':
                torch.cuda.manual_seed_all(cfg['seed'] + 1000003 * step)
            b = runner.batch(fit, plan[0][step - 1], args.device)
            b['teacher'] = torch.from_numpy(np.array(teacher[plan[0][step-1]],dtype=np.float32,copy=True)).to(args.device)
            b = runner.augment(b, int(plan[1][step - 1]))
            model.train()
            factor = step / cfg['warmup'] if step <= cfg['warmup'] else .05 + .95 * .5 * (1 + math.cos(math.pi * (step - cfg['warmup']) / (cfg['updates'] - cfg['warmup'])))
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * factor
            pixels = b['fine'].shape[0] * b['fine'].shape[-2] * b['fine'].shape[-1]
            counters['empty_formal_training_samples'] += int((b['formal'].flatten(1).sum(1) == 0).sum())
            # A failed AMP attempt must not count as an additional BN update.
            # NAF has no such buffers; the matched U-TAE continuation does.
            before_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            for attempt in range(8):
                check_deadline()
                if attempt:
                    with torch.no_grad():
                        for name, value in model.named_buffers():
                            value.copy_(before_buffers[name])
                torch.manual_seed(cfg['seed'] + 1000003 * step)
                if args.device == 'cuda':
                    torch.cuda.manual_seed_all(cfg['seed'] + 1000003 * step)
                optimizer.zero_grad(set_to_none=True)
                counters['amp_attempts'] += 1
                counters['training_forward_calls'] += 1
                counters['attempted_forward_fine_pixels'] += pixels
                counters['attempted_history_frame_pixels'] += pixels * b['history'].shape[1]
                with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == 'cuda'):
                    prediction = runner.forward(model, b)
                    loss, ground_truth_loss, teacher_loss = mixed_loss(prediction,b,runner)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss, as in the original one-pass trainer')
                scaler.scale(loss).backward()
                counters['training_backward_calls'] += 1
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    if not scaler.is_enabled():
                        raise RuntimeError('Nonfinite gradient')
                    counters['amp_backoffs'] += 1
                    scaler.update(new_scale=scaler.get_scale() * .5)
                    continue
                scaler.step(optimizer)
                scaler.update()
                break
            else:
                raise RuntimeError('Eight original AMP backoffs exhausted')
            update = dict(loss=float(loss.detach()), gradient_norm=float(norm), amp_attempts=attempt + 1,
                          fine_shape=list(b['fine'].shape[-2:]), history_dropout=.25, emissivity_dropout=.25,
                          ground_truth_loss=float(ground_truth_loss.detach()),teacher_loss=float(teacher_loss.detach()))
            counters['optimizer_updates'] += 1
            counters['finite_loss_and_gradient_updates'] += 1
            counters['full160_updates'] += 1
            counters['successful_update_fine_pixels'] += pixels
            decay = min(.995, (1 + step) / (10 + step))
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 1 - decay)
                for eb, buf in zip(ema.buffers(), model.buffers()):
                    eb.copy_(buf)
            if step % 50 == 0 or smoke:
                row = dict(step=step, **update, actual_history_k=b['history'].shape[1],
                           lr=optimizer.param_groups[0]['lr'], seconds=elapsed(), counters=dict(counters))
                print(json.dumps(row), flush=True)
                with (args.output / 'progress.jsonl').open('a') as file:
                    file.write(json.dumps(row) + '\n')
            if val is not None and step % cfg['validation_interval'] == 0:
                validate(step)
        verify()
        if runner.digest(teacher_info['teacher']['path']) != teacher_info['teacher']['sha256']:
            raise ValueError('Teacher cache changed during training')
        if smoke:
            runner.save(args.output / 'best.pt', checkpoint(model, 5, 'raw', None))
            save_last(5)
            report = dict(status='smoke_complete', updates=5, fit_only=True, validation_opened=False,
                          test_opened=False, seconds=elapsed(), counters=counters, scientific_goal_complete=False)
            runner.dump(args.output / 'smoke.json', report)
        else:
            if (counters['full160_updates'], counters['successful_update_fine_pixels']) != (1000, 102400000):
                raise RuntimeError('Actual full160 training schedule differs from the fixed protocol')
            if counters['amp_attempts'] != counters['optimizer_updates'] + counters['amp_backoffs']:
                raise RuntimeError('Optimizer updates and AMP backoffs do not match attempted updates')
            if counters['finite_loss_and_gradient_updates'] != 1000:
                raise RuntimeError('All 1000 retained updates must have finite loss and gradients')
            if [(r['step'], r['weights']) for r in rows] != [(s, w) for s in range(0, 1001, 250) for w in ('raw', 'ema')]:
                raise RuntimeError('Expected exactly 10 validation choices')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            selected_ck = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(selected_ck['state_dict'], strict=True)
            selected = infer(model, amp=False, final=True)
            model.load_state_dict(original_ck['state_dict'], strict=True)
            initial = infer(model, amp=False, final=True)
            value, initial_value = selected['macro']['rmse'], initial['macro']['rmse']
            report = dict(status='complete', updates=1000, seconds=elapsed(), parameters=runtime['parameters'],
                          execution_context=execution_context,execution_start_unix=execution_start_unix,
                          execution_end_unix=time.time(),isolated_training_timing=False,
                          counters=counters, validation_weight_candidates=len(rows), best_amp_rmse=best,
                          selected_step=best_step, selected_weights=best_weights,
                          selected_checkpoint_sha256=runner.digest(args.output / 'best.pt'),
                          selected_fp32=selected, initialization_fp32=initial,
                          improvement_k=initial_value - value,
                          locked_alpha25_reference_rmse_k=reference_rmse,
                          improvement_over_locked_alpha25_905_k=reference_rmse-value if is_pilot else None,
                          pilot_reference_gate_assessed=is_pilot,
                          continuation_gate_pass=bool(is_pilot and value < .42 and reference_rmse-value >= .003),
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                          teacher_cache_sha256=teacher_info['teacher']['sha256'],
                          teacher_receipt_sha256=runner.digest(args.teacher_receipt),
                          student_inference_views=1, teacher_inference_used_at_validation=False,
                          opened_arrays=sorted(opened), validation_opened=True,
                          test_opened=False, scientific_goal_complete=False,
                          no_automatic_repetition=True, selection_not_changed_by_fp32_gate=True)
            check_deadline()
            runner.dump(args.output / 'complete.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        runner.dump(args.output / 'interrupted.json', dict(status='incomplete', error=repr(exc),
                    seconds=elapsed(), counters=counters, test_opened=False, scientific_goal_complete=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'smoke', 'train'))
    parser.add_argument('--architecture', choices=('naf_history', 'baseline'), default='naf_history')
    parser.add_argument('--original-seed', type=int, choices=(20260905, 20260912, 20260913), default=20260905)
    parser.add_argument('--seed', type=int, choices=(20260921,), default=20260921)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--teacher-receipt', type=Path)
    parser.add_argument('--companion-output', type=Path)
    parser.add_argument('--source-completion', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pilot-completion', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.mode == 'check':
        from synthetic_check import check
        print(json.dumps(check()), flush=True)
        return
    for key in ('checkpoint', 'source_completion', 'output', 'teacher_receipt', 'companion_output'):
        if getattr(args, key) is None:
            parser.error('--' + key.replace('_', '-') + ' is required')
        setattr(args, key, getattr(args, key).resolve())
    if args.pilot_completion is not None:
        args.pilot_completion = args.pilot_completion.resolve()
    run(args)


if __name__ == '__main__':
    main()
