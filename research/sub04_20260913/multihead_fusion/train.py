"""Four-head fusion adapter of the fixed paired-budget strong-teacher trainer.

Only Fit/Val observations are allowed. Original artifacts are immutable.
Both architectures use identical teacher, sample sequence and added budget.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse, copy, importlib.util, json, math, signal, subprocess, sys, time
import numpy as np
import torch

ADAPTER_HERE = Path(__file__).resolve().parent
HERE = ADAPTER_HERE.parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'sub04_20260911'
PACKAGE = ROOT / 'resources/historylst246'
sys.path[:0] = [str(OLD), str(PACKAGE)]
spec = importlib.util.spec_from_file_location('strong_kd_original_runner', PACKAGE / 'run.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
from naf_history.model import HistoryNAFReconstructor
from dropout_models import DropoutUTAE
from historylst.hotspots import add_hotspot_metrics
mspec = importlib.util.spec_from_file_location('four_head_fusion_train_model', ADAPTER_HERE / 'model.py')
four = importlib.util.module_from_spec(mspec); mspec.loader.exec_module(four)

RECIPE = dict(updates=3000, batch_size=4, lr=1e-4, warmup=100,
    weight_decay=1e-4, validation_interval=500, evaluation_batch=2,
    history_dropout=0., emissivity_dropout=0., teacher_weight=.9,
    seed=20260923, ema_max_decay=.995, gradient_clip=1.)

def read(path): return json.loads(Path(path).read_text())

def loss(pred, batch):
    # Mask before subtraction/squaring to keep off-support NaNs out of gradients.
    def rmse(target, mask):
        mask = mask.bool()
        error = torch.where(mask, pred.float(), 0.) - torch.where(mask, target.float(), 0.)
        return (error.square().sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1) + 1e-6).sqrt().mean()
    truth = rmse(batch['target'], batch['formal'])
    teacher = rmse(batch['teacher'], batch['support'])
    return .1 * truth + .9 * teacher, truth, teacher

def guard(teacher_path):
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if 'data' in path.parts and 'test' in path.parts:
                raise RuntimeError('Test observations forbidden during development')
    sys.addaudithook(audit)

def train(args):
    started = time.time()
    extension = read(HERE / 'five_hour_extension.json')
    deadline = min(extension['stop_search_by_unix'], started + 1500)
    def check():
        if time.time() >= deadline: raise TimeoutError('Finite 1500s job/search cutoff reached')
    def alarm(signum, frame): raise TimeoutError('Finite job deadline')
    check(); signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    cfg = dict(RECIPE, architecture=args.architecture, original_seed=args.original_seed,
        algorithm='AdamW_fused_cuda', protocol='four_head_teacher_kd_20260913_v1',
        source_attention_heads=4 if args.architecture == 'naf_history' else 1,
        initialization_transform='copy_score_last_conv_1_to_4' if args.architecture == 'naf_history' else 'identity')
    frozen_path = OLD / 'final_delivery_late_20260912/matched/continuation_selection_freeze.json'
    frozen = read(frozen_path)
    entry, = [x for x in frozen['checkpoints'] if (x['architecture'], x['seed']) == (args.architecture, args.original_seed)]
    checkpoint_path = Path(entry['checkpoint'])
    if r.digest(checkpoint_path) != entry['checkpoint_sha256']: raise ValueError('Initialization weight changed')
    completion = read(Path(entry['run']) / 'complete.json')
    if completion['status'] != 'complete' or completion['updates'] != 1000: raise ValueError('Real preceding matched KD completion required')
    teacher_receipt = read(args.teacher_receipt)
    item = teacher_receipt['teacher']; teacher_path = Path(item['path'])
    if r.digest(teacher_path) != item['sha256']: raise ValueError('Teacher hash mismatch')
    manifest = read(PACKAGE / 'manifest.json')
    if teacher_receipt['manifest_sha256'] != r.digest(PACKAGE / 'manifest.json'): raise ValueError('Teacher manifest changed')
    if teacher_receipt['scene_ids'] != [x['scene_id'] for x in manifest['roles']['fit']['scenes']]: raise ValueError('Teacher is not complete ordered Fit603')
    if teacher_receipt.get('labels_opened') is not False or teacher_receipt.get('test_opened') is not False:
        raise ValueError('Teacher generation must not open labels/Test')
    if (args.architecture, args.original_seed) != ('naf_history', 20260905):
        if args.pilot_completion is None: raise ValueError('Actual successful pilot required before matched continuations')
        pilot = read(args.pilot_completion)
        if (pilot.get('continuation_gate_pass') is not True or pilot.get('updates') != 3000
            or pilot.get('architecture_variant') != 'four_head_history_fusion'):
            raise ValueError('Pilot did not pass the prospective Val improvement gate')
        if pilot['teacher_receipt_sha256'] != r.digest(args.teacher_receipt): raise ValueError('Pilot teacher differs')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    if active.stdout.strip(): raise RuntimeError('GPU occupied; do not interrupt another task')
    guard(teacher_path); r.setup('cuda')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model = (four.FourHeadHistoryNAF if args.architecture == 'naf_history' else DropoutUTAE)(history_dropout=0., emissivity_dropout=0.)
    initial = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if args.architecture == 'naf_history':
        four.expand_single_head_state_dict(initial['state_dict'], model)
    else:
        model.load_state_dict(initial['state_dict'], strict=True)
    model.cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], fused=True)
    scaler = torch.amp.GradScaler('cuda')
    fit, val = r.Dataset(PACKAGE, 'fit', labels=True), r.Dataset(PACKAGE, 'validation', labels=True)
    if len(fit) != 603 or len(val) != 45: raise ValueError('Complete Fit603 and Val45 required')
    teacher = np.load(teacher_path, mmap_mode='r', allow_pickle=False)
    if teacher.shape != (603, 1, 160, 160): raise ValueError('Teacher shape differs')
    for field, support in zip(teacher, fit.arrays['support']):
        if not np.isfinite(field[np.asarray(support, bool)]).all(): raise ValueError('Teacher invalid on input support')
    rng = np.random.default_rng(cfg['seed']); ids, codes = [], []
    probabilities = fit.sampling_probabilities()
    for _ in range(cfg['updates']):
        ids.append(rng.choice(len(fit), cfg['batch_size'], p=probabilities))
        codes.append(int(rng.integers(8)))
    ids, codes = np.asarray(ids), np.asarray(codes)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / 'schedule.npz', fit_ids=ids, d4=codes)
    sources = {str(p): r.digest(p) for p in [Path(__file__), ADAPTER_HERE / 'model.py', ADAPTER_HERE / 'training_adapter_design.json',
        HERE / 'train_strong_kd.py', HERE / 'five_hour_extension.json',
        OLD / 'naf_history/model.py', OLD / 'dropout_models.py', PACKAGE / 'run.py',
        PACKAGE / 'historylst/model.py', PACKAGE / 'historylst/metrics.py', PACKAGE / 'historylst/data.py',
        args.teacher_receipt, frozen_path]}
    runtime = dict(config=cfg, initialization=str(checkpoint_path), initialization_sha256=entry['checkpoint_sha256'],
        architecture_variant='four_head_history_fusion' if args.architecture == 'naf_history' else 'original_utae',
        converted_parameter_keys=list(four.EXPANDED_KEYS) if args.architecture == 'naf_history' else [],
        teacher_receipt=str(args.teacher_receipt), teacher_receipt_sha256=r.digest(args.teacher_receipt),
        teacher=item, source_sha256=sources, manifest_sha256=r.digest(PACKAGE / 'manifest.json'),
        schedule_sha256=r.digest(args.output / 'schedule.npz'), parameters=sum(p.numel() for p in model.parameters()),
        actual_start_unix=started, deadline_unix=deadline, device='cuda', selected_teacher_used_at_inference=False,
        test_opened=False, student_inference_views=1, planned_validation_candidates=14)
    r.dump(args.output / 'run.json', runtime)
    counters = dict(updates=0, forward_calls=0, backward_calls=0, amp_backoffs=0, validation_forward_calls=0,
        final_fp32_forward_calls=0, finite_updates=0, successful_fine_pixels=0)
    rows, best = [], math.inf
    @torch.inference_mode()
    def evaluate(net, amp, final=False):
        net.eval(); parts = []
        for first in range(0, len(val), 2):
            check(); b = r.batch(val, np.arange(first, min(first + 2, len(val))), 'cuda')
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp): pred = r.forward(net, b)
            parts.append(pred.float().cpu().numpy())
            counters['final_fp32_forward_calls' if final else 'validation_forward_calls'] += 1
        pred = r.repair(np.concatenate(parts), val.arrays['coarse'], val.arrays['support'])
        result = r.score(pred, val.arrays['target'], val.arrays['formal'], val.records)
        if final: add_hotspot_metrics(result, pred, val.arrays['target'], val.arrays['formal'])
        return result
    def validate(step):
        nonlocal best
        for name, net in [('raw', model), ('ema', ema)]:
            score = evaluate(net, True)['macro']['rmse']
            if not math.isfinite(score): raise RuntimeError('Nonfinite validation metric')
            row = dict(step=step, weights=name, rmse=score, seconds=time.time() - started)
            rows.append(row)
            if score < best:
                best = score
                r.save(args.output / 'best.pt', dict(state_dict=r.state(net), config=cfg, step=step, weights=name,
                    validation_rmse=score, initialization=runtime['initialization'], source_sha256=sources,
                    teacher_receipt_sha256=runtime['teacher_receipt_sha256']))
            print(json.dumps(dict(validation=row, best=best)), flush=True)
        r.dump(args.output / 'validation.json', rows)
        r.save(args.output / 'last.pt', dict(state_dict=r.state(model), ema=r.state(ema), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), config=cfg, step=step, counters=dict(counters), validation=rows))
    try:
        validate(0)
        for step in range(1, cfg['updates'] + 1):
            check(); batch = r.batch(fit, ids[step - 1], 'cuda')
            batch['teacher'] = torch.from_numpy(np.array(teacher[ids[step - 1]], dtype=np.float32)).cuda()
            batch = r.augment(batch, int(codes[step - 1])); model.train()
            factor = step / cfg['warmup'] if step <= cfg['warmup'] else .05 + .95 * .5 * (1 + math.cos(math.pi * (step - cfg['warmup']) / (cfg['updates'] - cfg['warmup'])))
            for group in optimizer.param_groups: group['lr'] = cfg['lr'] * factor
            buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
            for attempt in range(8):
                check()
                if attempt:
                    with torch.no_grad():
                        for name, value in model.named_buffers(): value.copy_(buffers[name])
                torch.manual_seed(cfg['seed'] + 1000003 * step); torch.cuda.manual_seed_all(cfg['seed'] + 1000003 * step)
                optimizer.zero_grad(set_to_none=True); counters['forward_calls'] += 1
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = r.forward(model, batch); total, truth, distilled = loss(pred, batch)
                if not torch.isfinite(total): raise RuntimeError('Nonfinite mixed loss')
                scaler.scale(total).backward(); counters['backward_calls'] += 1; scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                if not torch.isfinite(norm):
                    counters['amp_backoffs'] += 1; scaler.update(new_scale=scaler.get_scale() * .5); continue
                scaler.step(optimizer); scaler.update(); break
            else: raise RuntimeError('Eight finite-gradient retries exhausted')
            counters['updates'] += 1; counters['finite_updates'] += 1; counters['successful_fine_pixels'] += 4 * 160 * 160
            decay = min(cfg['ema_max_decay'], (1 + step) / (10 + step))
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters()): e.lerp_(p, 1 - decay)
                for e, p in zip(ema.buffers(), model.buffers()): e.copy_(p)
            if step % 100 == 0:
                row = dict(step=step, seconds=time.time()-started, loss=float(total), teacher_loss=float(distilled), truth_loss=float(truth), counters=dict(counters))
                with (args.output / 'progress.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            if step % cfg['validation_interval'] == 0: validate(step)
        if [(x['step'], x['weights']) for x in rows] != [(s, w) for s in range(0, 3001, 500) for w in ('raw', 'ema')]:
            raise ValueError('The actual 14 candidates differ from the paired recipe')
        if counters['finite_updates'] != 3000: raise ValueError('Incomplete updates')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        selected = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['state_dict']); result = evaluate(model, False, True)
        initial_rmse = completion['selected_fp32']['macro']['rmse']
        for p, digest in sources.items():
            if r.digest(p) != digest: raise RuntimeError('Bound source changed during training')
        if r.digest(teacher_path) != item['sha256']: raise RuntimeError('Teacher changed during training')
        is_pilot = args.architecture == 'naf_history' and args.original_seed == 20260905
        report = dict(status='complete', architecture_variant=runtime['architecture_variant'],
            updates=3000, validation_weight_candidates=14, selected_step=selected['step'],
            selected_weights=selected['weights'], selected_checkpoint_sha256=r.digest(args.output / 'best.pt'),
            selected_fp32=result, initialization_fp32_rmse=initial_rmse, improvement_k=initial_rmse-result['macro']['rmse'],
            continuation_gate_pass=bool(is_pilot and result['macro']['rmse'] < .412 and initial_rmse-result['macro']['rmse'] >= .005),
            counters=counters, parameters=runtime['parameters'], seconds=time.time()-started,
            teacher_receipt_sha256=runtime['teacher_receipt_sha256'], actual_start_unix=started, actual_end_unix=time.time(),
            single_network_forward=True, test_opened=False, scientific_goal_complete=False)
        check(); r.dump(args.output / 'complete.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'selected_fp32'}), flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        r.dump(args.output / 'interrupted.json', dict(status='incomplete', error=repr(exc), counters=counters, seconds=time.time()-started))
        raise
    finally: signal.setitimer(signal.ITIMER_REAL, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['naf_history', 'baseline'], required=True)
    parser.add_argument('--original-seed', type=int, choices=[20260905, 20260912, 20260913], required=True)
    parser.add_argument('--teacher-receipt', type=Path, required=True)
    parser.add_argument('--pilot-completion', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.teacher_receipt = args.teacher_receipt.resolve(); args.output = args.output.resolve()
    train(args)
