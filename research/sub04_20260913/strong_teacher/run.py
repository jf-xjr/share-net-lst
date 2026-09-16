"""Fixed three frozen D4 students x all eight D4 views teacher, no selection.

Val45 must pass macro RMSE < .407 K before the Fit603 input-only cache can run.
An ensemble teacher is never reported as a single-forward goal model.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse, gc, importlib.util, json, signal, subprocess, sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/sub04_20260911'
sys.path.insert(0, str(OLD))
import d4_teacher_feasibility_20260912 as fixed
from historylst.hotspots import add_hotspot_metrics
p, aug, original = fixed.p, fixed.aug, fixed.original
FREEZE = OLD / 'final_delivery_late_20260912/matched/continuation_selection_freeze.json'
READER = OLD / 'd4_self_distillation_20260912_v1/summarize_matched_v2.py'
EXTENSION = HERE.parent / 'five_hour_extension.json'
AGGREGATION = 'registered_inverse_then_float64_equal_mean_then_original_support_repair'
SEEDS = [20260905, 20260912, 20260913]


def read(path): return json.loads(Path(path).read_text())
def bind(path): return dict(path=str(Path(path).resolve()), sha256=p.sha(path))
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def guard(manifest, split, output, access):
    fields = manifest['roles'][split]['fields']
    inputs = {(p.PACKAGE / fields[k]['path']).resolve() for k in p.runner.INPUTS}
    labels = {(p.PACKAGE / fields[k]['path']).resolve() for k in ('target', 'formal')}
    outputs = {(output / name).resolve() for name in ('teacher.npy', 'teacher.partial.npy')}
    opened = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        path = Path(os.fsdecode(args[0])).resolve()
        if 'test' in path.parts or ('data' in path.parts and ({'fit', 'validation'} - {split}) & set(path.parts)):
            raise RuntimeError('Wrong-split observation access: ' + str(path))
        if 'labels' in path.parts and (split == 'fit' or not access['labels'] or path not in labels):
            raise RuntimeError('Prediction must seal before approved Val labels; no Fit labels')
        if path.suffix in ('.npy', '.npz'):
            if path not in inputs | outputs | (labels if access['labels'] else set()):
                raise RuntimeError('Unapproved array ' + str(path))
            opened.add(str(path))
    sys.addaudithook(audit)
    return opened


def run(args):
    start = time.perf_counter()
    deadline_epoch = min(read(EXTENSION)['hard_deadline_unix'], 1789271700)
    available = min(180. if args.split == 'validation' else 600., deadline_epoch - time.time())
    if available <= 0: raise TimeoutError('Authorized GPU evaluation window ended')
    if args.output.exists(): raise FileExistsError('New output only; no overwrite/retry')
    def deadline():
        if time.perf_counter() - start >= available or time.time() >= deadline_epoch:
            raise TimeoutError('Finite teacher execution limit reached')
    def alarm(signum, frame): raise TimeoutError('Finite teacher process alarm')
    old_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, available)
    args.output.mkdir(parents=True, exist_ok=False)
    models, hooks, array = [], [], None
    access = dict(labels=False)
    counts = dict(model_forward_calls=0, image_forwards=0, historical_source_encodings=0)
    try:
        spec = importlib.util.spec_from_file_location('actual_six_student_freeze_reader', READER)
        loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
        frozen = loader.read_freeze(FREEZE)
        entries = [e for e in frozen['checkpoints'] if e['architecture'] == 'naf_history']
        if [e['seed'] for e in entries] != SEEDS: raise ValueError('All three frozen NAF seeds required')
        manifest = read(p.PACKAGE / 'manifest.json')
        source = {str(q.resolve()): p.sha(q) for q in (Path(__file__), HERE / 'design.json', EXTENSION,
            FREEZE, READER, Path(fixed.__file__), fixed.AUGMENT, OLD / 'naf_history/model.py',
            OLD / 'evaluate_finalist.py', OLD / 'evaluate_screens.py', p.PACKAGE / 'run.py',
            p.PACKAGE / 'historylst/model.py', p.PACKAGE / 'historylst/data.py',
            p.PACKAGE / 'historylst/metrics.py', p.PACKAGE / 'historylst/hotspots.py')}
        validation = None
        if args.split == 'fit':
            result_path = HERE / 'validation/results.json'
            validation = read(result_path)
            if (validation['status'] != 'complete_fixed24_teacher_feasibility'
                or validation['teacher_feasibility_gate_pass'] is not True
                or validation['macro']['rmse'] >= .407
                or validation['improvement_over_prior_teacher_k'] < .002
                or validation['freeze_sha256'] != p.sha(FREEZE)):
                raise ValueError('Actual fixed24 fullVal feasibility must pass before Fit cache')
            receipt_path = Path(validation['predictions_complete']['path'])
            receipt = read(receipt_path)
            if (p.sha(receipt_path) != validation['predictions_complete']['sha256']
                or receipt['labels_opened'] is not False or receipt['counters']['image_forwards'] != 1080
                or [e['checkpoint_sha256'] for e in receipt['checkpoints']] != [e['checkpoint_sha256'] for e in entries]):
                raise ValueError('Complete same-teacher Val prediction receipt required')
            for path, digest in receipt['source_sha256'].items():
                if p.sha(path) != digest: raise ValueError('Teacher feasibility source changed')
            source.update({str(q.resolve()): p.sha(q) for q in (result_path, receipt_path)})
        opened = guard(manifest, args.split, args.output, access)
        inputs = {k: dict(path=str((p.PACKAGE / manifest['roles'][args.split]['fields'][k]['path']).resolve()),
            sha256=manifest['roles'][args.split]['fields'][k]['sha256']) for k in p.runner.INPUTS}
        for item in inputs.values():
            deadline()
            if p.sha(item['path']) != item['sha256']: raise ValueError('Input identity changed')
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True)
        if gpu.stdout.strip(): raise RuntimeError('GPU occupied; no interruption or waiting')
        checkpoints = [{k: e[k] for k in ('architecture', 'seed', 'checkpoint', 'checkpoint_sha256')} for e in entries]
        write(args.output / 'started.json', dict(status='frozen_before_predictions_and_labels', split=args.split,
            checkpoints=checkpoints, views=list(range(8)), member_count=3, equal_view_count=24,
            freeze_sha256=p.sha(FREEZE), input_bindings=inputs, source_sha256=source,
            aggregation=AGGREGATION, device='cuda', fp32=True, tf32=False, batch=2,
            repair_dtype='float64', labels_opened=False, test_opened=False, goal_pass=False,
            deadline_unix=deadline_epoch, job_limit_seconds=available))
        p.setup('cuda'); torch.set_float32_matmul_precision('highest')
        def count_stem(module, argv):
            n, c, h, w = argv[0].shape
            if c != 9 or (h, w) != (160, 160) or n % 9: raise RuntimeError('Unexpected source encoding')
            counts['model_forward_calls'] += 1
            counts['image_forwards'] += n // 9
            counts['historical_source_encodings'] += n
        for entry in entries:
            model = original.load_model(entry, 'cuda').float().eval().requires_grad_(False)
            models.append(model); hooks.append(model.historical.register_forward_pre_hook(count_stem))
        data = p.Dataset(p.PACKAGE, args.split, labels=False)
        n = 45 if args.split == 'validation' else 603
        cities = 15 if args.split == 'validation' else 201
        if len(data) != n or len({r['city'] for r in data.records}) != cities: raise ValueError('Complete original cohort required')
        path, temporary = args.output / 'teacher.npy', args.output / 'teacher.partial.npy'
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64', shape=(n, 1, 160, 160))
        prediction_start = time.perf_counter()
        with torch.inference_mode():
            for first in range(0, n, 2):
                ids = np.arange(first, min(first + 2, n)); raw = data.batch(ids)
                batch = {k: torch.from_numpy(v).cuda() for k, v in raw.items()}
                total = np.zeros((len(ids), 1, 160, 160), np.float64)
                for model in models:
                    for code in range(8):
                        deadline()
                        with torch.autocast('cuda', enabled=False): out = model(**aug.transform_batch(batch, code))
                        if out.dtype != torch.float32: raise RuntimeError('FP32 teacher required')
                        restored = aug.inverse_field(out, code).cpu().numpy().astype(np.float64)
                        if not np.isfinite(restored).all(): raise RuntimeError('Nonfinite teacher output')
                        total += restored
                repaired = p.runner.repair(total / 24., raw['coarse'], raw['support'])
                mask = raw['support'].astype(bool)
                if not np.isfinite(repaired[mask]).all() or not np.isnan(repaired[~mask]).all(): raise RuntimeError('Invalid support')
                array[ids] = repaired
                if first % (10 if n == 45 else 100) == 0:
                    print(json.dumps(dict(event='fixed24_teacher', split=args.split, scenes=int(ids[-1] + 1), seconds=time.perf_counter()-start)), flush=True)
        prediction_seconds = time.perf_counter() - prediction_start
        if counts != dict(model_forward_calls=((n + 1)//2)*24, image_forwards=n*24, historical_source_encodings=n*24*9):
            raise RuntimeError('Incomplete 24-view actual forwards')
        array.flush(); del array; array = None; temporary.replace(path)
        for hook in hooks: hook.remove()
        hooks.clear()
        for file, digest in source.items():
            deadline()
            if p.sha(file) != digest: raise RuntimeError('Teacher source changed')
        for entry in entries:
            if p.sha(entry['checkpoint']) != entry['checkpoint_sha256']: raise RuntimeError('Teacher weight changed')
        teacher = dict(bind(path), shape=[n, 1, 160, 160], dtype='float64')
        receipt = dict(status='complete_fixed24_teacher_before_labels', split=args.split, teacher=teacher,
            checkpoints=checkpoints, views=list(range(8)), member_count=3, equal_view_count=24,
            aggregation=AGGREGATION, scene_ids=[r['scene_id'] for r in data.records],
            manifest_sha256=p.sha(p.PACKAGE / 'manifest.json'), input_bindings=inputs,
            source_sha256=source, freeze_sha256=p.sha(FREEZE), device='cuda', fp32=True, tf32=False,
            batch=2, repair_dtype='float64', counters=counts, prediction_seconds=prediction_seconds,
            seconds=time.perf_counter()-start, opened_arrays=sorted(opened), labels_opened=False,
            fit_labels_opened=False, validation_arrays_opened=args.split=='validation', test_opened=False,
            all_future_students_share_one_cache=True, generation_cost_is_additional=True, goal_pass=False,
            teacher_val_results=bind(HERE/'validation/results.json') if validation is not None else None)
        write(args.output/'predictions_complete.json', receipt)
        checked = np.load(path, mmap_mode='r', allow_pickle=False)
        if checked.shape != (n,1,160,160) or checked.dtype != np.float64 or p.sha(path) != teacher['sha256']:
            raise ValueError('Complete prediction seal required')
        if args.split == 'validation':
            access['labels'] = True
            target = np.load(p.PACKAGE/manifest['roles']['validation']['fields']['target']['path'], mmap_mode='r', allow_pickle=False)
            formal = np.load(p.PACKAGE/manifest['roles']['validation']['fields']['formal']['path'], mmap_mode='r', allow_pickle=False)
            scores = p.score(checked, target, formal, data.records); add_hotspot_metrics(scores, checked, target, formal)
            prior_path = OLD/'d4_teacher_val_20260912_v1/results.json'
            prior = read(prior_path)['macro']['d4mean']['rmse']
            gain = prior-scores['macro']['rmse']; gate = scores['macro']['rmse'] < .407 and gain >= .002
            write(args.output/'scores.json', scores)
            result = dict(status='complete_fixed24_teacher_feasibility', macro=scores['macro'],
                prior_teacher_rmse=prior, prior_teacher_results=bind(prior_path), improvement_over_prior_teacher_k=gain,
                teacher_feasibility_gate_pass=bool(gate), gate='macro RMSE < .407 K and prior_teacher improvement >= .002 K',
                predictions_complete=bind(args.output/'predictions_complete.json'), source_sha256=source,
                freeze_sha256=p.sha(FREEZE), counters=counts, seconds=time.perf_counter()-start,
                validation_labels_opened=True, test_opened=False, goal_pass=False,
                limitation='Fixed 24-view ensemble teacher feasibility only, not a goal-qualifying single-forward network')
            write(args.output/'results.json', result)
            print(json.dumps(dict(status=result['status'],macro=result['macro'],teacher_feasibility_gate_pass=bool(gate))),flush=True)
        else: print(json.dumps(dict(status=receipt['status'],teacher=teacher,counters=counts,seconds=receipt['seconds'])),flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        write(args.output/'failed.json',dict(status='failed',error=repr(exc),counters=counts,seconds=time.perf_counter()-start,labels_opened=access['labels'],automatic_retry=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        for hook in hooks: hook.remove()
        models.clear(); del array; gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','run')); parser.add_argument('--split',choices=('validation','fit'))
    parser.add_argument('--output',type=Path); args=parser.parse_args()
    if args.command=='check': fixed.check()
    else:
        if args.output is None or args.split is None: parser.error('Explicit split and new output required')
        args.output=args.output.resolve(); run(args)
