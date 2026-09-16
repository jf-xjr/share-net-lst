"""Six actual matched 3000/14 strong-KD completions -> immutable Val freeze.

Preparation creates no run manifest. Missing or failed runs cannot become a
freeze. This reader never starts training, inference, or Test evaluation.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import argparse, gc, hashlib, importlib.util, json, math, signal, sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/sub04_20260911'
PACKAGE = ROOT / 'resources/historylst246'
PRIOR_READER = OLD / 'd4_self_distillation_20260912_v1/summarize_matched_v2.py'
PRIOR_FREEZE = OLD / 'final_delivery_late_20260912/matched/continuation_selection_freeze.json'
TEACHER_RECEIPT = NEW / 'strong_teacher/fit/predictions_complete.json'
TRAINER = NEW / 'train_strong_kd.py'
EXTENSION = NEW / 'five_hour_extension.json'
PROTOCOL = 'final_confirmation_six_actual_selected_weights_20260913_v1'
METHOD = 'final_confirmation_selected_single_variant'
sys.path.insert(0,str(HERE))
import authorization as allocation
spec=importlib.util.spec_from_file_location('final_confirmation_reader_loader',HERE/'loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
ORDER = [(role, seed) for seed in (20260905, 20260912, 20260913) for role in ('baseline', 'naf_history')]
RECIPE = dict(updates=3000, batch_size=4, lr=1e-4, warmup=100, weight_decay=1e-4,
    validation_interval=500, evaluation_batch=2, history_dropout=0., emissivity_dropout=0.,
    teacher_weight=.9, seed=20260923, ema_max_decay=.995, gradient_clip=1.)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


prior = module(PRIOR_READER, 'strong_kd_previous_completion_reader')
p = prior.paired
read, sha, require = prior.read, prior.sha, prior.require


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def binding(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def fields(value):
    require(value['protocol'] == PROTOCOL and value['ensemble_used'] is False
        and value['test_opened'] is False and value['test_authorized'] is False,
        'Actual native six-student freeze required')
    require([(e['architecture'], e['seed']) for e in value['checkpoints']] == ORDER
        and all(e['effective_method'] == METHOD for e in value['checkpoints']), 'Exact six single-forward strong-KD students required')
    v = value['validation']
    require(v['status'] == 'six_actual_strong_teacher_kd_runs_complete'
        and v['mean_metrics']['naf_history']['rmse'] < .42
        and v['criteria']['observed_mean_rmse_reduction_k'] >= .01
        and v['criteria']['all_three_seed_macro_improvements'] is True
        and v['criteria']['network_numeric_criteria_pass'] is True
        and len(v['effects']['rmse']['paired_seed_differences']) == 3
        and all(x > 0 for x in v['effects']['rmse']['paired_seed_differences']), 'Common three-seed Val gates did not pass')


def verify_teacher(value, manifest):
    require(value['status'] == 'complete_fixed24_teacher_before_labels' and value['split'] == 'fit'
        and value['member_count'] == 3 and value['equal_view_count'] == 24 and value['views'] == list(range(8)),
        'Complete fixed 24-view teacher required')
    require(value['device'] == 'cuda' and value['fp32'] is True and value['tf32'] is False
        and value['batch'] == 2 and value['repair_dtype'] == 'float64'
        and value['aggregation'] == 'registered_inverse_then_float64_equal_mean_then_original_support_repair', 'Teacher numerical protocol changed')
    require(all(value[k] is False for k in ('labels_opened', 'fit_labels_opened', 'validation_arrays_opened', 'test_opened')),
        'Teacher must be Fit-input-only and label-free')
    require(value['scene_ids'] == [r['scene_id'] for r in manifest['roles']['fit']['scenes']]
        and len(value['scene_ids']) == 603 and value['manifest_sha256'] == sha(PACKAGE/'manifest.json'), 'Full Fit603 teacher identity differs')
    require(value['teacher']['shape'] == [603, 1, 160, 160] and value['teacher']['dtype'] == 'float64'
        and value['counters'] == dict(model_forward_calls=7248, image_forwards=14472, historical_source_encodings=130248)
        and math.isfinite(value['seconds']) and 0 < value['seconds'] <= 600, 'Teacher actual work/cost missing')
    for key in p.runner.INPUTS:
        a, b = value['input_bindings'][key], manifest['roles']['fit']['fields'][key]
        require(Path(a['path']).resolve() == (PACKAGE/b['path']).resolve() and a['sha256'] == b['sha256'], 'Teacher input binding changed')
    require(value['original_teacher_screen_gate_pass'] is False and value['user_goal_success_standard_changed'] is False,
        'Preserve the failed extra compute screen and unchanged user standard')
    override = value['explicit_compute_override_receipt']
    require(sha(override['path']) == override['sha256'], 'Explicit cache allocation receipt changed')
    authorization = read(override['path'])
    require(authorization['status'] == 'root_authorized_one_fit_cache_after_failed_compute_screen'
        and authorization['original_screen_gate_pass'] is False and authorization['user_goal1_standard_lowered'] is False,
        'Actual compute-allocation exception required')


def freeze(args):
    require(not args.output.exists(), 'New freeze output directory only')
    decision, pilot_evidence = allocation.validate(args.authorization)
    variant = decision['selected_variant']
    paths = {key: (args.runs_root/f'{key[0]}_{key[1]}').resolve() for key in ORDER}
    paths[('naf_history',20260905)] = pilot_evidence[variant]['completion_path'].parent
    # Fail before reading any large state if even one real completion is absent.
    for key, path in paths.items():
        require((path/'complete.json').is_file(), 'Missing actual completion: ' + str(path))
        require(not (path/'interrupted.json').exists(), 'An interrupted run cannot be frozen')
        require(read(path/'complete.json')['status'] == 'complete', 'Nonterminal run is not evidence')
    torch.set_num_threads(1)
    inherited = prior.read_freeze(PRIOR_FREEZE)
    previous = {(e['architecture'], e['seed']): e for e in inherited['checkpoints']}
    manifest = read(PACKAGE/'manifest.json'); records = manifest['roles']['validation']['scenes']
    require(len(records) == 45, 'Complete unchanged Val45 required')
    teacher = read(TEACHER_RECEIPT); verify_teacher(teacher, manifest)
    teacher_hash = sha(TEACHER_RECEIPT)
    require(teacher['freeze_sha256'] == sha(PRIOR_FREEZE)
        and [e['checkpoint_sha256'] for e in teacher['checkpoints']] ==
        [previous[('naf_history', seed)]['checkpoint_sha256'] for seed in p.SEEDS], 'The shared teacher members changed')
    # Training already bound the actual cache; verify its bytes once here. The
    # Test reader intentionally never reopens this Fit cache or any Fit inputs.
    require(sha(teacher['teacher']['path']) == teacher['teacher']['sha256'], 'Teacher cache changed before freeze')
    allowed = {(path/'schedule.npz').resolve() for path in paths.values()}
    def guard(event, argv):
        if event != 'open' or not isinstance(argv[0], (str, bytes)): return
        path = Path(os.fsdecode(argv[0])).resolve()
        if 'data' in path.parts or (path.suffix in ('.npy', '.npz') and path not in allowed):
            raise RuntimeError('Freeze reads only completion evidence and six schedule artifacts')
    sys.addaudithook(guard)
    evidence = dict(inherited['source_and_evidence_sha256'], **teacher['source_sha256'])
    evidence.update({str(q.resolve()): sha(q) for q in (Path(__file__), HERE/'run.py', HERE/'loader.py', HERE/'confirm_train.py', HERE/'authorization.py', args.authorization, PRIOR_FREEZE,
        PRIOR_READER, TRAINER, EXTENSION, TEACHER_RECEIPT, Path(p.__file__))})
    entries, costs, scores, done_by_key, common_schedule = [], [], {}, {}, None
    for key in ORDER:
        path = paths[key]; meta, done, rows = (read(path/name) for name in ('run.json', 'complete.json', 'validation.json'))
        cfg = meta['config']; initial_entry = previous[key]
        is_retained_pilot = key == ('naf_history',20260905)
        if is_retained_pilot:
            expected = pilot_evidence[variant]['runtime']['config']
        else:
            expected = dict(RECIPE, architecture=key[0], original_seed=key[1], algorithm='AdamW_fused_cuda')
            expected.update(loader.factory.metadata(decision, argparse.Namespace(architecture=key[0],original_seed=key[1],authorization=args.authorization)))
        require(cfg == expected and meta['device'] == 'cuda' and meta['planned_validation_candidates'] == 14
            and meta['student_inference_views'] == 1 and meta['selected_teacher_used_at_inference'] is False
            and meta['test_opened'] is False, 'Shared training protocol/single-view identity differs')
        require(meta['initialization'] == initial_entry['checkpoint'] and meta['initialization_sha256'] == initial_entry['checkpoint_sha256']
            and meta['manifest_sha256'] == sha(PACKAGE/'manifest.json'), 'Actual preceding matched student must initialize its own run')
        require(meta['teacher_receipt_sha256'] == teacher_hash and done['teacher_receipt_sha256'] == teacher_hash
            and meta['teacher'] == teacher['teacher'] and Path(meta['teacher_receipt']).resolve() == TEACHER_RECEIPT,
            'All six students must share the exact teacher')
        require(meta['source_sha256'].get(str(TRAINER)) == sha(TRAINER)
            and meta['source_sha256'].get(str(TEACHER_RECEIPT)) == teacher_hash
            and meta['source_sha256'].get(str(PRIOR_FREEZE)) == sha(PRIOR_FREEZE), 'Actual training code and provenance not bound')
        require(done['updates'] == 3000 and done['validation_weight_candidates'] == 14 and done['single_network_forward'] is True
            and done['test_opened'] is False and 0 < done['seconds'] <= 1500, 'Actual finite 3000/14 completion required')
        require(done['actual_start_unix'] == meta['actual_start_unix']
            and abs(done['actual_end_unix']-done['actual_start_unix']-done['seconds']) < .1
            and done['actual_end_unix'] <= meta['deadline_unix']
            and meta['deadline_unix'] <= read(EXTENSION)['stop_search_by_unix'], 'Actual finite execution span differs')
        require([(r['step'], r['weights']) for r in rows] == [(s,w) for s in range(0,3001,500) for w in ('raw','ema')]
            and all(math.isfinite(r['rmse']) for r in rows), 'All 14 actual fullVal selection records required')
        selected = min(rows, key=lambda x: x['rmse'])
        best = torch.load(path/'best.pt', map_location='cpu', weights_only=False)
        last = torch.load(path/'last.pt', map_location='cpu', weights_only=False)
        initial = torch.load(initial_entry['checkpoint'], map_location='cpu', weights_only=False)
        require(best['config'] == cfg and last['config'] == cfg and last['step'] == 3000 and last['validation'] == rows,
            'Actual selected and terminal state must bind this completed schedule')
        require((best['step'], best['weights'], best['validation_rmse']) == (selected['step'], selected['weights'], selected['rmse'])
            and (done['selected_step'], done['selected_weights']) == (best['step'], best['weights'])
            and sha(path/'best.pt') == done['selected_checkpoint_sha256']
            and best['teacher_receipt_sha256'] == teacher_hash and best['initialization'] == meta['initialization']
            and best['source_sha256'] == meta['source_sha256'], 'Original earliest AMP minimum must remain selected')
        model = loader.make_model(variant,key[0])
        loader.validate_state(model,best['state_dict'])
        loader.validate_state(model,last['state_dict'])
        loader.validate_state(model,last['ema'])
        require(done['parameters'] == meta['parameters'] == sum(p.numel() for p in model.parameters()), 'Real architecture parameters differ')
        del model
        if not is_retained_pilot:
            require(meta['source_sha256'].get(str(HERE/'confirm_train.py')) == sha(HERE/'confirm_train.py')
                and meta['source_sha256'].get(str(args.authorization)) == sha(args.authorization), 'New confirmation source/allocation not bound')
            require(all(done.get(k) == expected[k] for k in ('confirmation_variant','confirmation_authorization','original_selected_pilot_gate_pass','allocation_only_not_success')), 'Actual confirmation metadata missing')
        counters = done['counters']
        require(counters['updates'] == counters['finite_updates'] == 3000
            and counters['successful_fine_pixels'] == 3000*4*160*160
            and counters['forward_calls'] == counters['backward_calls'] == 3000+counters['amp_backoffs']
            and counters['validation_forward_calls'] == 14*23 and counters['final_fp32_forward_calls'] == 23,
            'Actual work counters incomplete')
        require(last['counters'] == dict(counters, final_fp32_forward_calls=0), 'Terminal versus final-score counters differ')
        for name in initial['state_dict']:
            if name.endswith('num_batches_tracked'):
                before = int(initial['state_dict'][name])
                require(int(last['state_dict'][name]) == int(last['ema'][name]) == before+3000
                    and int(best['state_dict'][name]) == before+best['step'], 'BN state must reflect successful updates only')
        with np.load(path/'schedule.npz', allow_pickle=False) as schedule:
            require(set(schedule.files) == {'fit_ids','d4'}, 'Unexpected training schedule keys')
            bindings = {}
            for name, shape, limit in [('fit_ids',(3000,4),603),('d4',(3000,),8)]:
                value = schedule[name]
                require(value.shape == shape and np.issubdtype(value.dtype,np.integer)
                    and value.min() >= 0 and value.max() < limit, 'Incomplete actual sample/augmentation schedule')
                bindings[name] = dict(shape=list(shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
        require(sha(path/'schedule.npz') == meta['schedule_sha256'], 'Saved schedule artifact changed')
        if common_schedule is not None: require(bindings == common_schedule, 'Different actual sample/D4 schedule across the six students')
        common_schedule = bindings
        prior.validate_scores(done['selected_fp32'], records)
        initial_done = read(Path(initial_entry['run'])/'complete.json')
        require(done['initialization_fp32_rmse'] == initial_done['selected_fp32']['macro']['rmse']
            and abs(done['improvement_k']-(done['initialization_fp32_rmse']-done['selected_fp32']['macro']['rmse'])) < 1e-12,
            'Initialization/gain record does not match the previous real student')
        for file, digest in meta['source_sha256'].items(): require(sha(file) == digest, 'A bound training source changed')
        evidence.update(meta['source_sha256'])
        evidence.update({str(path/name): sha(path/name) for name in ('run.json','complete.json','validation.json','best.pt','last.pt','schedule.npz')})
        entries.append(dict(architecture=key[0], seed=key[1], run=str(path), original_run=initial_entry['run'],
            checkpoint=str(path/'best.pt'), checkpoint_sha256=done['selected_checkpoint_sha256'], config=cfg,
            effective_method=METHOD, confirmation_variant=variant, retained_original_pilot=is_retained_pilot, original_checkpoint_sha256=initial_entry['checkpoint_sha256']))
        costs.append(dict(architecture=key[0], seed=key[1], seconds=done['seconds'], parameters=done['parameters'],
            counters=counters, selected_step=done['selected_step'], selected_weights=done['selected_weights'],
            actual_start_unix=done['actual_start_unix'], actual_end_unix=done['actual_end_unix']))
        scores[key], done_by_key[key] = done['selected_fp32'], done
        del best, last, initial; gc.collect()
    pilot = done_by_key[('naf_history',20260905)]
    require(pilot['continuation_gate_pass'] is decision['original_selected_pilot_gate_pass']
        and sha(paths[('naf_history',20260905)]/'complete.json') == decision['selected_pilot']['completion']['sha256'],
        'The retained pilot or its original gate changed')
    for key, done in done_by_key.items():
        if key != ('naf_history',20260905):
            require(done['actual_start_unix'] >= decision['authorized_at_unix']
                and done['continuation_gate_pass'] is False, 'Confirmation must follow real allocation without promoting the old pilot gate')
    ordered = sorted(costs,key=lambda x:x['actual_start_unix'])
    require(all(a['actual_end_unix'] <= b['actual_start_unix'] for a,b in zip(ordered,ordered[1:])), 'This recipe requires actual isolated sequential GPU jobs')
    result = p.compare_scores(scores,'naf_history')
    result.update(status='six_actual_strong_teacher_kd_runs_complete', split='validation',
        absolute_three_seed_mean_lt_042=result['mean_metrics']['naf_history']['rmse'] < .42,
        absolute_three_seed_mean_lt_040=result['mean_metrics']['naf_history']['rmse'] < .4,
        actual_continuation_costs=costs, total_process_seconds=sum(c['seconds'] for c in costs),
        teacher_generation_cost=dict(seconds=teacher['seconds'],counters=teacher['counters'],shared_once=True),
        total_additional_seconds_including_shared_teacher=sum(c['seconds'] for c in costs)+teacher['seconds'],
        matched_additional_budget='Each actual original seed/architecture: same 3000 updates, 14 choices, fixed teacher and sample/D4 schedule; sequential isolated GPU jobs.',
        previous_matched_1000_update_stage=inherited['validation'], historical_search_costs_equal=False,
        selected_variant=variant, original_selected_pilot_gate_pass=pilot['continuation_gate_pass'],
        continuation_authority='explicit_root_final_confirmation_budget_allocation',
        single_network_forward=True, scientific_goal_complete=False, test_opened=False)
    value = dict(protocol=PROTOCOL, frozen_at_utc=datetime.now(timezone.utc).isoformat(), frozen_at_unix=time.time(),
        checkpoints=entries, validation=result, source_and_evidence_sha256=evidence, common_schedule=common_schedule,
        teacher_receipt=binding(TEACHER_RECEIPT), teacher=teacher, previous_freeze=binding(PRIOR_FREEZE),
        confirmation_authorization=binding(args.authorization), selected_variant=variant,
        ensemble_used=False, test_opened=False, test_authorized=False, scientific_goal_complete=False)
    # Failed Val gates still produce a results artifact, but no prediction freeze.
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'results.json',result)
    fields(value)
    write(args.output/'continuation_selection_freeze.json',value)
    print(json.dumps(dict(status='six_actual_runs_frozen',output=str(args.output),test_authorized=False)))


def read_freeze(path):
    value = read(path); fields(value)
    decision, _ = allocation.validate(allocation.bound(value['confirmation_authorization']))
    require(value['selected_variant'] == decision['selected_variant']
        and all(e['confirmation_variant'] == decision['selected_variant'] for e in value['checkpoints']), 'The single selected variant differs')
    require(value['source_and_evidence_sha256'].get(str(Path(__file__).resolve())) == sha(__file__), 'Freeze reader source not bound')
    for file, digest in value['source_and_evidence_sha256'].items(): require(sha(file) == digest, 'Frozen evidence changed: '+file)
    for entry in value['checkpoints']:
        done = read(Path(entry['run'])/'complete.json')
        require(done['status']=='complete' and done['updates']==3000 and done['validation_weight_candidates']==14
            and done['single_network_forward'] is True and done['selected_checkpoint_sha256']==entry['checkpoint_sha256']
            and sha(entry['checkpoint'])==entry['checkpoint_sha256'], 'Frozen actual single-student completion changed')
    return value


def check():
    value = dict(protocol=PROTOCOL,ensemble_used=False,test_opened=False,test_authorized=False,
        checkpoints=[dict(architecture=r,seed=s,effective_method=METHOD) for r,s in ORDER],
        validation=dict(status='six_actual_strong_teacher_kd_runs_complete', mean_metrics=dict(naf_history=dict(rmse=.41)),
            criteria=dict(observed_mean_rmse_reduction_k=.05,all_three_seed_macro_improvements=True,network_numeric_criteria_pass=True),
            effects=dict(rmse=dict(paired_seed_differences=[.04,.05,.06]))))
    fields(value)
    for mutate in (lambda v:v['checkpoints'].pop(), lambda v:v['validation']['mean_metrics']['naf_history'].update(rmse=.43),
                   lambda v:v['validation']['criteria'].update(all_three_seed_macro_improvements=False)):
        candidate=json.loads(json.dumps(value));mutate(candidate)
        try: fields(candidate)
        except ValueError: pass
        else: raise AssertionError('A required six-student gate was bypassed')
    print(json.dumps(dict(status='synthetic_six_identity_absolute_and_paired_gates_pass',gpu_used=False,data_opened=False,actual_weights_opened=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','freeze'))
    parser.add_argument('--runs-root',type=Path,default=HERE/'runs');parser.add_argument('--output',type=Path);parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    if args.command=='check':check()
    else:
        if args.output is None or args.authorization is None:parser.error('Explicit root --authorization and new --output required')
        args.authorization=args.authorization.resolve()
        args.runs_root=args.runs_root.resolve();args.output=args.output.resolve()
        remaining=read(EXTENSION)['hard_deadline_unix']-time.time()
        if remaining<=0:raise TimeoutError('Five-hour freeze deadline reached')
        def alarm(signum,frame):raise TimeoutError('Five-hour freeze process alarm')
        old_handler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,remaining)
        try:freeze(args)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
