"""Explicit JSON manifest of six real completions -> Val summary/weight freeze.

No training, inference or Test. This native continuation freeze is deliberately
not accepted by the existing final_effective_evaluation v1 protocol.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,statistics,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];PACKAGE=ROOT/'resources/historylst246'
sys.path.insert(0,str(ROOT/'research/strong_history_20260911'))
import paired_evaluation as paired
import numpy as np
import torch
ORDER=[(role,seed) for seed in (20260905,20260912,20260913) for role in ('baseline','naf_history')]
METRICS=('rmse','mae','hotspot_iou','hotspot_mae')
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def validate_scores(score,records):
    require([(r['scene_id'],r['city'],r['region']) for r in score['scenes']]
        ==[(r['scene_id'],r['city'],r['region']) for r in records], 'Complete unchanged Val45 identities required')
    require(len(score['cities'])==15 and len(score['regions'])==3,'Complete cities/regions required')
    for city in score['cities']:
        rows=[r for r in score['scenes'] if r['city']==city['city']]
        require(len(rows)==3,'Three dates per city required')
        for k in METRICS:require(math.isfinite(city[k]) and abs(statistics.mean(r[k] for r in rows)-city[k])<1e-12,'City metric mismatch')
    for region,result in score['regions'].items():
        rows=[r for r in score['cities'] if r['region']==region]
        require(len(rows)==5,'Five cities per region required')
        for k in METRICS:require(abs(statistics.mean(r[k] for r in rows)-result[k])<1e-12,'Region metric mismatch')
    for k in METRICS:require(abs(statistics.mean(r[k] for r in score['regions'].values())-score['macro'][k])<1e-12,'Macro mismatch')

def run(args):
    require(not args.output.exists(),'New output directory only')
    rows=read(args.runs)['runs'];items={(r['architecture'],r['seed']):r for r in rows}
    require(len(rows)==6 and set(items)==set(ORDER),'Exactly six completed architecture/seed runs required')
    design=read(HERE/'design.json');manifest=read(PACKAGE/'manifest.json');records=manifest['roles']['validation']['scenes']
    original={(r['architecture'],r['seed']):r for r in design['original_sources']}
    execution_design_path=HERE/'shared_execution_design.json';execution_design=read(execution_design_path)
    require(execution_design['status']=='execution_change_declared_before_shared_runs'
        and execution_design['original_trainer_sha256']==sha(HERE/'train.py')
        and execution_design['shared_entry_sha256']==sha(HERE/'train_shared_gpu.py')
        and execution_design['planned_runs_sha256']==sha(HERE/'planned_runs.json'), 'Execution-context declaration changed')
    require(len(records)==45,'Original complete Val45 required')
    paths=[(args.runs.parent/Path(items[key]['run'])).resolve() for key in ORDER]
    # Only schedule arrays from those explicit completed runs can be read.
    allowed={(p/'schedule.npz').resolve() for p in paths}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix in ('.npy','.npz') and p not in allowed:raise RuntimeError('No observation/prediction/Test arrays allowed')
        if 'data' in p.parts or 'network_test' in p.parts:raise RuntimeError('No dataset or Test access')
    sys.addaudithook(guard)
    scores,entries,costs,files,done_by_key={},[],[],{},{}
    shared=None;teacher_binding=None
    teacher_path=args.teacher_receipt.resolve();teacher=read(teacher_path)
    teacher_hash=sha(teacher_path)
    require(teacher['status']=='complete_Fit603_D4_teacher_before_labels' and teacher['split']=='fit'
        and teacher['architecture']=='naf_history' and teacher['seed']==20260905 and teacher['alpha']==.25
        and teacher['checkpoint']['sha256']=='7e2c2fa8292054a21a90c7b35902850969f9ac4e55a2fde325800cf6f21c4309', 'Exact shared teacher required')
    require(teacher['views']==list(range(8)) and teacher['aggregation']=='registered_inverse_then_float64_equal_mean_then_original_support_repair'
        and teacher['device']=='cuda' and teacher['fp32'] is True and teacher['tf32'] is False
        and teacher['batch']==2 and teacher['repair_dtype']=='float64', 'Teacher execution protocol mismatch')
    require(teacher['scene_ids']==[r['scene_id'] for r in manifest['roles']['fit']['scenes']] and len(teacher['scene_ids'])==603
        and teacher['manifest_sha256']==sha(PACKAGE/'manifest.json'), 'Full Fit603 teacher identity required')
    require(all(teacher[k] is False for k in ('labels_opened','fit_labels_opened','validation_arrays_opened','test_opened')), 'Teacher labels/Test opened')
    require(teacher['teacher']['shape']==[603,1,160,160] and teacher['teacher']['dtype']=='float64', 'Complete teacher shape required')
    require(teacher['counters']['model_forward_calls']==2416 and teacher['counters']['image_forwards']==4824
        and teacher['counters']['historical_source_encodings']==43416 and math.isfinite(teacher['seconds']) and teacher['seconds']>0, 'Actual eight-view teacher cost required')
    for field in ('fine','coarse','support','context','emissivity','history'):
        actual=teacher['input_bindings'][field];expected=manifest['roles']['fit']['fields'][field]
        require(Path(actual['path']).resolve()==(PACKAGE/expected['path']).resolve() and actual['sha256']==expected['sha256'], 'Shared teacher input mismatch')
    for p,h in teacher['source_sha256'].items():require(sha(p)==h,'Teacher source/evidence changed')
    files.update(teacher['source_sha256']);files[str(teacher_path)]=teacher_hash
    files[teacher['checkpoint']['path']]=teacher['checkpoint']['sha256']
    require(sha(teacher['checkpoint']['path'])==teacher['checkpoint']['sha256'],'Teacher checkpoint changed')
    teacher_binding=dict(path=str(teacher_path),sha256=teacher_hash)
    for key,path in zip(ORDER,paths):
        meta,done,validation=(read(path/name) for name in ('run.json','complete.json','validation.json'))
        cfg=meta['config'];source=original[key];shared_mode=key[1]!=20260905
        run_limit=540 if shared_mode else 360
        expected=dict(design['recipe'],architecture=key[0],original_seed=key[1],protocol='d4_self_distillation_20260912_v1',
            algorithm='AdamW_fused_cuda',optimizer_fused=True,per_run_limit_seconds=run_limit,arm='full_control',initialization_sha256=source['checkpoint_sha256'])
        require(all(cfg.get(k)==v for k,v in expected.items()),'Recipe or original initialization mismatch')
        require(not meta['smoke_only'] and meta['device']=='cuda' and meta['fit_queries']==603 and meta['validation_queries']==45,
            'Real complete GPU training cohort required')
        require(meta['teacher']==teacher and meta['teacher_receipt_sha256']==teacher_hash
            and meta['source_sha256'].get(str(teacher_path))==teacher_hash
            and meta['teacher_cache_shared_for_both_architectures'] is True and meta['teacher_updated'] is False
            and cfg['teacher_cache_sha256']==teacher['teacher']['sha256']
            and done['teacher_cache_sha256']==teacher['teacher']['sha256']
            and done['teacher_receipt_sha256']==teacher_hash
            and done['student_inference_views']==1 and done['teacher_inference_used_at_validation'] is False, 'Changed teacher or non-single-view student')
        require(done['status']=='complete' and done['updates']==1000 and done['validation_weight_candidates']==10
            and done['selection_not_changed_by_fp32_gate'] and done['test_opened'] is False and 0<done['seconds']<=run_limit,
            'Incomplete/over-budget/altered selection run')
        if shared_mode:
            companion=paths[ORDER.index(('baseline' if key[0]=='naf_history' else 'naf_history',key[1]))]
            context=meta['execution_context']
            require(cfg['execution_mode']=='same_gpu_planned_architecture_pair' and cfg['companion_output']==str(companion)
                and context==done['execution_context'] and context['mode']=='same_gpu_planned_architecture_pair'
                and context['companion_output']==str(companion) and context['same_seed']==key[1]
                and context['per_run_limit_seconds']==540 and context['isolated_timing'] is False
                and meta['isolated_training_timing'] is False and done['isolated_training_timing'] is False
                and context['scientific_recipe_unchanged'] is True, 'Shared execution context does not match same-seed planned companion')
            start,end=done['execution_start_unix'],done['execution_end_unix']
            require(meta['execution_start_unix']==start and math.isfinite(start) and math.isfinite(end) and end>start
                and abs((end-start)-done['seconds'])<.1 and end<=design['hard_deadline_unix'], 'Actual shared execution span invalid')
            for filename in ('train_shared_gpu.py','shared_execution_design.json','planned_runs.json'):
                p=HERE/filename;require(meta['source_sha256'].get(str(p))==sha(p), 'Actual shared source/context not bound')
            for occupant in context['verified_active_companions_at_admission']:
                require(occupant['output']==str(companion) and occupant['seed']==key[1]
                    and occupant['architecture']!=key[0] and occupant['pid']>0, 'Unexpected admitted GPU occupant')
            require(len(context['verified_active_companions_at_admission'])<=1,'Multiple companions admitted')
        else:
            require('execution_mode' not in cfg and 'execution_context' not in meta
                and meta['source_sha256'].get(str(HERE/'train.py'))==execution_design['original_trainer_sha256']
                and str(HERE/'train_shared_gpu.py') not in meta['source_sha256'], 'Seed905 must retain original isolated entry')
            start=end=None
        require([(r['step'],r['weights']) for r in validation]==[(n,w) for n in range(0,1001,250) for w in ('raw','ema')],'Exactly10 opportunities required')
        require(all(math.isfinite(r['validation_macro_rmse']) and math.isfinite(r['seconds']) and r['seconds']>0 for r in validation), 'All ten complete Val results must be finite')
        best=min(validation,key=lambda r:r['validation_macro_rmse'])
        require((done['selected_step'],done['selected_weights'],done['best_amp_rmse'])==(best['step'],best['weights'],best['validation_macro_rmse']), 'Selected minimum mismatch')
        ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
        require(not ck['smoke_only'] and ck['config']==cfg and ck['source_sha256']==meta['source_sha256']
            and ck['initialization']==meta['initialization'] and ck['selection_metric']=='full9_macro_rmse'
            and (ck['step'],ck['weights'],ck['selection_score'])==(best['step'],best['weights'],best['validation_macro_rmse'])
            and sha(path/'best.pt')==done['selected_checkpoint_sha256'],'Actual selected checkpoint does not match receipts')
        require(all(torch.isfinite(v).all().item() for v in ck['state_dict'].values()), 'Nonfinite selected state')
        selected_state=ck['state_dict']
        del ck
        init=meta['initialization']
        original_complete=read(source['completion'])
        require(sha(source['checkpoint'])==source['checkpoint_sha256']
            and sha(source['completion'])==source['completion_sha256']
            and (original_complete['status'],original_complete['updates'],original_complete['validation_candidates'])==('complete',18000,50),
            'Original 18k/50 completion or original best has changed')
        require(init['checkpoint']==source['checkpoint'] and init['checkpoint_sha256']==source['checkpoint_sha256']
            and init['source_completion']==source['completion'] and init['source_completion_sha256']==source['completion_sha256']
            and init['strict_state_equality'] and init['full9_original_cpu_max_abs_difference_k']==0. and init['fresh_optimizer_ema_scaler'],
            'Each exact original best/completion required')
        original_ck=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        require(last['step']==1000 and last['config']==cfg and last['source_sha256']==meta['source_sha256']
            and last['schedule']==meta['schedule'] and last['smoke_only'] is False, 'Actual terminal raw/EMA state required')
        require(all(g.get('fused') is True for g in last['optimizer']['param_groups']), 'Both actual CUDA optimizers must be fused')
        bn=[]
        for name,value in original_ck['state_dict'].items():
            if name.endswith('num_batches_tracked'):
                before=int(value);raw=int(last['state_dict'][name]);ema=int(last['ema'][name]);selected=int(selected_state[name])
                require(raw-before==1000 and ema==raw and selected-before==done['selected_step'], 'BN retries or selection changed running update count')
                bn.append(dict(name=name,before=before,raw_after=raw,ema_after=ema,selected=selected))
        require(bool(bn)==(key[0]=='baseline'), 'Unexpected architecture BN-buffer contract')
        terminal_counters=last['counters']
        del original_ck,last,selected_state
        for source_path,h in design['source_sha256'].items():require(meta['source_sha256'].get(source_path)==h and sha(source_path)==h,'Shared semantic/source seal mismatch')
        for source_path,h in meta['source_sha256'].items():require(sha(source_path)==h,'Run source/evidence changed')
        files.update(meta['source_sha256'])
        schedule=meta['schedule']
        if shared is not None:require(schedule==shared,'Sample/D4/shared teacher schedule differs across paired repetitions')
        shared=schedule
        with np.load(path/'schedule.npz',allow_pickle=False) as plan:
            for field,digest_key,shape in (('fit_ids','fit_ids_sha256',(1000,4)),('d4','d4_sha256',(1000,))):
                value=plan[field];require(np.issubdtype(value.dtype,np.integer) and value.min()>=0 and value.max()<(603 if field=='fit_ids' else 8), 'Invalid schedule index')
                require(value.shape==shape and hashlib.sha256(value.tobytes()).hexdigest()==schedule[digest_key],'Actual saved schedule mismatch')
        require(schedule['teacher_cache_sha256']==teacher['teacher']['sha256'] and schedule['history_source_count']==9
            and schedule['full160_updates']==1000 and schedule['dropout_policy']=='history0.25_emissivity0.25', 'Schedule metadata mismatch')
        counters=done['counters'];attempts=counters['amp_attempts']
        require(all(terminal_counters[k]==v for k,v in counters.items() if k!='final_fp32_forward_calls')
            and terminal_counters['final_fp32_forward_calls']==0, 'Terminal pre-FP32 and completed counters differ')
        require(counters['optimizer_updates']==counters['finite_loss_and_gradient_updates']==counters['full160_updates']==1000
            and counters['skipped_optimizer_updates']==0 and counters['successful_update_fine_pixels']==102400000
            and attempts==1000+counters['amp_backoffs'] and counters['training_forward_calls']==counters['training_backward_calls']==attempts
            and counters['validation_forward_calls']==10*23 and counters['final_fp32_forward_calls']==2*23
            and counters['attempted_forward_fine_pixels']==attempts*4*160*160
            and counters['attempted_history_frame_pixels']==attempts*4*160*160*9,'Actual work counters inconsistent')
        for name in ('selected_fp32','initialization_fp32'):validate_scores(done[name],records)
        scores[key]=done['selected_fp32'];done_by_key[key]=done
        for name in ('run.json','complete.json','validation.json','best.pt','last.pt','schedule.npz'):files[str(path/name)]=sha(path/name)
        entries.append(dict(architecture=key[0],seed=key[1],run=str(path),checkpoint=str(path/'best.pt'),
            checkpoint_sha256=done['selected_checkpoint_sha256'],config=cfg,effective_method='d4_self_distillation',
            original_checkpoint_sha256=source['checkpoint_sha256'],original_completion_sha256=source['completion_sha256']))
        costs.append(dict(architecture=key[0],seed=key[1],seconds=done['seconds'],parameters=done['parameters'],counters=counters,
            selected_step=done['selected_step'],selected_weights=done['selected_weights'],bn_counters=bn,
            bn_retry_running_buffer_restoration_verified=True,optimizer_fused=True,
            execution_mode='same_gpu_planned_architecture_pair' if shared_mode else 'original_isolated_entry',
            per_run_limit_seconds=run_limit,execution_start_unix=start,execution_end_unix=end,
            isolated_training_timing=not shared_mode))
    pilot=done_by_key['naf_history',20260905];pilot_path=paths[ORDER.index(('naf_history',20260905))]/'complete.json'
    require(pilot['continuation_gate_pass'] is True and pilot['selected_fp32']['macro']['rmse']<.42
        and pilot['locked_alpha25_reference_rmse_k']-pilot['selected_fp32']['macro']['rmse']>=.003,'Original pilot admission gate did not pass')
    for key,path in zip(ORDER,paths):
        if key!=('naf_history',20260905):require(read(path/'run.json')['source_sha256'].get(str(pilot_path))==sha(pilot_path),'Follow-up lacks actual passing-pilot binding')
    shared_spans=[]
    for seed in (20260912,20260913):
        pair=[c for c in costs if c['seed']==seed]
        require(len(pair)==2 and all(c['execution_mode']=='same_gpu_planned_architecture_pair' for c in pair),'Both architectures must share context per seed')
        start=min(c['execution_start_unix'] for c in pair);end=max(c['execution_end_unix'] for c in pair)
        overlap=min(c['execution_end_unix'] for c in pair)-max(c['execution_start_unix'] for c in pair)
        require(overlap>0,'Claimed same-seed GPU sharing must have actual temporal overlap')
        shared_spans.append(dict(seed=seed,execution_start_unix=start,execution_end_unix=end,physical_span_seconds=end-start,
            overlap_seconds=overlap,sum_process_seconds=sum(c['seconds'] for c in pair)))
    require(shared_spans[0]['execution_end_unix']<=shared_spans[1]['execution_start_unix'],'The two seed pairs must run sequentially')
    isolated_seconds=sum(c['seconds'] for c in costs if c['seed']==20260905)
    result=paired.compare_scores(scores,'naf_history')
    result.update(status='six_actual_d4_self_distillation_runs_complete',split='validation',
        absolute_three_seed_mean_lt_042=result['mean_metrics']['naf_history']['rmse']<.42,
        absolute_three_seed_mean_lt_040=result['mean_metrics']['naf_history']['rmse']<.40,
        actual_continuation_costs=costs,total_process_seconds=sum(x['seconds'] for x in costs),
        teacher_generation_cost=dict(seconds=teacher['seconds'],counters=teacher['counters'],shared_once=True),
        total_additional_seconds_including_shared_teacher=sum(x['seconds'] for x in costs)+teacher['seconds'],
        teacher_cache_not_reopened_by_summarizer=True,execution_context_version=2,shared_gpu_pair_spans=shared_spans,
        isolated_seed905_process_seconds=isolated_seconds,
        actual_training_work_span_seconds_excluding_between_run_gaps=isolated_seconds+sum(x['physical_span_seconds'] for x in shared_spans),
        timing_scope='Process seconds include overlap and are not additive wall time. Shared pair physical spans use first-line start/final-report end; isolated905 timestamps unavailable. Work-span sum excludes between-run gaps and is not end-to-end wall time or isolated throughput.',
        matched_additional_budget='Same1000 updates/10choices per architecture/seed; seed905 isolated360s, seeds912/913 pairedGPU540s. Architectures matched within each seed; execution differs across seeds and historical total search costs remain unequal',
        test_opened=False,scientific_goal_complete=False,automatic_followup=False)
    result['eligible_for_separate_consumed_test_followup']=bool(result['absolute_three_seed_mean_lt_042']
        and result['criteria']['network_numeric_criteria_pass'])
    files[str(execution_design_path)]=sha(execution_design_path);files[str(HERE/'train_shared_gpu.py')]=sha(HERE/'train_shared_gpu.py');files[str(args.runs)]=sha(args.runs);files[str(HERE/'design.json')]=sha(HERE/'design.json');files[str(Path(__file__).resolve())]=sha(__file__)
    files[str(Path(paired.__file__).resolve())]=sha(paired.__file__)
    freeze=dict(protocol='d4_self_distillation_six_selected_val_weights_v2',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoints=entries,source_and_evidence_sha256=files,schedule=shared,validation=result,
        teacher_receipt=teacher_binding,teacher=teacher,execution_context_version=2,execution_design=dict(path=str(execution_design_path),sha256=sha(execution_design_path)),
        ensemble_used=False,test_opened=False,test_authorized=False,existing_final_evaluator_compatible=False,
        not_an_independent_holdout_claim=True,scientific_goal_complete=False)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in (('results.json',result),('continuation_selection_freeze.json',freeze)):
        with (args.output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='complete',output=str(args.output),absolute_three_seed_mean_lt_042=result['absolute_three_seed_mean_lt_042'],test_opened=False)))

def read_freeze(path):
    """Return bound freeze; no dataset, prediction, or teacher arrays opened."""
    frozen=read(path)
    require(frozen['protocol']=='d4_self_distillation_six_selected_val_weights_v2'
        and frozen['ensemble_used'] is False and frozen['test_opened'] is False
        and frozen['validation']['status']=='six_actual_d4_self_distillation_runs_complete', 'Wrong freeze protocol')
    require([(e['architecture'],e['seed']) for e in frozen['checkpoints']]==ORDER,'Exact six frozen students required')
    require(frozen['validation']['eligible_for_separate_consumed_test_followup'] is True
        and frozen['validation']['mean_metrics']['naf_history']['rmse']<.42
        and frozen['validation']['criteria']['all_three_seed_macro_improvements'] is True
        and frozen['validation']['criteria']['observed_mean_rmse_reduction_k']>=.01,'Val admission gate not passed')
    require(frozen['source_and_evidence_sha256'].get(str(Path(__file__).resolve()))==sha(__file__), 'Freeze writer source not bound')
    for p,h in frozen['source_and_evidence_sha256'].items():require(sha(p)==h,'Frozen source/evidence changed: '+p)
    require(frozen['execution_context_version']==2 and frozen['validation']['execution_context_version']==2,'Execution context version missing')
    for entry in frozen['checkpoints']:
        require(sha(entry['checkpoint'])==entry['checkpoint_sha256'],'Frozen weight changed')
        cfg=entry['config'];done=read(Path(entry['run'])/'complete.json')
        require(cfg['protocol']=='d4_self_distillation_20260912_v1' and cfg['updates']==1000 and cfg['optimizer_fused'] is True
            and done['status']=='complete' and done['updates']==1000 and done['validation_weight_candidates']==10
            and done['student_inference_views']==1 and done['teacher_inference_used_at_validation'] is False
            and done['selected_checkpoint_sha256']==entry['checkpoint_sha256']
            and done['teacher_receipt_sha256']==frozen['teacher_receipt']['sha256']
            and done['teacher_cache_sha256']==frozen['teacher']['teacher']['sha256']
            and cfg['per_run_limit_seconds']==(360 if entry['seed']==20260905 else 540)
            and (entry['seed']==20260905 or (cfg['execution_mode']=='same_gpu_planned_architecture_pair'
                and done['isolated_training_timing'] is False)), 'Frozen selected student mismatch')
    return frozen

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','freeze'))
    parser.add_argument('--runs',type=Path);parser.add_argument('--teacher-receipt',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.command=='check':print(json.dumps(dict(paired.check(),actual_runs_opened=False,weights_opened=False,gpu_used=False)))
    else:
        if args.runs is None or args.output is None or args.teacher_receipt is None:parser.error('Explicit --runs, --teacher-receipt and new --output required')
        args.runs=args.runs.resolve();args.output=args.output.resolve();torch.set_num_threads(1);run(args)
