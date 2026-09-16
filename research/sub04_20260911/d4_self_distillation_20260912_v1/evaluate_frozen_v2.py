"""Explicit consumed-Test30 follow-up for six actually completed D4-distilled students.

Only real matching completion freeze passing common three-seed Val gates is
accepted. Prediction is one forward per student, never the eight-view teacher.
All six FP32/batch1 -> FP64 repaired Test90 arrays seal before target/formal.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import argparse, copy, gc, importlib.util, json, signal, subprocess, sys, time
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;SUB04=HERE.parent
sys.path.insert(0,str(SUB04))
import evaluate_finalist as original
p=original.paired
from historylst.hotspots import add_hotspot_metrics
DEADLINE=1789218000
ORDER=[(role,seed) for seed in p.SEEDS for role in ('baseline','naf_history')]
PROTOCOL='d4_self_distillation_six_selected_val_weights_v2'


def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def identity(entries):return [(e['architecture'],e['seed'],e['checkpoint_sha256']) for e in entries]

def fields(value):
    if (value['protocol']!=PROTOCOL or value['ensemble_used'] is not False or value['test_opened'] is not False
        or [(e['architecture'],e['seed']) for e in value['checkpoints']]!=ORDER
        or any(e['effective_method']!='d4_self_distillation' for e in value['checkpoints'])):raise ValueError('Exact six selected student freeze required')
    v=value['validation'];effects=v['effects']['rmse'];means=v['mean_metrics']
    if (means['naf_history']['rmse']>=.42 or means['baseline']['rmse']-means['naf_history']['rmse']<.01
        or len(effects['paired_seed_differences'])!=3 or not all(x>0 for x in effects['paired_seed_differences'])
        or v['criteria']['network_numeric_criteria_pass'] is not True):raise ValueError('Three-seed common Val gates did not pass')


def reader():
    path=HERE/'summarize_matched_v2.py'
    spec=importlib.util.spec_from_file_location('strict_d4_student_completion_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_seal():
    return {str(q.resolve()):p.sha(q) for q in (Path(__file__),HERE/'summarize_matched_v2.py',SUB04/'evaluate_finalist.py',
        SUB04/'evaluate_screens.py',Path(p.__file__),p.PACKAGE/'historylst/data.py',p.PACKAGE/'historylst/metrics.py',
        p.PACKAGE/'historylst/hotspots.py',p.PACKAGE/'historylst/model.py',SUB04/'naf_history/model.py')}


def guard(manifest,output,value,access):
    role=manifest['roles']['test'];inputs={(p.PACKAGE/role['fields'][k]['path']).resolve() for k in p.runner.INPUTS}
    labels={(p.PACKAGE/role['fields'][k]['path']).resolve() for k in ('target','formal')}
    predictions={(output/f"{r}_{s}.npy").resolve() for r,s in ORDER}
    evidence={Path(q).resolve() for q in value['source_and_evidence_sha256']}
    opened=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if 'labels' in path.parts and (not access['labels'] or path not in labels):raise RuntimeError('All six predictions required before Test target/formal')
        if 'data' in path.parts and {'fit','validation'} & set(path.parts):raise RuntimeError('No Fit/Val data in student Test inference')
        if path.suffix in ('.npy','.npz'):
            allowed=inputs|predictions|evidence|(labels if access['labels'] else set())
            if path not in allowed:raise RuntimeError('Unapproved array: '+str(path))
            opened.add(str(path))
    sys.addaudithook(audit);return opened


def execute(args):
    start=time.perf_counter();remaining=DEADLINE-time.time()
    if remaining<=0:raise TimeoutError('13:00 UTC deadline reached')
    def deadline():
        if time.time()>=DEADLINE:raise TimeoutError('13:00 UTC deadline reached')
    def alarm(signum,frame):raise TimeoutError('13:00 UTC process alarm')
    old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,remaining)
    loader=reader();value=loader.read_freeze(args.freeze);fields(value)
    digest=p.sha(args.freeze);sources=source_seal();entries=copy.deepcopy(value['checkpoints'])
    manifest=read(p.PACKAGE/'manifest.json');records=manifest['roles']['test']['scenes']
    if len(records)!=90 or len({r['city'] for r in records})!=30 or len({r['region'] for r in records})!=3:raise ValueError('Require complete consumed Test90/30city/3region')
    access=dict(labels=False);opened=guard(manifest,args.output,value,access);model=None
    common=dict(split='test',cohort_identity='Consumed 30-city follow-up',test_previously_consumed=True,
        independent_holdout=False,student_inference_views=1,teacher_eight_views_used_at_inference=False,
        model_or_method_reselected=False,scientific_goal_complete=False,freeze_sha256=digest,evaluator_source_sha256=sources)
    try:
        if args.command=='predict':
            if args.output.exists():raise FileExistsError('New output only; no overwrite/retry')
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if gpu.stdout.strip():raise RuntimeError('GPU occupied; no wait or interruption')
            p.setup('cuda');torch.set_float32_matmul_precision('highest');args.output.mkdir(parents=True,exist_ok=False)
            receipt=dict(common,stage='predicting',device='cuda',dtype='float32',tf32=False,batch=1,repair_dtype='float64',
                labels_opened=False,entries=entries,scene_ids=[r['scene_id'] for r in records],
                explicit_predict_call_authorizes_this_frozen_test_execution=True,
                shared_teacher_receipt=value['teacher_receipt'],inherited_validation_costs=value['validation'].get('actual_continuation_costs'))
            write(args.output/'prediction_started.json',receipt)
            data=p.Dataset(p.PACKAGE,'test',labels=False)
            with torch.inference_mode():
                for entry in entries:
                    deadline();model=original.load_model(entry,'cuda').float().eval();parts=[];began=time.perf_counter()
                    for i in range(90):
                        deadline();batch=p.runner.batch(data,[i],'cuda')
                        with torch.autocast('cuda',enabled=False):out=p.runner.forward(model,batch)
                        if out.dtype!=torch.float32:raise RuntimeError('FP32 student required')
                        parts.append(out.cpu().numpy())
                    prediction=p.runner.repair(np.concatenate(parts),data.arrays['coarse'],data.arrays['support'])
                    support=np.asarray(data.arrays['support'],bool)
                    if not np.isfinite(prediction[support]).all() or not np.isnan(prediction[~support]).all():raise RuntimeError('Invalid support')
                    name=f"{entry['architecture']}_{entry['seed']}.npy";path=args.output/name
                    with path.open('xb') as f:np.save(f,prediction,allow_pickle=False)
                    entry.update(prediction=name,prediction_sha256=p.sha(path),shape=[90,1,160,160],output_dtype='float64',
                        image_forwards=90,inference_seconds_including_io=time.perf_counter()-began,timing_is_benchmark=False)
                    print(json.dumps(dict(event='student_prediction_sealed',architecture=entry['architecture'],seed=entry['seed'])),flush=True)
                    del model,prediction,parts;model=None;gc.collect();torch.cuda.empty_cache()
            if p.sha(args.freeze)!=digest or source_seal()!=sources:raise RuntimeError('Frozen sources changed')
            loader.read_freeze(args.freeze)
            receipt.update(stage='all_predictions_sealed',seconds=time.perf_counter()-start,total_image_forwards=540,opened_arrays=sorted(opened))
            write(args.output/'predictions_complete.json',receipt)
        else:
            if (args.output/'results.json').exists():raise FileExistsError('Never rescore an existing result')
            receipt_path=args.output/'predictions_complete.json';receipt=read(receipt_path)
            if (receipt['stage']!='all_predictions_sealed' or receipt['freeze_sha256']!=digest
                or receipt['device']!='cuda' or receipt['dtype']!='float32' or receipt['tf32'] is not False or receipt['batch']!=1
                or receipt['labels_opened'] is not False or receipt['evaluator_source_sha256']!=sources
                or identity(receipt['entries'])!=identity(entries) or receipt['scene_ids']!=[r['scene_id'] for r in records]
                or receipt['total_image_forwards']!=540 or receipt['student_inference_views']!=1):raise ValueError('All six same frozen student predictions must seal first')
            arrays={}
            for entry in receipt['entries']:
                deadline();path=args.output/entry['prediction']
                if p.sha(path)!=entry['prediction_sha256']:raise ValueError('Prediction hash mismatch')
                a=np.load(path,mmap_mode='r',allow_pickle=False)
                if a.shape!=(90,1,160,160) or a.dtype!=np.float64:raise ValueError('Incomplete repaired prediction')
                arrays[(entry['architecture'],entry['seed'])]=a
            access['labels']=True
            target=np.load(p.PACKAGE/manifest['roles']['test']['fields']['target']['path'],mmap_mode='r',allow_pickle=False)
            formal=np.load(p.PACKAGE/manifest['roles']['test']['fields']['formal']['path'],mmap_mode='r',allow_pickle=False)
            scores={}
            for key,prediction in arrays.items():
                deadline();result=p.score(prediction,target,formal,records);add_hotspot_metrics(result,prediction,target,formal)
                scores[key]=result;write(args.output/f'{key[0]}_{key[1]}_scores.json',result)
            result=p.compare_scores(scores,'naf_history');result.update(common,prediction_receipt_sha256=p.sha(receipt_path),
                absolute_three_seed_mean_lt_042=result['mean_metrics']['naf_history']['rmse']<.42,
                absolute_three_seed_mean_lt_040=result['mean_metrics']['naf_history']['rmse']<.4,
                full_goal_complete=False,usage_cost_branch_evaluated=False,opened_arrays=sorted(opened),
                limitation='Consumed Test30 follow-up, not independent confirmation; no Test-driven method/weight selection. Three-seed bootstrap is descriptive.')
            if p.sha(args.freeze)!=digest or source_seal()!=sources:raise RuntimeError('Frozen source changed')
            loader.read_freeze(args.freeze);write(args.output/'results.json',result)
            print(json.dumps(dict(mean_metrics=result['mean_metrics'],criteria=result['criteria'],goal_pass=False)),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        del model;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def check():
    value=dict(protocol=PROTOCOL,ensemble_used=False,test_opened=False,
        checkpoints=[dict(architecture=r,seed=s,effective_method='d4_self_distillation') for r,s in ORDER],
        validation=dict(mean_metrics=dict(baseline=dict(rmse=.46),naf_history=dict(rmse=.41)),
            effects=dict(rmse=dict(paired_seed_differences=[.04,.05,.06])),criteria=dict(network_numeric_criteria_pass=True)))
    fields(value);value['validation']['mean_metrics']['naf_history']['rmse']=.43
    try:fields(value)
    except ValueError:pass
    else:raise AssertionError('Absolute gate must reject')
    print(json.dumps(dict(status='import_and_synthetic_freeze_gates_pass',data_opened=False,weights_opened=False,gpu_used=False,test_opened=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','predict','score'))
    parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.command=='check':check()
    else:
        if args.freeze is None or args.output is None:parser.error('Real completed --freeze and explicit --output required')
        args.freeze=args.freeze.resolve();args.output=args.output.resolve();execute(args)
