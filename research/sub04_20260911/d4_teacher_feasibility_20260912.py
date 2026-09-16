"""One frozen alpha25 NAF905 D4-mean Val45 TEACHER feasibility, never a goal model.

No source/weight/view subset search. Both original orientation and eight-view
mean predictions seal before Val labels. run is explicit idle GPU, <=180s and
hard13:00 UTC; check runs only synthetic inversion/solar-vector checks.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
import argparse, gc, importlib.util, json, signal, subprocess, sys, time
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate_finalist as original
p=original.paired
AUGMENT=p.PACKAGE/'reference_models/code/g246_8h_augment.py'
spec=importlib.util.spec_from_file_location('registered_d4_teacher_augmentation',AUGMENT)
aug=importlib.util.module_from_spec(spec);spec.loader.exec_module(aug)
from historylst.hotspots import add_hotspot_metrics
DEADLINE=1789218000
WEIGHT_SHA='7e2c2fa8292054a21a90c7b35902850969f9ac4e55a2fde325800cf6f21c4309'


def read(path):return json.loads(Path(path).read_text())

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def binding(path):return dict(path=str(Path(path).resolve()),sha256=p.sha(path))

def guard(manifest,output,access):
    fields=manifest['roles']['validation']['fields']
    inputs={(p.PACKAGE/fields[k]['path']).resolve() for k in p.runner.INPUTS}
    labels={(p.PACKAGE/fields[k]['path']).resolve() for k in ('target','formal')}
    results={(output/(k+s)).resolve() for k in ('code0','d4mean') for s in ('.npy','.partial.npy')}
    opened=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if 'test' in path.parts or ('data' in path.parts and 'fit' in path.parts):raise RuntimeError('Only Val45 allowed; no Fit/Test')
        if 'labels' in path.parts and (not access['labels'] or path not in labels):raise RuntimeError('Both predictions must seal before approved Val labels')
        if path.suffix in ('.npy','.npz'):
            if path not in inputs|results|(labels if access['labels'] else set()):raise RuntimeError('Unapproved array '+str(path))
            opened.add(str(path))
    sys.addaudithook(audit);return opened


def run(output):
    started=time.perf_counter();available=min(180.,DEADLINE-time.time())
    if available<=0:raise TimeoutError('13:00 UTC deadline reached')
    if output.exists():raise FileExistsError('New output only; no overwrite/retry')
    def deadline():
        if time.perf_counter()-started>=available or time.time()>=DEADLINE:raise TimeoutError('180s/global13:00 deadline reached')
    def timeout(signum,frame):raise TimeoutError('180s/global13:00 process alarm')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,available)
    output.mkdir(parents=True,exist_ok=False);model=None;hook=None;access=dict(labels=False)
    counts=dict(model_forward_calls=0,image_forwards=0,historical_source_encodings=0)
    try:
        manifest=read(p.PACKAGE/'manifest.json');opened=guard(manifest,output,access)
        plan_path=HERE/'weight_interpolation_20260912_v1/interpolation_plan.json';plan=read(plan_path)
        entry,=[e for e in plan['candidates']['0.25'] if (e['architecture'],e['seed'])==('naf_history',20260905)]
        if entry['checkpoint_sha256']!=WEIGHT_SHA or p.sha(entry['checkpoint'])!=WEIGHT_SHA:raise ValueError('Exact frozen alpha25 NAF905 required')
        source={str(q.resolve()):p.sha(q) for q in (Path(__file__),AUGMENT,plan_path,HERE/'naf_history/model.py',
            HERE/'evaluate_finalist.py',HERE/'evaluate_screens.py',p.PACKAGE/'run.py',p.PACKAGE/'manifest.json',
            p.PACKAGE/'historylst/model.py',p.PACKAGE/'historylst/data.py',p.PACKAGE/'historylst/metrics.py',p.PACKAGE/'historylst/hotspots.py')}
        input_bindings={k:dict(path=str(p.PACKAGE/manifest['roles']['validation']['fields'][k]['path']),
            sha256=manifest['roles']['validation']['fields'][k]['sha256']) for k in p.runner.INPUTS}
        for value in input_bindings.values():
            deadline()
            if p.sha(value['path'])!=value['sha256']:raise ValueError('Val input changed')
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if gpu.stdout.strip():raise RuntimeError('GPU occupied; no interruption/wait/retry')
        design=dict(status='frozen_before_prediction_and_labels',checkpoint=binding(entry['checkpoint']),source_sha256=source,
            input_bindings=input_bindings,teacher='one frozen alpha25 NAF905 with all eight fixed D4 transforms, no weight/view subset selection',
            views=list(range(8)),device='cuda',fp32=True,tf32=False,batch=2,repair_dtype='float64',
            aggregation='Registered inverse transform of every view -> FP64 arithmetic mean -> original support repair',
            vectors='Context6/7 solar east/north follow registered transform_context; no ad hoc vector remapping',
            candidates=['same_run_code0','fixed_D4_mean_teacher'],feasibility_gate='code0 macro RMSE - D4mean macro RMSE >=.003K AND D4mean macro RMSE <.42K',
            eight_views_not_a_single_forward_student=True,goal_pass=False,training_started=False,automatic_followup=False,
            job_limit_seconds=available,global_deadline_utc='2026-09-12T13:00:00Z',labels_opened=False,test_opened=False)
        write(output/'design.json',design)
        p.setup('cuda');torch.set_float32_matmul_precision('highest')
        model=original.load_model(entry,'cuda').float().eval().requires_grad_(False)
        data=p.Dataset(p.PACKAGE,'validation',labels=False)
        if len(data)!=45 or len({r['city'] for r in data.records})!=15:raise ValueError('FullVal45 required')
        def counted_stem(module,args):
            batch,channels,h,w=args[0].shape
            if channels!=9 or (h,w)!=(160,160) or batch%9:raise RuntimeError('Unexpected actual source encoding')
            counts['model_forward_calls']+=1;counts['image_forwards']+=batch//9;counts['historical_source_encodings']+=batch
        hook=model.historical.register_forward_pre_hook(counted_stem)
        arrays={k:np.empty((45,1,160,160),np.float64) for k in ('code0','d4mean')};prediction_start=time.perf_counter()
        with torch.inference_mode():
            for first in range(0,45,2):
                ids=np.arange(first,min(first+2,45));raw=data.batch(ids);batch={k:torch.from_numpy(v).cuda() for k,v in raw.items()}
                total=np.zeros((len(ids),1,160,160),np.float64)
                for code in range(8):
                    deadline();transformed=aug.transform_batch(batch,code)
                    with torch.autocast('cuda',enabled=False):pred=model(**transformed)
                    if pred.dtype!=torch.float32:raise RuntimeError('Only FP32 teacher prediction allowed')
                    restored=aug.inverse_field(pred,code).cpu().numpy().astype(np.float64)
                    if not np.isfinite(restored).all():raise RuntimeError('Nonfinite teacher output')
                    if code==0:arrays['code0'][ids]=p.runner.repair(restored,raw['coarse'],raw['support'])
                    total+=restored
                arrays['d4mean'][ids]=p.runner.repair(total/8.,raw['coarse'],raw['support'])
                if first%10==0:print(json.dumps(dict(event='teacher_predictions',scenes=int(ids[-1]+1),seconds=time.perf_counter()-started)),flush=True)
        prediction_seconds=time.perf_counter()-prediction_start
        if counts!=dict(model_forward_calls=184,image_forwards=360,historical_source_encodings=3240):raise RuntimeError('Incomplete eight-view work')
        hook.remove();hook=None;entries={}
        for key,value in arrays.items():
            path=output/(key+'.npy');tmp=output/(key+'.partial.npy')
            with tmp.open('wb') as f:np.save(f,value,allow_pickle=False)
            tmp.replace(path);entries[key]=dict(binding(path),shape=list(value.shape),dtype='float64')
        for path,digest in source.items():
            deadline()
            if p.sha(path)!=digest:raise RuntimeError('Source changed')
        if p.sha(entry['checkpoint'])!=WEIGHT_SHA:raise RuntimeError('Frozen weight changed')
        receipt=dict(status='both_complete_before_labels',entries=entries,checkpoint=binding(entry['checkpoint']),source_sha256=source,
            design_sha256=p.sha(output/'design.json'),manifest_sha256=p.sha(p.PACKAGE/'manifest.json'),
            scene_ids=[r['scene_id'] for r in data.records],device='cuda',fp32=True,tf32=False,batch=2,repair_dtype='float64',
            counters=counts,prediction_seconds=prediction_seconds,opened_arrays=sorted(opened),labels_opened=False,test_opened=False)
        write(output/'predictions_complete.json',receipt)
        # Re-open and verify complete outputs before changing the label access state.
        for key,value in entries.items():
            if p.sha(value['path'])!=value['sha256']:raise RuntimeError('Prediction seal changed')
            checked=np.load(value['path'],mmap_mode='r',allow_pickle=False)
            if checked.shape!=(45,1,160,160) or checked.dtype!=np.float64:raise RuntimeError('Incomplete prediction shape')
        deadline();access['labels']=True
        target=np.load(p.PACKAGE/manifest['roles']['validation']['fields']['target']['path'],mmap_mode='r',allow_pickle=False)
        formal=np.load(p.PACKAGE/manifest['roles']['validation']['fields']['formal']['path'],mmap_mode='r',allow_pickle=False)
        scores={k:add_hotspot_metrics(p.runner.score(v,target,formal,data.records),v,target,formal) for k,v in arrays.items()}
        gain=scores['code0']['macro']['rmse']-scores['d4mean']['macro']['rmse'];gate=gain>=.003 and scores['d4mean']['macro']['rmse']<.42
        write(output/'scores.json',scores)
        write(output/'results.json',dict(status='complete_teacher_feasibility_only',macro={k:v['macro'] for k,v in scores.items()},
            teacher_gain_k=gain,teacher_feasibility_gate_pass=bool(gate),gate_is_not_goal_pass=True,goal_pass=False,
            single_forward_student_produced=False,training_started=False,automatic_followup=False,
            predictions_complete=binding(output/'predictions_complete.json'),scores=binding(output/'scores.json'),
            source_sha256=source,counters=counts,seconds=time.perf_counter()-started,validation_labels_opened=True,test_opened=False,
            limitation='Adaptive Val development teacher diagnostic only; eight forwards cannot substitute for a single-forward student or three-seed goal evidence'))
        print(json.dumps(dict(status='complete',code0=scores['code0']['macro']['rmse'],d4mean=scores['d4mean']['macro']['rmse'],gain=gain,teacher_feasibility_gate_pass=bool(gate))),flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        write(output/'failed.json',dict(status='failed',error=repr(exc),counters=counts,seconds=time.perf_counter()-started,
            validation_labels_opened=access['labels'],test_opened=False,automatic_retry=False));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if hook is not None:hook.remove()
        del model;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def check():
    batch=dict(fine=torch.arange(52*8*8,dtype=torch.float32).reshape(1,52,8,8),coarse=torch.arange(4,dtype=torch.float32).reshape(1,1,2,2),
        support=torch.ones(1,1,8,8),context=torch.arange(15,dtype=torch.float32)[None],emissivity=torch.ones(1,4,8,8),history=torch.ones(1,9,9,8,8))
    for code in range(8):
        changed=aug.transform_batch(batch,code);restored=aug.transform_batch(changed,aug.inverse_code(code))
        for key in batch:assert torch.equal(restored[key],batch[key])
        assert torch.equal(aug.inverse_field(changed['fine'][:,:1],code),batch['fine'][:,:1])
    print(json.dumps(dict(status='registered_D4_inverse_and_Context_solar_check_pass',data_opened=False,weights_opened=False,gpu_used=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','run'));parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.command=='check':check()
    elif args.output is None:parser.error('Explicit new --output required')
    else:run(args.output.resolve())
