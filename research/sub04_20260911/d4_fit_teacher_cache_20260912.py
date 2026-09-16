"""One fixed alpha25 NAF905 all-eight-D4 Fit603 teacher cache; no labels/Val/Test arrays.

Uses the already verified Val teacher implementation and binds its actual JSON
receipts without reopening its predictions or labels. Every future student
shares this one cache; generation cost is additional, never hidden training.
"""
from pathlib import Path
import argparse, gc, json, os, signal, subprocess, sys, time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import d4_teacher_feasibility_20260912 as fixed
import numpy as np
import torch
p,aug,original=fixed.p,fixed.aug,fixed.original
DEFAULT_FEASIBILITY=HERE/'d4_teacher_val_20260912_v1/results.json'
AGGREGATION='registered_inverse_then_float64_equal_mean_then_original_support_repair'


def guard(manifest,output):
    allowed={(p.PACKAGE/manifest['roles']['fit']['fields'][k]['path']).resolve() for k in p.runner.INPUTS}
    allowed.update((output/name).resolve() for name in ('teacher.npy','teacher.partial.npy'))
    opened=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if 'labels' in path.parts or ('data' in path.parts and {'validation','test'} & set(path.parts)):
            raise RuntimeError('Only Fit inputs allowed; no ground-truth or Val/Test arrays')
        if path.suffix in ('.npy','.npz'):
            if path not in allowed:raise RuntimeError('Unapproved array '+str(path))
            opened.add(str(path))
    sys.addaudithook(audit);return opened


def run(args):
    start=time.perf_counter();available=min(180.,fixed.DEADLINE-time.time())
    if available<=0:raise TimeoutError('13:00 UTC deadline reached')
    if args.output.exists():raise FileExistsError('New cache only; no retry/overwrite')
    def deadline():
        if time.perf_counter()-start>=available or time.time()>=fixed.DEADLINE:raise TimeoutError('180s/global13:00 deadline reached')
    def alarm(signum,frame):raise TimeoutError('180s/global13:00 process alarm')
    old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,available)
    args.output.mkdir(parents=True,exist_ok=False);model=None;hook=None;array=None
    counts=dict(model_forward_calls=0,image_forwards=0,historical_source_encodings=0)
    try:
        manifest=fixed.read(p.PACKAGE/'manifest.json');opened=guard(manifest,args.output)
        feasibility=fixed.read(args.feasibility)
        bound=feasibility['predictions_complete'];receipt=fixed.read(bound['path'])
        if p.sha(bound['path'])!=bound['sha256']:raise ValueError('Actual Val feasibility receipt changed')
        if (feasibility['status']!='complete_teacher_feasibility_only' or feasibility['teacher_feasibility_gate_pass'] is not True
            or feasibility['teacher_gain_k']<.003 or feasibility['macro']['d4mean']['rmse']>=.42
            or abs(feasibility['macro']['code0']['rmse']-feasibility['macro']['d4mean']['rmse']-feasibility['teacher_gain_k'])>1e-12
            or receipt['status']!='both_complete_before_labels' or receipt['checkpoint']['sha256']!=fixed.WEIGHT_SHA
            or receipt['counters']['image_forwards']!=360 or receipt['fp32'] is not True or receipt['tf32'] is not False
            or receipt['labels_opened'] is not False or receipt['test_opened'] is not False):raise ValueError('Require actual successful fixed D4 Val teacher feasibility')
        if receipt['manifest_sha256']!=p.sha(p.PACKAGE/'manifest.json'):raise ValueError('Teacher cohort manifest changed')
        for path,digest in receipt['source_sha256'].items():
            if p.sha(path)!=digest:raise ValueError('Verified teacher implementation changed')
        plan_path=HERE/'weight_interpolation_20260912_v1/interpolation_plan.json';plan=fixed.read(plan_path)
        entry,=[e for e in plan['candidates']['0.25'] if (e['architecture'],e['seed'])==('naf_history',20260905)]
        if entry['checkpoint_sha256']!=fixed.WEIGHT_SHA or p.sha(entry['checkpoint'])!=fixed.WEIGHT_SHA:raise ValueError('Exact alpha25 NAF905 required')
        source=dict(receipt['source_sha256'],**{str(q.resolve()):p.sha(q) for q in
            (Path(__file__),Path(fixed.__file__),args.feasibility,Path(bound['path']),fixed.AUGMENT)})
        inputs={k:dict(path=str(p.PACKAGE/manifest['roles']['fit']['fields'][k]['path']),
            sha256=manifest['roles']['fit']['fields'][k]['sha256']) for k in p.runner.INPUTS}
        for item in inputs.values():
            deadline()
            if p.sha(item['path'])!=item['sha256']:raise ValueError('Fit input changed')
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if gpu.stdout.strip():raise RuntimeError('GPU occupied; no waiting or interruption')
        design=dict(status='frozen_before_Fit_array_loading',checkpoint=fixed.binding(entry['checkpoint']),
            architecture='naf_history',seed=20260905,alpha=.25,views=list(range(8)),aggregation=AGGREGATION,
            source_sha256=source,input_bindings=inputs,manifest_sha256=p.sha(p.PACKAGE/'manifest.json'),
            feasibility_results=fixed.binding(args.feasibility),feasibility_predictions_receipt=bound,
            device='cuda',fp32=True,tf32=False,batch=2,repair_dtype='float64',cache_consumers='All subsequently authorized students share this exact teacher cache',
            shared_teacher_does_not_assert_students_completed=True,generation_cost_is_additional=True,
            pilot_limit_seconds=available,global_deadline_utc='2026-09-12T13:00:00Z',
            labels_opened=False,fit_labels_opened=False,validation_arrays_opened=False,test_opened=False)
        fixed.write(args.output/'design.json',design)
        p.setup('cuda');torch.set_float32_matmul_precision('highest')
        model=original.load_model(entry,'cuda').float().eval().requires_grad_(False)
        data=p.Dataset(p.PACKAGE,'fit',labels=False)
        if len(data)!=603 or len({r['city'] for r in data.records})!=201:raise ValueError('CompleteFit603/201 required')
        def count_stem(module,argv):
            n,c,h,w=argv[0].shape
            if c!=9 or (h,w)!=(160,160) or n%9:raise RuntimeError('Unexpected actual teacher shape')
            counts['model_forward_calls']+=1;counts['image_forwards']+=n//9;counts['historical_source_encodings']+=n
        hook=model.historical.register_forward_pre_hook(count_stem)
        path=args.output/'teacher.npy';temporary=args.output/'teacher.partial.npy'
        array=np.lib.format.open_memmap(temporary,mode='w+',dtype='float64',shape=(603,1,160,160));began=time.perf_counter()
        with torch.inference_mode():
            for first in range(0,603,2):
                ids=np.arange(first,min(first+2,603));raw=data.batch(ids);b={k:torch.from_numpy(v).cuda() for k,v in raw.items()}
                total=np.zeros((len(ids),1,160,160),np.float64)
                for code in range(8):
                    deadline()
                    with torch.autocast('cuda',enabled=False):out=model(**aug.transform_batch(b,code))
                    if out.dtype!=torch.float32:raise RuntimeError('FP32 teacher required')
                    inv=aug.inverse_field(out,code).cpu().numpy().astype(np.float64)
                    if not np.isfinite(inv).all():raise RuntimeError('Nonfinite raw teacher output')
                    total+=inv
                repaired=p.runner.repair(total/8.,raw['coarse'],raw['support']);mask=raw['support'].astype(bool)
                if not np.isfinite(repaired[mask]).all() or not np.isnan(repaired[~mask]).all():raise RuntimeError('Teacher support invalid')
                array[ids]=repaired
                if first%100==0:print(json.dumps(dict(event='Fit_teacher',scenes=int(ids[-1]+1),seconds=time.perf_counter()-start)),flush=True)
        prediction_seconds=time.perf_counter()-began
        if counts!=dict(model_forward_calls=2416,image_forwards=4824,historical_source_encodings=43416):raise RuntimeError('Incomplete fixed8 work')
        array.flush();del array;array=None;temporary.replace(path)
        hook.remove();hook=None
        for file,digest in source.items():
            deadline()
            if p.sha(file)!=digest:raise RuntimeError('Source/feasibility receipt changed')
        if p.sha(entry['checkpoint'])!=fixed.WEIGHT_SHA:raise RuntimeError('Weight changed')
        final=np.load(path,mmap_mode='r',allow_pickle=False)
        if final.shape!=(603,1,160,160) or final.dtype!=np.float64:raise RuntimeError('Incomplete cache shape')
        for pred,support in zip(final,data.arrays['support']):
            mask=np.asarray(support,bool)
            if not np.isfinite(pred[mask]).all() or not np.isnan(pred[~mask]).all():raise RuntimeError('Final cache support mismatch')
        teacher=dict(fixed.binding(path),shape=[603,1,160,160],dtype='float64');deadline()
        record=dict(status='complete_Fit603_D4_teacher_before_labels',split='fit',teacher=teacher,
            checkpoint=fixed.binding(entry['checkpoint']),architecture='naf_history',seed=20260905,alpha=.25,
            views=list(range(8)),aggregation=AGGREGATION,scene_ids=[r['scene_id'] for r in data.records],
            manifest_sha256=p.sha(p.PACKAGE/'manifest.json'),input_bindings=inputs,source_sha256=source,
            design=fixed.binding(args.output/'design.json'),feasibility_results=fixed.binding(args.feasibility),
            feasibility_predictions_receipt=bound,device='cuda',fp32=True,tf32=False,batch=2,repair_dtype='float64',
            counters=counts,prediction_seconds=prediction_seconds,seconds=time.perf_counter()-start,
            opened_arrays=sorted(opened),labels_opened=False,fit_labels_opened=False,validation_arrays_opened=False,test_opened=False,
            all_future_students_share_one_cache=True,generation_cost_is_additional=True,goal_pass=False)
        fixed.write(args.output/'predictions_complete.json',record)
        print(json.dumps(dict(status=record['status'],teacher=teacher,seconds=record['seconds'],receipt=str(args.output/'predictions_complete.json'))),flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        fixed.write(args.output/'failed.json',dict(status='failed',error=repr(exc),counters=counts,seconds=time.perf_counter()-start,
            labels_opened=False,fit_labels_opened=False,validation_arrays_opened=False,test_opened=False,automatic_retry=False));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if hook is not None:hook.remove()
        del model,array;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','run'))
    parser.add_argument('--output',type=Path);parser.add_argument('--feasibility',type=Path,default=DEFAULT_FEASIBILITY);args=parser.parse_args()
    if args.command=='check':fixed.check()
    elif args.output is None:parser.error('Explicit new --output required')
    else:args.output=args.output.resolve();args.feasibility=args.feasibility.resolve();run(args)
