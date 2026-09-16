"""Fixed query-history product capacity pilot:1500 FP32 updates, eight fullVal candidates.

This entry only starts through an explicit invocation. Only original Fit target/formal supply the half-weight GT objective;
the unchanged complete validation split selects weights. No Test or auto-queue.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,math,signal,subprocess,sys,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;NEW=HERE.parent;ROOT=NEW.parents[1]
RECOVERY=NEW/'compact_recovery';PACKAGE=ROOT/'resources/historylst246'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('query_product_actual_parent_reader',RECOVERY/'final_reader.py')
parent_reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent_reader)
loader=parent_reader.loader;old=loader.trainer;r=old.r
from historylst.hotspots import add_hotspot_metrics
read=parent_reader.read;sha=parent_reader.sha;require=parent_reader.require;binding=parent_reader.binding
PROTOCOL='compact_query_product_fp32_half_teacher_half_GT_1500_8_v1'
PARAMETERS={'compact':5977045}
sys.path.insert(0,str(NEW))
from compact_query_product.model import QueryProductCompactHistoryNAF,load_compact_parent_state
PILOT=HERE/'runs/compact_20260905/complete.json'


def construct(architecture):
    require(architecture in PARAMETERS,'Only unchanged compact allowed')
    return QueryProductCompactHistoryNAF(history_dropout=0.,emissivity_dropout=0.).float()


def objective(prediction,batch):
    def rmse(target,mask):
        mask=mask.bool()
        residual=torch.where(mask,prediction.float(),0.)-torch.where(mask,target.float(),0.)
        return (residual.square().sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)+1e-6).sqrt().mean()
    truth=rmse(batch['target'],batch['formal'])
    teacher=rmse(batch['teacher'],batch['support'])
    return .5*truth+.5*teacher,truth,teacher


def gate(parent_rmse,selected_rmse):
    require(math.isfinite(parent_rmse) and math.isfinite(selected_rmse),'Finite actual Val scores required')
    gain=parent_rmse-selected_rmse
    return dict(parent_validation_rmse=parent_rmse,selected_validation_rmse=selected_rmse,
        validation_gain_k=gain,minimum_validation_gain_k=.001,continuation_gate_pass=gain>=.001)


def source_seal():
    paths=[Path(__file__),HERE/'model.py',HERE/'CPU_checks.json',HERE/'design.json',RECOVERY/'selection.json',Path(r.__file__),
        NEW/'train_strong_kd.py',NEW/'multihead_fusion/model.py',old.OLD/'naf_history/model.py',
        old.OLD/'dropout_models.py',PACKAGE/'historylst/data.py',PACKAGE/'historylst/model.py',
        PACKAGE/'historylst/metrics.py',PACKAGE/'historylst/hotspots.py']
    return dict(parent_reader.source_seal(),**{str(p.resolve()):sha(p) for p in paths})


def checked_parent(architecture,seed):
    selection=parent_reader.read_selection(RECOVERY/'selection.json')
    require(selection['selected_family']=='compact','Actual fixed compact family selection required')
    checked=parent_reader.checked_run(RECOVERY/'runs'/f'{architecture}_{seed}',architecture,seed)
    if (architecture,seed)==('compact',20260905):
        pilot=read(HERE/'design.json')['pilot'];path=Path(checked['path'])
        require(sha(path/'best.pt')==pilot['parent_checkpoint_sha256']
            and sha(path/'complete.json')==pilot['parent_complete_sha256']
            and checked['completion']['selected_fp32']['macro']['rmse']==pilot['parent_validation_rmse'],
            'Pilot must retain the exact completed original compact905best')
    return checked


def checked_pilot(path):
    path=Path(path).resolve();require(path==PILOT.resolve(),'Only the one registered actual pilot permits expansion')
    done=read(path);directory=path.parent
    require(done['status']=='complete' and done['architecture']=='compact' and done['original_seed']==20260905
        and done['updates']==1500 and done['validation_weight_candidates']==8
        and done['config']['protocol']==PROTOCOL and done['Fit_labels_opened'] is True
        and done['test_opened'] is False and not (directory/'interrupted.json').exists(), 'Actual complete1500/8 pilot required')
    for p,h in done['source_sha256'].items():require(sha(p)==h,'Pilot source changed')
    require(sha(directory/'run.json')==done['run_sha256'] and sha(directory/'best.pt')==done['selected_checkpoint_sha256'],'Actual pilot artifacts changed')
    rows=read(directory/'validation.json')
    require([(x['step'],x['weights']) for x in rows]==[(s,w) for s in (0,500,1000,1500) for w in ('raw','ema')],'Pilot eight exact candidates required')
    require(all(math.isfinite(x['rmse']) for x in rows),'Nonfinite pilot candidate')
    selected=min(rows,key=lambda x:x['rmse']);saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    require((saved['step'],saved['weights'],saved['validation_rmse'])==(selected['step'],selected['weights'],selected['rmse'])
        and saved['config']==done['config'],'Actual earliest minimum required')
    model=construct('compact');loader.validate_state(model,saved['state_dict'])
    actual=gate(read(HERE/'design.json')['pilot']['parent_validation_rmse'],done['selected_fp32']['macro']['rmse'])
    require(all(done[k]==v for k,v in actual.items()) and actual['continuation_gate_pass'] is True,'Original pilot .001 gate failed')
    require(abs(selected['rmse']-actual['selected_validation_rmse'])<1e-5,'Selected FP32 repeat differs')
    counters=done['counters'];require(counters['updates']==counters['finite_updates']==1500
        and counters['forward_calls']==counters['backward_calls']==3000 and counters['amp_backoffs']==0
        and counters['validation_forward_calls']==9*23,'Pilot real complete work differs')
    return binding(path)


def input_guard(manifest,teacher_path,output,parent_schedule):
    allowed={teacher_path.resolve(),(output/'schedule.npz').resolve(),parent_schedule.resolve()}
    for split,names in [('fit',tuple(r.INPUTS)+('target','formal','valid')),('validation',tuple(r.INPUTS)+('target','formal','valid'))]:
        allowed.update((PACKAGE/manifest['roles'][split]['fields'][k]['path']).resolve() for k in names)
    opened=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if 'data' in p.parts and 'test' in p.parts:
            raise RuntimeError('No Test observations in query-product training')
        if p.suffix in ('.npy','.npz'):
            if p not in allowed:raise RuntimeError('Unapproved polish array: '+str(p))
            opened.add(str(p))
    sys.addaudithook(audit);return opened


def train(args):
    started=time.time();design=read(HERE/'design.json');budget=read(NEW/'four_hour_extension.json')
    require(design['protocol']==PROTOCOL and design['budget']['stop_training_unix']==budget['stop_training_search_by_unix'], 'Fixed polish design/deadline required')
    deadline=min(started+700,budget['stop_training_search_by_unix'])
    def check():
        if time.time()>=deadline:raise TimeoutError('700-second query-product cap or unchanged training stop reached')
    def alarm(*_):raise TimeoutError('Bounded polish training deadline')
    check();require(not args.output.exists(),'Unique new run only; never restart existing training')
    torch.set_num_threads(2)
    parent=checked_parent(args.architecture,args.original_seed);parent_path=Path(parent['path'])/'best.pt'
    pilot_binding=None
    if (args.architecture,args.original_seed)!=('compact',20260905):pilot_binding=checked_pilot(PILOT)
    else:require(args.output.resolve()==PILOT.parent.resolve(),'One registered905pilot output only')
    cfg=dict(design['recipe'],protocol=PROTOCOL,architecture=args.architecture,original_seed=args.original_seed,
        teacher_used_at_inference=False,inference_views=1,
        loader_class='compact_query_product.model.QueryProductCompactHistoryNAF',parameter_groups=design['parameter_groups'])
    require(sha(HERE/'model.py')=='4ec15b7d27c8dedbfe59f336bb1387034cd20645b8e7266eda7f0e632e633648'
        and sha(HERE/'CPU_checks.json')=='86be867dd63c866b000e196a4197f531e34fae1bba077052804c06ad8a9b44af',
        'Root-approved exact model and actual parent identity/gradient checks required')
    require(cfg['updates']==1500 and cfg['teacher_weight']==.5 and cfg['ground_truth_weight']==.5
        and cfg['batch_size']==4 and cfg['microbatch_size']==2 and cfg['gradient_accumulation']==2
        and cfg['lr']==5e-5 and cfg['query_product_lr']==5e-4 and cfg['warmup']==50 and cfg['cosine_floor']==.1
        and cfg['validation_interval']==500 and cfg['weight_decay']==1e-4 and cfg['ema_max_decay']==.995,
        'Exactly the fixed query-product1500/8 recipe is required')
    teacher_receipt=ROOT/design['teacher']['receipt'];receipt=read(teacher_receipt);teacher_item=receipt['teacher'];teacher_path=Path(teacher_item['path'])
    require(sha(teacher_receipt)==design['teacher']['receipt_sha256'] and sha(teacher_path)==teacher_item['sha256']==design['teacher']['array_sha256'],'Unchanged fixed24 teacher required')
    manifest=read(PACKAGE/'manifest.json')
    require(receipt['labels_opened'] is False and receipt['test_opened'] is False
        and receipt['manifest_sha256']==sha(PACKAGE/'manifest.json')
        and receipt['scene_ids']==[x['scene_id'] for x in manifest['roles']['fit']['scenes']],'Complete original ordered input-only teacher required')
    with np.load(Path(parent['path'])/'schedule.npz',allow_pickle=False) as preceding:
        expected_ids=preceding['fit_ids'][:1500].copy();expected_codes=preceding['d4'][:1500].copy()
    sources=source_seal();sources.update({str(Path(parent['path'])/n):sha(Path(parent['path'])/n) for n in ('complete.json','run.json','best.pt','schedule.npz')});sources[str(teacher_receipt.resolve())]=sha(teacher_receipt)
    if pilot_binding:sources[pilot_binding['path']]=pilot_binding['sha256']
    check();active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    require(not active.stdout.strip(),'GPU occupied; never interrupt an external job')
    opened=input_guard(manifest,teacher_path,args.output,Path(parent['path'])/'schedule.npz');r.setup('cuda')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    torch.manual_seed(cfg['schedule_seed']);np.random.seed(cfg['schedule_seed'])
    model=construct(args.architecture);initial=torch.load(parent_path,map_location='cpu',weights_only=False);initialization_mapping=load_compact_parent_state(model,initial['state_dict'])
    require(sum(p.numel() for p in model.parameters())==PARAMETERS[args.architecture],'Unchanged architecture parameter count required')
    require(not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),'Microbatch accumulation requires the original BN-free compact model')
    model.cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
    q_names={f'fusion.{i}.query_product.{part}' for i in range(4) for part in ('weight','bias')}
    named=dict(model.named_parameters());require(q_names<=set(named),'Four exact new query projections required')
    groups=[dict(params=[v for k,v in named.items() if k not in q_names],lr=cfg['lr'],registered_base_lr=cfg['lr'],name='backbone'),
        dict(params=[named[k] for k in sorted(q_names)],lr=cfg['query_product_lr'],registered_base_lr=cfg['query_product_lr'],name='query_product')]
    require(sum(v.numel() for v in groups[0]['params'])==5911525 and sum(v.numel() for v in groups[1]['params'])==65520,'Exact old/new parameter partition required')
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'],fused=True)
    fit,val=r.Dataset(PACKAGE,'fit',labels=True),r.Dataset(PACKAGE,'validation',labels=True)
    require(len(fit)==603 and len(val)==45 and set(fit.arrays)==set(r.INPUTS)|{'target','formal','valid'},'Complete original supervised Fit and fullVal45 required')
    teacher=np.load(teacher_path,mmap_mode='r',allow_pickle=False);require(teacher.shape==(603,1,160,160) and str(teacher.dtype)==teacher_item['dtype'],'Actual common complete teacher shape/dtype')
    rng=np.random.default_rng(cfg['schedule_seed']);probabilities=fit.sampling_probabilities();ids=[];codes=[]
    for _ in range(1500):
        ids.append(rng.choice(len(fit),4,p=probabilities));codes.append(int(rng.integers(8)))
    ids,codes=np.asarray(ids),np.asarray(codes)
    require(np.array_equal(ids,expected_ids) and np.array_equal(codes,expected_codes),'Exact original sampler/D4 prefix differs')
    args.output.mkdir(parents=True,exist_ok=False);np.savez(args.output/'schedule.npz',fit_ids=ids,d4=codes)
    runtime=dict(config=cfg,source_sha256=sources,initialization=str(parent_path),initialization_sha256=sha(parent_path),
        initialization_mapping=initialization_mapping,parent_complete=binding(Path(parent['path'])/'complete.json'),
        preceding_updates=6000,preceding_validation_weight_candidates=26,parent_validation_rmse=parent['completion']['selected_fp32']['macro']['rmse'],
        teacher_receipt=binding(teacher_receipt),teacher=teacher_item,teacher_new_generation_seconds=0,
        schedule_sha256=sha(args.output/'schedule.npz'),schedule_content_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('fit_ids',ids),('d4',codes)]},
        parameters=PARAMETERS[args.architecture],manifest_sha256=sha(PACKAGE/'manifest.json'),actual_start_unix=started,deadline_unix=deadline,
        validation_candidates=8,validation_precision='FP32 TF32 off; exact original FP64 repair',
        single_network_forward=True,Fit_labels_opened=True,test_opened=False,teacher_used_at_inference=False,pilot_completion=pilot_binding,training_precision='FP32 TF32off autocast disabled',microbatch_size=2,gradient_accumulation=2)
    r.dump(args.output/'run.json',runtime)
    counters=dict(updates=0,finite_updates=0,forward_calls=0,backward_calls=0,amp_backoffs=0,validation_forward_calls=0);rows=[];best=math.inf
    @torch.inference_mode()
    def evaluate(net,hotspots=False):
        net.eval();parts=[]
        for first in range(0,len(val),2):
            check();b=r.batch(val,np.arange(first,min(first+2,len(val))),'cuda')
            with torch.autocast('cuda',enabled=False):prediction=r.forward(net,b)
            require(prediction.dtype==torch.float32,'Actual Val FP32 required');parts.append(prediction.cpu().numpy());counters['validation_forward_calls']+=1
        prediction=r.repair(np.concatenate(parts),val.arrays['coarse'],val.arrays['support'])
        score=r.score(prediction,val.arrays['target'],val.arrays['formal'],val.records)
        if hotspots:add_hotspot_metrics(score,prediction,val.arrays['target'],val.arrays['formal'])
        return score
    def validate(step):
        nonlocal best
        for name,net in [('raw',model),('ema',ema)]:
            score=evaluate(net)['macro']['rmse'];require(math.isfinite(score),'Nonfinite Val score')
            row=dict(step=step,weights=name,rmse=score,seconds=time.time()-started);rows.append(row)
            if score<best:
                best=score;r.save(args.output/'best.pt',dict(state_dict=r.state(net),config=cfg,step=step,weights=name,validation_rmse=score,source_sha256=sources,initialization=str(parent_path)))
            print(json.dumps(dict(validation=row,best=best)),flush=True)
        r.dump(args.output/'validation.json',rows)
        r.save(args.output/'last.pt',dict(state_dict=r.state(model),ema=r.state(ema),optimizer=optimizer.state_dict(),config=cfg,step=step,counters=dict(counters),validation=rows))
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:
        validate(0)
        require(abs(rows[0]['rmse']-runtime['parent_validation_rmse'])<1e-5,'Strict parent initial FP32 score differs')
        for step in range(1,1501):
            check();model.train()
            factor=step/50 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/1450))
            for group in optimizer.param_groups:group['lr']=group['registered_base_lr']*factor
            torch.manual_seed(cfg['schedule_seed']+1000003*step);torch.cuda.manual_seed_all(cfg['schedule_seed']+1000003*step)
            optimizer.zero_grad(set_to_none=True);loss_value=0.;truth_value=0.;teacher_value=0.
            for micro in range(2):
                check();selected_ids=ids[step-1,micro*2:(micro+1)*2]
                batch=r.batch(fit,selected_ids,'cuda')
                batch['teacher']=torch.from_numpy(np.array(teacher[selected_ids],dtype=np.float32)).cuda()
                batch=r.augment(batch,int(codes[step-1]))
                with torch.autocast('cuda',enabled=False):
                    pred=r.forward(model,batch);counters['forward_calls']+=1
                    require(pred.dtype==torch.float32,'All training forwards must be real FP32')
                    loss,truth,teacher_loss=objective(pred,batch)
                require(loss.dtype==torch.float32 and bool(torch.isfinite(loss)),'Finite fixed half-GT half-teacher per-scene RMSE required')
                (loss*.5).backward();counters['backward_calls']+=1;loss_value+=float(loss.detach())*.5;truth_value+=float(truth.detach())*.5;teacher_value+=float(teacher_loss.detach())*.5
                del pred,loss,truth,teacher_loss,batch
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            require(bool(torch.isfinite(norm)),'Nonfinite FP32 gradients: stop without retry or budget extension')
            optimizer.step()
            counters['updates']+=1;counters['finite_updates']+=1;decay=min(.995,(1+step)/(10+step))
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
                for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
            if step%100==0:
                row=dict(step=step,seconds=time.time()-started,loss=loss_value,teacher_scene_rmse=teacher_value,ground_truth_scene_rmse=truth_value,teacher_weight=.5,ground_truth_weight=.5,learning_rate=optimizer.param_groups[0]['lr'],query_product_learning_rate=optimizer.param_groups[1]['lr'],counters=dict(counters))
                with (args.output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if step%500==0:validate(step)
        require([(x['step'],x['weights']) for x in rows]==[(s,w) for s in (0,500,1000,1500) for w in ('raw','ema')]
            and counters['finite_updates']==1500,'Actual1500/8 completion required')
        selected=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False);loader.validate_state(model,selected['state_dict']);result=evaluate(model,True)
        require(abs(result['macro']['rmse']-selected['validation_rmse'])<1e-5,'Selected exact FP32 repeat differs')
        for path,digest in sources.items():require(sha(path)==digest,'Bound source/parent changed '+path)
        require(sha(teacher_path)==teacher_item['sha256'],'Teacher changed during training')
        outcome=gate(runtime['parent_validation_rmse'],result['macro']['rmse'])
        if (args.architecture,args.original_seed)!=('compact',20260905):outcome['continuation_gate_pass']=False
        ended=time.time();check()
        report=dict(status='complete',config=cfg,architecture=args.architecture,original_seed=args.original_seed,updates=1500,validation_weight_candidates=8,
            selected_step=selected['step'],selected_weights=selected['weights'],selected_checkpoint_sha256=sha(args.output/'best.pt'),selected_fp32=result,
            parameters=PARAMETERS[args.architecture],counters=counters,seconds=ended-started,actual_start_unix=started,actual_end_unix=ended,
            run_sha256=sha(args.output/'run.json'),source_sha256=sources,parent_complete=runtime['parent_complete'],teacher_receipt_sha256=sha(teacher_receipt),
            preceding_updates=6000,preceding_validation_weight_candidates=26,single_network_forward=True,Fit_labels_opened=True,test_opened=False,
            teacher_used_at_inference=False,goal1_pass=False,opened_arrays=sorted(opened),**outcome,
            note='Pilot expansion is a budget screen, not final user success. No automatic expansion, freeze or Test. Nonpilot continuation_gate_pass is inapplicable and remains false.')
        r.dump(args.output/'complete.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('selected_fp32','source_sha256','config','opened_arrays')}),flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);r.dump(args.output/'interrupted.json',dict(status='incomplete',error=repr(exc),counters=counters,seconds=time.time()-started));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--architecture',choices=tuple(PARAMETERS),required=True)
    parser.add_argument('--original-seed',type=int,choices=parent_reader.p.SEEDS,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output=args.output.resolve();train(args)
