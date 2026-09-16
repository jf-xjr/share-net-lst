"""Fixed annealed-teacher recovery; same six inputs, one network at inference."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse, copy, importlib.util, json, math, signal, subprocess, sys, time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = NEW.parents[1]
OLD = NEW.parent / 'sub04_20260911'
PACKAGE = ROOT / 'resources/historylst246'
sys.path[:0] = [str(HERE), str(NEW), str(OLD), str(PACKAGE)]
spec = importlib.util.spec_from_file_location('annealed_original_helpers', NEW/'train_strong_kd.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
r = base.r
from model import CompactFourHeadHistoryNAF, load_parent_state
from multihead_fusion.model import FourHeadHistoryNAF
from dropout_models import DropoutUTAE
from historylst.hotspots import add_hotspot_metrics

PARAMETERS = {'compact': 5911525, 'full': 9310501, 'baseline': 2985697}

def read(path): return json.loads(Path(path).read_text())
def require(value, message):
    if not value: raise ValueError(message)
def construct(architecture):
    factory = {'compact': CompactFourHeadHistoryNAF, 'full': FourHeadHistoryNAF, 'baseline': DropoutUTAE}[architecture]
    return factory(history_dropout=0., emissivity_dropout=0.)

def objective(prediction, batch, step, updates):
    def rmse(target, mask):
        mask = mask.bool()
        residual = torch.where(mask, prediction.float(), 0.) - torch.where(mask, target.float(), 0.)
        return (residual.square().sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)+1e-6).sqrt().mean()
    truth = rmse(batch['target'], batch['formal'])
    teacher = rmse(batch['teacher'], batch['support'])
    weight = .9 - .8 * step / updates
    return (1-weight)*truth + weight*teacher, truth, teacher, weight

def train(args):
    started = time.time(); design = read(HERE/'design.json')
    budget = read(NEW/'four_hour_extension.json')
    deadline = min(budget['stop_training_search_by_unix'], started+2700)
    def check():
        if time.time() >= deadline: raise TimeoutError('Fixed 2700s job or training deadline reached')
    def alarm(*_): raise TimeoutError('Fixed training execution deadline reached')
    check(); require(not args.output.exists(), 'New run directory only; never restart completed training')
    require(design['protocol']=='compact_or_full_annealed_teacher_recovery_v1', 'Unexpected design')
    cfg = dict(design['recipe'], architecture=args.architecture, original_seed=args.original_seed,
               protocol=design['protocol'], teacher_used_at_inference=False, inference_views=1,
               loader_class={'compact':'compact_recovery.model.CompactFourHeadHistoryNAF',
                             'full':'multihead_fusion.model.FourHeadHistoryNAF',
                             'baseline':'dropout_models.DropoutUTAE'}[args.architecture])
    require(cfg['updates']==6000 and cfg['teacher_start']==.9 and cfg['teacher_end']==.1,
            'One fixed 6000-update annealing recipe')
    freeze_path = NEW/'final_confirmation/frozen/continuation_selection_freeze.json'
    frozen = read(freeze_path)
    role = 'baseline' if args.architecture=='baseline' else 'naf_history'
    parent, = [x for x in frozen['checkpoints'] if (x['architecture'],x['seed'])==(role,args.original_seed)]
    parent_path = Path(parent['checkpoint'])
    require(r.digest(parent_path)==parent['checkpoint_sha256'], 'Actual parent bytes changed')
    teacher_receipt_path = NEW/'strong_teacher/fit/predictions_complete.json'
    receipt = read(teacher_receipt_path); teacher_item = receipt['teacher']
    teacher_path = Path(teacher_item['path']); manifest = read(PACKAGE/'manifest.json')
    require(receipt['labels_opened'] is False and receipt['test_opened'] is False, 'Original input-only teacher required')
    require(receipt['manifest_sha256']==r.digest(PACKAGE/'manifest.json') and
            receipt['scene_ids']==[x['scene_id'] for x in manifest['roles']['fit']['scenes']], 'Original complete Fit teacher order')
    require(r.digest(teacher_path)==teacher_item['sha256'], 'Teacher array changed')
    active = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], capture_output=True,text=True,check=True)
    require(not active.stdout.strip(), 'GPU occupied; do not interrupt another task')
    base.guard(teacher_path); r.setup('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['schedule_seed']); np.random.seed(cfg['schedule_seed'])
    model = construct(args.architecture)
    initial = torch.load(parent_path,map_location='cpu',weights_only=False)
    if args.architecture=='compact': initialization = load_parent_state(model,initial['state_dict'])
    else:
        model.load_state_dict(initial['state_dict'],strict=True)
        initialization = dict(mapping='strict_identity',parent_keys=len(initial['state_dict']))
    require(sum(p.numel() for p in model.parameters())==PARAMETERS[args.architecture], 'Actual parameter count differs')
    model.cuda(); ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=True)
    scaler = torch.amp.GradScaler('cuda')
    fit,val = r.Dataset(PACKAGE,'fit',labels=True),r.Dataset(PACKAGE,'validation',labels=True)
    require(len(fit)==603 and len(val)==45, 'Complete original Fit603/Val45 required')
    teacher = np.load(teacher_path,mmap_mode='r',allow_pickle=False)
    require(teacher.shape==(603,1,160,160), 'Complete teacher shape')
    rng = np.random.default_rng(cfg['schedule_seed']); ids=[]; codes=[]
    for _ in range(cfg['updates']):
        ids.append(rng.choice(len(fit),cfg['batch_size'],p=fit.sampling_probabilities()))
        codes.append(int(rng.integers(8)))
    ids,codes = np.asarray(ids),np.asarray(codes)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez(args.output/'schedule.npz',fit_ids=ids,d4=codes)
    source_paths = [Path(__file__),HERE/'model.py',HERE/'design.json',NEW/'four_hour_extension.json',
        NEW/'network_acceptance_20260913_revision.json',NEW/'train_strong_kd.py',
        NEW/'multihead_fusion/model.py',OLD/'naf_history/model.py',OLD/'dropout_models.py',
        PACKAGE/'run.py',PACKAGE/'historylst/model.py',PACKAGE/'historylst/metrics.py',PACKAGE/'historylst/data.py',
        freeze_path,teacher_receipt_path]
    sources = {str(p.resolve()):r.digest(p) for p in source_paths}
    runtime = dict(config=cfg,source_sha256=sources,initialization=str(parent_path),
        initialization_sha256=parent['checkpoint_sha256'],initialization_mapping=initialization,
        teacher_receipt=str(teacher_receipt_path),teacher_receipt_sha256=r.digest(teacher_receipt_path),teacher=teacher_item,
        schedule_sha256=r.digest(args.output/'schedule.npz'),parameters=PARAMETERS[args.architecture],
        manifest_sha256=r.digest(PACKAGE/'manifest.json'),actual_start_unix=started,deadline_unix=deadline,
        validation_candidates=26,validation_precision='FP32 TF32 off; exact original FP64 repair',
        single_network_forward=True,test_opened=False,teacher_used_at_inference=False)
    r.dump(args.output/'run.json',runtime)
    counters=dict(updates=0,finite_updates=0,forward_calls=0,backward_calls=0,amp_backoffs=0,validation_forward_calls=0)
    rows=[]; best=math.inf
    @torch.inference_mode()
    def evaluate(net,hotspots=False):
        net.eval(); parts=[]
        for first in range(0,len(val),cfg['evaluation_batch']):
            check(); b=r.batch(val,np.arange(first,min(first+cfg['evaluation_batch'],len(val))),'cuda')
            parts.append(r.forward(net,b).float().cpu().numpy()); counters['validation_forward_calls']+=1
        prediction=r.repair(np.concatenate(parts),val.arrays['coarse'],val.arrays['support'])
        score=r.score(prediction,val.arrays['target'],val.arrays['formal'],val.records)
        if hotspots: add_hotspot_metrics(score,prediction,val.arrays['target'],val.arrays['formal'])
        return score
    def validate(step):
        nonlocal best
        for name,net in [('raw',model),('ema',ema)]:
            score=evaluate(net)['macro']['rmse']; require(math.isfinite(score),'Nonfinite Val score')
            row=dict(step=step,weights=name,rmse=score,seconds=time.time()-started); rows.append(row)
            if score<best:
                best=score
                r.save(args.output/'best.pt',dict(state_dict=r.state(net),config=cfg,step=step,weights=name,
                    validation_rmse=score,source_sha256=sources,initialization=str(parent_path)))
            print(json.dumps(dict(validation=row,best=best)),flush=True)
        r.dump(args.output/'validation.json',rows)
        r.save(args.output/'last.pt',dict(state_dict=r.state(model),ema=r.state(ema),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),config=cfg,step=step,counters=dict(counters),validation=rows))
    signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:
        validate(0)
        for step in range(1,cfg['updates']+1):
            check(); batch=r.batch(fit,ids[step-1],'cuda')
            batch['teacher']=torch.from_numpy(np.array(teacher[ids[step-1]],dtype=np.float32)).cuda()
            batch=r.augment(batch,int(codes[step-1])); model.train()
            factor=step/cfg['warmup'] if step<=cfg['warmup'] else .05+.95*.5*(1+math.cos(math.pi*(step-cfg['warmup'])/(cfg['updates']-cfg['warmup'])))
            for group in optimizer.param_groups: group['lr']=cfg['lr']*factor
            buffers={k:v.detach().clone() for k,v in model.named_buffers()}
            for attempt in range(8):
                check()
                if attempt:
                    with torch.no_grad():
                        for name,value in model.named_buffers(): value.copy_(buffers[name])
                torch.manual_seed(cfg['schedule_seed']+1000003*step)
                torch.cuda.manual_seed_all(cfg['schedule_seed']+1000003*step)
                optimizer.zero_grad(set_to_none=True); counters['forward_calls']+=1
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=r.forward(model,batch); total,truth,distilled,weight=objective(pred,batch,step,cfg['updates'])
                require(bool(torch.isfinite(total)),'Nonfinite annealed loss')
                scaler.scale(total).backward(); counters['backward_calls']+=1; scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                if not torch.isfinite(norm):
                    counters['amp_backoffs']+=1; scaler.update(new_scale=scaler.get_scale()*.5); continue
                scaler.step(optimizer); scaler.update(); break
            else: raise RuntimeError('Eight finite-gradient retries exhausted')
            counters['updates']+=1; counters['finite_updates']+=1
            decay=min(cfg['ema_max_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,1-decay)
                for a,b in zip(ema.buffers(),model.buffers()): a.copy_(b)
            if step%100==0:
                row=dict(step=step,seconds=time.time()-started,loss=float(total),truth_loss=float(truth),
                    teacher_loss=float(distilled),teacher_weight=weight,learning_rate=optimizer.param_groups[0]['lr'],counters=dict(counters))
                with (args.output/'progress.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if step%cfg['validation_interval']==0: validate(step)
        require([(x['step'],x['weights']) for x in rows]==[(s,w) for s in range(0,6001,500) for w in ('raw','ema')], 'All26 identical candidate opportunities required')
        require(counters['finite_updates']==6000,'Incomplete successful updates')
        selected=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(selected['state_dict'],strict=True); result=evaluate(model,True)
        require(abs(result['macro']['rmse']-selected['validation_rmse'])<1e-5,'Selected exact FP32 check differs')
        for path,digest in sources.items(): require(r.digest(path)==digest,'Bound source changed '+path)
        require(r.digest(teacher_path)==teacher_item['sha256'],'Teacher changed during training')
        report=dict(status='complete',config=cfg,architecture=args.architecture,original_seed=args.original_seed,
            updates=6000,validation_weight_candidates=26,selected_step=selected['step'],selected_weights=selected['weights'],
            selected_checkpoint_sha256=r.digest(args.output/'best.pt'),selected_fp32=result,parameters=PARAMETERS[args.architecture],
            counters=counters,seconds=time.time()-started,actual_start_unix=started,actual_end_unix=time.time(),
            run_sha256=r.digest(args.output/'run.json'),source_sha256=sources,teacher_receipt_sha256=r.digest(teacher_receipt_path),
            single_network_forward=True,test_opened=False,teacher_used_at_inference=False,goal1_pass=False,
            note='Completed training only; final tiered acceptance requires frozen full Test three-seed comparison and actual costs')
        r.dump(args.output/'complete.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('selected_fp32','source_sha256','config')}),flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        r.dump(args.output/'interrupted.json',dict(status='incomplete',error=repr(exc),counters=counters,seconds=time.time()-started))
        raise
    finally: signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',choices=['compact','full','baseline'],required=True)
    parser.add_argument('--original-seed',type=int,choices=[20260905,20260912,20260913],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output=args.output.resolve(); train(args)
