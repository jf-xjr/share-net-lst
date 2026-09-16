"""Standalone train/predict/score entry point; no original project imports."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
from pathlib import Path
import argparse,copy,csv,hashlib,json,math,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from historylst.data import Dataset,INPUTS
from historylst.metrics import repair,score
from historylst.model import HistoryUTAE

def dump(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def save(path,data):
    path=Path(path);temp=path.with_suffix('.partial');torch.save(data,temp);temp.replace(path)

def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()

def setup(device):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True

def batch(data,ids,device):return {k:torch.from_numpy(a).to(device) for k,a in data.batch(ids).items()}

def augment(b,code):
    out={k:torch.rot90(v,code&3,(-2,-1)) if k!='context' else v.clone() for k,v in b.items()}
    if code&4:out={k:torch.flip(v,[-1]) if k!='context' else v for k,v in out.items()}
    east,north=b['context'][:,6],b['context'][:,7]
    ex,ny=[(east,north),(-north,east),(-east,-north),(north,-east)][code&3]
    out['context'][:,6]=-ex if code&4 else ex;out['context'][:,7]=ny
    return {k:v.contiguous() for k,v in out.items()}

def forward(model,b):return model(**{k:b[k] for k in INPUTS})

def loss_fn(p,b):
    e=(p.float()-b['target'].float()).square();mask=b['formal'].bool()
    mse=torch.where(mask,e,0.).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)
    return (mse+1e-6).sqrt().mean()

@torch.inference_mode()
def inference(model,data,device,batch_size,amp=False):
    model.eval();parts=[]
    for start in range(0,len(data),batch_size):
        b=batch(data,np.arange(start,min(start+batch_size,len(data))),device)
        with torch.autocast(device,dtype=torch.float16,enabled=amp):p=forward(model,b)
        parts.append(p.float().cpu().numpy())
    return repair(np.concatenate(parts),data.arrays['coarse'],data.arrays['support'])

def train(a):
    cfg=json.loads(a.config.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise FileExistsError('Completed run must not be overwritten')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=HistoryUTAE().to(a.device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=a.device=='cuda')
    data=Dataset(a.root,'fit',labels=True)
    val=None if a.mode=='smoke' else Dataset(a.root,'validation',labels=True)
    steps=cfg['updates'];rng=np.random.default_rng(cfg['seed']);prob=data.sampling_probabilities()
    ids=[];codes=[]
    for _ in range(steps):ids.append(rng.choice(len(data),cfg['batch_size'],p=prob));codes.append(int(rng.integers(8)))
    best=math.inf;start_step=0;rows=[]
    if a.resume:
        ck=torch.load(out/'last.pt',map_location=a.device,weights_only=False)
        assert ck['config']==cfg
        model.load_state_dict(ck['state_dict']);ema.load_state_dict(ck['ema']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        start_step=ck['step'];best=ck['best'];rows=json.loads((out/'validation.json').read_text())
    elif (out/'run.json').exists():raise FileExistsError('Use --resume for an existing run')
    source_files=[Path(__file__)]+sorted((a.root/'historylst').rglob('*.py'))
    runtime=dict(config=cfg,device=a.device,gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,start_time=time.time(),
        manifest_sha256=digest(a.root/'manifest.json'),code_sha256={str(p.relative_to(a.root)):digest(p) for p in source_files},
        initialization='Independent random initialization; no original network/checkpoint',
        visible_training_inputs=list(INPUTS),fit_queries=len(data),validation_queries=len(val) if val else 0)
    if not a.resume:dump(out/'run.json',runtime)
    def validate(step):
        nonlocal best
        for name,net in [('raw',model),('ema',ema)]:
            prediction=inference(net,val,a.device,cfg['evaluation_batch'],amp=a.device=='cuda')
            result=score(prediction,val.arrays['target'],val.arrays['formal'],val.records)
            value=result['macro']['rmse'];row=dict(step=step,weights=name,rmse=value,seconds=time.perf_counter()-began)
            rows.append(row)
            if value<best:
                best=value;save(out/'best.pt',dict(state_dict=state(net),step=step,weights=name,validation_rmse=value,config=cfg))
            print(json.dumps(dict(validation=row,best=best)),flush=True)
        dump(out/'validation.json',rows)
    began=time.perf_counter();previous_seconds=0.
    if a.resume:previous_seconds=ck.get('elapsed_seconds',0.)
    if val is not None and start_step==0:validate(0)
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    max_steps=min(5,steps) if a.mode=='smoke' else steps
    for step in range(start_step+1,max_steps+1):
        for attempt in range(8):
            torch.manual_seed(cfg['seed']+1000003*step)
            if a.device=='cuda':torch.cuda.manual_seed_all(cfg['seed']+1000003*step)
            b=augment(batch(data,ids[step-1],a.device),codes[step-1]);model.train();optimizer.zero_grad(set_to_none=True)
            factor=step/cfg['warmup'] if step<=cfg['warmup'] else .05+.95*.5*(1+math.cos(math.pi*(step-cfg['warmup'])/(steps-cfg['warmup'])))
            for g in optimizer.param_groups:g['lr']=cfg['lr']*factor
            with torch.autocast(a.device,dtype=torch.float16,enabled=a.device=='cuda'):
                p=forward(model,b);loss=loss_fn(p,b)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):
                if not scaler.is_enabled():raise RuntimeError('Nonfinite gradient')
                scaler.update(new_scale=scaler.get_scale()*.5);continue
            scaler.step(optimizer);scaler.update();break
        else:raise RuntimeError('Eight AMP backoffs exhausted')
        decay=min(.995,(1+step)/(10+step))
        with torch.no_grad():
            for ep,param in zip(ema.parameters(),model.parameters()):ep.lerp_(param,1-decay)
            for eb,buf in zip(ema.buffers(),model.buffers()):eb.copy_(buf)
        if step%50==0 or a.mode=='smoke':
            row=dict(step=step,loss=float(loss.detach()),lr=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-began,amp_attempts=attempt+1)
            print(json.dumps(row),flush=True)
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if val is not None and step%cfg['validation_interval']==0:validate(step)
        if val is not None and (step%cfg['validation_interval']==0 or step==steps):
            save(out/'last.pt',dict(state_dict=state(model),ema=state(ema),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),step=step,best=best,config=cfg,elapsed_seconds=previous_seconds+time.perf_counter()-began))
    elapsed=previous_seconds+time.perf_counter()-began
    if a.mode=='smoke':
        report=dict(status='smoke_complete',updates=max_steps,seconds=elapsed,parameters=runtime['parameters'],peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
            fit_only=True,checkpoint_retained=False,finite_loss_and_gradients=True)
        dump(out/'smoke.json',report);print(json.dumps(report),flush=True)
    else:
        assert len(rows)==2*(1+steps//cfg['validation_interval'])
        report=dict(status='complete',updates=steps,seconds=elapsed,best_validation_rmse=best,validation_candidates=len(rows),parameters=runtime['parameters'],peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None)
        dump(out/'complete.json',report);print(json.dumps(report),flush=True)

def predict(a):
    data=Dataset(a.root,a.split,labels=False);model=HistoryUTAE().to(a.device)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'],strict=True)
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'predictions.npy').exists():raise FileExistsError('Existing predictions')
    t=time.perf_counter();p=inference(model,data,a.device,1,amp=False)
    np.save(a.output/'predictions.npy',p)
    dump(a.output/'prediction.json',dict(split=a.split,queries=len(data),seconds=time.perf_counter()-t,device=a.device,checkpoint_sha256=digest(a.checkpoint),prediction_sha256=digest(a.output/'predictions.npy'),labels_opened=False))

def evaluate(a):
    receipt=json.loads((a.output/'prediction.json').read_text())
    assert receipt['split']==a.split and receipt['prediction_sha256']==digest(a.output/'predictions.npy')
    data=Dataset(a.root,a.split,labels=True);p=np.load(a.output/'predictions.npy',mmap_mode='r')
    results=score(p,data.arrays['target'],data.arrays['formal'],data.records)
    dump(a.output/'metrics.json',results);print(json.dumps(results['macro']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','train','predict','score'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--split',default='test',choices=['fit','validation','test'])
    args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve();setup(args.device)
    if args.mode in ('smoke','train'):train(args)
    elif args.mode=='predict':predict(args)
    else:evaluate(args)
