"""Train the pinned two-stage thermal fusion comparator on the common task."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,math,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]/'resources/historylst246'
sys.path[:0]=[str(HERE),str(PACKAGE)]
from thst_adapter import CommonInputTHST,auxiliary,prediction_kelvin
from historylst.data import Dataset
from historylst.metrics import repair,score
from historylst.hotspots import add_hotspot_metrics
spec=importlib.util.spec_from_file_location('thst_published_loss',HERE/'vendor/thstnet/tools/pytorch_ssim.py')
ss=importlib.util.module_from_spec(spec);spec.loader.exec_module(ss)
spec=importlib.util.spec_from_file_location('thst_task_runner',PACKAGE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class ThermalInputs:
    def __init__(self,split):
        root=HERE/'thst_inputs'/split
        self.receipt=json.loads((root/'complete.json').read_text())
        self.fine=np.load(root/'fine_history.npy',mmap_mode='r')
        self.coarse=np.load(root/'coarse_history.npy',mmap_mode='r')
        self.available=np.load(root/'available.npy',mmap_mode='r')


def make_batch(data,cache,ids,refs,top,left,code):
    out={}
    for key,value in data.arrays.items():
        if key=='context':a=np.array(value[ids],copy=True)
        elif key=='coarse':a=np.array(value[ids,...,top//4:top//4+32,left//4:left//4+32],copy=True)
        else:a=np.array(value[ids,...,top:top+128,left:left+128],copy=True)
        out[key]=torch.from_numpy(a).cuda()
    c0=np.array(cache.coarse[ids,refs,top//4:top//4+32,left//4:left//4+32],copy=True)
    out['ref_coarse']=torch.from_numpy(np.repeat(np.repeat(c0,4,1),4,2)[:,None]).cuda()
    out['ref_fine']=torch.from_numpy(np.array(cache.fine[ids,refs,top:top+128,left:left+128],copy=True)[:,None]).cuda()
    out=r.augment(out,code)
    c1=out['coarse'].float().clone()
    for j in range(len(c1)):
        finite=torch.isfinite(c1[j]);fill=c1[j][finite].median()
        c1[j]=torch.where(finite,c1[j],fill)
    out['current_coarse']=torch.nn.functional.interpolate((c1-250)/100,size=(128,128),mode='nearest')
    return out


def objective(prediction,batch):
    mask=batch['formal'].bool()
    p=torch.where(mask,(prediction.float()-250)/100,0.)
    y=torch.where(mask,(batch['target'].float()-250)/100,0.)
    charbonnier=((y-p).square()+1e-6).sqrt().mean()
    structural=1-ss.msssim(p,y,val_range=1,normalize='relu')
    return charbonnier+structural


@torch.inference_mode()
def predict(model,data,cache):
    model.eval();results=[]
    for i in range(len(data)):
        values={k:torch.from_numpy(np.array(v[i:i+1],copy=True)).cuda()
                for k,v in data.arrays.items() if k in r.INPUTS}
        aux=auxiliary(values)
        current=values['coarse'].float();finite=torch.isfinite(current)
        current=torch.where(finite,current,current[finite].median())
        c1=torch.nn.functional.interpolate((current-250)/100,size=(160,160),mode='nearest')
        refs=np.flatnonzero(cache.available[i]);refs=refs if len(refs) else np.array([0])
        c0=torch.from_numpy(np.array(cache.coarse[i,refs],copy=True)[:,None]).cuda()
        c0=torch.nn.functional.interpolate(c0,size=(160,160),mode='nearest')
        f0=torch.from_numpy(np.array(cache.fine[i,refs],copy=True)[:,None]).cuda()
        image=torch.zeros((len(refs),1,160,160),device='cuda');count=torch.zeros_like(image)
        for top,left in ((0,0),(0,32),(32,0),(32,32)):
            for start in range(0,len(refs),2):
                stop=min(start+2,len(refs));n=stop-start
                out=model(c0[start:stop,:,top:top+128,left:left+128],
                          f0[start:stop,:,top:top+128,left:left+128],
                          c1[:,:,top:top+128,left:left+128].expand(n,-1,-1,-1),
                          aux[:,:,top:top+128,left:left+128].expand(n,-1,-1,-1))
                image[start:stop,:,top:top+128,left:left+128]+=100*out.float()+250
                count[start:stop,:,top:top+128,left:left+128]+=1
        image/=count
        coverage=values['history'][0,refs,2:3].float()
        denom=coverage.sum(0,keepdim=True)
        weights=torch.where(denom>0,coverage/denom.clamp_min(1e-8),torch.full_like(coverage,1/len(refs)))
        results.append((image*weights).sum(0,keepdim=True).cpu().numpy())
    return repair(np.concatenate(results),data.arrays['coarse'],data.arrays['support'])


def sources():
    paths=[Path(__file__),HERE/'thst_adapter.py',HERE/'prepare_thst_inputs.py',HERE/'THST_ADAPTATION.md',PACKAGE/'run.py']
    paths+=sorted((HERE/'vendor/thstnet').rglob('*.py'))
    paths+=[HERE/'vendor_deps/timm/models/layers'/name for name in ('drop.py','helpers.py','weight_init.py')]
    paths+=sorted((PACKAGE/'historylst').rglob('*.py'))
    return {str(p):r.digest(p) for p in paths}


def train(args):
    cfg=dict(seed=20260914,stage=args.stage,updates=6000,micro_batch=4,accumulation=1,
             lr=.0001,optimizer='Adam fused',weight_decay=0.,gradient_clip=1.,
             scheduler='ReduceLROnPlateau, macro RMSE min, factor .5, patience 2',
             validation_interval=1000,training_crop=128,validation_scene=160,
             loss='published Charbonnier + published MS-SSIM implementation; formal mask; common projection',
             history='random available reference in training; all 9 references in validation/prediction',
             auxiliary='155 channels, zero-initialized projections into three patch embeddings')
    out=HERE/('thst_smoke_batch4' if args.mode=='smoke' else 'thst')/f'stage{args.stage}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise FileExistsError('Completed stage')
    seal=sources();torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=CommonInputTHST()
    parent=None
    if args.stage==2 and args.mode!='smoke':
        parent=HERE/'thst/stage1/best.pt';saved=torch.load(parent,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True)
    model.set_stage(args.stage);model.cuda()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],fused=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2)
    scaler=torch.amp.GradScaler('cuda')
    fit=Dataset(PACKAGE,'fit',labels=True);cache=ThermalInputs('fit')
    val=None if args.mode=='smoke' else Dataset(PACKAGE,'validation',labels=True)
    vcache=None if val is None else ThermalInputs('validation')
    rng=np.random.default_rng(cfg['seed']+args.stage*101)
    probability=fit.sampling_probabilities()
    ids=[];refs=[];crops=[];codes=[]
    for step in range(cfg['updates']):
        ii=rng.choice(len(fit),4,p=probability);rr=[]
        for idx in ii:
            options=np.flatnonzero(cache.available[idx]);rr.append(int(rng.choice(options)) if len(options) else 0)
        ids.append(ii);refs.append(rr);crops.append(rng.integers(0,3,size=2)*16);codes.append(int(rng.integers(8)))
    ids,refs,crops,codes=map(np.asarray,(ids,refs,crops,codes))
    start_step=0;best=float('inf');rows=[]
    if args.resume:
        saved=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        if saved['config']!=cfg or saved['sources']!=seal:raise ValueError('Resume contract changed')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']);scheduler.load_state_dict(saved['scheduler'])
        start_step=saved['step'];best=saved['best'];rows=saved['validation']
    elif (out/'run.json').exists():raise FileExistsError('Use explicit resume')
    if not args.resume:
        np.savez(out/'schedule.npz',ids=ids,references=refs,crops=crops,d4=codes)
        r.dump(out/'run.json',dict(config=cfg,sources=seal,manifest_sha256=r.digest(PACKAGE/'manifest.json'),
                                  parent_sha256=r.digest(parent) if parent else None,
                                  parameters=sum(p.numel() for p in model.parameters()),
                                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  schedule_sha256=r.digest(out/'schedule.npz'),test_opened=False))
    started=time.time();torch.cuda.reset_peak_memory_stats()
    def validate(step):
        nonlocal best
        prediction=predict(model,val,vcache)
        result=score(prediction,val.arrays['target'],val.arrays['formal'],val.records)
        value=result['macro']['rmse'];rows.append(dict(step=step,rmse=value,seconds=time.time()-started))
        if value<best:
            best=value;r.save(out/'best.pt',dict(state_dict=r.state(model),config=cfg,step=step,validation_rmse=value,sources=seal))
        if step:scheduler.step(value)
        r.dump(out/'validation.json',rows)
        print(json.dumps(dict(validation=rows[-1],best=best)),flush=True)
    if val is not None and not start_step:validate(0)
    steps=2 if args.mode=='smoke' else cfg['updates']
    for step in range(start_step+1,steps+1):
        model.train();losses=[]
        for attempt in range(8):
            optimizer.zero_grad(set_to_none=True);losses=[]
            torch.manual_seed(cfg['seed']+1000003*step)
            for micro in range(cfg['accumulation']):
                take=slice(micro*cfg['micro_batch'],(micro+1)*cfg['micro_batch'])
                b=make_batch(fit,cache,ids[step-1,take],refs[step-1,take],*crops[step-1],int(codes[step-1]))
                with torch.autocast('cuda',dtype=torch.float16):
                    prediction=prediction_kelvin(model,b,b['ref_coarse'],b['ref_fine'],b['current_coarse'])
                with torch.autocast('cuda',enabled=False):loss=objective(prediction,b)
                if not torch.isfinite(loss):raise ValueError('Nonfinite THST loss')
                scaler.scale(loss/cfg['accumulation']).backward();losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if torch.isfinite(norm):scaler.step(optimizer);scaler.update();break
            scaler.update(new_scale=scaler.get_scale()/2)
        else:raise ValueError('Repeated THST nonfinite gradient')
        if step%50==0 or args.mode=='smoke':
            row=dict(step=step,loss=float(np.mean(losses)),seconds=time.time()-started,
                     lr=optimizer.param_groups[0]['lr'],amp_attempts=attempt+1)
            print(json.dumps(row),flush=True)
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if val is not None and step%cfg['validation_interval']==0:
            validate(step)
            r.save(out/'last.pt',dict(state_dict=r.state(model),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                                     scheduler=scheduler.state_dict(),step=step,best=best,config=cfg,sources=seal,validation=rows))
    if sources()!=seal:raise ValueError('Source changed while executing')
    report=dict(status='smoke_complete' if args.mode=='smoke' else 'complete',updates=steps,
                seconds=time.time()-started,best_validation_rmse=best if rows else None,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),finite_gradients=True)
    r.dump(out/('smoke.json' if args.mode=='smoke' else 'complete.json'),report);print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','train','predict'])
    parser.add_argument('--stage',type=int,choices=[1,2],default=2);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--split',choices=['validation','test'],default='test');args=parser.parse_args()
    r.setup('cuda');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode in ('smoke','train'):train(args)
    else:
        out=HERE/'thst'/f'stage{args.stage}';checkpoint=out/'best.pt'
        model=CommonInputTHST();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);model.set_stage(args.stage);model.cuda()
        data=Dataset(PACKAGE,args.split,labels=False);cache=ThermalInputs(args.split)
        p=predict(model,data,cache);path=out/(args.split+'_predictions.npy')
        if path.exists():raise FileExistsError('Predictions already sealed')
        np.save(path,p);r.dump(path.with_suffix('.json'),dict(checkpoint_sha256=r.digest(checkpoint),
            prediction_sha256=r.digest(path),labels_opened=False,queries=len(data),split=args.split))


if __name__=='__main__':main()
