"""Metric-aligned short adaptation of the pinned professional THST structure."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,copy,json,math,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import train_thst as th
from thst_reference_variants import matched_cache
r=th.r;OUT=HERE/'thst_task_loss'

def train(resume):
    OUT.mkdir(exist_ok=True)
    if (OUT/'training_complete.json').exists():return
    cfg=dict(seed=20260914,updates=1500,batch_size=4,lr=.0001,warmup=50,weight_decay=.0001,
             loss='common formal-normalized scene RMSE',ema='min(.995,(1+step)/(10+step))',
             crop=128,validation_interval=500,validation_candidates=7,teacher=False)
    sources=th.sources();sources.update({str(p):r.digest(p) for p in (Path(__file__),HERE/'THST_TASK_LOSS_PROTOCOL.md',HERE/'thst_reference_variants.py')})
    parent=HERE/'thst/stage2/best.pt';assert (parent.parent/'complete.json').exists()
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=th.CommonInputTHST();base=torch.load(parent,map_location='cpu',weights_only=False)
    model.load_state_dict(base['state_dict']);model.set_stage(2);model.cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=True)
    scaler=torch.amp.GradScaler('cuda')
    fit=th.Dataset(th.PACKAGE,'fit',labels=True);cache=th.ThermalInputs('fit')
    val=th.Dataset(th.PACKAGE,'validation',labels=True);vcache=th.ThermalInputs('validation')
    rng=np.random.default_rng(cfg['seed']+303);probability=fit.sampling_probabilities()
    ids=[];refs=[];crops=[];codes=[]
    for _ in range(cfg['updates']):
        ii=rng.choice(len(fit),4,p=probability);rr=[]
        for i in ii:
            available=np.flatnonzero(cache.available[i]);rr.append(int(rng.choice(available)) if len(available) else 0)
        ids.append(ii);refs.append(rr);crops.append(rng.integers(0,3,size=2)*16);codes.append(int(rng.integers(8)))
    ids,refs,crops,codes=map(np.asarray,(ids,refs,crops,codes))
    start_step=0;best=float('inf');rows=[]
    if resume:
        ck=torch.load(OUT/'last.pt',map_location='cuda',weights_only=False)
        assert ck['config']==cfg and ck['sources']==sources
        model.load_state_dict(ck['state_dict']);ema.load_state_dict(ck['ema']);optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler']);start_step=ck['step'];best=ck['best'];rows=ck['validation']
    elif (OUT/'run.json').exists():raise FileExistsError('Use --resume')
    else:
        np.savez(OUT/'schedule.npz',ids=ids,references=refs,crops=crops,d4=codes)
        r.dump(OUT/'run.json',dict(config=cfg,sources=sources,parent_sha256=r.digest(parent),
            parent_step=base['step'],parent_validation_rmse=base['validation_rmse'],manifest_sha256=r.digest(th.PACKAGE/'manifest.json'),
            schedule_sha256=r.digest(OUT/'schedule.npz'),trainable_parameters=sum(p.numel() for p in parameters),test_labels_opened=False))
    started=time.time();torch.cuda.reset_peak_memory_stats()
    def validate(step):
        nonlocal best
        choices=[('raw',model)] if step==0 else [('raw',model),('ema',ema)]
        for name,net in choices:
            p=th.predict(net,val,vcache);value=th.score(p,val.arrays['target'],val.arrays['formal'],val.records)['macro']['rmse']
            row=dict(step=step,weights=name,rmse=value,seconds=time.time()-started);rows.append(row)
            if value<best:
                best=value;r.save(OUT/'best.pt',dict(state_dict=r.state(net),config=cfg,step=step,weights=name,validation_rmse=value,sources=sources))
            print(json.dumps(dict(validation=row,best=best)),flush=True)
        r.dump(OUT/'validation.json',rows)
    if not start_step:validate(0)
    for step in range(start_step+1,cfg['updates']+1):
        factor=step/cfg['warmup'] if step<=cfg['warmup'] else .05+.95*.5*(1+math.cos(math.pi*(step-cfg['warmup'])/(cfg['updates']-cfg['warmup'])))
        for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
        for attempt in range(8):
            torch.manual_seed(cfg['seed']+1000003*step);model.train();optimizer.zero_grad(set_to_none=True)
            b=th.make_batch(fit,cache,ids[step-1],refs[step-1],*crops[step-1],int(codes[step-1]))
            with torch.autocast('cuda',dtype=torch.float16):
                prediction=th.prediction_kelvin(model,b,b['ref_coarse'],b['ref_fine'],b['current_coarse'])
            with torch.autocast('cuda',enabled=False):loss=r.loss_fn(prediction,b)
            if not torch.isfinite(loss):raise ValueError('Nonfinite common RMSE')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if torch.isfinite(norm):scaler.step(optimizer);scaler.update();break
            scaler.update(new_scale=scaler.get_scale()/2)
        else:raise ValueError('Repeated nonfinite gradient')
        decay=min(.995,(1+step)/(10+step))
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-decay)
            for eb,bf in zip(ema.buffers(),model.buffers()):eb.copy_(bf)
        if step%50==0:
            row=dict(step=step,loss=float(loss.detach()),lr=optimizer.param_groups[0]['lr'],seconds=time.time()-started,amp_attempts=attempt+1)
            with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%500==0:
            validate(step)
            r.save(OUT/'last.pt',dict(state_dict=r.state(model),ema=r.state(ema),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                config=cfg,sources=sources,step=step,best=best,validation=rows))
    assert len(rows)==7 and all(r.digest(p)==h for p,h in sources.items())
    r.dump(OUT/'training_complete.json',dict(status='complete',updates=1500,best_validation_rmse=best,
        validation_candidates=len(rows),seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()))

def predictions():
    if (OUT/'predictions_complete.json').exists():return
    checkpoint=OUT/'best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=th.CommonInputTHST();model.load_state_dict(ck['state_dict']);model.set_stage(2);model.cuda()
    val=th.Dataset(th.PACKAGE,'validation',labels=False);cache=th.ThermalInputs('validation');single,reference_rows=matched_cache(val,cache)
    arrays={'all':th.predict(model,val,cache),'matched':th.predict(model,val,single)}
    labels=th.Dataset(th.PACKAGE,'validation',labels=True)
    values={k:th.score(p,labels.arrays['target'],labels.arrays['formal'],val.records)['macro']['rmse'] for k,p in arrays.items()}
    assert abs(values['all']-ck['validation_rmse'])<1e-6
    native=json.loads((HERE/'thst_reference/selection.json').read_text())
    selected=min(values,key=values.get);native_value=native['validation_rmse'][native['selected']]
    selection=dict(reference=selected,validation_rmse=values,checkpoint_sha256=r.digest(checkpoint),
        native_reference=native['selected'],native_validation_rmse=native_value,
        final_family='task_loss' if values[selected]<native_value else 'native',
        protocol_sha256=r.digest(HERE/'THST_TASK_LOSS_PROTOCOL.md'),source_sha256=r.digest(__file__),validation_references=reference_rows)
    freeze=OUT/'selection.json'
    if freeze.exists():assert json.loads(freeze.read_text())==selection
    else:r.dump(freeze,selection)
    for name,p in arrays.items():np.save(OUT/f'validation_{name}.npy',p)
    test=th.Dataset(th.PACKAGE,'test',labels=False);cache=th.ThermalInputs('test');single,rows=matched_cache(test,cache)
    receipts=[]
    for name,c in [('all',cache),('matched',single)]:
        start=time.time();p=th.predict(model,test,c);path=OUT/f'test_{name}.npy';np.save(path,p)
        receipts.append(dict(reference=name,path=str(path),sha256=r.digest(path),seconds=time.time()-start))
    r.dump(OUT/'predictions_complete.json',dict(queries=len(test),scene_order=[q['scene_id'] for q in test.records],
        checkpoint_sha256=r.digest(checkpoint),selection_sha256=r.digest(freeze),entries=receipts,test_labels_opened=False,test_references=rows))
    print(json.dumps(dict(reference=selected,validation_rmse=values,final_family=selection['final_family'])),flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    r.setup('cuda');torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train(args.resume);predictions()
if __name__=='__main__':main()
