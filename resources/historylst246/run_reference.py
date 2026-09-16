"""Replay original baselines on the portable interface; CPU FP32, inputs only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'reference_models/code'))
from historylst.data import Dataset
from historylst.metrics import repair

def center(value,support):
    n,_,h,w=value.shape;shape=(n,1,h//4,4,w//4,4)
    count=support.reshape(shape).sum((3,5))
    mean=np.where(support,value,0).reshape(shape).sum((3,5))/np.maximum(count,1)
    return np.where(support,value-mean.repeat(4,2).repeat(4,3),0.)

def q(value,coarse,support):
    observed=np.isfinite(coarse).repeat(4,2).repeat(4,3)
    return np.where(support,np.where(observed,center(value,support),value),0.)

def template(anchor,inputs,alpha,beta):
    h=inputs['history'].astype(np.float64);coverage=h[:,:,2:3];den=coverage.sum(1)
    raw=np.divide((5*h[:,:,1:2]*coverage).sum(1),den,out=np.zeros_like(den),where=den>0)
    c,s=inputs['coarse'],inputs['support']
    return repair(anchor+alpha*q(raw,c,s)-beta*q((den>0)*center(anchor,s),c,s),c,s)

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=Dataset(a.root,a.split,labels=False)
    inventory=json.loads((HERE/'reference_models/manifest.json').read_text())
    spec=inventory['models'][a.model]
    ck=torch.load(HERE/'reference_models'/spec['checkpoint'],map_location='cpu',weights_only=False)
    from train_g246_8h import create_model,forward_batch,repair_numpy
    model=create_model(ck['model_spec']['family'],ck['model_spec']['width']).eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    output=np.lib.format.open_memmap(a.output,mode='w+',dtype='float64',shape=(len(data),1,160,160))
    start=time.perf_counter()
    for i in range(len(data)):
        inputs=data.batch([i])
        if a.model=='current_only':inputs['history'][:]=0
        keys=[k for k in inputs if k!='history'] if a.model=='template' else list(inputs)
        batch={k:torch.from_numpy(inputs[k]) for k in keys}
        with torch.inference_mode():raw=forward_batch(model,batch).float().numpy()
        anchor=repair_numpy(raw,inputs['coarse'],inputs['support'])
        if a.model=='template':
            alpha,beta=spec['coefficients_alpha_beta'];prediction=template(anchor,inputs,alpha,beta)
        else:prediction=np.where(inputs['support'],anchor,np.nan)
        output[i]=prediction[0]
    output.flush()
    receipt=dict(model=a.model,split=a.split,queries=len(data),labels_opened=False,
        device='cpu',forward_dtype='float32',seconds=time.perf_counter()-start,
        source_checkpoint_sha256=spec['source_sha256'],training_lineage=spec['lineage'])
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=HERE)
    p.add_argument('--split',choices=['fit','validation','test'],default='test')
    p.add_argument('--model',choices=['current_only','full_history','without_explicit','released_history','template'],required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
