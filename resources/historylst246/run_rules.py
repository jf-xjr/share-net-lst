"""Input-only examples of two previously fixed historical mosaic references."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from historylst.data import Dataset
from historylst.metrics import repair

def blocks(x):return np.asarray(x).reshape(40,4,40,4)
def lift(x):return x.repeat(4,0).repeat(4,1)
def close(x,c,s):return np.nan_to_num(repair(x[None,None],c[None,None],s[None,None])[0,0])

def predict(inputs,record,mode):
    h=np.asarray(inputs['history'],dtype=np.float64)
    s=np.asarray(inputs['support'][0],bool);c=np.asarray(inputs['coarse'][0],np.float64)
    base=close(np.asarray(inputs['fine'][0],np.float64),c,s)
    selected=np.zeros_like(s);delta=np.zeros_like(base)
    candidates=[]
    for source in record['history']:
        j=source['slot_index']
        if not (h[j,2]>0).any() or (mode=='recent_coarse' and j<6):continue
        t=datetime.fromisoformat(source['datetime'].replace('Z','+00:00'))
        candidates.append((-t.timestamp(),source['item_id'],j))
    for _,_,j in sorted(candidates):
        valid=s&(h[j,2]>=.75)
        difference=np.where(valid,300+20*h[j,0]-base,0.)
        count=blocks(valid).sum((1,3));total=blocks(difference).sum((1,3))
        mean=np.divide(total,count,out=np.zeros_like(total),where=count>0)
        residual=np.where(valid,difference-lift(mean),0.)
        transferred=close(base+residual,c,s)
        first=valid&~selected;delta[first]=transferred[first]-base[first];selected|=valid
    return repair((base+delta)[None,None],c[None,None],s[None,None])[0]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--split',choices=['fit','validation','test'],default='test')
    p.add_argument('--mode',choices=['recent_coarse','all9_coarse'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=Dataset(a.root,a.split,labels=False)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out=np.lib.format.open_memmap(a.output,mode='w+',dtype='float64',shape=(len(data),1,160,160))
    for i,record in enumerate(data.records):out[i]=predict({k:v[i] for k,v in data.arrays.items()},record,a.mode)
    out.flush();print(json.dumps(dict(mode=a.mode,queries=len(data),labels_opened=False)))
