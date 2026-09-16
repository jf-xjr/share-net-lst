"""Cache finite thermal reference pairs from registered past observations."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]/'resources/historylst246'
sys.path.insert(0,str(PACKAGE))
from historylst.data import Dataset


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['fit','validation','test'],required=True)
    args=parser.parse_args();data=Dataset(PACKAGE,args.split,labels=False)
    out=HERE/'thst_inputs'/args.split;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        old=json.loads((out/'complete.json').read_text())
        assert old['source_sha256']==digest(__file__) and old['manifest_sha256']==digest(PACKAGE/'manifest.json')
        print('Complete input cache already exists');return
    mode='r+' if (out/'fine_history.npy').exists() else 'w+'
    fine=np.lib.format.open_memmap(out/'fine_history.npy',mode=mode,dtype=np.float32,shape=(len(data),9,160,160))
    coarse=np.lib.format.open_memmap(out/'coarse_history.npy',mode=mode,dtype=np.float32,shape=(len(data),9,40,40))
    available=np.lib.format.open_memmap(out/'available.npy',mode=mode,dtype=np.bool_,shape=(len(data),9))
    start=time.time()
    for i in range(len(data)):
        h=np.asarray(data.arrays['history'][i],np.float64)
        support=np.asarray(data.arrays['support'][i,0],bool)
        current=np.asarray(data.arrays['coarse'][i,0],float)
        current_fill=np.where(np.isfinite(current),current,np.nanmedian(current))
        base=np.repeat(np.repeat(current_fill,4,0),4,1)
        count=support.reshape(40,4,40,4).sum((1,3))
        for j in range(9):
            valid=h[j,2]>0;observed=valid&support
            available[i,j]=observed.any()
            temperature=20*h[j,0]+300
            fill=float(np.median(temperature[observed])) if observed.any() else float(np.nanmedian(current))
            finite=np.where(valid,temperature,fill) if observed.any() else base
            parent=np.where(support,finite,0).reshape(40,4,40,4).sum((1,3))/np.maximum(count,1)
            parent=np.where(count>0,parent,fill)
            fine[i,j]=(finite-250)/100;coarse[i,j]=(parent-250)/100
        if (i+1)%50==0:print(json.dumps(dict(split=args.split,done=i+1,seconds=time.time()-start)),flush=True)
    fine.flush();coarse.flush();available.flush()
    report=dict(split=args.split,queries=len(data),source_sha256=digest(__file__),
                manifest_sha256=digest(PACKAGE/'manifest.json'),labels_opened=False,
                files={k:digest(out/k) for k in ('fine_history.npy','coarse_history.npy','available.npy')},seconds=time.time()-start)
    (out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
