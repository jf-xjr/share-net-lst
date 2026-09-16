"""Replay the previously fixed past-only all-pair ubESTARFM adaptation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'fusion_reference'))
from historylst.data import Dataset
from past_ubestarfm import predict_scene

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=HERE)
    p.add_argument('--split',choices=['fit','validation','test'],default='test')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0,help='Fit-only mechanical check; never a truncated evaluation')
    a=p.parse_args()
    if a.limit and a.split!='fit':raise ValueError('--limit is only allowed for Fit smoke checks')
    data=Dataset(a.root,a.split,labels=False);n=min(len(data),a.limit) if a.limit else len(data)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    output=np.lib.format.open_memmap(a.output,mode='w+',dtype='float64',shape=(n,1,160,160))
    began=time.perf_counter();rows=[]
    for i in range(n):
        inputs=data.batch([i]);prediction,details=predict_scene(**inputs,mode='pool',workers=a.workers,return_details=True)
        output[i]=np.where(inputs['support'],prediction,np.nan)[0]
        rows.append(dict(scene_id=data.records[i]['scene_id'],**details))
        if (i+1)%15==0:print(json.dumps(dict(queries=i+1,seconds=time.perf_counter()-began)),flush=True)
    output.flush();receipt=dict(method='past-only all-pair ubESTARFM adaptation',queries=n,split=a.split,
        seconds=time.perf_counter()-began,labels_opened=False,fit_only_smoke=bool(a.limit),supply=rows)
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='supply'}))
