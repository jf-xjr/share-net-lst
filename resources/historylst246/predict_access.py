"""Seal three fixed-weight access contrasts before scoring; inputs only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from historylst.data import Dataset
from historylst.model import HistoryUTAE
from historylst.metrics import repair

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=HERE);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['fit','validation','test'],default='test')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=Dataset(a.root,a.split,labels=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=HistoryUTAE().eval();model.load_state_dict(ck['state_dict'],strict=True)
    receipts={}
    for mode,keep in [('full9',9),('keep7',7),('zero',0)]:
        path=a.output/(mode+'.npy');out=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(len(data),1,160,160));start=time.perf_counter()
        for i in range(len(data)):
            inputs=data.batch([i]);inputs['history'][:,keep:]=0
            b={k:torch.from_numpy(v) for k,v in inputs.items()}
            with torch.inference_mode():raw=model(**b).float().numpy()
            out[i]=repair(raw,inputs['coarse'],inputs['support'])[0]
        out.flush();receipts[mode]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.perf_counter()-start)
        print(json.dumps(dict(mode=mode,**receipts[mode])),flush=True)
    receipt=dict(status='all_predictions_sealed',split=a.split,queries=len(data),labels_opened=False,device='cpu',forward_dtype='float32',
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),selected_step=ck['step'],selected_weights=ck['weights'],predictions=receipts,
        interpretation='Fixed-weight input access, not separately trained no-history or seven-slot models')
    (a.output/'predictions_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
