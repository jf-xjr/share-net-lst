"""Timing-only completion of the existing temporal-attention comparator."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,pathlib,sys,time
import numpy as np
import torch
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'resources/historylst246')]
from train_mocolsk import r
from historylst.model import HistoryUTAE
from historylst.data import Dataset
def main():
    assert (HERE/'mocolsk/analysis/complete.json').exists(),'Finish all training and comparator timing first'
    out=HERE/'utae_inference_benchmark.json'
    if out.exists():raise FileExistsError(out)
    r.setup('cuda');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Dataset(ROOT/'resources/historylst246','validation',labels=False)
    original=json.loads((ROOT/'research/near_neighbor_attribution_20260914/analysis_augmented/original_inference_benchmark.json').read_text())
    receipt=json.loads((ROOT/'research/sub04_20260913/compact_query_product/final_evaluation/test/predictions_complete.json').read_text())
    e=next(e for e in receipt['entries'] if e['architecture']=='baseline' and e['seed']==20260905)
    path=pathlib.Path(e['checkpoint']);assert r.digest(path)==e['checkpoint_sha256']
    model=HistoryUTAE();ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'],strict=True)
    del ck;model=model.cuda().eval();cases=[]
    class OneScene:
        def __init__(self,i):self.arrays={k:v[i:i+1] for k,v in data.arrays.items()};self.records=[data.records[i]]
        def __len__(self):return 1
        def batch(self,ids):return {k:np.array(v[ids],copy=True) for k,v in self.arrays.items()}
    for case in original['results']['final_0.426']['cases']:
        d=OneScene(case['index']);assert d.records[0]['scene_id']==case['scene_id']
        def infer():return r.inference(model,d,'cuda',1,amp=False)
        for _ in range(3):p=infer()
        durations=[];differences=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            torch.cuda.synchronize();begin=time.perf_counter();q=infer();torch.cuda.synchronize();durations.append(time.perf_counter()-begin)
            delta=float(np.nanmax(abs(p-q)));differences.append(delta)
            assert np.array_equal(np.isfinite(p),np.isfinite(q))
            assert delta<.0002,delta
        cases.append(dict(scene_id=case['scene_id'],index=case['index'],references=case['references'],seconds=durations,
          median_seconds=float(np.median(durations)),peak_cuda_bytes=torch.cuda.max_memory_allocated(),repeat_maximum_difference_K=differences))
    result=dict(checkpoint=str(path),checkpoint_sha256=r.digest(path),parameters=sum(p.numel() for p in model.parameters()),cases=cases,
      median_across_case_medians=float(np.median([x['median_seconds'] for x in cases])))
    report={k:v for k,v in original.items() if k not in ['results','source_sha256']}
    report.update(results={'original_UTAE':result},repeat_check='Initial 1e-6 K bound replaced by the established 2e-4 K FP32 operation bound; actual repeat differences retained. No training or Test predictions repeated.',source_sha256={str(p):r.digest(p) for p in [pathlib.Path(__file__),ROOT/'resources/historylst246/historylst/model.py']})
    r.dump(out,report);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
