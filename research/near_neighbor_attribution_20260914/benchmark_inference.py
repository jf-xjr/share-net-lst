"""Same end-to-end FP32 timing after training queues finish; inputs only."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import train_thst as th
from thst_reference_variants import matched_cache
from attribution_models import construct
r=th.r
class OneScene:
    def __init__(self,data,index):
        self.arrays={k:v[index:index+1] for k,v in data.arrays.items()};self.records=[data.records[index]]
    def __len__(self):return 1
    def batch(self,ids):return {k:np.array(v[ids],copy=True) for k,v in self.arrays.items()}
def main(checkpoints_override=None,output_override=None):
    r.setup('cuda');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=th.Dataset(th.PACKAGE,'validation',labels=False);cache=th.ThermalInputs('validation')
    matched,_=matched_cache(data,cache)
    order=np.argsort(np.asarray(cache.available).sum(1),kind='stable')
    indices=[int(order[0]),int(order[len(order)//2]),int(order[-1])]
    original=HERE.parents[1]/'research/sub04_20260913/compact_query_product/final_evaluation/test'
    receipt=json.loads((original/'predictions_complete.json').read_text())
    entry=next(e for e in receipt['entries'] if e['architecture']=='naf_history' and e['seed']==20260905)
    assert r.digest(entry['checkpoint'])==entry['checkpoint_sha256']
    checkpoints={'final_0.426':Path(entry['checkpoint']),'THST_common':HERE/'thst/stage2/best.pt',
                 'THST_matched':HERE/'thst/stage2/best.pt'}
    if checkpoints_override is not None:checkpoints=checkpoints_override
    results={}
    for name,path in checkpoints.items():
        torch.cuda.empty_cache();model=construct('learned') if name=='final_0.426' else th.CommonInputTHST()
        saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict'],strict=True)
        if name.startswith('THST'):model.set_stage(2)
        model=model.cuda().eval();rows=[]
        for index in indices:
            d=OneScene(data,index)
            used=matched if name.endswith('_matched') else cache
            c=SimpleNamespace(fine=used.fine[index:index+1],coarse=used.coarse[index:index+1],available=used.available[index:index+1])
            def predict():return r.inference(model,d,'cuda',1,amp=False) if name=='final_0.426' else th.predict(model,d,c)
            for _ in range(3):p=predict()
            durations=[];torch.cuda.reset_peak_memory_stats()
            for _ in range(10):
                torch.cuda.synchronize();start=time.perf_counter();q=predict();torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
                assert np.allclose(p,q,atol=1e-6,rtol=0,equal_nan=True)
            rows.append(dict(scene_id=d.records[0]['scene_id'],index=index,references=int(used.available[index].sum()),
                seconds=durations,median_seconds=float(np.median(durations)),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        results[name]=dict(checkpoint=str(path),checkpoint_sha256=r.digest(path),
                          parameters=sum(p.numel() for p in model.parameters()),cases=rows,
                          median_across_case_medians=float(np.median([v['median_seconds'] for v in rows])))
        del model,saved;torch.cuda.empty_cache()
    out=Path(output_override) if output_override is not None else HERE/'analysis/inference_benchmark.json'
    r.dump(out,dict(results=results,labels_opened=False,split='validation',scene_size=160,
        precision='FP32, TF32 disabled; common FP64 support projection',
        timed='CPU prepared input arrays through host-to-GPU transfers, complete method inference, projection, CPU output',
        selection='lowest, median, highest number of scene-available histories in Val45',
        warmup=3,repetitions=10,gpu=torch.cuda.get_device_name(0),source_sha256=r.digest(__file__)))
    print(json.dumps({k:dict(parameters=v['parameters'],median_seconds=v['median_across_case_medians']) for k,v in results.items()}),flush=True)
if __name__=='__main__':main()
