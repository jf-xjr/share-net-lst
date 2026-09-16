"""Complete the fixed native/calibrated experiment and freeze choice before Test."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,pathlib,subprocess,sys,time
import numpy as np
import torch
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'mocolsk'
sys.path[:0]=[str(HERE),str(ROOT/'research/near_neighbor_attribution_20260914'),str(ROOT/'resources/historylst246')]
from evaluate_comparisons import evaluate,mean_scores,bootstrap,digest,dump
from historylst.data import Dataset
from train_mocolsk import r
from mocolsk_model import CommonMoCoLSK
def call(*args):
    subprocess.run([sys.executable,str(HERE/'train_mocolsk.py'),*args],check=True)
def finish():
    deadline=time.monotonic()+7200
    while not (OUT/'native/complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Native training did not complete')
        time.sleep(10)
    if not (OUT/'calibrated/complete.json').exists():call('train','--phase','calibrated')
    selection={}
    for phase in ['native','calibrated']:
        p=OUT/phase/'best.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
        selection[phase]=dict(checkpoint=str(p),checkpoint_sha256=digest(p),validation_rmse=ck['validation_rmse'],step=ck['step'],weights=ck['weights'])
        del ck
    phase=min(selection,key=lambda p:selection[p]['validation_rmse'])
    freeze=dict(criterion='Minimum complete Val45 macro RMSE; native wins exact tie',selected_phase=phase,
      candidates=selection,test_labels_opened=False,protocol_sha256=digest(HERE/'PROTOCOL.md'),source_sha256=digest(__file__))
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==freeze
    else:dump(path,freeze)
    for p in ['native','calibrated']:
        if not (OUT/p/'predict/predictions.npy').exists():call('predict','--phase',p)
    # Both full Test packets are fixed before either score is computed.
    seal={p:digest(OUT/p/'predict/predictions.npy') for p in ['native','calibrated']}
    dump(OUT/'predictions_complete.json',dict(predictions_sha256=seal,selection_sha256=digest(path),labels_opened=False))
    data=Dataset(ROOT/'resources/historylst246','test',labels=True);scores={};members={}
    for p in ['native','calibrated']:
        arr=np.load(OUT/p/'predict/predictions.npy');members[p]=evaluate(arr,data)
        scores['MoCoLSK_'+p]=mean_scores([members[p]])
    scores['MoCoLSK_selected']=scores['MoCoLSK_'+phase]
    analysis=OUT/'analysis';dump(analysis/'scores.json',scores);dump(analysis/'members.json',members)
    final=json.loads((ROOT/'research/near_neighbor_attribution_20260914/analysis_augmented/scores.json').read_text())['final_0.426']
    pairs={p+'_minus_final':bootstrap(scores['MoCoLSK_'+p],final) for p in ['native','calibrated','selected']}
    dump(analysis/'paired.json',pairs)
    print(json.dumps({'selected':phase,'metrics':{p:v['macro'] for p,v in scores.items()},'paired':pairs['selected_minus_final']}),flush=True)
    benchmark(selection[phase],analysis)
    dump(analysis/'complete.json',dict(status='complete',selection_sha256=digest(path),prediction_sha256=seal,
      source_sha256=digest(__file__),outputs={p.name:digest(p) for p in analysis.iterdir() if p.name!='complete.json'}))
def benchmark(choice,out):
    r.setup('cuda');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Dataset(ROOT/'resources/historylst246','validation',labels=False)
    old=json.loads((ROOT/'research/near_neighbor_attribution_20260914/analysis_augmented/original_inference_benchmark.json').read_text())
    rows=[];torch.cuda.empty_cache();model=CommonMoCoLSK();ck=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict']);del ck;model=model.cuda().eval()
    class OneScene:
        def __init__(self,i):self.arrays={k:v[i:i+1] for k,v in data.arrays.items()};self.records=[data.records[i]]
        def __len__(self):return 1
        def batch(self,ids):return {k:np.array(v[ids],copy=True) for k,v in self.arrays.items()}
    for case in old['results']['final_0.426']['cases']:
        index=case['index'];d=OneScene(index);assert d.records[0]['scene_id']==case['scene_id']
        def infer():return r.inference(model,d,'cuda',1,amp=False)
        for _ in range(3):p=infer()
        durations=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();q=infer();torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
            assert np.allclose(p,q,atol=1e-6,rtol=0,equal_nan=True)
        rows.append(dict(scene_id=case['scene_id'],index=index,references=case['references'],seconds=durations,
          median_seconds=float(np.median(durations)),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    result=dict(checkpoint=choice['checkpoint'],checkpoint_sha256=choice['checkpoint_sha256'],
      parameters=sum(p.numel() for p in model.parameters()),cases=rows,
      median_across_case_medians=float(np.median([x['median_seconds'] for x in rows])))
    report={k:v for k,v in old.items() if k not in ['results','source_sha256']}
    report.update(results={'MoCoLSK_selected':result},source_sha256=digest(__file__))
    dump(out/'inference_benchmark.json',report)
    print(json.dumps({'timing_seconds':result['median_across_case_medians'],'parameters':result['parameters']}),flush=True)
if __name__=='__main__':finish()
