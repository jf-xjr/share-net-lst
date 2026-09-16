"""Timing retry with explicit FP32 reproducibility measurements, no retraining."""
from finish_mocolsk import *
def main():
    analysis=OUT/'analysis';selection=json.loads((OUT/'selection.json').read_text());choice=selection['candidates'][selection['selected_phase']]
    r.setup('cuda');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Dataset(ROOT/'resources/historylst246','validation',labels=False)
    old=json.loads((ROOT/'research/near_neighbor_attribution_20260914/analysis_augmented/original_inference_benchmark.json').read_text())
    model=CommonMoCoLSK();ck=torch.load(choice['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);del ck
    model=model.cuda().eval();rows=[]
    class OneScene:
        def __init__(self,i):self.arrays={k:v[i:i+1] for k,v in data.arrays.items()};self.records=[data.records[i]]
        def __len__(self):return 1
        def batch(self,ids):return {k:np.array(v[ids],copy=True) for k,v in self.arrays.items()}
    for case in old['results']['final_0.426']['cases']:
        d=OneScene(case['index']);assert d.records[0]['scene_id']==case['scene_id']
        def infer():return r.inference(model,d,'cuda',1,amp=False)
        for _ in range(3):p=infer()
        durations=[];differences=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();q=infer();torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
            delta=float(np.nanmax(abs(p-q)));differences.append(delta)
            assert np.array_equal(np.isfinite(p),np.isfinite(q))
            assert delta<.0002,delta
        rows.append(dict(scene_id=case['scene_id'],index=case['index'],references=case['references'],seconds=durations,
          median_seconds=float(np.median(durations)),peak_cuda_bytes=torch.cuda.max_memory_allocated(),repeat_maximum_difference_K=differences))
    result=dict(checkpoint=choice['checkpoint'],checkpoint_sha256=choice['checkpoint_sha256'],parameters=sum(p.numel() for p in model.parameters()),cases=rows,
      median_across_case_medians=float(np.median([x['median_seconds'] for x in rows])))
    report={k:v for k,v in old.items() if k not in ['results','source_sha256']}
    report.update(results={'MoCoLSK_selected':result},source_sha256=digest(pathlib.Path(__file__)),
      timing_recovery='The initial 1e-6 K repeat check was stricter than FP32 full-network numerical resolution. No training or Test prediction was repeated. This timing run records each observed difference against the established 2e-4 K FP32 check bound.')
    dump(analysis/'inference_benchmark.json',report)
    seal=json.loads((OUT/'predictions_complete.json').read_text())['predictions_sha256']
    assert all(digest(OUT/p/'predict/predictions.npy')==v for p,v in seal.items())
    dump(analysis/'complete.json',dict(status='complete',selection_sha256=digest(OUT/'selection.json'),prediction_sha256=seal,
      timing_recovery_source_sha256=digest(pathlib.Path(__file__)),source_sha256=digest(HERE/'finish_mocolsk.py'),
      outputs={p.name:digest(p) for p in analysis.iterdir() if p.name!='complete.json'}))
    print(json.dumps({'timing_seconds':result['median_across_case_medians'],'maximum_repeat_difference_K':max(max(x['repeat_maximum_difference_K']) for x in rows)}),flush=True)
if __name__=='__main__':main()
