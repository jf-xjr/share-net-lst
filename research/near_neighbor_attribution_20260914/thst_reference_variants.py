"""Input-only reference matching, Val policy selection, then Test prediction."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import train_thst as th
r=th.r
def matched_cache(data,cache):
    available=np.zeros_like(cache.available);rows=[]
    for i in range(len(data)):
        refs=np.flatnonzero(cache.available[i]);scores=[]
        h=data.arrays['history'][i];support=data.arrays['support'][i,0].astype(bool)
        query=data.arrays['coarse'][i,0]
        for j in refs:
            seen=((h[j,2]>0)&support).reshape(40,4,40,4).sum((1,3))>0
            mask=seen&np.isfinite(query);correlation=None
            if mask.sum()>=64:
                x=np.asarray(query[mask],float);y=np.asarray(cache.coarse[i,j][mask],float)
                x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
                if den>1e-12:correlation=float(np.dot(x,y)/den)
            age=float(np.median(h[j,8]));scores.append(dict(reference=int(j),correlation=correlation,age=age,parents=int(mask.sum())))
        if len(refs):
            chosen=max(scores,key=lambda v:(v['correlation'] if v['correlation'] is not None else -2.,-v['age']))['reference']
            available[i,chosen]=True
        else:chosen=0
        rows.append(dict(scene_id=data.records[i]['scene_id'],reference=chosen,candidates=scores))
    return SimpleNamespace(fine=cache.fine,coarse=cache.coarse,available=available),rows
def main():
    r.setup('cuda');torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=HERE/'thst_reference';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    path=HERE/'thst/stage2/best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
    model=th.CommonInputTHST();model.load_state_dict(ck['state_dict']);model.set_stage(2);model.cuda()
    data=th.Dataset(th.PACKAGE,'validation',labels=False);cache=th.ThermalInputs('validation')
    single,rows=matched_cache(data,cache)
    vp={'all':th.predict(model,data,cache),'matched':th.predict(model,data,single)}
    labels=th.Dataset(th.PACKAGE,'validation',labels=True)
    scores={k:th.score(v,labels.arrays['target'],labels.arrays['formal'],labels.records)['macro']['rmse'] for k,v in vp.items()}
    assert abs(scores['all']-ck['validation_rmse'])<1e-6
    for name,p in vp.items():np.save(out/f'validation_{name}.npy',p)
    selection=dict(selected=min(scores,key=scores.get),validation_rmse=scores,checkpoint_sha256=r.digest(path),
        source_sha256=r.digest(__file__),protocol_sha256=r.digest(HERE/'THST_REFERENCE_VARIANTS.md'),validation_references=rows)
    freeze=out/'selection.json'
    if freeze.exists():assert json.loads(freeze.read_text())==selection
    else:r.dump(freeze,selection)
    data=th.Dataset(th.PACKAGE,'test',labels=False);cache=th.ThermalInputs('test');single,rows=matched_cache(data,cache)
    start=time.time();p=th.predict(model,data,single);path=out/'test_matched.npy';np.save(path,p)
    r.dump(out/'complete.json',dict(selection_sha256=r.digest(freeze),prediction_sha256=r.digest(path),
        checkpoint_sha256=selection['checkpoint_sha256'],labels_opened_for_test=False,queries=len(data),
        seconds=time.time()-start,scene_order=[v['scene_id'] for v in data.records],test_references=rows))
    print(json.dumps(dict(selected=selection['selected'],validation_rmse=scores)),flush=True)
if __name__=='__main__':main()
