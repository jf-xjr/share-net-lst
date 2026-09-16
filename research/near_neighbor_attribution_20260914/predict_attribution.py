"""Seal all matched selections, then predict without opening target arrays."""
from pathlib import Path
import importlib.util,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]/'resources/historylst246'
sys.path[:0]=[str(HERE),str(PACKAGE)]
from attribution_models import construct
from historylst.data import Dataset
spec=importlib.util.spec_from_file_location('prediction_runner',PACKAGE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def main():
    r.setup('cuda');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=HERE/'attribution_predictions';out.mkdir(exist_ok=True)
    entries=[]
    for seed in (20260914,20260915):
        for variant in ('coverage','learned'):
            run=HERE/'attribution'/f'{variant}_{seed}'
            complete=json.loads((run/'complete.json').read_text())
            assert complete['status']=='complete' and complete['updates']==12000
            implementation=json.loads((run/'implementation.json').read_text())
            assert all(r.digest(p)==h for p,h in implementation.items())
            checkpoint=run/'best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            entries.append(dict(seed=seed,variant=variant,checkpoint=str(checkpoint),
                checkpoint_sha256=r.digest(checkpoint),step=ck['step'],weights=ck['weights'],
                validation_rmse=ck['validation_rmse'],completion=complete))
    freeze=dict(entries=entries,manifest_sha256=r.digest(PACKAGE/'manifest.json'),
                prediction_source_sha256=r.digest(__file__),labels_opened=False)
    path=out/'selection_freeze.json'
    if path.exists():assert json.loads(path.read_text())==freeze
    else:r.dump(path,freeze)
    data=Dataset(PACKAGE,'test',labels=False);receipts=[]
    for e in entries:
        path=out/f"{e['variant']}_{e['seed']}.npy";receipt=path.with_suffix('.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text());assert saved['prediction_sha256']==r.digest(path)
            assert saved['checkpoint_sha256']==e['checkpoint_sha256'];receipts.append(saved);continue
        if path.exists():raise FileExistsError('Unsealed prediction requires inspection')
        model=construct(e['variant']).cuda()
        ck=torch.load(e['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict'],strict=True)
        start=time.time();p=r.inference(model,data,'cuda',1,amp=False);np.save(path,p)
        saved=dict(**e,prediction=str(path),prediction_sha256=r.digest(path),
                   seconds=time.time()-start,labels_opened=False,queries=len(data),
                   scene_order=[q['scene_id'] for q in data.records])
        r.dump(receipt,saved);receipts.append(saved);del model
        print(json.dumps({k:saved[k] for k in ('variant','seed','seconds')}),flush=True)
    r.dump(out/'predictions_complete.json',dict(entries=receipts,queries=len(data),
        labels_opened=False,selection_freeze_sha256=r.digest(out/'selection_freeze.json')))

if __name__=='__main__':main()
