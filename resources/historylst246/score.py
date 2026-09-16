"""Score any implementation's ordered Kelvin predictions; NumPy only."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from historylst.data import Dataset
from historylst.metrics import score, repair
from historylst.hotspots import add_hotspot_metrics

def evaluate(root, split, predictions, project=False):
    data=Dataset(root,split,labels=True)
    p=np.load(predictions,mmap_mode='r',allow_pickle=False)
    expected=data.arrays['target'].shape
    if p.shape != expected:raise ValueError(f'Expected {expected}, received {p.shape}')
    if project:p=repair(p,data.arrays['coarse'],data.arrays['support'])
    results=score(p,data.arrays['target'],data.arrays['formal'],data.records)
    add_hotspot_metrics(results,p,data.arrays['target'],data.arrays['formal'])
    s=np.asarray(data.arrays['support']);c=np.asarray(data.arrays['coarse'])
    cells=np.where(s,p,0).reshape(len(data),1,40,4,40,4)
    counts=s.reshape(len(data),1,40,4,40,4).sum((3,5))
    means=cells.sum((3,5))/np.maximum(counts,1);observed=np.isfinite(c)&(counts>0)
    results['consistency_max_abs_k']=float(np.max(np.abs(means[observed]-c[observed])))
    results['evaluation']=dict(split=split,projection_applied_by_scorer=project,
        aggregation='scene metric -> city mean -> region mean -> macro',
        prediction_sha256=hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
        test_status=data.manifest['test_status'],kelvin_units=True)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--split',choices=['fit','validation','test'],default='test')
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',action='store_true',help='Explicitly apply common support projection before scoring')
    a=p.parse_args();r=evaluate(a.root,a.split,a.predictions,a.project)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r['macro']))
