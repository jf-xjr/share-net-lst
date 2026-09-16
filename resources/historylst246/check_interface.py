"""Mechanical interface checks; Fit inputs only, no query labels or scores."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from historylst.data import Dataset
from historylst.model import HistoryUTAE
from historylst.metrics import repair,score

def main(root):
    torch.set_num_threads(2);torch.manual_seed(71)
    data=Dataset(root,'fit',labels=False)
    assert set(data.arrays)=={'fine','coarse','support','context','emissivity','history'}
    b={k:torch.from_numpy(v) for k,v in data.batch([0]).items()}
    model=HistoryUTAE().eval()
    # A nonzero regression head makes the missing-value check non-vacuous.
    torch.nn.init.normal_(model.core.out_conv.weight,std=.05)
    with torch.inference_mode():
        original=model(**b).numpy()
        changed={k:v.clone() for k,v in b.items()};h=changed['history']
        missing=h[:,:,2:3]==0
        assert missing.any()
        h[:,:,:2]=torch.where(missing,torch.full_like(h[:,:,:2],12345.),h[:,:,:2])
        h[:,:,3:4]=torch.where(missing,torch.full_like(h[:,:,3:4],54321.),h[:,:,3:4])
        altered=model(**changed).numpy()
        assert np.array_equal(original,altered)
        missing_b={k:v.clone() for k,v in b.items()};missing_b['history'].zero_()
        no_history=model(**missing_b).numpy();assert np.isfinite(no_history).all()
    rng=np.random.default_rng(83)
    p=rng.normal(300,5,(2,1,160,160));s=rng.random(p.shape)>.35
    c=rng.normal(300,4,(2,1,40,40));c[:,:,0,:]=np.nan;s[:,:,4:8,4:8]=False
    out=repair(p,c,s);count=s.reshape(2,1,40,4,40,4).sum((3,5))
    means=np.where(s,out,0).reshape(2,1,40,4,40,4).sum((3,5))/np.maximum(count,1)
    observed=np.isfinite(c)&(count>0);error=float(np.max(abs(means[observed]-c[observed])))
    assert error<1e-10 and np.isnan(out[~s]).all()
    # Unsupported parents must not receive an invented mean constraint.
    assert np.array_equal(out[:,:,:4,:][s[:,:,:4,:]],p[:,:,:4,:][s[:,:,:4,:]])
    try:score(np.zeros((1,1,1,1)),np.zeros((2,1,1,1)),np.ones((2,1,1,1),bool),[])
    except ValueError:pass
    else:raise AssertionError('Scorer silently accepted a truncated split')
    return dict(status='pass',labels_opened=False,fit_queries_read=1,
        missing_temperature_and_QA_invariant=True,all_history_missing_finite=True,
        partial_support_projection_max_abs_k=error,unobserved_parent_unchanged=True,
        truncated_predictions_rejected=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=main(a.root)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
