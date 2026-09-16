"""Paired within-region city comparison of two common-task score files."""
from pathlib import Path
import argparse,json
import numpy as np

def compare(first,second,seed=20260911,draws=20000):
    a={r['city']:r for r in first['cities']};b={r['city']:r for r in second['cities']}
    if set(a)!=set(b):raise ValueError('Different city cohorts')
    cities=sorted(a)
    if any(a[c]['region']!=b[c]['region'] for c in cities):raise ValueError('Different city-region assignments')
    regions=('china','europe','us');groups=[[i for i,c in enumerate(cities) if a[c]['region']==g] for g in regions]
    if any(not g for g in groups) or sum(map(len,groups))!=len(cities):raise ValueError('Expected all three task regions')
    rng=np.random.default_rng(seed);samples=[rng.choice(g,(draws,len(g))) for g in groups];result={}
    for k in ('rmse','mae','hotspot_iou','hotspot_mae'):
        if any(k not in a[c] or k not in b[c] for c in cities):continue
        delta=np.array([a[c][k]-b[c][k] for c in cities]);boots=np.mean([delta[d].mean(1) for d in samples],axis=0)
        sign=1 if k=='hotspot_iou' else -1
        result[k]=dict(a_minus_b=float(np.mean([delta[g].mean() for g in groups])),pointwise_city_95=np.quantile(boots,[.025,.975]).tolist(),
            cities_better=int((sign*delta>1e-10).sum()),cities_worse=int((sign*delta<-1e-10).sum()),
            region_a_minus_b={r:float(delta[g].mean()) for r,g in zip(regions,groups)})
    return {'contrasts':result,'seed':seed,'draws':draws,'interpretation':'First minus second; pointwise city intervals, not family-wise or training-seed intervals. Inputs must use the same task/support and aggregation.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=compare(json.loads(a.first.read_text()),json.loads(a.second.read_text()))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['contrasts']['rmse']))
