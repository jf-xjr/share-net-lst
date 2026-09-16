"""Common actual-support projection and original macro scoring."""
from collections import defaultdict
import numpy as np

def repair(prediction,coarse,support):
    p=np.asarray(prediction,dtype=np.float64).copy();m=np.asarray(support,dtype=bool)
    n,c,h,w=p.shape
    pb=p.reshape(n,c,h//4,4,w//4,4);mb=m.reshape(n,c,h//4,4,w//4,4)
    count=mb.sum((3,5));mean=np.where(mb,pb,0).sum((3,5))/np.maximum(count,1)
    delta=np.where(np.isfinite(coarse)&(count>0),coarse-mean,0)
    pb+=delta[:,:,:,None,:,None]
    return np.where(m,p,np.nan)

def score(prediction,target,formal,records):
    if prediction.shape != target.shape or formal.shape != target.shape:
        raise ValueError('Prediction, target and formal support shapes must match exactly')
    if len(prediction) != len(records):
        raise ValueError('Prediction count must match the complete ordered split')
    scenes=[]
    for p,t,m,r in zip(prediction,target,formal,records):
        mask=np.asarray(m,bool)
        if not mask.any():raise ValueError('Empty scoring scene')
        if not np.isfinite(p[mask]).all():raise ValueError('Nonfinite prediction on scoring support')
        if not np.isfinite(t[mask]).all():raise ValueError('Nonfinite reference on scoring support')
        e=p[mask].astype(np.float64)-t[mask].astype(np.float64)
        scenes.append(dict(scene_id=r['scene_id'],city=r['city'],region=r['region'],pixels=int(mask.sum()),
            rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(np.mean(e)),mse=float(np.mean(e**2))))
    cities=[]
    for city in sorted({r['city'] for r in scenes}):
        rows=[r for r in scenes if r['city']==city]
        cities.append(dict(city=city,region=rows[0]['region'],**{k:float(np.mean([r[k] for r in rows])) for k in ('rmse','mae','bias','mse')}))
    regions={g:{k:float(np.mean([r[k] for r in cities if r['region']==g])) for k in ('rmse','mae','bias','mse')} for g in sorted({r['region'] for r in cities})}
    return dict(macro={k:float(np.mean([r[k] for r in regions.values()])) for k in ('rmse','mae','bias','mse')},regions=regions,cities=cities,scenes=scenes)
