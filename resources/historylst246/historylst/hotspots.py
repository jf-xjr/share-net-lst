"""Scene-relative top-decile surface hotspots under the fixed formal support."""
import math
import numpy as np

def add_hotspot_metrics(result,predictions,target,formal):
    """Add IoU and reference-hotspot MAE to a validated core score in place."""
    for row,p,t,m in zip(result['scenes'],predictions,target,formal):
        m=np.asarray(m,bool);p=np.asarray(p[m],np.float64);t=np.asarray(t[m],np.float64)
        n=math.ceil(len(t)/10)
        ref=np.argsort(-t,kind='stable')[:n];pred=np.argsort(-p,kind='stable')[:n]
        shared=len(np.intersect1d(ref,pred,assume_unique=True))
        row['hotspot_iou']=float(shared/(2*n-shared))
        row['hotspot_mae']=float(np.mean(abs(p[ref]-t[ref])))
    names=('hotspot_iou','hotspot_mae')
    for city in result['cities']:
        rows=[r for r in result['scenes'] if r['city']==city['city']]
        for k in names:city[k]=float(np.mean([r[k] for r in rows]))
    for region,row in result['regions'].items():
        cities=[c for c in result['cities'] if c['region']==region]
        for k in names:row[k]=float(np.mean([c[k] for c in cities]))
    for k in names:result['macro'][k]=float(np.mean([r[k] for r in result['regions'].values()]))
    return result
