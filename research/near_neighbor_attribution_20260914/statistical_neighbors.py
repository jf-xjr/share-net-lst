"""Yoo/GRF and EOFRV task adaptations; prediction reads inputs only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys,tempfile,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PACKAGE=ROOT/'resources/historylst246'
E3=ROOT/'research/thermal_reuse_20260907/e3'
sys.path[:0]=[str(PACKAGE),str(E3)]
from historylst.data import Dataset
from historylst.metrics import repair
from lasso_kernel_selection import select_kernels
RSCRIPT=E3/'runtime/r_env/bin/Rscript'
RCORE=E3/'grf_reconstruct.R'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def aggregate(x,support):
    """Aggregate exactly the fine predictor used at inference on query support."""
    x=np.asarray(x,np.float64)
    m=np.asarray(support,bool)
    count=m.reshape(40,4,40,4).sum((1,3))
    mean=np.where(m[None],x,0).reshape(len(x),40,4,40,4).sum((2,4))/np.maximum(count,1)[None]
    return mean,count


def impute(x,support):
    out=np.array(x,dtype=np.float64,copy=True)
    for row in out:
        valid=np.isfinite(row)&support
        fill=float(np.median(row[valid])) if valid.any() else 0.
        row[~np.isfinite(row)]=fill
    return out


def prepare(data,index):
    b={k:np.asarray(v[index]) for k,v in data.arrays.items()}
    support=b['support'][0].astype(bool);h=np.asarray(b['history'],np.float64)
    valid=h[:,2]>0
    query_time=datetime.fromisoformat(data.records[index]['query_datetime'].replace('Z','+00:00'))
    for source in data.records[index]['history']:
        if source.get('datetime') and datetime.fromisoformat(source['datetime'].replace('Z','+00:00'))>=query_time:
            raise ValueError('Historical source is not strictly before target acquisition')
    available=np.flatnonzero(np.any(valid&support[None],axis=(1,2)))
    temperature=np.where(valid,20*h[:,0]+300,np.nan)
    temperature=impute(temperature[available],support)
    current=np.concatenate((b['fine'][1:],b['emissivity'],
                           np.broadcast_to(b['context'][:,None,None],(15,160,160))),axis=0)
    current=impute(current,support)
    tc,count=aggregate(temperature,support)
    train=np.flatnonzero(np.isfinite(b['coarse'][0]).ravel()&(count.ravel()>0))
    names=[f'history_{int(i)}_temperature' for i in available]
    selection=None
    if len(available):
        try:selection=select_kernels(tc.reshape(len(available),-1)[:,train].T,
                                    b['coarse'][0].ravel()[train],train,kernel_names=names)
        except ValueError as exc:
            # Input geometry failure is retained; it cannot remove a test scene.
            selection=dict(status='insufficient_coarse_selection_support',reason=str(exc),
                           selected_indices_zero_based=[],selected_kernel_names=[])
    if selection is None:selection=dict(status='no_history',selected_indices_zero_based=[],selected_kernel_names=[])
    selected=np.asarray(selection['selected_indices_zero_based'],int)
    return b,support,h,temperature,current,train,available,names,selected,selection


def matrix_design(prepared,mode):
    b,support,h,temperature,current,train,available,names,selected,selection=prepared
    feature_names=[f'current_{i}' for i in range(len(current))]
    roles=['auxiliary']*len(current)
    if mode=='yoo':
        fine=np.concatenate((current,temperature[selected]),axis=0)
        feature_names += [names[i] for i in selected]
        roles += ['thermal']*len(selected)
    elif mode=='allinputs':
        # Full available historical field information, including all temperature
        # candidates. LASSO determines LLF correction variables, not access.
        extra=np.array(h[:,1:],copy=True)
        extra[:,0]=np.where(h[:,2]>0,extra[:,0],0.)
        extra[:,2]=np.where(h[:,2]>0,extra[:,2],0.)
        extra[:,3]=np.where(h[:,5]>0,extra[:,3],0.)
        extra=impute(extra.reshape(-1,160,160),support)
        fine=np.concatenate((current,extra,temperature),axis=0)
        feature_names += [f'history_{j}_field_{k}' for j in range(9) for k in range(1,9)] + names
        roles += ['auxiliary']*len(extra)+['thermal' if i in selected else 'auxiliary' for i in range(len(available))]
    else:raise ValueError(mode)
    coarse,_=aggregate(fine,support)
    x=coarse.reshape(len(fine),-1)[:,train].T
    fine_ids=np.flatnonzero(support)
    z=fine.reshape(len(fine),-1)[:,fine_ids].T
    # Constants within this query do not enter tree splits or linear correction.
    keep=np.ptp(x,axis=0)>1e-10
    if not keep.any():keep[0]=True
    return (x[:,keep],b['coarse'][0].ravel()[train].astype(float),z[:,keep],fine_ids,
            [v for v,k in zip(feature_names,keep) if k],[v for v,k in zip(roles,keep) if k])


def forest_predict(design,workdir):
    x,y,z,fine_ids,names,roles=design
    if not all(np.isfinite(a).all() for a in (x,y,z)):raise ValueError('Nonfinite model matrix')
    with tempfile.TemporaryDirectory(prefix='historylst_grf_') as scratch:
        p=Path(scratch)
        for name,a,columns in [('train_x',x,names),('train_y',y[:,None],['coarse_K']),('predict_x',z,names)]:
            np.savetxt(p/(name+'.csv'),a,delimiter=',',header=','.join(columns),comments='',fmt='%.17g')
        with (p/'columns.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['name','role']);writer.writerows(zip(names,roles))
        command=[str(RSCRIPT),'--vanilla',str(RCORE),'--train-x',str(p/'train_x.csv'),
                 '--train-y',str(p/'train_y.csv'),'--predict-x',str(p/'predict_x.csv'),
                 '--columns',str(p/'columns.csv'),'--output',str(p/'r_output')]
        receipt={name:digest(p/name) for name in ('train_x.csv','train_y.csv','predict_x.csv','columns.csv')}
        result=subprocess.run(command,capture_output=True,text=True)
        (workdir/'r_stdout.txt').write_text(result.stdout);(workdir/'r_stderr.txt').write_text(result.stderr)
        if result.returncode:raise RuntimeError(f'R failure: {workdir}')
        pred=np.loadtxt(p/'r_output/predictions.csv',delimiter=',',skiprows=1)
        metadata=json.loads((p/'r_output/run_metadata.json').read_text())
        metadata['matrix_csv_sha256']=receipt
        dump(workdir/'r_metadata.json',metadata)
    raw=np.zeros((3,1,160,160),np.float64)
    for j in range(3):raw[j,0].ravel()[fine_ids]=pred[:,j+1]
    return raw


def eofrv(prepared,record):
    b,support,h,temperature,current,train,available,names,selected,selection=prepared
    # The provided parent anomaly yields the actual historical parent mean at
    # observed pixels; no fine target or target QA is consulted here.
    temp=20*h[:,0]+300;anom=5*h[:,1];valid=(h[:,2]>0)&support[None]
    historical_parent=temp-anom
    doy=(np.mod(np.arctan2(h[:,6],h[:,7]),2*np.pi)/(2*np.pi)*365.25)
    # Exact DOY comes from the recorded source acquisition when available.
    for j,source in enumerate(record['history']):
        date=source.get('datetime') or source.get('query_datetime')
        if date:doy[j]=datetime.fromisoformat(date.replace('Z','+00:00')).timetuple().tm_yday
    current_doy=datetime.fromisoformat(record['query_datetime'].replace('Z','+00:00')).timetuple().tm_yday
    coarse=np.repeat(np.repeat(b['coarse'][0],4,axis=0),4,axis=1)
    # Linear rescaling of polynomial predictors improves numerical conditioning
    # while preserving the original degree-two least-squares function class.
    d=(doy-183.)/183.;t=(historical_parent-300.)/20.
    design=np.stack((np.ones_like(t),d,d*d,t,t*t),-1)
    dq=(current_doy-183.)/183.;tq=(np.nan_to_num(coarse,nan=300.)-300.)/20.
    query=np.stack((np.ones_like(tq),np.full_like(tq,dq),np.full_like(tq,dq*dq),tq,tq*tq),-1)
    lambdas=(.01,.1,1.,10.,100.)
    result=np.zeros((160,160));regularized={v:result.copy() for v in lambdas};rank_counts={}
    # At most 512 observation patterns; fit many pixels in batched linear algebra.
    pattern=(valid.astype(np.uint16)*(1<<np.arange(9))[:,None,None]).sum(0)
    for key in np.unique(pattern[support]):
        pixels=(pattern==key)&support;ids=np.flatnonzero([(int(key)>>j)&1 for j in range(9)])
        if len(ids)==0:rank_counts['no_history']=rank_counts.get('no_history',0)+int(pixels.sum());continue
        x=design[ids][:,pixels,:].transpose(1,0,2);y=anom[ids][:,pixels].T
        # Original polynomial whenever identifiable; mean template is the
        # predeclared fallback for sparse/rank-deficient local archives.
        ranks=np.linalg.matrix_rank(x)
        good=(ranks==5)&(len(ids)>=5)
        values=y.mean(1)
        if good.any():
            coef=np.matmul(np.linalg.pinv(x[good]),y[good,:,None])[...,0]
            values[good]=(query[pixels][good]*coef).sum(1)
        result[pixels]=values
        xm=x.mean(1,keepdims=True);ym=y.mean(1,keepdims=True)
        xc=x-xm;yc=y-ym
        gram=np.swapaxes(xc,1,2)@xc
        rhs=np.swapaxes(xc,1,2)@yc[:,:,None]
        eye=np.eye(5)[None]
        for penalty in lambdas:
            coef=np.linalg.solve(gram+penalty*eye,rhs)[...,0]
            regularized[penalty][pixels]=ym[:,0]+((query[pixels]-xm[:,0])*coef).sum(1)
        rank_counts['polynomial']=rank_counts.get('polynomial',0)+int(good.sum())
        rank_counts['mean_fallback']=rank_counts.get('mean_fallback',0)+int((~good).sum())
    raw=np.asarray(b['fine'][:1],np.float64)+result[None]
    return {'eofrv':raw,**{f'eofrv_ridge_{penalty:g}':np.asarray(b['fine'][:1],np.float64)+v[None]
                          for penalty,v in regularized.items()}},rank_counts


def run_one(split,index):
    start=time.time();data=Dataset(PACKAGE,split,labels=False)
    out=HERE/'statistics_v2'/split/f'{index:03d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    prepared=prepare(data,index);b=prepared[0];outputs={};metrics={}
    for mode in ('yoo','allinputs'):
        folder=out/mode;folder.mkdir(exist_ok=True)
        pred=forest_predict(matrix_design(prepared,mode),folder)
        fixed=repair(pred,np.repeat(b['coarse'][None],3,0),np.repeat(b['support'][None],3,0))
        for j,kind in enumerate(('aux_rf','rf','llf')):
            outputs[mode+'_'+kind]=fixed[j]
            outputs[mode+'_'+kind+'_raw']=pred[j]
    estimates,disposition=eofrv(prepared,data.records[index])
    for name,raw in estimates.items():
        outputs[name+'_raw']=raw
        outputs[name]=repair(raw[None],b['coarse'][None],b['support'][None])[0]
    for key,value in outputs.items():
        if not key.endswith('_raw') and not np.isfinite(value[b['support'].astype(bool)]).all():
            raise ValueError('Nonfinite supported prediction')
    np.savez_compressed(out/'predictions.npz',**outputs)
    receipt=dict(index=index,split=split,scene_id=data.records[index]['scene_id'],seconds=time.time()-start,
                 selection=prepared[-1],eofrv_disposition=disposition,labels_opened=False,
                 prediction_sha256=digest(out/'predictions.npz'))
    dump(out/'complete.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],required=True)
    parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();data=Dataset(PACKAGE,args.split,labels=False)
    out=HERE/'statistics_v2'/args.split;out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),E3/'lasso_kernel_selection.py',RCORE,PACKAGE/'manifest.json',PACKAGE/'historylst/data.py',PACKAGE/'historylst/metrics.py']
    seal={str(p):digest(p) for p in sources}
    path=out/'source_freeze.json'
    if path.exists() and json.loads(path.read_text())!=seal:raise ValueError('Source freeze differs')
    if not path.exists():dump(path,seal)
    indices=list(range(len(data) if args.limit is None else min(args.limit,len(data))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,args.split,i):i for i in indices}
        for future in as_completed(futures):
            r=future.result();print(json.dumps({k:r[k] for k in ('index','split','seconds')}),flush=True)
    if any(digest(p)!=h for p,h in seal.items()):raise ValueError('Sources changed during predictions')
    if len(indices)==len(data):
        rows=[json.loads((out/f'{i:03d}'/'complete.json').read_text()) for i in indices]
        dump(out/'predictions_complete.json',dict(queries=len(rows),labels_opened=False,scenes=rows,source_sha256=seal))


if __name__=='__main__':main()
