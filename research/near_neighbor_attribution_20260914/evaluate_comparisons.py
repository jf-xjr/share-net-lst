"""Common scoring, Val-only ridge selection, and stratified paired analysis."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PACKAGE=ROOT/'resources/historylst246'
sys.path.insert(0,str(PACKAGE))
from historylst.data import Dataset
from historylst.metrics import score
from historylst.hotspots import add_hotspot_metrics
FIELDS=('rmse','mae','bias','mse','hotspot_iou','hotspot_mae')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def audit(prediction,data):
    p=np.asarray(prediction);support=data.arrays['support'].astype(bool)
    assert p.shape==support.shape and len(p)==len(data)
    assert np.isfinite(p[support]).all()
    pb=np.where(support,p,0).reshape(len(p),1,40,4,40,4)
    count=support.reshape(len(p),1,40,4,40,4).sum((3,5))
    mean=pb.sum((3,5))/np.maximum(count,1)
    coarse=data.arrays['coarse'];mask=np.isfinite(coarse)&(count>0)
    error=float(np.max(abs(mean[mask]-coarse[mask])))
    assert error<1e-8,('Coarse mismatch',error)
    return dict(queries=len(p),supported_pixels=int(support.sum()),maximum_coarse_error_K=error)
def evaluate(p,data):
    check=audit(p,data)
    result=score(p,data.arrays['target'],data.arrays['formal'],data.records)
    add_hotspot_metrics(result,p,data.arrays['target'],data.arrays['formal'])
    result['audit']=check;return result
def stats_arrays(split):
    data=Dataset(PACKAGE,split,labels=False);folder=HERE/'statistics_v2'/split
    receipt=json.loads((folder/'predictions_complete.json').read_text())
    assert receipt['queries']==len(data) and len(receipt['scenes'])==len(data)
    arrays={}
    for i,(r,q) in enumerate(zip(receipt['scenes'],data.records)):
        path=folder/f'{i:03d}'/'predictions.npz'
        assert r['scene_id']==q['scene_id'] and r['index']==i
        assert r['prediction_sha256']==digest(path)
        with np.load(path) as archive:
            for name in archive.files:
                if name.endswith('_raw'):continue
                arrays.setdefault(name,[]).append(archive[name])
    return {name:np.stack(v) for name,v in arrays.items()}
def statistics(split):
    if split=='test':selection=json.loads((HERE/'statistics_v2/selection.json').read_text())
    arrays=stats_arrays(split);data=Dataset(PACKAGE,split,labels=True)
    results={name:evaluate(p,data) for name,p in arrays.items()}
    folder=HERE/'statistics_v2'/split;dump(folder/'scores.json',results)
    if split=='validation':
        keys=[f'eofrv_ridge_{v:g}' for v in (.01,.1,1.,10.,100.)]
        choice=min(keys,key=lambda name:results[name]['macro']['rmse'])
        selection=dict(selected_eofrv=choice,criterion='complete Val45 region-macro RMSE',
            candidate_scores={name:results[name]['macro']['rmse'] for name in keys},
            validation_predictions_receipt_sha256=digest(folder/'predictions_complete.json'),
            scoring_source_sha256=digest(__file__),analysis_protocol_sha256=digest(HERE/'ANALYSIS_PROTOCOL.md'))
        path=HERE/'statistics_v2/selection.json'
        if path.exists():assert json.loads(path.read_text())==selection
        else:dump(path,selection)
    print(json.dumps(dict(split=split,selected_eofrv=selection['selected_eofrv'],
                         metrics={k:v['macro'] for k,v in results.items()})),flush=True)
    return results,arrays
def mean_scores(results):
    merged={}
    merged['macro']={k:float(np.mean([r['macro'][k] for r in results])) for k in FIELDS}
    merged['regions']={g:{k:float(np.mean([r['regions'][g][k] for r in results])) for k in FIELDS}
                       for g in results[0]['regions']}
    merged['cities']=[]
    for city in results[0]['cities']:
        rows=[next(c for c in r['cities'] if c['city']==city['city']) for r in results]
        merged['cities'].append(dict(city=city['city'],region=city['region'],
                                    **{k:float(np.mean([c[k] for c in rows])) for k in FIELDS}))
    merged['members']=[r['macro'] for r in results];return merged
def bootstrap(a,b,metric='rmse'):
    ca={r['city']:r for r in a['cities']};cb={r['city']:r for r in b['cities']}
    assert ca.keys()==cb.keys();rng=np.random.default_rng(20260914)
    groups={g:[] for g in sorted({r['region'] for r in ca.values()})}
    rows=[]
    for city in sorted(ca):
        delta=ca[city][metric]-cb[city][metric];g=ca[city]['region'];groups[g].append(delta)
        rows.append(dict(city=city,region=g,delta=delta))
    draws=np.zeros(10000)
    for v in groups.values():
        v=np.asarray(v);draws+=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)/len(groups)
    return dict(metric=metric,definition='first minus second',
        delta=float(np.mean([np.mean(v) for v in groups.values()])),
        confidence_interval_95=np.quantile(draws,[.025,.975]).tolist(),
        positive_cities=sum(r['delta']>0 for r in rows),cities=len(rows),
        regional_delta={g:float(np.mean(v)) for g,v in groups.items()},city_deltas=rows)
def input_strata(data):
    h=np.asarray(data.arrays['history']);valid=h[:,:,2]>0;count=valid.sum(1)
    cov=h[:,:,2].mean(1);parent=20*h[:,:,0]+300-5*h[:,:,1]
    current=np.repeat(np.repeat(np.asarray(data.arrays['coarse'])[:,0],4,1),4,2)
    lo=np.where(valid,parent,np.inf).min(1);hi=np.where(valid,parent,-np.inf).max(1)
    gap=np.where(valid,abs(parent-current[:,None]),np.inf).min(1)
    enough=(count>=2)&np.isfinite(current)
    return {'history_0':count==0,'history_1_2':(count>=1)&(count<=2),
        'history_3_5':(count>=3)&(count<=5),'history_6_9':count>=6,
        'coverage_low':cov<.25,'coverage_middle':(cov>=.25)&(cov<.75),'coverage_high':cov>=.75,
        'state_below':enough&(current<lo-1),'state_inside':enough&(current>=lo-1)&(current<=hi+1),
        'state_above':enough&(current>hi+1),
        'nearest_gap_0_2':(count>0)&(gap<=2),'nearest_gap_2_5':(count>0)&(gap>2)&(gap<=5),
        'nearest_gap_over5':(count>0)&np.isfinite(gap)&(gap>5)}
def stratified(arrays,data):
    strata=input_strata(data);result={}
    for name,mask in strata.items():
        mask=mask[:,None]&data.arrays['formal'].astype(bool)
        eligible=np.flatnonzero(mask.sum((1,2,3))>=32)
        row=dict(scenes=len(eligible),pixels=int(mask[eligible].sum()),
                 cities=len({data.records[i]['city'] for i in eligible}),
                 regions=len({data.records[i]['region'] for i in eligible}),methods={})
        if len(eligible):
            records=[data.records[i] for i in eligible]
            for method,members in arrays.items():
                scores=[score(p[eligible],data.arrays['target'][eligible],mask[eligible],records) for p in members]
                row['methods'][method]={k:float(np.mean([r['macro'][k] for r in scores])) for k in ('rmse','mae','bias','mse')}
        result[name]=row
    return result
def all_results():
    output=HERE/'analysis';output.mkdir(exist_ok=True)
    stats,sa=statistics('test');data=Dataset(PACKAGE,'test',labels=True)
    choice=json.loads((HERE/'statistics_v2/selection.json').read_text())['selected_eofrv']
    groups={k:[v] for k,v in stats.items()};arrays={k:[v] for k,v in sa.items()}
    original=ROOT/'research/sub04_20260913/compact_query_product/final_evaluation/test'
    receipt=json.loads((original/'predictions_complete.json').read_text())
    assert receipt['scene_ids']==[r['scene_id'] for r in data.records]
    for e in receipt['entries']:
        path=original/e['prediction'];assert digest(path)==e['prediction_sha256']
        name='final_0.426' if e['architecture']=='naf_history' else 'original_UTAE'
        p=np.load(path);groups.setdefault(name,[]).append(evaluate(p,data));arrays.setdefault(name,[]).append(p)
    receipt=json.loads((HERE/'attribution_predictions/predictions_complete.json').read_text())
    attribution_rows=[]
    for e in receipt['entries']:
        assert e['scene_order']==[r['scene_id'] for r in data.records]
        path=Path(e['prediction']);assert digest(path)==e['prediction_sha256'];p=np.load(path)
        name=e['variant'];s=evaluate(p,data)
        groups.setdefault(name,[]).append(s);arrays.setdefault(name,[]).append(p)
        attribution_rows.append(dict(variant=name,seed=e['seed'],selected_step=e['step'],
                                    weights=e['weights'],validation_rmse=e['validation_rmse'],scores=s))
    path=HERE/'thst/stage2/test_predictions.npy';receipt=json.loads(path.with_suffix('.json').read_text())
    assert receipt['queries']==len(data) and receipt['split']=='test' and receipt['labels_opened'] is False
    assert digest(path.parent/'best.pt')==receipt['checkpoint_sha256']
    assert read_manifest_hash(path.parent/'run.json')==digest(PACKAGE/'manifest.json')
    assert digest(path)==receipt['prediction_sha256'];p=np.load(path)
    groups['THST_common']=[evaluate(p,data)];arrays['THST_common']=[p]
    refroot=HERE/'thst_reference';refreceipt=json.loads((refroot/'complete.json').read_text())
    assert refreceipt['scene_order']==[q['scene_id'] for q in data.records]
    assert digest(refroot/'test_matched.npy')==refreceipt['prediction_sha256']
    assert digest(refroot/'selection.json')==refreceipt['selection_sha256']
    thst_choice=json.loads((refroot/'selection.json').read_text())['selected']
    p=np.load(refroot/'test_matched.npy');groups['THST_matched']=[evaluate(p,data)];arrays['THST_matched']=[p]
    name='THST_common' if thst_choice=='all' else 'THST_matched'
    groups['THST_selected']=groups[name];arrays['THST_selected']=arrays[name]
    combined={k:mean_scores(v) for k,v in groups.items()}
    assert abs(combined['final_0.426']['macro']['rmse']-.42613199570471244)<1e-9
    pairs={'coverage_minus_learned':bootstrap(combined['coverage'],combined['learned'])}
    for name in ('yoo_llf','allinputs_llf',choice,'THST_selected','original_UTAE'):
        pairs[name+'_minus_final']=bootstrap(combined[name],combined['final_0.426'])
    for seed in (20260914,20260915):
        pair=[next(r['scores'] for r in attribution_rows if r['variant']==v and r['seed']==seed)
              for v in ('coverage','learned')]
        pairs[f'coverage_minus_learned_{seed}']=bootstrap(*pair)
    selected={k:arrays[k] for k in ('final_0.426','coverage','learned','yoo_llf','allinputs_llf',choice,'THST_selected')}
    strata=stratified(selected,data)
    dump(output/'scores.json',combined);dump(output/'paired.json',pairs)
    dump(output/'attribution_members.json',attribution_rows);dump(output/'strata.json',strata)
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','members',*FIELDS]);writer.writeheader()
        for name,r in combined.items():writer.writerow(dict(method=name,members=len(groups[name]),**r['macro']))
    dump(output/'complete.json',dict(queries=len(data),selected_eofrv=choice,selected_thst_reference=thst_choice,
        sources={str(p):digest(p) for p in (Path(__file__),HERE/'ANALYSIS_PROTOCOL.md',PACKAGE/'historylst/metrics.py',PACKAGE/'historylst/hotspots.py')},
        outputs={p.name:digest(p) for p in output.iterdir() if p.name!='complete.json' and p.is_file()}))
    print(json.dumps({k:v['macro'] for k,v in combined.items()}),flush=True)
def read_manifest_hash(path):return json.loads(path.read_text())['manifest_sha256']
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['statistics_validation','statistics_test','all']);args=parser.parse_args()
    if args.mode=='all':all_results()
    else:statistics(args.mode.removeprefix('statistics_'))
