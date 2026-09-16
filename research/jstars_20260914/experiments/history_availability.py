"""Frozen-network archive/recent history removal, followed by paired scoring."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from datetime import datetime
import importlib.util, json, sys, time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PACKAGE=ROOT/'resources/historylst246'
sys.path[:0]=[str(ROOT/'research/sub04_20260913'),str(PACKAGE),
              str(ROOT/'research/near_neighbor_attribution_20260914')]
from compact_query_product.model import QueryProductCompactHistoryNAF
from historylst.data import Dataset
from historylst.metrics import repair
from evaluate_comparisons import evaluate,mean_scores,bootstrap,digest,dump
spec=importlib.util.spec_from_file_location('availability_portable',PACKAGE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
OUT=HERE/'history_availability'
ORIGINAL=ROOT/'research/sub04_20260913/compact_query_product/final_evaluation/test'


def main():
    OUT.mkdir(exist_ok=True)
    receipt=json.loads((ORIGINAL/'predictions_complete.json').read_text())
    data=Dataset(PACKAGE,'test',labels=False)
    assert receipt['scene_ids']==[q['scene_id'] for q in data.records]
    source_hash=digest(__file__);manifest_hash=digest(PACKAGE/'manifest.json')
    for q in data.records:
        now=datetime.fromisoformat(q['query_datetime'].replace('Z','+00:00'))
        assert [h['slot_index'] for h in q['history']]==list(range(9))
        for h in q['history']:
            if h['datetime'] is None:continue
            date=datetime.fromisoformat(h['datetime'].replace('Z','+00:00'))
            assert date<now
            if h['slot_index']<6:assert date.year in (2018,2019,2020)
            else:assert 7.9 <= (now-date).total_seconds()/86400 <= 64.1
    entries=[e for e in receipt['entries'] if e['architecture']=='naf_history']
    assert len(entries)==3
    variants={'archive_only':[6,7,8],'recent_only':[0,1,2,3,4,5]}
    freeze=dict(source_sha256=source_hash,manifest_sha256=manifest_hash,
                original_receipt_sha256=digest(ORIGINAL/'predictions_complete.json'),
                variants=variants,labels_opened=False,slots_and_past_dates_verified=True,
                scene_ids=[q['scene_id'] for q in data.records],
                checkpoints=[{k:e[k] for k in ('seed','checkpoint','checkpoint_sha256')} for e in entries])
    path=OUT/'freeze.json'
    if path.exists():assert json.loads(path.read_text())==freeze
    else:dump(path,freeze)
    r.setup('cuda');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    all_receipts=[]
    for entry in entries:
        assert digest(entry['checkpoint'])==entry['checkpoint_sha256']
        model=QueryProductCompactHistoryNAF(history_dropout=0.,emissivity_dropout=0.).cuda().eval()
        model.load_state_dict(torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
        original_path=ORIGINAL/entry['prediction']
        assert digest(original_path)==entry['prediction_sha256']
        with torch.inference_mode():
            batch=r.batch(data,[0],'cuda')
            p=r.forward(model,batch).float().cpu().numpy()
        p=repair(p,data.arrays['coarse'][:1],data.arrays['support'][:1])
        observed=data.arrays['support'][:1].astype(bool)
        difference=abs(p-np.load(original_path,mmap_mode='r')[:1])
        error=float(np.max(difference[observed]))
        assert error<.0002,error
        for variant,removed in variants.items():
            target=OUT/f"{variant}_{entry['seed']}.npy";record=target.with_suffix('.json')
            if record.exists():
                saved=json.loads(record.read_text());assert saved['prediction_sha256']==digest(target)
                assert saved['freeze_sha256']==digest(path);all_receipts.append(saved);continue
            assert not target.exists()
            parts=[];began=time.perf_counter()
            with torch.inference_mode():
                for i in range(len(data)):
                    batch=r.batch(data,[i],'cuda');batch['history'][:,removed]=0
                    parts.append(r.forward(model,batch).float().cpu().numpy())
            prediction=repair(np.concatenate(parts),data.arrays['coarse'],data.arrays['support'])
            np.save(target,prediction)
            saved=dict(variant=variant,seed=entry['seed'],prediction=str(target),
                       prediction_sha256=digest(target),freeze_sha256=digest(path),
                       checkpoint_sha256=entry['checkpoint_sha256'],queries=len(data),
                       labels_opened=False,original_first_scene_maximum_difference_K=error,
                       seconds=time.perf_counter()-began)
            dump(record,saved);all_receipts.append(saved)
            print(json.dumps(saved),flush=True)
        del model;torch.cuda.empty_cache()
    dump(OUT/'predictions_complete.json',dict(entries=all_receipts,labels_opened=False,freeze_sha256=digest(path)))
    del data
    data=Dataset(PACKAGE,'test',labels=True);members={key:[] for key in ['all_history',*variants]}
    for entry in entries:members['all_history'].append(evaluate(np.load(ORIGINAL/entry['prediction']),data))
    for record in all_receipts:members[record['variant']].append(evaluate(np.load(record['prediction']),data))
    combined={key:mean_scores(value) for key,value in members.items()}
    assert abs(combined['all_history']['macro']['rmse']-.42613199570471244)<1e-9
    paired={key+'_minus_all':bootstrap(combined[key],combined['all_history']) for key in variants}
    dump(OUT/'scores.json',combined);dump(OUT/'members.json',members);dump(OUT/'paired.json',paired)
    dump(OUT/'complete.json',dict(status='complete',new_prediction_scenes=540,
         scores={key:value['macro'] for key,value in combined.items()},
         pairs={key:{k:v for k,v in value.items() if k!='city_deltas'} for key,value in paired.items()},
         source_unchanged=digest(__file__)==source_hash,manifest_unchanged=digest(PACKAGE/'manifest.json')==manifest_hash))
    print((OUT/'complete.json').read_text(),flush=True)

if __name__=='__main__':main()
