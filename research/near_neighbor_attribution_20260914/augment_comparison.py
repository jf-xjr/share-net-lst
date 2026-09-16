"""Add the Val-selected task-loss THST check to the complete comparison."""
from pathlib import Path
import csv,json,shutil,sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import evaluate_comparisons as e
def read(p):return json.loads(p.read_text())
def main():
    base=HERE/'analysis';out=HERE/'analysis_augmented';out.mkdir(exist_ok=True)
    assert (base/'complete.json').exists()
    extra=HERE/'thst_task_loss';selection=read(extra/'selection.json');receipt=read(extra/'predictions_complete.json')
    assert receipt['selection_sha256']==e.digest(extra/'selection.json')
    data=e.Dataset(e.PACKAGE,'test',labels=True)
    assert receipt['scene_order']==[q['scene_id'] for q in data.records]
    scores=read(base/'scores.json');paired=read(base/'paired.json');strata=read(base/'strata.json')
    arrays={}
    for row in receipt['entries']:
        assert e.digest(row['path'])==row['sha256']
        name='THST_task_loss_'+row['reference'];p=np.load(row['path']);arrays[name]=[p]
        scores[name]=e.mean_scores([e.evaluate(p,data)])
        paired[name+'_minus_final']=e.bootstrap(scores[name],scores['final_0.426'])
    extra_strata=e.stratified(arrays,data)
    for k,row in extra_strata.items():
        assert row['scenes']==strata[k]['scenes'] and row['pixels']==strata[k]['pixels']
        strata[k]['methods'].update(row['methods'])
    scores['THST_original_selected']=scores['THST_selected']
    chosen='THST_task_loss_'+selection['reference'] if selection['final_family']=='task_loss' else 'THST_original_selected'
    scores['THST_selected']=scores[chosen]
    paired['THST_selected_minus_final']=e.bootstrap(scores['THST_selected'],scores['final_0.426'])
    for row in strata.values():
        if row['scenes'] and selection['final_family']=='task_loss':row['methods']['THST_selected']=row['methods'][chosen]
    e.dump(out/'scores.json',scores);e.dump(out/'paired.json',paired);e.dump(out/'strata.json',strata)
    for name in ('attribution_members.json','inference_benchmark.json'):
        shutil.copy2(base/name,out/('original_'+name if name=='inference_benchmark.json' else name))
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method','members',*e.FIELDS]);w.writeheader()
        for name,row in scores.items():w.writerow(dict(method=name,members=len(row['members']),**row['macro']))
    e.dump(out/'complete.json',dict(queries=len(data),thst_selected=chosen,selection=selection,
        sources={str(p):e.digest(p) for p in (Path(__file__),HERE/'THST_TASK_LOSS_PROTOCOL.md',base/'complete.json',extra/'predictions_complete.json')},
        primary_selection='Val45 only; original and task-loss adaptation retained'))
    import plot_results
    plot_results.OUT=out;plot_results.main()
    from benchmark_inference import main as benchmark
    benchmark(checkpoints_override={'THST_task_loss_all':extra/'best.pt','THST_task_loss_matched':extra/'best.pt'},
              output_override=out/'task_loss_inference_benchmark.json')
    print(json.dumps(dict(thst_selected=chosen,rmse=scores[chosen]['macro']['rmse'])),flush=True)
if __name__=='__main__':main()
