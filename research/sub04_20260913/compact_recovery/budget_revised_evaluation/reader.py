"""Actual completed compact6000 vs sealed baseline3000, plus baseline6000 seed905.

Training budgets are explicitly unequal. No interrupted weight is admissible.
Seven single-network predictions must seal before the shared Test labels open.
"""
from pathlib import Path
import argparse,copy,importlib.util,json,time
import torch
HERE=Path(__file__).resolve().parent;RECOVERY=HERE.parent;NEW=RECOVERY.parent
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
current=module(RECOVERY/'final_reader.py','budget_revised_actual_current_reader')
old=current.previous;p=current.p;ROOT=current.ROOT;OLD=current.OLD;PACKAGE=current.PACKAGE
read=current.read;sha=current.sha;write=current.write;require=current.require;binding=current.binding
PROTOCOL='actual_compact6000_vs_existing_baseline3000_with_one6000_sensitivity_v1'
ORDER=current.ORDER;ALL_ORDER=ORDER+[('baseline_sensitivity',20260905)]
EXTENSION=current.EXTENSION;RULE=current.RULE
NOTE='Three compact6000/26 are compared with three previously completed baseline3000/14 continuations. Historical common parent stages are retained. Additional recovery training/selection budgets are unequal. The completed baseline6000 seed905 is sensitivity only; no 3-seed matched6000 claim. Interrupted baseline912 and unrun baseline913 are excluded.'
def source_seal():
 paths=[Path(__file__),HERE/'run.py',RECOVERY/'final_reader.py',RECOVERY/'final_loader.py',RECOVERY/'final_run.py',
  NEW/'final_confirmation/reader.py',NEW/'final_confirmation/loader.py',EXTENSION,RULE,PACKAGE/'manifest.json']
 return {str(q.resolve()):sha(q) for q in paths}
def fields(value):
 require(value['protocol']==PROTOCOL and value['selected_family']=='compact' and value['ensemble_used'] is False
  and value['test_opened'] is False and value['test_authorized'] is False and value['matched_training_budget'] is False,'Explicit budget-revised single-model freeze required')
 require([(e['architecture'],e['seed']) for e in value['checkpoints']]==ALL_ORDER,'Exactly six main weights and one completed sensitivity weight required')
 for e in value['checkpoints']:
  expected='sealed_baseline3000' if e['architecture']=='baseline' else 'completed_recovery6000'
  require(e['completion_cohort']==expected,'Wrong actual completion cohort')
 v=value['validation'];gains=v['effects']['rmse']['paired_seed_differences']
 require(v['mean_metrics']['naf_history']['rmse']<.425 and len(gains)==3 and all(x>0 for x in gains)
  and v['criteria']['observed_mean_rmse_reduction_k']>=.01 and v['criteria']['network_numeric_criteria_pass'] is True,
  'Original absolute and paired numeric Val gates failed; budget status is separate')
def load_model(entry,device):
 if entry['completion_cohort']=='sealed_baseline3000':return old.loader.load_model(entry,device)
 value=dict(entry)
 if value['architecture']=='baseline_sensitivity':value['architecture']='baseline'
 return current.loader.load_model(value,device)
def recovery_entry(check,role,seed):
 d=check['completion'];meta=check['meta'];path=Path(check['path'])
 return dict(architecture=role,seed=seed,run=str(path),original_run=check['parent']['run'],checkpoint=str(path/'best.pt'),
  checkpoint_sha256=d['selected_checkpoint_sha256'],config=meta['config'],effective_method=current.METHOD,
  selected_family='compact',parameters=d['parameters'],original_checkpoint_sha256=check['parent']['checkpoint_sha256'],
  completion_cohort='completed_recovery6000',actual_updates=6000,actual_validation_candidates=26)
def freeze(args):
 require(not args.output.exists(),'New actual freeze directory required')
 selection=current.read_selection(args.selection);require(selection['selected_family']=='compact','Actual original compact family selection required')
 inherited=old.read_freeze(current.PARENT);require(sha(current.PARENT)==current.PARENT_SHA,'Original six freeze changed')
 runs={s:RECOVERY/'runs'/f'compact_{s}' for s in p.SEEDS};sensitivity=RECOVERY/'runs/baseline_20260905'
 for path in list(runs.values())+[sensitivity]:
  require((path/'complete.json').is_file() and not (path/'interrupted.json').exists(),'Only actual terminal complete weights: '+str(path))
 torch.set_num_threads(1)
 checked={s:current.checked_run(path,'compact',s) for s,path in runs.items()}
 strong=current.checked_run(sensitivity,'baseline',20260905)
 evidence=dict(inherited['source_and_evidence_sha256'],**source_seal())
 evidence[str(current.PARENT)]=sha(current.PARENT);evidence[str(args.selection.resolve())]=sha(args.selection)
 entries=[];scores={};costs=[];common=None
 for role,seed in ORDER:
  if role=='baseline':
   e=copy.deepcopy(next(e for e in inherited['checkpoints'] if (e['architecture'],e['seed'])==(role,seed)))
   e.update(completion_cohort='sealed_baseline3000',actual_updates=3000,actual_validation_candidates=14)
   done=read(Path(e['run'])/'complete.json');scores[(role,seed)]=done['selected_fp32']
   model=old.loader.load_model(e,'cpu');del model
  else:
   v=checked[seed];done=v['completion'];e=recovery_entry(v,role,seed);scores[(role,seed)]=done['selected_fp32'];evidence.update(v['evidence'])
   if common is not None:require(common==v['schedule'],'Three compact recovery schedules differ')
   common=v['schedule']
   if seed==20260905:require(binding(Path(v['path'])/'complete.json')==selection['pilots']['compact']['completion'],'Original selected compact pilot changed')
  entries.append(e);costs.append(dict(architecture=role,seed=seed,seconds=done['seconds'],updates=e['actual_updates'],validation_candidates=e['actual_validation_candidates'],parameters=done['parameters'],counters=done['counters']))
 evidence.update(strong['evidence']);extra=recovery_entry(strong,'baseline_sensitivity',20260905);entries.append(extra)
 validation=p.compare_scores(scores,'naf_history');validation.update(status='six_actual_completed_unequal_budget_main_weights',
  split='validation',matched_training_budget=False,budget_limitation=NOTE,actual_continuation_costs=costs,
  total_process_seconds=sum(x['seconds'] for x in costs),teacher_generation_cost=inherited['validation']['teacher_generation_cost'],
  total_additional_seconds_including_shared_teacher=sum(x['seconds'] for x in costs),matched_additional_budget=False,
  sensitivity=dict(architecture='baseline',seed=20260905,macro=strong['completion']['selected_fp32']['macro'],
   updates=6000,validation_candidates=26,seconds=strong['completion']['seconds'],three_seed_confirmation=False))
 value=dict(protocol=PROTOCOL,frozen_at_unix=time.time(),checkpoints=entries,validation=validation,selected_family='compact',
  selection=binding(args.selection),previous_freeze=binding(current.PARENT),teacher_receipt=inherited['teacher_receipt'],
  source_and_evidence_sha256=evidence,matched_training_budget=False,budget_limitation=NOTE,
  ensemble_used=False,test_opened=False,test_authorized=False,scientific_goal_complete=False)
 fields(value);args.output.mkdir(parents=True);write(args.output/'results.json',validation)
 write(args.output/'continuation_selection_freeze.json',value)
 print(json.dumps(dict(status='seven_actual_weights_frozen',mean_metrics=validation['mean_metrics'],matched_training_budget=False,test_authorized=False)))
def read_freeze(path):
 value=read(path);fields(value)
 require(value['source_and_evidence_sha256'].get(str(Path(__file__).resolve()))==sha(__file__),'New reader source not bound')
 for q,h in value['source_and_evidence_sha256'].items():require(sha(q)==h,'Frozen evidence changed: '+q)
 selection=current.read_selection(value['selection']['path'])
 require(selection['selected_family']=='compact' and sha(value['selection']['path'])==value['selection']['sha256'],'Original family selection changed')
 for e in value['checkpoints']:
  run=Path(e['run']);d=read(run/'complete.json')
  require(not (run/'interrupted.json').exists() and d['status']=='complete' and d['updates']==e['actual_updates']
   and d['validation_weight_candidates']==e['actual_validation_candidates'] and d['selected_checkpoint_sha256']==e['checkpoint_sha256']
   and sha(e['checkpoint'])==e['checkpoint_sha256'],'Actual completion or selected weight changed')
 return value
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selection',type=Path,default=RECOVERY/'selection.json');parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();freeze(args)
