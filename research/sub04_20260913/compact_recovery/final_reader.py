"""Actual two-pilot selection and six completed 6000/26 runs; no inference/Test."""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,math,os,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;NEW=HERE.parent;ROOT=HERE.parents[2];OLD=ROOT/'research/sub04_20260911';PACKAGE=ROOT/'resources/historylst246'
sys.dont_write_bytecode=True
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
previous=module(NEW/'final_confirmation/reader.py','recovery_previous_common_metric_reader')
loader=module(HERE/'final_loader.py','recovery_strict_frozen_loader')
p=previous.p;read=previous.read;sha=previous.sha;require=previous.require;binding=previous.binding;write=previous.write
PROTOCOL='six_actual_tiered_annealed_recovery_weights_v1';METHOD=loader.METHOD
EXTENSION=NEW/'four_hour_extension.json';RULE=NEW/'network_acceptance_20260913_revision.json';DESIGN=HERE/'design.json'
PARENT=NEW/'final_confirmation/frozen/continuation_selection_freeze.json';PARENT_SHA='a84894d0e02e55e7dfdc36ec33b3098bedd67837b4a3ba0a97257b2df3cbbade'
TEACHER=NEW/'strong_teacher/fit/predictions_complete.json';TEACHER_SHA='48bfd6b2bdbf672cf6bdc10cb08a77a9811fc3be76b66dc29b584a5bc8857d58'
ORDER=[(role,seed) for seed in p.SEEDS for role in ('baseline','naf_history')]
def source_seal():
 return {str(q.resolve()):sha(q) for q in (Path(__file__),HERE/'final_run.py',HERE/'final_loader.py',HERE/'train.py',HERE/'model.py',DESIGN,EXTENSION,RULE,PACKAGE/'manifest.json',Path(p.__file__),NEW/'final_confirmation/reader.py')}
def tier(parameters):
 rule=read(RULE)
 require(rule['tiers']==[dict(parameters_lt=6000000,three_seed_macro_rmse_lt_k=.425),dict(parameters_lt=10000000,three_seed_macro_rmse_lt_k=.42)],'Exact user parameter-tier revision required')
 for row in rule['tiers']:
  if parameters<row['parameters_lt']:return row['three_seed_macro_rmse_lt_k']
 raise ValueError('No qualifying user parameter tier')
def checked_run(path,architecture,seed):
 path=Path(path).resolve();require((path/'complete.json').is_file() and not (path/'interrupted.json').exists(),'Missing or interrupted actual run: '+str(path))
 require(sha(PARENT)==PARENT_SHA and sha(TEACHER)==TEACHER_SHA,'Original parent or fixed teacher receipt changed')
 meta,done,rows=[read(path/n) for n in ('run.json','complete.json','validation.json')]
 cfg=dict(read(DESIGN)['recipe'],architecture=architecture,original_seed=seed,protocol=read(DESIGN)['protocol'],teacher_used_at_inference=False,inference_views=1,
  loader_class={'compact':'compact_recovery.model.CompactFourHeadHistoryNAF','full':'multihead_fusion.model.FourHeadHistoryNAF','baseline':'dropout_models.DropoutUTAE'}[architecture])
 require(meta['config']==done['config']==cfg and done['status']=='complete' and done['architecture']==architecture and done['original_seed']==seed,'Actual role/seed or fixed recipe differs')
 role='baseline' if architecture=='baseline' else 'naf_history';parent,=[e for e in read(PARENT)['checkpoints'] if (e['architecture'],e['seed'])==(role,seed)]
 require(meta['initialization']==parent['checkpoint'] and meta['initialization_sha256']==parent['checkpoint_sha256'] and sha(parent['checkpoint'])==parent['checkpoint_sha256'],'Same corresponding completed parent required')
 require(meta['manifest_sha256']==sha(PACKAGE/'manifest.json') and meta['teacher_receipt_sha256']==done['teacher_receipt_sha256']==TEACHER_SHA and meta['teacher']==read(TEACHER)['teacher'],'Common manifest or teacher differs')
 require(meta['validation_candidates']==26 and meta['validation_precision']=='FP32 TF32 off; exact original FP64 repair' and meta['single_network_forward'] is True and meta['test_opened'] is False and meta['teacher_used_at_inference'] is False,'Actual complete single-view FP32 validation protocol required')
 require(done['updates']==6000 and done['validation_weight_candidates']==26 and done['single_network_forward'] is True and done['test_opened'] is False and done['teacher_used_at_inference'] is False,'Actual6000/26 terminal completion required')
 require(done['run_sha256']==sha(path/'run.json') and done['source_sha256']==meta['source_sha256'],'Completion/runtime source binding differs')
 for q,h in meta['source_sha256'].items():require(sha(q)==h,'Bound training source changed: '+q)
 require(meta['source_sha256'].get(str(HERE/'train.py'))==sha(HERE/'train.py') and meta['source_sha256'].get(str(HERE/'model.py'))==sha(HERE/'model.py'),'Exact shared training/model code missing')
 require(0<done['seconds']<=2700 and done['actual_start_unix']==meta['actual_start_unix'] and abs(done['actual_end_unix']-done['actual_start_unix']-done['seconds'])<.1 and done['actual_end_unix']<=meta['deadline_unix']<=read(EXTENSION)['stop_training_search_by_unix'],'Actual bounded execution span differs')
 require([(x['step'],x['weights']) for x in rows]==[(n,w) for n in range(0,6001,500) for w in ('raw','ema')] and all(math.isfinite(x['rmse']) for x in rows),'Exact26 complete FP32 candidates required')
 selected=min(rows,key=lambda x:x['rmse']);best=torch.load(path/'best.pt',map_location='cpu',weights_only=False);last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
 require(best['config']==last['config']==cfg and last['step']==6000 and last['validation']==rows,'Actual selected/last state metadata differs')
 require((best['step'],best['weights'],best['validation_rmse'])==(selected['step'],selected['weights'],selected['rmse']) and (done['selected_step'],done['selected_weights'])==(best['step'],best['weights']) and sha(path/'best.pt')==done['selected_checkpoint_sha256'],'Earliest complete FP32 minimum was not retained')
 require(best['source_sha256']==meta['source_sha256'] and best['initialization']==meta['initialization'],'Selected state source binding differs')
 model=loader.trainer.construct(architecture)
 for state in (best['state_dict'],last['state_dict'],last['ema']):loader.validate_state(model,state)
 count=sum(v.numel() for v in model.parameters());require(count==meta['parameters']==done['parameters']==loader.trainer.PARAMETERS[architecture],'Actual parameter count differs')
 initial=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
 if architecture=='compact':
  checked=loader.trainer.load_parent_state(model,initial['state_dict']);require(meta['initialization_mapping']==checked,'Fixed parent key extraction receipt differs')
 else:require(meta['initialization_mapping']==dict(mapping='strict_identity',parent_keys=len(initial['state_dict'])),'Original full/baseline mapping differs')
 counters=done['counters'];require(counters['updates']==counters['finite_updates']==6000 and counters['forward_calls']==counters['backward_calls']==6000+counters['amp_backoffs'] and counters['validation_forward_calls']==27*23,'Actual successful work or complete27 FP32 passes differs')
 require(last['counters']==dict(counters,validation_forward_calls=26*23),'Terminal versus selected verification counters differ')
 for name in initial['state_dict']:
  if name.endswith('num_batches_tracked'):
   before=int(initial['state_dict'][name]);require(int(last['state_dict'][name])==int(last['ema'][name])==before+6000 and int(best['state_dict'][name])==before+best['step'],'BN successful-update count differs')
 with np.load(path/'schedule.npz',allow_pickle=False) as schedule:
  require(set(schedule.files)=={'fit_ids','d4'},'Unexpected saved schedule keys');content={}
  for name,shape,limit in [('fit_ids',(6000,4),603),('d4',(6000,),8)]:
   a=schedule[name];require(a.shape==shape and np.issubdtype(a.dtype,np.integer) and a.min()>=0 and a.max()<limit,'Invalid complete sampling/D4 schedule');content[name]=dict(shape=list(shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
 require(sha(path/'schedule.npz')==meta['schedule_sha256'],'Schedule artifact changed')
 records=read(PACKAGE/'manifest.json')['roles']['validation']['scenes'];previous.prior.validate_scores(done['selected_fp32'],records)
 require(abs(done['selected_fp32']['macro']['rmse']-selected['rmse'])<1e-5,'Fresh final selected FP32 score mismatch')
 evidence=dict(meta['source_sha256']);evidence.update({str(path/n):sha(path/n) for n in ('run.json','complete.json','validation.json','best.pt','last.pt','schedule.npz')});evidence.update({str(Path(parent['checkpoint'])):parent['checkpoint_sha256'],str(Path(parent['run'])/'complete.json'):sha(Path(parent['run'])/'complete.json')})
 del model,best,last,initial;gc.collect()
 return dict(path=str(path),architecture=architecture,seed=seed,meta=meta,completion=done,parent=parent,evidence=evidence,schedule=content)
def selection_value(runs_root):
 checked={family:checked_run(Path(runs_root)/(family+'_20260905'),family,20260905) for family in ('compact','full')}
 values={k:v['completion']['selected_fp32']['macro']['rmse']-tier(v['completion']['parameters']) for k,v in checked.items()}
 family=min(checked,key=lambda k:(values[k],checked[k]['completion']['parameters']))
 old=read(Path(checked['full']['parent']['run'])/'complete.json')['selected_fp32']['macro']['rmse']-.42
 return family,values,old,checked
def select(args):
 require(not args.output.exists(),'Unique actual two-pilot selection only');family,values,old,checked=selection_value(args.runs_root)
 result=dict(status='two_actual_6000_26_pilots_selected_by_fixed_Val_minus_tier',protocol=PROTOCOL,selected_family=family,adjusted_val_scores=values,existing_full905_adjusted_val=old,
  development_gate_pass=values[family]<old,created_at_unix=time.time(),runs_root=str(args.runs_root.resolve()),source_sha256=source_seal(),pilots={k:dict(completion=binding(Path(v['path'])/'complete.json'),checkpoint=binding(Path(v['path'])/'best.pt')) for k,v in checked.items()},Test_opened=False)
 write(args.output,result);print(json.dumps(dict(selected_family=family,adjusted_val_scores=values,development_gate_pass=result['development_gate_pass'])))
def read_selection(path):
 s=read(path);require(s['status']=='two_actual_6000_26_pilots_selected_by_fixed_Val_minus_tier' and s['protocol']==PROTOCOL and s['development_gate_pass'] is True and s['Test_opened'] is False,'Actual two-complete-pilot development gate required')
 for q,h in s['source_sha256'].items():require(sha(q)==h,'Selection source changed')
 for v in s['pilots'].values():
  for b in v.values():require(sha(b['path'])==b['sha256'],'Actual pilot changed')
 scores={}
 for family,item in s['pilots'].items():
  d=read(item['completion']['path']);require(d['status']=='complete' and d['updates']==6000 and d['validation_weight_candidates']==26,'Missing actual pilot completion');scores[family]=d['selected_fp32']['macro']['rmse']-tier(d['parameters'])
 require(scores==s['adjusted_val_scores'] and s['selected_family']==min(scores,key=lambda k:(scores[k],loader.trainer.PARAMETERS[k])) and scores[s['selected_family']]<s['existing_full905_adjusted_val'],'Predeclared two-family selection changed')
 return s
def fields(value):
 require(value['protocol']==PROTOCOL and value['ensemble_used'] is False and value['test_opened'] is False and value['test_authorized'] is False,'Actual single-view six freeze required')
 require([(e['architecture'],e['seed']) for e in value['checkpoints']]==ORDER and all(e['effective_method']==METHOD and e['selected_family']==value['selected_family'] for e in value['checkpoints']),'Actual matching six identities/family required')
 v=value['validation'];threshold=tier(loader.trainer.PARAMETERS[value['selected_family']]);effects=v['effects']['rmse']['paired_seed_differences']
 require(v['status']=='six_actual_6000_26_recovery_runs_complete' and v['mean_metrics']['naf_history']['rmse']<threshold and v['criteria']['observed_mean_rmse_reduction_k']>=.01 and len(effects)==3 and all(x>0 for x in effects) and v['criteria']['network_numeric_criteria_pass'] is True,'Tiered absolute and matched three-seed Val gates failed')
def freeze(args):
 require(not args.output.exists(),'Unique new freeze directory only');selection=read_selection(args.selection);family=selection['selected_family'];runs={key:Path(args.runs_root)/(('baseline' if key[0]=='baseline' else family)+'_'+str(key[1])) for key in ORDER}
 require(all((q/'complete.json').is_file() for q in runs.values()),'All six actual completions required before freeze')
 torch.set_num_threads(1);evidence=source_seal();evidence[str(args.selection.resolve())]=sha(args.selection);teacher=read(TEACHER);require(sha(TEACHER)==TEACHER_SHA,'Fixed teacher changed');evidence[str(TEACHER)]=TEACHER_SHA
 checked={key:checked_run(path,'baseline' if key[0]=='baseline' else family,key[1]) for key,path in runs.items()}
 common=None;scores={};entries=[];costs=[]
 for key,v in checked.items():
  if common is not None:require(common==v['schedule'],'Six actual schedules differ')
  common=v['schedule'];d=v['completion'];meta=v['meta'];path=Path(v['path']);evidence.update(v['evidence']);scores[key]=d['selected_fp32']
  if key==('naf_history',20260905):require(binding(path/'complete.json')==selection['pilots'][family]['completion'],'Selected original pilot must be retained')
  else:require(d['actual_start_unix']>=selection['created_at_unix'],'Matched confirmation must follow fixed actual pilot selection')
  entries.append(dict(architecture=key[0],seed=key[1],run=str(path),original_run=v['parent']['run'],checkpoint=str(path/'best.pt'),checkpoint_sha256=d['selected_checkpoint_sha256'],config=meta['config'],effective_method=METHOD,selected_family=family,retained_original_pilot=key==('naf_history',20260905),parameters=d['parameters'],original_checkpoint_sha256=v['parent']['checkpoint_sha256']))
  costs.append(dict(architecture=key[0],seed=key[1],seconds=d['seconds'],parameters=d['parameters'],counters=d['counters'],actual_start_unix=d['actual_start_unix'],actual_end_unix=d['actual_end_unix']))
 ordered=sorted(costs,key=lambda c:c['actual_start_unix']);require(all(a['actual_end_unix']<=b['actual_start_unix'] for a,b in zip(ordered,ordered[1:])),'Actual training jobs must be sequential/isolated')
 result=p.compare_scores(scores,'naf_history');threshold=tier(loader.trainer.PARAMETERS[family]);result.update(status='six_actual_6000_26_recovery_runs_complete',split='validation',selected_family=family,parameter_count=loader.trainer.PARAMETERS[family],user_absolute_threshold_k=threshold,
  tier_absolute_pass=result['mean_metrics']['naf_history']['rmse']<threshold,actual_continuation_costs=costs,total_process_seconds=sum(c['seconds'] for c in costs),teacher_generation_cost=dict(seconds=teacher['seconds'],counters=teacher['counters'],shared_once=True,reused_from_prior_stage=True),
  total_additional_seconds_including_shared_teacher=sum(c['seconds'] for c in costs),matched_additional_budget='Each actual role/seed:6000 updates,26 FP32 choices,same teacher and sample/D4 schedule',historical_search_costs_equal=False,test_opened=False,scientific_goal_complete=False)
 value=dict(protocol=PROTOCOL,frozen_at_unix=time.time(),checkpoints=entries,validation=result,selected_family=family,selection=binding(args.selection),source_and_evidence_sha256=evidence,common_schedule=common,teacher_receipt=binding(TEACHER),teacher=teacher,previous_freeze=binding(PARENT),ensemble_used=False,test_opened=False,test_authorized=False,scientific_goal_complete=False)
 args.output.mkdir(parents=True);write(args.output/'results.json',result);fields(value);write(args.output/'continuation_selection_freeze.json',value);print(json.dumps(dict(status='six_actual_recovery_runs_frozen',selected_family=family,threshold_k=threshold,test_authorized=False)))
def read_freeze(path):
 v=read(path);fields(v);selection=read_selection(v['selection']['path']);require(sha(v['selection']['path'])==v['selection']['sha256'] and selection['selected_family']==v['selected_family'],'Frozen family/selection changed')
 require(v['source_and_evidence_sha256'].get(str(Path(__file__).resolve()))==sha(__file__),'Reader source not bound')
 for q,h in v['source_and_evidence_sha256'].items():require(sha(q)==h,'Frozen evidence changed: '+q)
 for e in v['checkpoints']:
  d=read(Path(e['run'])/'complete.json');require(d['status']=='complete' and d['updates']==6000 and d['validation_weight_candidates']==26 and d['selected_checkpoint_sha256']==e['checkpoint_sha256'] and sha(e['checkpoint'])==e['checkpoint_sha256'],'Actual frozen completion changed')
 return v
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('select','freeze','check'));parser.add_argument('--runs-root',type=Path,default=HERE/'runs');parser.add_argument('--output',type=Path);parser.add_argument('--selection',type=Path);args=parser.parse_args()
 if args.command=='check':
  require(tier(5911525)==.425 and tier(9310501)==.42,'Strict user tiers');print(json.dumps(dict(status='CPU_import_tiers_pass_no_observations_or_freeze',GPU_used=False)));return
 require(args.output is not None,'Explicit new output required');require(time.time()<read(EXTENSION)['hard_deadline_unix'],'Four-hour deadline passed')
 if args.command=='freeze':require(args.selection is not None,'Actual two-pilot selection required');freeze(args)
 else:select(args)
if __name__=='__main__':main()
