"""Actual three QP1500 confirmations versus three sealed baseline3000 weights."""
from pathlib import Path
import argparse,copy,gc,hashlib,importlib.util,math,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;QP=HERE.parent;NEW=QP.parent;RECOVERY=NEW/'compact_recovery'
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
current=module(RECOVERY/'final_reader.py','QP_final_original_macro_reader')
old=current.previous;p=current.p;ROOT=current.ROOT;OLD=current.OLD;PACKAGE=current.PACKAGE
read=current.read;sha=current.sha;write=current.write;require=current.require;binding=current.binding
sys.path.insert(0,str(NEW))
from compact_query_product.model import QueryProductCompactHistoryNAF,load_compact_parent_state
PROTOCOL='actual_three_QP1500_vs_sealed_baseline3000_unequal_budget_v1'
METHOD='compact_query_product_single_forward';PARAMETERS=5977045;ORDER=current.ORDER
EXTENSION=current.EXTENSION;RULE=current.RULE;ALLOCATION=QP/'confirmation_allocation.json'
PILOT=QP/'runs/compact_20260905/complete.json';PILOT_SHA='71353e1ac0de99603837c2cc34f152919c096f01407ec5648b01ad8f215fd886'
LEGACY_RECEIPT=RECOVERY/'budget_revised_evaluation/test/predictions_complete.json'
NOTE='Three QP1500/8 updates follow corresponding completed compact6000/26 parents; the three baseline3000/14 weights are reused. Additional training/selection budgets are unequal. The original QP905 .001 gain screen failed and remains false; two confirmations have separate root budget authorization. Consumed Test30 is not independent validation. No ensemble or Test-driven weight selection.'
def source_seal():
 paths=[Path(__file__),HERE/'run.py',QP/'model.py',QP/'train.py',QP/'design.json',QP/'CPU_checks.json',QP/'metadata_erratum.json',QP/'completion_metadata_supplement.json',ALLOCATION,NEW/'final_hour_reallocation.json',
  RECOVERY/'final_reader.py',RECOVERY/'final_loader.py',NEW/'final_confirmation/reader.py',NEW/'final_confirmation/loader.py',EXTENSION,RULE,PACKAGE/'manifest.json',
  OLD/'d4_self_distillation_20260912_v1/evaluate_frozen_v2.py',OLD/'benchmark_final_frozen_20260912.py',LEGACY_RECEIPT,
  QP/'continue_confirm.py',QP/'continue_tail.py',QP/'final_evaluation_reserve_allocation.json']
 return {str(q.resolve()):sha(q) for q in paths}
def construct():return QueryProductCompactHistoryNAF(history_dropout=0.,emissivity_dropout=0.).float()
def validate_state(model,state):
 expected=model.state_dict();require(set(state)==set(expected),'Complete QP state keys required')
 for k,v in state.items():require(isinstance(v,torch.Tensor) and v.shape==expected[k].shape and v.dtype==expected[k].dtype and bool(torch.isfinite(v).all()),'QP tensor schema/finite mismatch '+k)
 model.load_state_dict(state,strict=True)
 require(sum(v.numel() for v in model.parameters())==PARAMETERS,'Actual QP parameter count differs')
def load_model(entry,device):
 if entry['completion_cohort']=='sealed_baseline3000':return old.loader.load_model(entry,device)
 require(entry['effective_method']==METHOD and entry['parameters']==PARAMETERS and sha(entry['checkpoint'])==entry['checkpoint_sha256'],'Exact QP frozen factory/weight required')
 saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False);require(saved['config']==entry['config'],'QP selected config differs')
 model=construct();validate_state(model,saved['state_dict']);return model.to(device).eval()
def check_erratum():
 e=read(QP/'metadata_erratum.json');s=read(QP/'completion_metadata_supplement.json')
 require(sha(PILOT)==PILOT_SHA and read(PILOT)['continuation_gate_pass'] is False,'Actual failed original905 pilot changed')
 require(e['field']=='deployment.architecture_changed' and e['recorded_value'] is False and e['correct_value'] is True and e['actual_parameters']==PARAMETERS,'Architecture metadata erratum required')
 require(s['completion']==binding(PILOT) and s['metadata_erratum']==binding(QP/'metadata_erratum.json') and s['actual_architecture_changed'] is True and s['continuation_gate_pass'] is False,'Actual completion supplement differs')
 for k in ('design','model','model_CPU_receipt','trainer'):require(sha(e[k]['path'])==e[k]['sha256'],'Erratum source changed')
def checked_run(seed):
 path=QP/'runs'/f'compact_{seed}';require((path/'complete.json').is_file() and not (path/'interrupted.json').exists(),'Actual completed QP run required: '+str(seed))
 meta,done,rows=[read(path/n) for n in ('run.json','complete.json','validation.json')];design=read(QP/'design.json')
 cfg=dict(design['recipe'],protocol=design['protocol'],architecture='compact',original_seed=seed,teacher_used_at_inference=False,inference_views=1,loader_class='compact_query_product.model.QueryProductCompactHistoryNAF',parameter_groups=design['parameter_groups'])
 require(meta['config']==done['config']==cfg and done['status']=='complete' and done['architecture']=='compact' and done['original_seed']==seed,'Actual unchanged QP numerical recipe/role required')
 require(done['updates']==1500 and done['validation_weight_candidates']==8 and done['parameters']==meta['parameters']==PARAMETERS,'Actual QP1500/8/5977045 required')
 require(done['run_sha256']==sha(path/'run.json') and done['source_sha256']==meta['source_sha256'],'QP run/completion source differs')
 for q,h in meta['source_sha256'].items():require(sha(q)==h,'Bound QP source changed: '+q)
 require(meta['source_sha256'].get(str(QP/'train.py'))=='d0c6fc43803cc9fce23c708dbb1d29970c993d7ba2f07884bd341053fd8ec7b4' and meta['source_sha256'].get(str(QP/'model.py'))=='4ec15b7d27c8dedbfe59f336bb1387034cd20645b8e7266eda7f0e632e633648','Original exact QP implementation required')
 parent=current.checked_run(RECOVERY/'runs'/f'compact_{seed}','compact',seed);parent_path=Path(parent['path'])
 require(meta['initialization']==str(parent_path/'best.pt') and meta['initialization_sha256']==parent['completion']['selected_checkpoint_sha256'] and meta['parent_complete']==done['parent_complete']==binding(parent_path/'complete.json'),'QP corresponding6000 parent differs')
 require(meta['preceding_updates']==done['preceding_updates']==6000 and meta['preceding_validation_weight_candidates']==done['preceding_validation_weight_candidates']==26,'QP true preceding budget differs')
 require(meta['teacher_receipt']==binding(current.TEACHER) and done['teacher_receipt_sha256']==current.TEACHER_SHA and meta['teacher']==read(current.TEACHER)['teacher'],'Unchanged24 teacher required')
 require(meta['manifest_sha256']==sha(PACKAGE/'manifest.json') and meta['validation_candidates']==8 and meta['validation_precision']=='FP32 TF32 off; exact original FP64 repair' and meta['training_precision']=='FP32 TF32off autocast disabled','Original complete FP32 input/precision protocol')
 for item in (meta,done):require(item['single_network_forward'] is True and item['test_opened'] is False and item['teacher_used_at_inference'] is False,'Single forward/no new Test training required')
 require([(x['step'],x['weights']) for x in rows]==[(n,w) for n in (0,500,1000,1500) for w in ('raw','ema')] and all(math.isfinite(x['rmse']) for x in rows),'Exact eight complete Val candidates required')
 selected=min(rows,key=lambda x:x['rmse']);best=torch.load(path/'best.pt',map_location='cpu',weights_only=False);last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
 require(best['config']==last['config']==cfg and last['step']==1500 and last['validation']==rows,'QP complete last state differs')
 require((best['step'],best['weights'],best['validation_rmse'])==(selected['step'],selected['weights'],selected['rmse']) and (done['selected_step'],done['selected_weights'])==(best['step'],best['weights']) and sha(path/'best.pt')==done['selected_checkpoint_sha256'],'QP earliest selected complete candidate differs')
 require(best['source_sha256']==meta['source_sha256'] and best['initialization']==meta['initialization'],'QP selected provenance differs')
 model=construct();initial=torch.load(parent_path/'best.pt',map_location='cpu',weights_only=False)
 require(load_compact_parent_state(model,initial['state_dict'])==meta['initialization_mapping'],'QP strict original-key initialization mapping differs')
 for state in (best['state_dict'],last['state_dict'],last['ema']):validate_state(model,state)
 counters=done['counters'];require(counters['updates']==counters['finite_updates']==1500 and counters['forward_calls']==counters['backward_calls']==3000 and counters['amp_backoffs']==0 and counters['validation_forward_calls']==9*23,'Actual QP finite work/selected repeat counts differ')
 require(last['counters']==dict(counters,validation_forward_calls=8*23),'QP last-vs-selected counters differ')
 with np.load(path/'schedule.npz',allow_pickle=False) as saved,np.load(parent_path/'schedule.npz',allow_pickle=False) as original:
  require(set(saved.files)=={'fit_ids','d4'},'Only original sampling and D4')
  schedule={}
  for k,shape in (('fit_ids',(1500,4)),('d4',(1500,))):
   a=saved[k];require(a.shape==shape and np.array_equal(a,original[k][:1500]),'QP exact corresponding1500 sampling prefix differs');schedule[k]=hashlib.sha256(a.tobytes()).hexdigest()
 require(sha(path/'schedule.npz')==meta['schedule_sha256'] and schedule==meta['schedule_content_sha256'],'QP schedule hashes differ')
 require(0<done['seconds']<=700 and done['actual_start_unix']==meta['actual_start_unix'] and abs(done['actual_end_unix']-done['actual_start_unix']-done['seconds'])<.1 and done['actual_end_unix']<=meta['deadline_unix']<=done['actual_start_unix']+700,'QP actual finite execution span differs')
 if seed==20260905:
  require(sha(path/'complete.json')==PILOT_SHA and meta['deadline_unix']<=read(EXTENSION)['stop_training_search_by_unix'],'Retained exact original905 pilot required')
 else:
  a=read(ALLOCATION)
  for item in (meta,done):
   require(item['confirmation_allocation']==binding(ALLOCATION) and item['confirmation_runner']==binding(QP/'continue_confirm.py') and item['original_pilot_gate_pass'] is False and item['matched_additional_training_budget'] is False and item['numerical_recipe_unchanged'] is True and item['metadata_erratum']==binding(QP/'metadata_erratum.json'),'Actual explicit confirmation authority required')
  require(done['actual_start_unix']>=a['created_unix'] and a['original_pilot_completion']==binding(PILOT) and a['original_pilot_gate_pass'] is False and a['budget_added_seconds']==0,'Confirmation preserves original failed screen and existing budget')
  if seed==20260912:require(meta['deadline_unix']<=a['training_stop_unix'],'912 unchanged authorized cutoff required')
  else:
   # A later separately sealed tail authority is admissible, never an edit to912.
   if 'tail_allocation' in done:
    t=read(done['tail_allocation']['path']);require(binding(done['tail_allocation']['path'])==done['tail_allocation']==meta['tail_allocation'] and done['tail_runner']==meta['tail_runner']==binding(QP/'continue_tail.py'),'Actual tail authority binding required')
    require(t['status']=='explicit_root_reallocation_of_final_evaluation_reserve' and t['previous_confirmation_allocation']==binding(ALLOCATION) and t['budget_added_seconds']==0 and t['hard_deadline_unix']==read(EXTENSION)['hard_deadline_unix'] and t['maximum_job_seconds']==700 and t['original_pilot_gate_pass'] is False and t['final_numeric_gates_unchanged'] is True and t['no_Test_weight_or_method_selection'] is True,'Actual tail budget only, unchanged hard deadline and final gates')
    require(meta['deadline_unix']<=t['effective_training_stop_unix']==1789301796 and done['actual_start_unix']>=t['created_unix'],'Actual913 tail execution span differs')
    for item in (meta,done):require(item['effective_training_stop_unix']==t['effective_training_stop_unix'] and item['final_hard_deadline_unix']==t['hard_deadline_unix'] and item['user_budget_added_seconds']==0 and item['preceding_confirmation']==binding(QP/'runs/compact_20260912/complete.json'),'Actual tail predecessor/deadline differs')
   else:require(meta['deadline_unix']<=a['training_stop_unix'],'913 without tail amendment retains original cutoff')
 require(done['actual_end_unix']<=read(EXTENSION)['hard_deadline_unix'],'Hard budget exceeded')
 records=read(PACKAGE/'manifest.json')['roles']['validation']['scenes'];old.prior.validate_scores(done['selected_fp32'],records)
 require(abs(done['selected_fp32']['macro']['rmse']-selected['rmse'])<1e-5,'QP actual selected FP32 repeat differs')
 evidence=dict(parent['evidence'],**meta['source_sha256']);evidence.update({str(path/n):sha(path/n) for n in ('run.json','complete.json','validation.json','best.pt','last.pt','schedule.npz')})
 del model,initial,best,last;gc.collect()
 return dict(path=str(path),meta=meta,completion=done,parent=parent,evidence=evidence,schedule=schedule)
def fields(v):
 require(v['protocol']==PROTOCOL and v['selected_family']=='compact_query_product' and v['ensemble_used'] is False and v['test_opened'] is False and v['test_authorized'] is False and v['matched_training_budget'] is False,'Explicit actual unequal-budget QP freeze required')
 require([(e['architecture'],e['seed']) for e in v['checkpoints']]==ORDER,'Exactly three QP and three baseline identities required')
 for e in v['checkpoints']:
  expected=('sealed_baseline3000',3000,14) if e['architecture']=='baseline' else ('completed_QP1500',1500,8)
  require((e['completion_cohort'],e['actual_updates'],e['actual_validation_candidates'])==expected,'Actual completed cohort mismatch')
  if e['architecture']=='naf_history':require(e['parameters']==PARAMETERS and e['effective_method']==METHOD,'Strict new QP factory required')
 val=v['validation'];g=val['effects']['rmse']['paired_seed_differences']
 require(val['mean_metrics']['naf_history']['rmse']<current.tier(PARAMETERS) and len(g)==3 and all(x>0 for x in g) and val['criteria']['observed_mean_rmse_reduction_k']>=.01 and val['criteria']['network_numeric_criteria_pass'] is True,'Original three-seed Val absolute/paired gates failed')
def freeze(args):
 require(not args.output.exists(),'Unique new freeze only');require(time.time()<read(EXTENSION)['hard_deadline_unix'],'Hard deadline reached');check_erratum()
 require(all((QP/'runs'/f'compact_{s}'/'complete.json').is_file() for s in p.SEEDS),'All three actual QP completions required; no placeholder weights')
 torch.set_num_threads(1);inherited=old.read_freeze(current.PARENT);require(sha(current.PARENT)==current.PARENT_SHA,'Original baseline freeze changed')
 checked={s:checked_run(s) for s in p.SEEDS};evidence=dict(inherited['source_and_evidence_sha256'],**source_seal());evidence[str(current.PARENT)]=current.PARENT_SHA
 entries=[];scores={};costs=[];prior_costs=[];common=None
 for role,seed in ORDER:
  if role=='baseline':
   e=copy.deepcopy(next(e for e in inherited['checkpoints'] if (e['architecture'],e['seed'])==(role,seed)));e.update(completion_cohort='sealed_baseline3000',actual_updates=3000,actual_validation_candidates=14)
   done=read(Path(e['run'])/'complete.json');model=old.loader.load_model(e,'cpu');del model
  else:
   c=checked[seed];done=c['completion'];meta=c['meta'];path=Path(c['path']);evidence.update(c['evidence'])
   if common is not None:require(common==c['schedule'],'Same QP1500 original schedule required')
   common=c['schedule'];e=dict(architecture=role,seed=seed,run=str(path),original_run=c['parent']['path'],checkpoint=str(path/'best.pt'),checkpoint_sha256=done['selected_checkpoint_sha256'],config=meta['config'],effective_method=METHOD,selected_family='compact_query_product',parameters=PARAMETERS,original_checkpoint_sha256=meta['initialization_sha256'],completion_cohort='completed_QP1500',actual_updates=1500,actual_validation_candidates=8)
   prior_costs.append(dict(seed=seed,updates=6000,validation_candidates=26,seconds=c['parent']['completion']['seconds'],completion=binding(Path(c['parent']['path'])/'complete.json')))
  entries.append(e);scores[(role,seed)]=done['selected_fp32'];costs.append(dict(architecture=role,seed=seed,seconds=done['seconds'],updates=e['actual_updates'],validation_candidates=e['actual_validation_candidates'],parameters=done['parameters'],counters=done['counters']))
 teacher_cost=inherited['validation']['teacher_generation_cost']
 stage_seconds=dict(QP1500=sum(x['seconds'] for x in costs if x['architecture']=='naf_history'),preceding_compact6000=sum(x['seconds'] for x in prior_costs),reused_baseline3000=sum(x['seconds'] for x in costs if x['architecture']=='baseline'),shared_teacher_generation_once=teacher_cost['seconds'])
 validation=p.compare_scores(scores,'naf_history');validation.update(status='six_actual_QP1500_and_baseline3000_weights',split='validation',matched_training_budget=False,budget_limitation=NOTE,actual_continuation_costs=costs,preceding_compact_training_costs=prior_costs,total_process_seconds=sum(x['seconds'] for x in costs),teacher_generation_cost=teacher_cost,training_stage_seconds=stage_seconds,reported_stage_seconds_including_shared_teacher=sum(stage_seconds.values()),total_additional_seconds_including_shared_teacher=sum(stage_seconds.values()),training_cost_scope='Only QP1500, corresponding compact6000, reused baseline3000 and shared teacher once; earlier ancestors and discarded searches excluded. Sums are process elapsed, not node wall-clock. total_process_seconds covers only the six direct compared continuation stages.',matched_additional_budget=False,goal1_pass=False,actual_cost_pending=True)
 value=dict(protocol=PROTOCOL,frozen_at_unix=time.time(),checkpoints=entries,validation=validation,selected_family='compact_query_product',source_and_evidence_sha256=evidence,matched_training_budget=False,budget_limitation=NOTE,teacher_receipt=inherited['teacher_receipt'],previous_freeze=binding(current.PARENT),confirmation_allocation=binding(ALLOCATION),metadata_erratum=binding(QP/'metadata_erratum.json'),ensemble_used=False,test_opened=False,test_authorized=False,scientific_goal_complete=False)
 fields(value);args.output.mkdir(parents=True);write(args.output/'results.json',validation);write(args.output/'continuation_selection_freeze.json',value)
 print({'status':'six_actual_QP_baseline_weights_frozen','test_authorized':False,'matched_training_budget':False})
def read_freeze(path):
 value=read(path);fields(value);check_erratum()
 require(value['source_and_evidence_sha256'].get(str(Path(__file__).resolve()))==sha(__file__),'QP reader source not sealed')
 for q,h in value['source_and_evidence_sha256'].items():require(sha(q)==h,'Frozen evidence changed: '+q)
 for e in value['checkpoints']:
  directory=Path(e['run']);d=read(directory/'complete.json')
  require(not (directory/'interrupted.json').exists() and d['status']=='complete' and d['updates']==e['actual_updates'] and d['validation_weight_candidates']==e['actual_validation_candidates'] and d['selected_checkpoint_sha256']==e['checkpoint_sha256'] and sha(e['checkpoint'])==e['checkpoint_sha256'],'Frozen actual completion/weight changed')
 return value
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output=args.output.resolve();freeze(args)
