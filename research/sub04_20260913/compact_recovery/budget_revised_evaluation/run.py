"""Original Test90 seven sealed predictions; original six-model Val45x10 timing.

Only the fixed prediction count is adapted in the Test core. Main metrics still
use the original three-seed six-entry comparator; the seventh is sensitivity.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,hashlib,importlib.util,json,os,signal,sys,time
HERE=Path(__file__).resolve().parent
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
reader=module(HERE/'reader.py','budget_revised_seven_reader');p=reader.p
def authorization(path,freeze):
 if path is None:raise ValueError('Explicit root post-freeze seven-weight consumed-Test authorization required')
 value=reader.read_freeze(freeze);a=reader.read(path)
 reader.require(a['status']=='explicit_root_authorization_after_seven_weight_freeze'
  and a['freeze_sha256']==reader.sha(freeze) and value['frozen_at_unix']<=a['authorized_at_unix']<=time.time()
  and a['test_previously_consumed'] is True and a['weight_or_method_selection'] is False
  and a['single_forward_views']==1 and a['matched_training_budget'] is False
  and a['source_sha256']==reader.source_seal(),'Actual source-bound budget-revised Test authorization required')
 return value
def adapted_execute(core):
 source=Path(core.__file__).read_text();tree=ast.parse(source)
 function,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute']
 before=ast.dump(function);changed=[]
 class Count(ast.NodeTransformer):
  def visit_Constant(self,node):
   if type(node.value) is int and node.value==540:
    changed.append(node.lineno);return ast.copy_location(ast.Constant(630),node)
   return node
 function=Count().visit(copy.deepcopy(function));reader.require(len(changed)==2,'Exactly two original six-image counters must be updated')
 transformed=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));exec(compile(transformed,core.__file__,'exec'),core.__dict__)
 return dict(original_core_sha256=reader.sha(core.__file__),change='only two literal total-image counts540 ->630 in execute; no metric/forward/label-order change',
  changed_original_lines=changed,original_execute_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
  adapted_execute_AST_sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest())
def evaluate(args):
 frozen=authorization(args.test_authorization,args.freeze)
 core=module(reader.OLD/'d4_self_distillation_20260912_v1/evaluate_frozen_v2.py','budget_revised_original_Test90_core')
 changes=adapted_execute(core);original_seal=core.source_seal
 core.source_seal=lambda:dict(original_seal(),**reader.source_seal(),**{str(args.test_authorization.resolve()):reader.sha(args.test_authorization)})
 core.reader=lambda:reader;core.fields=reader.fields;core.ORDER=reader.ALL_ORDER
 core.DEADLINE=reader.read(reader.EXTENSION)['hard_deadline_unix']
 original_compare=core.p.compare_scores
 def compare(scores,role):
  reader.require(set(scores)==set(reader.ALL_ORDER),'All seven actual scores required')
  result=original_compare({key:scores[key] for key in reader.ORDER},role)
  sensitivity=scores[('baseline_sensitivity',20260905)]['macro']
  network=scores[('naf_history',20260905)]['macro'];old=scores[('baseline',20260905)]['macro']
  result['same_seed_stronger_baseline_sensitivity']=dict(seed=20260905,completed_updates=6000,validation_candidates=26,
   macro=sensitivity,compact_macro=network,old_baseline_macro=old,
   compact_rmse_gain_vs_completed6000_baseline_k=sensitivity['rmse']-network['rmse'],
   completed6000_baseline_rmse_gain_vs_old3000_k=old['rmse']-sensitivity['rmse'],
   three_seed_confirmation=False,model_selected_on_Test=False)
  return result
 core.p=SimpleNamespace(**vars(core.p));core.p.compare_scores=compare
 original_write=core.write
 def write(path,value):
  if Path(path).name in ('prediction_started.json','predictions_complete.json','results.json'):
   value=dict(value,effective_training_method=reader.PROTOCOL,selected_family='compact',actual_parameters=5911525,
    teacher_twenty_four_views_used_at_inference=False,explicit_test_authorization=reader.binding(args.test_authorization),
    main_comparison_model_count=6,sensitivity_model_count=1,total_frozen_model_count=7,
    matched_training_budget=False,budget_limitation=reader.NOTE,core_adaptation=changes,user_tier_rule=reader.binding(reader.RULE))
   if Path(path).name=='results.json':
    absolute=value['mean_metrics']['naf_history']['rmse']<.425
    numeric=bool(absolute and value['criteria']['network_numeric_criteria_pass'])
    value.update(user_absolute_threshold_k=.425,tier_absolute_pass=absolute,tier_and_paired_numeric_pass=numeric,
     numeric_goal1_pass=numeric,matched_budget_scientific_confirmation=False,goal1_fair_training_confirmation_pass=False,
     actual_common_costs_still_required=True,goal1_pass=False,
     limitation=reader.NOTE+' Consumed Test30, not independent holdout; no Test-driven selection. Three-seed bootstrap is descriptive.')
  original_write(path,value)
 core.write=write;old_loader=core.original.load_model;core.original.load_model=reader.load_model
 try:core.execute(args)
 finally:core.original.load_model=old_loader
def benchmark(args):
 core=module(reader.OLD/'benchmark_final_frozen_20260912.py','budget_revised_original_Val45_paired10')
 core.DEADLINE=reader.read(reader.EXTENSION)['hard_deadline_unix'];raw=reader.read_freeze(args.freeze)
 schedules={Path(q).resolve() for q in raw['source_and_evidence_sha256'] if Path(q).suffix=='.npz'}
 def adapted(path):
  value=reader.read_freeze(path);value['checkpoints']=value['checkpoints'][:6]
  value['evidence_sha256']=value['source_and_evidence_sha256']
  value['methods']={'baseline':'existing_final_confirmation3000','naf_history':'actual_compact_recovery6000'}
  value['training_cost']={k:value['validation'][k] for k in ('actual_continuation_costs','total_process_seconds','teacher_generation_cost','total_additional_seconds_including_shared_teacher','matched_additional_budget')}
  value['cost_note']=reader.NOTE+' Cost measures the six main models only. The seventh sensitivity model is not included.'
  return value
 def seal():return dict(core.v1.source_seal(),**reader.source_seal(),**{str(Path(core.__file__).resolve()):reader.sha(core.__file__)})
 adapter=SimpleNamespace(__file__=str(Path(__file__).resolve()),original=SimpleNamespace(load_model=reader.load_model),read_freeze=adapted,source_seal=seal)
 def guard(manifest):
  allowed={(p.PACKAGE/manifest['roles']['validation']['fields'][k]['path']).resolve() for k in p.runner.INPUTS};opened=set()
  def audit(event,args):
   if event!='open' or not isinstance(args[0],(str,bytes)):return
   path=Path(os.fsdecode(args[0])).resolve()
   if 'labels' in path.parts or ('data' in path.parts and {'fit','test'}&set(path.parts)):raise RuntimeError('Timing only original Val inputs')
   if path.suffix in ('.npy','.npz'):
    if path not in allowed|schedules:raise RuntimeError('Unapproved timing array')
    opened.add(str(path))
  sys.addaudithook(audit);return opened
 core.reader=lambda version:adapter;core.input_guard=guard
 remaining=core.DEADLINE-time.time()
 if remaining<=0:raise TimeoutError('Hard deadline reached')
 old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded common timing deadline')));signal.setitimer(signal.ITIMER_REAL,remaining)
 try:core.run(SimpleNamespace(freeze=args.freeze,output=args.output,freeze_reader=reader.PROTOCOL))
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','predict','score','benchmark'))
 parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--test-authorization',type=Path);args=parser.parse_args()
 if args.command=='check':
  core=module(reader.OLD/'d4_self_distillation_20260912_v1/evaluate_frozen_v2.py','CPU_budget_revised_core_check')
  result=adapted_execute(core);print(json.dumps(dict(status='CPU_import_and_exact_counter_adaptation_pass',adaptation=result,GPU_used=False,Test_opened=False)));return
 if args.freeze is None or args.output is None:parser.error('Actual --freeze and --output required')
 args.freeze=args.freeze.resolve();args.output=args.output.resolve()
 if args.test_authorization is not None:args.test_authorization=args.test_authorization.resolve()
 (benchmark if args.command=='benchmark' else evaluate)(args)
if __name__=='__main__':main()
