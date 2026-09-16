"""Explicit consumed-Test follow-up and original six-weight complete eager timing."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,hashlib,importlib.util,json,os,shutil,signal,sys,time
HERE=Path(__file__).resolve().parent
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
reader=module(HERE/'reader.py','QP_final_six_reader');p=reader.p
def authorization(path,freeze):
 reader.require(path is not None,'Explicit root authorization after actual six-weight freeze required')
 value=reader.read_freeze(freeze);a=reader.read(path)
 reader.require(a['status']=='explicit_root_authorization_after_QP_three_seed_freeze' and a['freeze_sha256']==reader.sha(freeze)
  and value['frozen_at_unix']<=a['authorized_at_unix']<=time.time()<reader.read(reader.EXTENSION)['hard_deadline_unix']
  and a['test_previously_consumed'] is True and a['weight_or_method_selection'] is False and a['single_forward_views']==1
  and a['matched_training_budget'] is False and a['source_sha256']==reader.source_seal(),'Actual post-freeze source-bound consumed-Test authorization required')
 return value
def legacy_entries(value,records):
 receipt=reader.read(reader.LEGACY_RECEIPT)
 reader.require(receipt['stage']=='all_predictions_sealed' and receipt['labels_opened'] is False and receipt['device']=='cuda' and receipt['dtype']=='float32' and receipt['tf32'] is False and receipt['batch']==1 and receipt['repair_dtype']=='float64' and receipt['scene_ids']==[r['scene_id'] for r in records],'Actual same-protocol sealed baseline Test90 predictions required')
 out={}
 for entry in value['checkpoints']:
  if entry['architecture']!='baseline':continue
  source,=[e for e in receipt['entries'] if (e['architecture'],e['seed'])==('baseline',entry['seed'])]
  reader.require(source['checkpoint_sha256']==entry['checkpoint_sha256'] and source['config']==entry['config'] and source['shape']==[90,1,160,160] and source['output_dtype']=='float64' and source['image_forwards']==90,'Exact baseline prediction identity differs')
  out[entry['seed']]=source
 return out
def adapt_execute(core):
 tree=ast.parse(Path(core.__file__).read_text());fun,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute'];before=ast.dump(fun);changed=[]
 class Adapt(ast.NodeTransformer):
  def visit_Constant(self,node):
   if type(node.value) is int and node.value==540:changed.append('actual_forward_count');return ast.copy_location(ast.Constant(270),node)
   return node
  def visit_For(self,node):
   node=self.generic_visit(node)
   if isinstance(node.target,ast.Name) and node.target.id=='entry' and isinstance(node.iter,ast.Name) and node.iter.id=='entries':
    changed.append('reuse_three_baselines');node.body=ast.parse("if entry['architecture']=='baseline':\n _reuse_baseline(entry,args.output,value,records)\n continue").body+node.body
   return node
 fun=Adapt().visit(copy.deepcopy(fun));reader.require(changed.count('actual_forward_count')==2 and changed.count('reuse_three_baselines')==1,'Only two counts and one baseline reuse branch may change')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),core.__file__,'exec'),core.__dict__)
 return dict(original_core_sha256=reader.sha(core.__file__),changes=changed,original_execute_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),adapted_execute_AST_sha256=hashlib.sha256(ast.dump(fun).encode()).hexdigest(),new_QP_image_forwards=270,reused_baseline_images=270,all_six_arrays_sealed_before_labels=True)
def evaluate(args):
 frozen=authorization(args.test_authorization,args.freeze)
 core=module(reader.OLD/'d4_self_distillation_20260912_v1/evaluate_frozen_v2.py','QP_original_Test90_core');changes=adapt_execute(core)
 original_seal=core.source_seal
 core.source_seal=lambda:dict(original_seal(),**reader.source_seal(),**{str(args.test_authorization.resolve()):reader.sha(args.test_authorization)})
 core.reader=lambda:reader;core.fields=reader.fields;core.ORDER=reader.ORDER;core.DEADLINE=reader.read(reader.EXTENSION)['hard_deadline_unix']
 records=reader.read(reader.PACKAGE/'manifest.json')['roles']['test']['scenes'];legacy=legacy_entries(frozen,records)
 original_guard=core.guard
 def guard(manifest,output,value,access):
  guarded=copy.deepcopy(value)
  for e in legacy.values():guarded['source_and_evidence_sha256'][str((reader.LEGACY_RECEIPT.parent/e['prediction']).resolve())]=e['prediction_sha256']
  return original_guard(manifest,output,guarded,access)
 core.guard=guard
 def reuse(entry,output,value,scene_records):
  e=legacy[entry['seed']];source=reader.LEGACY_RECEIPT.parent/e['prediction'];target=output/f"baseline_{entry['seed']}.npy";began=time.perf_counter()
  reader.require(not target.exists() and reader.sha(source)==e['prediction_sha256'],'Original sealed baseline bytes changed')
  shutil.copyfile(source,target);reader.require(reader.sha(target)==e['prediction_sha256'],'Reused baseline copy changed')
  entry.update(prediction=target.name,prediction_sha256=e['prediction_sha256'],shape=[90,1,160,160],output_dtype='float64',image_forwards=0,reused_images=90,
   prediction_reused=True,reused_prediction=reader.binding(source),reused_receipt=reader.binding(reader.LEGACY_RECEIPT),copy_seconds=time.perf_counter()-began,timing_is_benchmark=False)
 core._reuse_baseline=reuse
 if args.command=='score':
  receipt=reader.read(args.output/'predictions_complete.json')
  reader.require(receipt['total_reused_images']==270 and sum(e['image_forwards'] for e in receipt['entries'])==270 and sum(e.get('reused_images',0) for e in receipt['entries'])==270,'Truthful270 new/270 reused complete prediction counts required')
  for e in receipt['entries']:
   if e['architecture']=='baseline':reader.require(e['reused_receipt']==reader.binding(reader.LEGACY_RECEIPT) and e['prediction_sha256']==legacy[e['seed']]['prediction_sha256'] and e['image_forwards']==0,'Exact existing baseline prediction reuse required')
   else:reader.require(e['image_forwards']==90,'Each actual QP weight must have all90 new forwards')
 original_write=core.write
 def write(path,value):
  if Path(path).name in ('prediction_started.json','predictions_complete.json','results.json'):
   value=dict(value,effective_training_method=reader.PROTOCOL,selected_family='compact_query_product',actual_parameters=reader.PARAMETERS,
    explicit_test_authorization=reader.binding(args.test_authorization),matched_training_budget=False,budget_limitation=reader.NOTE,core_adaptation=changes,user_tier_rule=reader.binding(reader.RULE),total_reused_images=270,total_frozen_model_count=6,teacher_twenty_four_views_used_at_inference=False)
   if Path(path).name=='results.json':
    threshold=reader.current.tier(reader.PARAMETERS);absolute=value['mean_metrics']['naf_history']['rmse']<threshold;numeric=bool(absolute and value['criteria']['network_numeric_criteria_pass'])
    value.update(user_absolute_threshold_k=threshold,tier_absolute_pass=absolute,tier_and_paired_numeric_pass=numeric,numeric_goal1_pass=numeric,
     matched_budget_scientific_confirmation=False,goal1_fair_training_confirmation_pass=False,actual_common_costs_still_required=True,goal1_pass=False,limitation=reader.NOTE+' Three observed-seed/city intervals are descriptive.')
  original_write(path,value)
 core.write=write;original_loader=core.original.load_model;core.original.load_model=reader.load_model
 try:core.execute(args)
 finally:core.original.load_model=original_loader
def benchmark(args):
 core=module(reader.OLD/'benchmark_final_frozen_20260912.py','QP_original_Val45_paired10');core.DEADLINE=reader.read(reader.EXTENSION)['hard_deadline_unix'];raw=reader.read_freeze(args.freeze)
 schedules={Path(q).resolve() for q in raw['source_and_evidence_sha256'] if Path(q).suffix=='.npz'}
 def adapted(path):
  value=reader.read_freeze(path);value['evidence_sha256']=value['source_and_evidence_sha256'];value['methods']={'baseline':'existing_final_confirmation3000','naf_history':reader.METHOD}
  value['training_cost']={k:value['validation'][k] for k in ('actual_continuation_costs','preceding_compact_training_costs','total_process_seconds','teacher_generation_cost','training_stage_seconds','reported_stage_seconds_including_shared_teacher','total_additional_seconds_including_shared_teacher','training_cost_scope','matched_additional_budget')};value['cost_note']=reader.NOTE
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
 remaining=core.DEADLINE-time.time();reader.require(remaining>0,'Hard deadline reached')
 previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded final common timing deadline')));signal.setitimer(signal.ITIMER_REAL,remaining)
 try:core.run(SimpleNamespace(freeze=args.freeze,output=args.output,freeze_reader=reader.PROTOCOL))
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
def check():
 torch=reader.torch;torch.set_num_threads(1)
 core=module(reader.OLD/'d4_self_distillation_20260912_v1/evaluate_frozen_v2.py','QP_CPU_counter_adapter');changes=adapt_execute(core)
 available=[];missing=[]
 for seed in p.SEEDS:
  path=reader.QP/'runs'/f'compact_{seed}'/'best.pt'
  if not path.is_file() or not (path.parent/'complete.json').is_file():missing.append(seed);continue
  saved=torch.load(path,map_location='cpu',weights_only=False);model=reader.construct();reader.validate_state(model,saved['state_dict'])
  wrong=dict(saved['state_dict']);wrong.pop('fusion.0.query_product.weight')
  try:reader.validate_state(model,wrong)
  except ValueError:pass
  else:raise AssertionError('Missing query weight must reject')
  available.append(dict(seed=seed,checkpoint_sha256=reader.sha(path),actual_complete=(path.parent/'complete.json').is_file(),parameters=sum(v.numel() for v in model.parameters())));del model,saved
 reader.require(not torch.cuda.is_initialized(),'CPU checks must not initialize GPU')
 print(json.dumps(dict(status='CPU_strict_actual_available_weights_and_AST_reuse_PASS',adaptation=changes,available=available,missing_actual_weights=missing,missing_weights_not_fabricated=True,GPU_used=False,observations_opened=False)))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=('check','freeze','predict','score','benchmark'));parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--test-authorization',type=Path);args=parser.parse_args()
 if args.command=='check':check();return
 if args.output is None:parser.error('Explicit new output required')
 args.output=args.output.resolve()
 if args.command=='freeze':reader.freeze(args);return
 if args.freeze is None:parser.error('Actual frozen weights required')
 args.freeze=args.freeze.resolve()
 if args.test_authorization is not None:args.test_authorization=args.test_authorization.resolve()
 (benchmark if args.command=='benchmark' else evaluate)(args)
if __name__=='__main__':main()
