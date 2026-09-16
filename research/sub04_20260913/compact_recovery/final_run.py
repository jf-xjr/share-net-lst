"""Unchanged Test90 and Val45 ten-repeat cores with new actual 6000/26 factories."""
from pathlib import Path
from types import SimpleNamespace
import argparse,importlib.util,json,os,signal,sys,time
HERE=Path(__file__).resolve().parent
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
reader=load(HERE/'final_reader.py','recovery_actual_six_completion_reader')
loader=reader.loader;p=reader.p
def authorization(path,freeze):
 if path is None:raise ValueError('Separate explicit root --test-authorization required after the real freeze')
 value=reader.read_freeze(freeze);a=reader.read(path)
 reader.require(a['status']=='explicit_root_authorization_after_six_weight_freeze' and a['freeze_sha256']==reader.sha(freeze)
  and value['frozen_at_unix']<=a['authorized_at_unix']<=time.time() and a['test_previously_consumed'] is True
  and a['weight_or_method_selection'] is False and a['single_forward_views']==1
  and a['source_sha256']==reader.source_seal(),'Actual post-freeze, source-bound consumed-Test authorization required')
 return value
def evaluate(args):
 frozen=authorization(args.test_authorization,args.freeze)
 core=load(reader.OLD/'d4_self_distillation_20260912_v1/evaluate_frozen_v2.py','recovery_unchanged_complete_Test90_core')
 original_seal=core.source_seal
 core.source_seal=lambda:dict(original_seal(),**reader.source_seal(),**{str(args.test_authorization.resolve()):reader.sha(args.test_authorization)})
 core.reader=lambda:reader;core.fields=reader.fields;core.DEADLINE=reader.read(reader.EXTENSION)['hard_deadline_unix']
 original_write=core.write
 def write(path,value):
  if Path(path).name in ('prediction_started.json','predictions_complete.json','results.json'):
   value=dict(value,effective_training_method=reader.METHOD,selected_family=frozen['selected_family'],
    teacher_twenty_four_views_used_at_inference=False,explicit_test_authorization=reader.binding(args.test_authorization),
    actual_parameters=loader.trainer.PARAMETERS[frozen['selected_family']],user_tier_rule=reader.binding(reader.RULE))
   if Path(path).name=='results.json':
    threshold=reader.tier(value['actual_parameters']);absolute=value['mean_metrics']['naf_history']['rmse']<threshold
    value.update(user_absolute_threshold_k=threshold,tier_absolute_pass=absolute,
     tier_and_paired_numeric_pass=bool(absolute and value['criteria']['network_numeric_criteria_pass']),
     actual_common_costs_still_required=True,goal1_pass=False)
  original_write(path,value)
 core.write=write;old_loader=core.original.load_model;core.original.load_model=loader.load_model
 try:core.execute(args)
 finally:core.original.load_model=old_loader
def benchmark(args):
 core=load(reader.OLD/'benchmark_final_frozen_20260912.py','recovery_unchanged_full_Val45_paired10_cost')
 core.DEADLINE=reader.read(reader.EXTENSION)['hard_deadline_unix'];raw=reader.read_freeze(args.freeze)
 schedules={Path(q).resolve() for q in raw['source_and_evidence_sha256'] if Path(q).suffix=='.npz'}
 def adapted(path):
  v=reader.read_freeze(path);v['evidence_sha256']=v['source_and_evidence_sha256'];v['methods']={r:reader.METHOD for r in ('baseline','naf_history')}
  v['training_cost']={k:v['validation'][k] for k in ('actual_continuation_costs','total_process_seconds','teacher_generation_cost','total_additional_seconds_including_shared_teacher','matched_additional_budget')}
  v['cost_note']='Six actual matching6000/26 recovery runs; inherited fixed24 teacher reused with zero new cache generation cost; prior teacher cost separately disclosed. Historical search unequal.'
  return v
 def seal():return dict(core.v1.source_seal(),**reader.source_seal(),**{str(Path(core.__file__).resolve()):reader.sha(core.__file__)})
 adapter=SimpleNamespace(__file__=str(Path(__file__).resolve()),original=SimpleNamespace(load_model=loader.load_model),read_freeze=adapted,source_seal=seal)
 def guard(manifest):
  allowed={(p.PACKAGE/manifest['roles']['validation']['fields'][k]['path']).resolve() for k in p.runner.INPUTS};opened=set()
  def audit(event,argv):
   if event!='open' or not isinstance(argv[0],(str,bytes)):return
   q=Path(os.fsdecode(argv[0])).resolve()
   if 'labels' in q.parts or ('data' in q.parts and {'fit','test'}&set(q.parts)):raise RuntimeError('Timing permits original Val inputs only')
   if q.suffix in ('.npy','.npz'):
    if q not in allowed|schedules:raise RuntimeError('Unapproved timing array')
    opened.add(str(q))
  sys.addaudithook(audit);return opened
 core.reader=lambda version:adapter;core.input_guard=guard
 remaining=core.DEADLINE-time.time()
 if remaining<=0:raise TimeoutError('Original extended hard deadline reached')
 old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded common timing deadline')));signal.setitimer(signal.ITIMER_REAL,remaining)
 try:core.run(SimpleNamespace(freeze=args.freeze,output=args.output,freeze_reader=reader.PROTOCOL))
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','predict','score','benchmark'));parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--test-authorization',type=Path);args=parser.parse_args()
 if args.command=='check':
  reader.require(reader.tier(5911525)==.425 and reader.tier(9310501)==.42,'Original exact tiers');print(json.dumps(dict(status='CPU_import_and_tiers_pass_no_inference',Test_authorization_created=False,GPU_used=False)));return
 if args.freeze is None or args.output is None:parser.error('Actual --freeze and new --output required')
 args.freeze=args.freeze.resolve();args.output=args.output.resolve()
 if args.test_authorization is not None:args.test_authorization=args.test_authorization.resolve()
 (benchmark if args.command=='benchmark' else evaluate)(args)
if __name__=='__main__':main()
