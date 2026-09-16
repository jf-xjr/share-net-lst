"""Two explicitly allocated actual QP confirmations; original failed gate preserved."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,time
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
spec=importlib.util.spec_from_file_location('unchanged_QP_numerical_trainer',HERE/'train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
AUTH=HERE/'confirmation_allocation.json';SAM=NEW/'compact_sam/runs/compact_20260905/complete.json'
SAM_SHA='6ad91c91d541e76ab484c56d8b53fe3e5f83f878ffc25a8fbfc1e0c4271387b3'
PILOT_SHA='71353e1ac0de99603837c2cc34f152919c096f01407ec5648b01ad8f215fd886'
def authority(args):
 a=base.read(AUTH);p=base.read(base.PILOT);s=base.read(SAM);budget=base.read(NEW/'final_hour_reallocation.json')
 base.require(a['status']=='explicit_root_conditional_budget_override_for_best_actual_Val_candidate'
  and a['budget_added_seconds']==0 and a['seeds']==[20260912,20260913] and a['original_pilot_gate_pass'] is False
  and a['baseline_retraining'] is False and a['matched_additional_training_budget'] is False
  and a['method_selection_uses_Test'] is False and a['methods_or_weights_may_be_chosen_from_new_Test'] is False,
  'Actual explicit conditional budget allocation required')
 base.require(a['hard_deadline_unix']==budget['hard_deadline_unix']==base.read(NEW/'four_hour_extension.json')['hard_deadline_unix']
  and a['training_stop_unix']==budget['new_training_stop_unix'] and a['maximum_job_seconds']==700,'Fixed actual budget differs')
 base.require(base.sha(SAM)==SAM_SHA and s['status']=='complete' and s['updates']==800 and s['validation_weight_candidates']==10
  and not (SAM.parent/'interrupted.json').exists(),'Real completed SAM condition required')
 base.require(base.sha(base.PILOT)==PILOT_SHA==a['original_pilot_completion']['sha256']
  and Path(a['original_pilot_completion']['path']).resolve()==base.PILOT.resolve()
  and p['status']=='complete' and p['updates']==1500 and p['validation_weight_candidates']==8
  and p['continuation_gate_pass'] is False and p['validation_gain_k']==a['actual_pilot_gain_k']
  and p['minimum_validation_gain_k']==a['original_minimum_gain_k']==.001
  and s['selected_validation_rmse']>=p['selected_validation_rmse'],'Keep failed original QP gate; actual SAM must not be better')
 for item in (p,s):
  for path,digest in item['source_sha256'].items():base.require(base.sha(path)==digest,'Bound actual pilot source changed')
 base.require(base.sha(base.PILOT.parent/'best.pt')==p['selected_checkpoint_sha256']
  and base.sha(SAM.parent/'best.pt')==s['selected_checkpoint_sha256'],'Actual selected pilot weights changed')
 base.require(args.original_seed in a['seeds'] and args.architecture=='compact'
  and args.output.resolve()==(HERE/'runs'/f'compact_{args.original_seed}').resolve(),'Exactly the two allocated corresponding-seed outputs')
 if args.original_seed==20260913:
  prior=HERE/'runs/compact_20260912';d=base.read(prior/'complete.json')
  base.require(d['status']=='complete' and d['updates']==1500 and d['validation_weight_candidates']==8
   and d['confirmation_allocation']['sha256']==base.sha(AUTH) and not (prior/'interrupted.json').exists(),'Actual912 completion before913')
 base.require(time.time()<a['training_stop_unix'],'Actual training stop reached')
 return a
def run(args):
 a=authority(args);source=Path(base.__file__).read_text();tree=ast.parse(source)
 function,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train'];changed=[]
 class Adapt(ast.NodeTransformer):
  def visit_Assign(self,node):
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='deadline':
    changed.append('deadline');return ast.copy_location(ast.parse("deadline=min(started+700,_actual_allocation['training_stop_unix'])").body[0],node)
   return self.generic_visit(node)
  def visit_Call(self,node):
   if isinstance(node.func,ast.Name) and node.func.id=='checked_pilot':
    changed.append('execution_authority');node=copy.deepcopy(node);node.func.id='_authorized_completed_pilot';return node
   return self.generic_visit(node)
 transformed=Adapt().visit(copy.deepcopy(function));base.require(changed==['deadline','execution_authority'],'Only two explicit authority/deadline substitutions allowed')
 base._actual_allocation=a
 def allowed(path):
  authority(args);base.require(Path(path).resolve()==base.PILOT.resolve(),'Retain actual failed original pilot');return base.binding(path)
 base._authorized_completed_pilot=allowed
 old_seal=base.source_seal
 base.source_seal=lambda:dict(old_seal(),**{str(p.resolve()):base.sha(p) for p in (Path(__file__),AUTH,SAM,NEW/'final_hour_reallocation.json',HERE/'metadata_erratum.json')})
 old_dump=base.r.dump
 def dump(path,value):
  if Path(path).name in ('run.json','complete.json','interrupted.json'):
   value=dict(value,confirmation_allocation=base.binding(AUTH),confirmation_runner=base.binding(__file__),
    execution_authority='explicit_root_budget_override_after_actual_SAM',original_pilot_gate_pass=False,
    matched_additional_training_budget=False,numerical_recipe_unchanged=True,metadata_erratum=base.binding(HERE/'metadata_erratum.json'),
    adaptation=dict(changed_nodes=changed,original_trainer_sha256=base.sha(base.__file__),
     adapted_train_AST_sha256=hashlib.sha256(ast.dump(transformed).encode()).hexdigest()))
  return old_dump(path,value)
 base.r.dump=dump
 compiled=ast.fix_missing_locations(ast.Module(body=[transformed],type_ignores=[]));exec(compile(compiled,base.__file__,'exec'),base.__dict__)
 base.train(args)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--architecture',choices=('compact',),default='compact')
 parser.add_argument('--original-seed',type=int,choices=(20260912,20260913),required=True);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();args.output=args.output.resolve();run(args)
