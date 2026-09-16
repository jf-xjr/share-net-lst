"""Reuse the original KD function with one explicit allocation-gate replacement.

The old pilot, old gate value, old source and every optimizer/update/validation
statement remain unchanged. Only the pre-training continuation permission block
is replaced by a validator for the root's final-confirmation allocation.
"""
from pathlib import Path
import argparse,ast,importlib.util,inspect,sys,textwrap
import torch
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
sys.path[:0]=[str(HERE),str(NEW)]
import authorization as auth
PARENT=NEW/'train_strong_kd.py';PARENT_SHA='c80551e32eb85f091743421d73508c1ed5a0a6040aa324e054cc86e08ecbfc69'
spec=importlib.util.spec_from_file_location('final_confirmation_original_loop',PARENT)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from multihead_fusion.model import FourHeadHistoryNAF,expand_single_head_state_dict
spec=importlib.util.spec_from_file_location('final_confirmation_refined_model',NEW/'refinement/model.py')
refined=importlib.util.module_from_spec(spec);spec.loader.exec_module(refined)

class FourHeadInitialization(FourHeadHistoryNAF):
 def __init__(self,**kw):super().__init__(**kw);self._loaded=False;self._inside=False
 def load_state_dict(self,state,strict=True,assign=False):
  if self._inside:return super().load_state_dict(state,strict=strict,assign=assign)
  if not strict or assign:raise ValueError('Strict copying only')
  if not self._loaded:
   self._inside=True
   try:expand_single_head_state_dict(state,self)
   finally:self._inside=False
   self._loaded=True;return torch.nn.modules.module._IncompatibleKeys([],[])
  return super().load_state_dict(state,strict=True)

class RefinementInitialization(refined.RefinedHistoryNAF):
 def __init__(self,**kw):super().__init__(**kw);self._loaded=False;self._inside=False
 def load_state_dict(self,state,strict=True,assign=False):
  if self._inside:return super().load_state_dict(state,strict=strict,assign=assign)
  if not strict or assign:raise ValueError('Strict copying only')
  if not self._loaded:
   self._inside=True
   try:self.load_parent_state(state)
   finally:self._inside=False
   self._loaded=True;return torch.nn.modules.module._IncompatibleKeys([],[])
  return super().load_state_dict(state,strict=True)

def selected_factory(variant):
 return dict(standard=base.HistoryNAFReconstructor,fourhead=FourHeadHistoryNAF,refinement=refined.RefinedHistoryNAF)[variant]
ORIGINAL_NAF=base.HistoryNAFReconstructor
def model_factory(variant,role,**kw):
 if role=='baseline':return base.DropoutUTAE(**kw)
 return dict(standard=ORIGINAL_NAF,fourhead=FourHeadInitialization,refinement=RefinementInitialization)[variant](**kw)

def metadata(a,args):
 return dict(protocol=auth.PROTOCOL,confirmation_variant=a['selected_variant'],
  architecture=args.architecture,original_seed=args.original_seed,
  loader_class=('dropout_models.DropoutUTAE' if args.architecture=='baseline' else
   dict(standard='naf_history.model.HistoryNAFReconstructor',fourhead='multihead_fusion.model.FourHeadHistoryNAF',refinement='refinement.model.RefinedHistoryNAF')[a['selected_variant']]),
  confirmation_authorization=dict(path=str(args.authorization),sha256=auth.sha(args.authorization)),
  original_selected_pilot_gate_pass=a['original_selected_pilot_gate_pass'],allocation_only_not_success=True)

def require_allocation(args):
 a,e=auth.validate(args.authorization,args.teacher_receipt)
 if args.architecture not in ('naf_history','baseline') or args.original_seed not in (20260905,20260912,20260913):raise ValueError('Only the matched architectures/seeds')
 if (args.architecture,args.original_seed)==('naf_history',20260905):raise ValueError('Retain the actual first-seed pilot; never retrain it here')
 if args.output.exists():raise FileExistsError('New confirmation output only')
 return a,e

def adapted_train_function():
 if auth.sha(PARENT)!=PARENT_SHA:raise ValueError('Hash-pinned original trainer changed')
 tree=ast.parse(textwrap.dedent(inspect.getsource(base.train)));fn=tree.body[0]
 target="(args.architecture, args.original_seed) != ('naf_history', 20260905)"
 matches=[i for i,node in enumerate(fn.body) if isinstance(node,ast.If) and ast.unparse(node.test)==target]
 if len(matches)!=1:raise RuntimeError('Expected precisely one original continuation allocation block')
 fn.body[matches[0]]=ast.Expr(ast.Call(func=ast.Name(id='_require_confirmation_allocation',ctx=ast.Load()),args=[ast.Name(id='args',ctx=ast.Load())],keywords=[]))
 ast.fix_missing_locations(tree);namespace=dict(base.__dict__);namespace['_require_confirmation_allocation']=require_allocation
 exec(compile(tree,str(PARENT),'exec'),namespace)
 return namespace['train'],ast.unparse(tree)

def train(args):
 a,_=require_allocation(args);constructed=[];old_dump=base.r.dump
 function,transformed=adapted_train_function()
 def naf_factory(**kw):
  model=model_factory(a['selected_variant'],'naf_history',**kw);constructed.append(model);return model
 function.__globals__['HistoryNAFReconstructor']=naf_factory
 extra_sources={str(p.resolve()):auth.sha(p) for p in (Path(__file__),HERE/'authorization.py',NEW/'multihead_fusion/model.py',NEW/'refinement/model.py',args.authorization)}
 def dump(path,value):
  if Path(path).name=='run.json':
   value['config'].update(metadata(a,args));value['source_sha256'].update(extra_sources)
   value.update(metadata(a,args));value['continuation_permission']='explicit_root_allocation_preserving_original_pilot_gate'
   expected=auth.PARAMETERS['baseline' if args.architecture=='baseline' else a['selected_variant']]
   if value['parameters']!=expected:raise RuntimeError('Actual factory parameters differ')
  elif Path(path).name in ('complete.json','interrupted.json'):value.update(metadata(a,args))
  return old_dump(path,value)
 base.r.dump=dump
 try:return function(args)
 finally:base.r.dump=old_dump

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--architecture',choices=['baseline','naf_history'],required=True)
 p.add_argument('--original-seed',type=int,choices=[20260905,20260912,20260913],required=True)
 p.add_argument('--authorization',type=Path,required=True);p.add_argument('--teacher-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 for key in ('authorization','teacher_receipt','output'):setattr(args,key,getattr(args,key).resolve())
 train(args)
if __name__=='__main__':main()
