"""Strict single-model loaders for the two fixed recovery candidates and baseline."""
from pathlib import Path
import hashlib,importlib.util
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compact_recovery_frozen_training_factories',HERE/'train.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
METHOD='fixed_block_recovery_annealed_teacher_6000_26'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def make_model(family,role):
 if family not in ('compact','full') or role not in ('baseline','naf_history'):raise ValueError('Unregistered frozen family or role')
 return trainer.construct('baseline' if role=='baseline' else family).float()
def validate_state(model,state):
 expected=model.state_dict()
 if set(state)!=set(expected):raise ValueError('Strict selected architecture keys differ')
 for k,v in state.items():
  if not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype:raise ValueError('Selected tensor schema differs: '+k)
  if v.is_floating_point() and not torch.isfinite(v).all():raise ValueError('Nonfinite selected tensor: '+k)
 model.load_state_dict(state,strict=True)
def load_model(entry,device):
 if entry['effective_method']!=METHOD or sha(entry['checkpoint'])!=entry['checkpoint_sha256']:raise ValueError('Exact recovery freeze required')
 saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
 if saved['config']!=entry['config']:raise ValueError('Selected checkpoint config changed')
 model=make_model(entry['selected_family'],entry['architecture']);validate_state(model,saved['state_dict'])
 expected=trainer.PARAMETERS['baseline' if entry['architecture']=='baseline' else entry['selected_family']]
 if sum(p.numel() for p in model.parameters())!=expected:raise ValueError('Parameter tier factory mismatch')
 return model.to(device).eval()
