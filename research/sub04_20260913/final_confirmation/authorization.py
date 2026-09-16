"""Validate an explicit root allocation after three real completed pilots."""
from pathlib import Path
import hashlib,json,math,time
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
VARIANTS=('standard','fourhead','refinement')
RECIPE=dict(updates=3000,batch_size=4,lr=1e-4,warmup=100,weight_decay=1e-4,
 validation_interval=500,evaluation_batch=2,history_dropout=0.,emissivity_dropout=0.,
 teacher_weight=.9,seed=20260923,ema_max_decay=.995,gradient_clip=1.)
PROTOCOL='root_final_confirmation_3000_14_20260913_v1'
PARAMETERS=dict(standard=9310105,fourhead=9310501,refinement=9380953,baseline=2985697)
FINAL_CRITERIA=dict(test_three_seed_mean_rmse_lt_k=.42,paired_mean_rmse_gain_ge_k=.01,
 all_three_seed_gains_positive=True,single_forward_views=1)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bound(item):
 p=Path(item['path']).resolve()
 if sha(p)!=item['sha256']:raise ValueError('Bound artifact changed: '+str(p))
 return p
def pilot(variant,item,teacher_sha):
 path=bound(item['completion']);checkpoint=bound(item['checkpoint'])
 if checkpoint!=path.parent/'best.pt':raise ValueError('Selected checkpoint must belong to the real pilot')
 done=read(path);runtime=read(path.parent/'run.json');cfg=runtime['config'];rows=read(path.parent/'validation.json')
 if (done['status']!='complete' or done['updates']!=3000 or done['validation_weight_candidates']!=14
  or done['parameters']!=PARAMETERS[variant] or done['selected_checkpoint_sha256']!=item['checkpoint']['sha256']
  or done['teacher_receipt_sha256']!=teacher_sha or runtime['teacher_receipt_sha256']!=teacher_sha
  or cfg['original_seed']!=20260905 or any(cfg.get(k)!=v for k,v in RECIPE.items())
  or done['counters']['updates']!=3000 or done['counters']['finite_updates']!=3000
  or done['test_opened'] is not False or done['single_network_forward'] is not True):
  raise ValueError('Actual complete common-recipe single-seed pilot required')
 if [(x['step'],x['weights']) for x in rows]!=[(s,w) for s in range(0,3001,500) for w in ('raw','ema')]:
  raise ValueError('Actual fourteen pilot candidates required')
 if variant=='standard' and cfg['protocol']!='strong_teacher_kd_20260913_v1':raise ValueError('Wrong standard pilot')
 if variant=='fourhead' and (cfg['protocol']!='four_head_teacher_kd_20260913_v1' or done.get('architecture_variant')!='four_head_history_fusion'):raise ValueError('Wrong four-head pilot')
 if variant=='refinement' and (cfg['protocol']!='refinement4_strong_teacher_kd_20260913_v1' or cfg.get('refinement_blocks')!=4):raise ValueError('Wrong refinement pilot')
 if not math.isfinite(done['selected_fp32']['macro']['rmse']):raise ValueError('Finite actual FP32 Val score required')
 for p,digest in runtime['source_sha256'].items():
  if sha(p)!=digest:raise ValueError('Pilot-bound source changed: '+p)
 return dict(completion_path=path,checkpoint_path=checkpoint,done=done,runtime=runtime)
def validate(path,teacher_receipt=None):
 path=Path(path).resolve();a=read(path)
 if (a.get('status')!='explicit_root_budget_authorization_final_six_confirmation'
  or a.get('selection_rule')!='lowest_actual_fp32_Val45_macro_rmse_among_three_completed_pilots'
  or set(a.get('pilots',{}))!=set(VARIANTS) or a.get('recipe')!=RECIPE
  or a.get('final_user_criteria')!=FINAL_CRITERIA
  or a.get('old_pilot_gate_modified') is not False or a.get('user_success_standard_lowered') is not False
  or a.get('allocation_only_not_success') is not True or a.get('automatic_teacher_search') is not False):
  raise ValueError('Explicit unchanged-standard root allocation receipt required')
 tp=bound(a['teacher_receipt'])
 if tp!=NEW/'strong_teacher/fit/predictions_complete.json':raise ValueError('Only existing shared fixed24 teacher permitted')
 if teacher_receipt is not None and Path(teacher_receipt).resolve()!=tp:raise ValueError('Caller teacher differs')
 evidence={v:pilot(v,item,a['teacher_receipt']['sha256']) for v,item in a['pilots'].items()}
 selected=min(VARIANTS,key=lambda v:evidence[v]['done']['selected_fp32']['macro']['rmse'])
 if a['selected_variant']!=selected or a['selected_pilot']!=a['pilots'][selected]:raise ValueError('Root selection must be the actual minimum of the three fixed pilots')
 if a['original_selected_pilot_gate_pass'] is not evidence[selected]['done']['continuation_gate_pass']:
  raise ValueError('The original pilot gate must be recorded truthfully without promotion')
 if not max(x['done']['actual_end_unix'] for x in evidence.values())<=a['authorized_at_unix']<=time.time():
  raise ValueError('Authorization must follow all three real completions')
 if a['authorized_at_unix']>=read(NEW/'five_hour_extension.json')['stop_search_by_unix']:
  raise ValueError('No new allocation beyond the original search deadline')
 return a,evidence

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--receipt',type=Path,required=True)
 args=parser.parse_args();value,evidence=validate(args.receipt)
 print(json.dumps(dict(status='actual_root_confirmation_allocation_validated',
  selected_variant=value['selected_variant'],original_selected_pilot_gate_pass=value['original_selected_pilot_gate_pass'],
  pilot_macro_rmse={k:v['done']['selected_fp32']['macro']['rmse'] for k,v in evidence.items()},
  GPU_used=False,training_started=False,freeze_created=False)))
