"""Continue the independent THST stream, then seal matched NAF predictions."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def dump(value):
    p=HERE/'deep_queue.json';t=p.with_suffix('.partial')
    t.write_text(json.dumps(value,indent=2));t.replace(p)
def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
state=dict(pid=os.getpid(),status='running',started=time.time(),jobs=[]);dump(state)
jobs=[]
for stage in (1,2):
    out=HERE/'thst'/f'stage{stage}'
    command=[sys.executable,str(HERE/'train_thst.py'),'train','--stage',str(stage)]
    jobs.append((f'thst_stage{stage}',out/'complete.json',command))
jobs.append(('thst_prediction',HERE/'thst/stage2/test_predictions.json',
             [sys.executable,str(HERE/'train_thst.py'),'predict','--stage','2','--split','test']))
jobs.append(('thst_reference_variants',HERE/'thst_reference/complete.json',
             [sys.executable,str(HERE/'thst_reference_variants.py')]))
jobs.append(('attribution_prediction',HERE/'attribution_predictions/predictions_complete.json',
             [sys.executable,str(HERE/'predict_attribution.py')]))
for name,done,command in jobs:
    if name=='attribution_prediction':
        state['status']='waiting_for_attribution';dump(state)
        while True:
            upstream=json.loads((HERE/'attribution_queue.json').read_text())
            if upstream['status']=='complete':break
            if upstream['status']=='failed' or not live(upstream['pid']):
                state['status']='upstream_failed';dump(state);raise RuntimeError('Attribution queue stopped')
            time.sleep(5)
        state['status']='running';dump(state)
    external=done.parent/'external_process.json'
    if name=='thst_stage1' and external.exists():
        e=json.loads(external.read_text())
        while not done.exists() and live(e['pid']):
            stat=Path(f"/proc/{e['pid']}/stat")
            try:same=stat.read_text().split()[21]==e['proc_start_ticks']
            except FileNotFoundError:same=False
            if not same:break
            time.sleep(5)
    if done.exists():state['jobs'].append(dict(name=name,status='already_complete'));dump(state);continue
    if name in ('thst_stage1','thst_stage2'):
        if (done.parent/'last.pt').exists():command.append('--resume')
        elif (done.parent/'run.json').exists():raise RuntimeError('Existing stage lacks resume checkpoint')
    done.parent.mkdir(parents=True,exist_ok=True)
    with (done.parent/'queue_stdout.log').open('a') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        job=dict(name=name,pid=p.pid,status='running',started=time.time());state['jobs'].append(job);dump(state)
        print(json.dumps(job),flush=True);code=p.wait()
    job.update(returncode=code,finished=time.time(),status='complete' if code==0 and done.exists() else 'failed')
    dump(state);print(json.dumps(job),flush=True)
    if job['status']=='failed':state['status']='failed';dump(state);sys.exit(code or 1)
state.update(status='complete',finished=time.time());dump(state)
