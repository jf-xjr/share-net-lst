"""Complete Val selection before starting Test's frozen statistical comparison."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def dump(value):
    p=HERE/'statistics_queue.json';t=p.with_suffix('.partial')
    t.write_text(json.dumps(value,indent=2));t.replace(p)
state=dict(pid=os.getpid(),status='waiting_for_validation',started=time.time(),jobs=[]);dump(state)
while not (HERE/'statistics_v2/validation/predictions_complete.json').exists():
    try:os.kill(29442,0)
    except ProcessLookupError:
        state['status']='validation_process_stopped';dump(state);raise RuntimeError('Inspect unfinished Val before resuming')
    time.sleep(5)
state['status']='running';dump(state)
jobs=[('validation_score',[sys.executable,str(HERE/'evaluate_comparisons.py'),'statistics_validation']),
      ('test_prediction',[sys.executable,str(HERE/'statistical_neighbors.py'),'--split','test','--workers','2']),
      ('test_score',[sys.executable,str(HERE/'evaluate_comparisons.py'),'statistics_test'])]
for name,command in jobs:
    with (HERE/(name+'.log')).open('a') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        job=dict(name=name,pid=p.pid,status='running',started=time.time());state['jobs'].append(job);dump(state)
        print(json.dumps(job),flush=True);code=p.wait()
    job.update(returncode=code,finished=time.time(),status='complete' if code==0 else 'failed')
    dump(state);print(json.dumps(job),flush=True)
    if code:state['status']='failed';dump(state);sys.exit(code)
state.update(status='complete',finished=time.time());dump(state)
