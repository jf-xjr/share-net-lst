"""Serial GPU queue with resumable, immutable individual runs."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def write(state):
    path=HERE/'attribution_queue.json';temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(state,indent=2));temp.replace(path)

state=dict(pid=os.getpid(),started=time.time(),status='running',jobs=[])
write(state)
for seed in (20260914,20260915):
    for variant in ('coverage','learned'):
        out=HERE/'attribution'/f'{variant}_{seed}'
        if (out/'complete.json').exists():
            state['jobs'].append(dict(seed=seed,variant=variant,status='already_complete'))
            write(state);continue
        command=[sys.executable,str(HERE/'train_attribution.py'),'train','--variant',variant,'--seed',str(seed)]
        if (out/'last.pt').exists():command.append('--resume')
        elif (out/'run.json').exists():raise RuntimeError('Existing incomplete run has no resume checkpoint')
        out.mkdir(parents=True,exist_ok=True)
        with (out/'stdout.log').open('a') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            job=dict(seed=seed,variant=variant,pid=p.pid,started=time.time(),status='running')
            state['jobs'].append(job);write(state)
            print(json.dumps(job),flush=True)
            result=p.wait()
        job.update(returncode=result,finished=time.time(),status='complete' if result==0 else 'failed')
        write(state);print(json.dumps(job),flush=True)
        if result!=0:state['status']='failed';write(state);sys.exit(result)
state.update(status='complete',finished=time.time());write(state)
