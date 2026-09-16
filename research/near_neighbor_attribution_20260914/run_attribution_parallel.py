"""Two-worker continuation of the same immutable matched training runs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,threading,time
HERE=Path(__file__).resolve().parent
LOCK=threading.Lock()
state=dict(pid=os.getpid(),started=time.time(),status='running',workers=2,jobs=[])
def write():
    p=HERE/'attribution_queue.json';t=p.with_suffix('.partial')
    t.write_text(json.dumps(state,indent=2));t.replace(p)
def live(e):
    try:
        stat=Path(f"/proc/{e['pid']}/stat").read_text().split()
        return stat[21]==e['proc_start_ticks'] and stat[2]!='Z'
    except FileNotFoundError:return False
def job(seed,variant):
    out=HERE/'attribution'/f'{variant}_{seed}';out.mkdir(parents=True,exist_ok=True)
    row=dict(seed=seed,variant=variant,status='initializing',started=time.time())
    with LOCK:state['jobs'].append(row);write()
    external=out/'external_process.json'
    if external.exists() and not (out/'complete.json').exists():
        e=json.loads(external.read_text())
        with LOCK:row.update(status='inherited_running',pid=e['pid']);write()
        while live(e):time.sleep(5)
    if (out/'complete.json').exists():
        result=json.loads((out/'complete.json').read_text())
        assert result['status']=='complete' and result['updates']==12000
        with LOCK:row.update(status='complete',finished=time.time(),reused_completed=True);write()
        return
    command=[sys.executable,str(HERE/'train_attribution.py'),'train','--variant',variant,'--seed',str(seed)]
    if (out/'last.pt').exists():command.append('--resume')
    elif (out/'run.json').exists():raise RuntimeError('Incomplete run lacks resume checkpoint')
    with (out/'stdout.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        with LOCK:row.update(status='running',pid=process.pid);write()
        print(json.dumps(row),flush=True);code=process.wait()
    with LOCK:
        row.update(status='complete' if code==0 and (out/'complete.json').exists() else 'failed',returncode=code,finished=time.time());write()
    print(json.dumps(row),flush=True)
    if row['status']!='complete':raise RuntimeError(f'{variant}_{seed} failed')
def main():
    write()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(job,seed,variant) for seed in (20260914,20260915) for variant in ('coverage','learned')]
            for future in pending:future.result()
    except BaseException:
        with LOCK:state.update(status='failed',finished=time.time());write()
        raise
    with LOCK:state.update(status='complete',finished=time.time());write()
if __name__=='__main__':main()
