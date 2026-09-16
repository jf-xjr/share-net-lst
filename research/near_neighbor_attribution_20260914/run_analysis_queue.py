"""Score the complete frozen comparison and export figures when dependencies finish."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
state=dict(pid=os.getpid(),started=time.time(),status='waiting',jobs=[])
def write():
    p=HERE/'analysis_queue.json';t=p.with_suffix('.partial')
    t.write_text(json.dumps(state,indent=2));t.replace(p)
write()
dependencies=[HERE/'statistics_queue.json',HERE/'deep_queue.json']
while True:
    statuses={}
    for path in dependencies:
        value=json.loads(path.read_text());statuses[path.name]=value['status']
        if value['status']=='complete':continue
        try:os.kill(value['pid'],0)
        except ProcessLookupError:
            state.update(status='dependency_stopped',dependency=str(path));write();raise RuntimeError(str(path))
        if 'failed' in value['status'] or 'stopped' in value['status']:
            state.update(status='dependency_failed',dependency=str(path));write();raise RuntimeError(str(path))
    if all(v=='complete' for v in statuses.values()):break
    state['dependencies']=statuses;write();time.sleep(5)
state['status']='running';write()
for script,args in [('evaluate_comparisons.py',['all']),('plot_results.py',[]),('benchmark_inference.py',[])]:
    with (HERE/(script+'.log')).open('a') as f:
        p=subprocess.Popen([sys.executable,str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT)
        job=dict(script=script,pid=p.pid,started=time.time(),status='running');state['jobs'].append(job);write()
        result=p.wait()
    job.update(status='complete' if result==0 else 'failed',returncode=result,finished=time.time());write()
    print(json.dumps(job),flush=True)
    if result:state['status']='failed';write();sys.exit(result)
state.update(status='complete',finished=time.time());write()
