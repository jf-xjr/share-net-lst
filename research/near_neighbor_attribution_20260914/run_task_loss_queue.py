"""Use completed THST weights for a short metric-aligned comparator check."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
state=dict(pid=os.getpid(),started=time.time(),status='waiting_for_thst',jobs=[])
def write():
    p=HERE/'task_loss_queue.json';t=p.with_suffix('.partial');t.write_text(json.dumps(state,indent=2));t.replace(p)
def wait_for(done,queue):
    while not done.exists():
        q=json.loads(queue.read_text())
        if 'failed' in q['status'] or 'stopped' in q['status']:raise RuntimeError(str(queue))
        os.kill(q['pid'],0);time.sleep(5)
def run(name,command):
    with (HERE/(name+'.log')).open('a') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        row=dict(name=name,pid=p.pid,status='running',started=time.time());state['jobs'].append(row);write()
        print(json.dumps(row),flush=True);code=p.wait()
    row.update(status='complete' if code==0 else 'failed',returncode=code,finished=time.time());write()
    print(json.dumps(row),flush=True)
    if code:raise RuntimeError(name)
def main():
    write()
    try:
        wait_for(HERE/'thst_reference/complete.json',HERE/'deep_queue.json')
        state['status']='running';write()
        command=[sys.executable,str(HERE/'train_thst_task_loss.py')]
        if (HERE/'thst_task_loss/last.pt').exists():command.append('--resume')
        run('thst_task_loss',command)
        state['status']='waiting_for_base_analysis';write()
        while True:
            q=json.loads((HERE/'analysis_queue.json').read_text())
            if q['status']=='complete':break
            if q['status']=='failed' or 'stopped' in q['status']:raise RuntimeError('Base analysis stopped')
            os.kill(q['pid'],0);time.sleep(5)
        state['status']='running';write()
        run('augment_comparison',[sys.executable,str(HERE/'augment_comparison.py')])
    except BaseException:
        state.update(status='failed',finished=time.time());write();raise
    state.update(status='complete',finished=time.time());write()
if __name__=='__main__':main()
