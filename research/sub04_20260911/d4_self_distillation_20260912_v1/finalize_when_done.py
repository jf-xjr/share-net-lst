"""Finite authorized post-training pipeline; never launches or kills training."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE.parent/'final_delivery_late_20260912')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    receipt=out/'pipeline.json'
    if receipt.exists():raise FileExistsError('Existing pipeline receipt; no automatic resume')
    deadline=min(1789218000,json.loads((HERE/'design.json').read_text())['hard_deadline_unix'])
    plan=json.loads((HERE/'planned_runs.json').read_text());runs=plan['runs']
    expected={(a,s) for a in ('baseline','naf_history') for s in (20260905,20260912,20260913)}
    if len(runs)!=6 or {(r['architecture'],r['seed']) for r in runs}!=expected:raise ValueError('Exact six planned runs required')
    state=dict(status='waiting_for_real_completions_and_idle_gpu',started_unix=time.time(),deadline_unix=deadline,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stages=[],training_started=False)
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(receipt)
    def remaining():
        left=deadline-time.time()
        if left<=0:raise TimeoutError('13:00 UTC hard deadline reached')
        return left
    def idle():
        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
            stdin=subprocess.DEVNULL,capture_output=True,text=True,check=True,timeout=min(2.,remaining()))
        return not q.stdout.strip()
    def stage(name,argv):
        remaining();row=dict(stage=name,argv=argv,start_unix=time.time(),log=str(out/(name+'.log')))
        state['status']='running_'+name;state['stages'].append(row);save();child=None
        try:
            with (out/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;save();code=child.wait(timeout=remaining());row['exit_code']=code
                if code!=0:raise RuntimeError(name+' returned '+str(code))
        except BaseException as exc:
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait();row['own_subprocess_group_killed']=True
            row['exit_code']=None if child is None else child.returncode;row['error']=repr(exc);raise
        finally:
            row['end_unix']=time.time();row['seconds']=row['end_unix']-row['start_unix'];save()
    save()
    try:
        while True:
            remaining();complete=True
            for run in runs:
                path=Path(run['run'])
                if (path/'interrupted.json').exists():raise RuntimeError('Training interrupted: '+str(path))
                if not (path/'complete.json').exists():complete=False;continue
                try:done=json.loads((path/'complete.json').read_text())
                except json.JSONDecodeError:complete=False;continue
                if done.get('status')!='complete' or done.get('updates')!=1000 or done.get('validation_weight_candidates')!=10:
                    raise RuntimeError('Incomplete training receipt: '+str(path))
            if complete and idle():break
            time.sleep(min(.25,remaining()))
        state['all_six_complete_and_gpu_idle_unix']=time.time();save()
        freeze=out/'matched/continuation_selection_freeze.json';py=sys.executable
        stage('freeze',[py,str(HERE/'summarize_matched_v2.py'),'freeze','--runs',str(HERE/'planned_runs.json'),
            '--teacher-receipt',plan['teacher_receipt'],'--output',str(out/'matched')])
        for command in ('predict','score'):
            stage('test_'+command,[py,str(HERE/'evaluate_frozen_v2.py'),command,'--freeze',str(freeze),'--output',str(out/'test')])
        stage('benchmark',[py,str(HERE/'benchmark_short_v3.py'),'run','--freeze',str(freeze),'--output',str(out/'benchmark')])
        remaining();state['status']='all_requested_stages_completed'
    except BaseException as exc:
        state['status']='stopped_at_deadline' if time.time()>=deadline else 'failed';state['error']=repr(exc);raise
    finally:
        state['ended_unix']=time.time();save()

if __name__=='__main__':main()
