"""Run exactly five missing matched confirmations after real two-pilot selection.

Explicit run invocation is required. Never starts automatically, repeats an
existing output, selects another family, freezes weights, or evaluates Test.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('queue_actual_recovery_reader',HERE/'final_reader.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)


def queue(family):
    if family not in ('compact','full'):raise ValueError('Actual chosen family required')
    return [(family,20260912),('baseline',20260905),(family,20260913),
            ('baseline',20260912),('baseline',20260913)]


def atomic_write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    os.replace(temporary,path)


def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True)
    if result.stdout.strip():raise RuntimeError('GPU occupied; stop without waiting or interrupting external work')


def run(args):
    import torch
    torch.set_num_threads(1)
    require=reader.require
    require(not args.output.exists(),'New runner directory only; no restart or resume')
    selection=reader.read_selection(args.selection)
    require(args.runs_root.resolve()==Path(selection['runs_root']).resolve(),'Use the selected actual pilot run root')
    family=selection['selected_family'];planned=queue(family)
    paths=[args.runs_root/(architecture+'_'+str(seed)) for architecture,seed in planned]
    require(all(not p.exists() for p in paths),'Every planned output must be absent; no completed or partial run may be restarted')
    retained=reader.checked_run(args.runs_root/(family+'_20260905'),family,20260905)
    require(reader.binding(Path(retained['path'])/'complete.json')==selection['pilots'][family]['completion'],
        'Retain exactly the actual selected first seed')
    sources=dict(reader.source_seal(),**{str(Path(__file__).resolve()):reader.sha(__file__)})
    selection_binding=reader.binding(args.selection)
    stop=reader.read(reader.EXTENSION)['stop_training_search_by_unix']
    require(time.time()<stop,'Original 11:26:36 UTC training stop reached')
    gpu_idle()
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',runner_pid=os.getpid(),selected_family=family,
        selection=selection_binding,source_sha256=sources,started_at_unix=time.time(),
        stop_training_by_unix=stop,per_run_cap_seconds=2700,
        planned=[dict(architecture=a,seed=s,output=str(p.resolve())) for (a,s),p in zip(planned,paths)],
        completed=[],active_child=None,Test_opened=False,automatic_retry=False,
        freeze_created=False,automatic_finalization=False)
    reader.write(args.output/'started.json',state)
    atomic_write(args.output/'progress.json',state)
    child=None
    def stop_own_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    def interrupted(signum,frame):
        raise InterruptedError('Runner received signal '+str(signum))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for (architecture,seed),output in zip(planned,paths):
            require(time.time()<stop,'Original training deadline reached; no next job')
            require(not output.exists(),'Never restart an existing completed or partial output')
            require(reader.binding(args.selection)==selection_binding,'Selection changed during queue')
            for path,digest in sources.items():require(reader.sha(path)==digest,'Frozen queue/training source changed')
            gpu_idle()
            log_path=args.output/(architecture+'_'+str(seed)+'.log')
            command=[sys.executable,'-B',str(HERE/'train.py'),'--architecture',architecture,
                '--original-seed',str(seed),'--output',str(output.resolve())]
            launched=time.time()
            with log_path.open('x') as log:
                child=subprocess.Popen(command,cwd=reader.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_child']=dict(pid=child.pid,architecture=architecture,seed=seed,
                    output=str(output.resolve()),log=str(log_path.resolve()),command=command,
                    launched_at_unix=launched,deadline_unix=min(stop,launched+2700))
                atomic_write(args.output/'progress.json',state)
                print(json.dumps(dict(event='actual_child_started',**state['active_child'])),flush=True)
                while True:
                    remaining=min(stop,launched+2700)-time.time()
                    if remaining<=0:raise TimeoutError('Actual child2700s cap or unchanged training stop reached')
                    try:
                        code=child.wait(timeout=min(30,remaining));break
                    except subprocess.TimeoutExpired:
                        state['last_monitor_unix']=time.time()
                        atomic_write(args.output/'progress.json',state)
            require(code==0,'Training exited nonzero: '+str(code)+'; queue stops without retry')
            finished=time.time()
            gpu_idle()
            checked=reader.checked_run(output,architecture,seed)
            complete=checked['completion']
            require(complete['actual_start_unix']>=selection['created_at_unix'],'Confirmation must follow actual selection')
            require(checked['schedule']==retained['schedule'],'Matched exact scene/D4 content changed')
            event=dict(architecture=architecture,seed=seed,output=str(output.resolve()),
                process_exit_code=code,process_elapsed_seconds=finished-launched,
                actual_training_seconds=complete['seconds'],completion=reader.binding(output/'complete.json'),
                checkpoint=reader.binding(output/'best.pt'),selected_fp32=complete['selected_fp32']['macro'],
                updates=6000,validation_weight_candidates=26,counters=complete['counters'])
            state['completed'].append(event);state['active_child']=None
            reader.write(args.output/(architecture+'_'+str(seed)+'_verified.json'),event)
            atomic_write(args.output/'progress.json',state)
            print(json.dumps(dict(event='actual_child_completed_and_verified',**event)),flush=True)
            child=None
        state.update(status='five_actual_matched_confirmations_complete',finished_at_unix=time.time())
        reader.write(args.output/'complete.json',state)
        atomic_write(args.output/'progress.json',state)
    except BaseException as exc:
        stop_own_child()
        state.update(status='failed_stopped_no_retry',error=repr(exc),finished_at_unix=time.time())
        if child is not None and state['active_child'] is not None:
            state['active_child']['terminal_exit_code']=child.poll()
        reader.write(args.output/'failed.json',state)
        atomic_write(args.output/'progress.json',state)
        raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','run'))
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--runs-root',type=Path,default=HERE/'runs')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='check':
        assert len(queue('compact'))==5 and len(set(queue('full')))==5
        assert queue('compact')[0]==('compact',20260912)
        assert not any(seed==20260905 and role!='baseline' for role,seed in queue('full'))
        print(json.dumps(dict(status='CPU_fixed_five_job_queue_check_pass',GPU_used=False,training_started=False,Test_opened=False)))
        return
    if args.selection is None or args.output is None:parser.error('Actual --selection and new --output required')
    run(args)


if __name__=='__main__':main()
