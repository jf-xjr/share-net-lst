"""Thin adapter: unchanged common Val45 eager cost protocol, six frozen D4 students.

Only freeze reading/provenance adapts. Timed requests, models, sample/repeat
schedule, batch1 FP32, stage timers and macro aggregation are reused unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, copy, importlib.util, json, os, signal, sys, time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;SUB04=HERE.parent
sys.path.insert(0,str(SUB04))
import benchmark_final_frozen_20260912 as benchmark
spec=importlib.util.spec_from_file_location('d4_benchmark_strict_freeze_reader',HERE/'summarize_matched_v2.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
p=benchmark.p


def adapted_freeze(path):
    original=summary.read_freeze(path)
    value=copy.deepcopy(original)
    value['evidence_sha256']=original['source_and_evidence_sha256']
    value['methods']={role:'d4_self_distillation' for role in ('baseline','naf_history')}
    value['training_cost']={k:original['validation'][k] for k in
        ('actual_continuation_costs','total_process_seconds','teacher_generation_cost','total_additional_seconds_including_shared_teacher','matched_additional_budget')}
    value['cost_note']='Actual six matched1000-update/10-choice continuations plus the single shared eight-view teacher cache. '
    value['cost_note']+='These are additional costs; original training, discarded work and historical search remain separate. Concurrent per-run elapsed sums are not node wall-clock.'
    return value


def source_seal():
    return dict(benchmark.v1.source_seal(),**{str(q.resolve()):p.sha(q) for q in
        (Path(__file__),Path(benchmark.__file__),HERE/'summarize_matched_v2.py')})


def run(args):
    benchmark.deadline()
    raw=json.loads(args.freeze.read_text())
    # read_freeze hashes the six sealed training schedule artifacts. They are
    # provenance bytes, not observations, and must not be mistaken for labels.
    schedules={(Path(e['run'])/'schedule.npz').resolve() for e in raw['checkpoints']}
    if len(schedules)!=6 or any(str(q) not in raw['source_and_evidence_sha256'] for q in schedules):
        raise ValueError('Six hash-bound schedule artifacts required')
    def input_guard(manifest):
        allowed={(p.PACKAGE/manifest['roles']['validation']['fields'][k]['path']).resolve() for k in p.runner.INPUTS}
        opened=set()
        def audit(event,argv):
            if event!='open' or not isinstance(argv[0],(str,bytes)):return
            path=Path(os.fsdecode(argv[0])).resolve()
            if 'labels' in path.parts or ('data' in path.parts and {'fit','test'} & set(path.parts)):
                raise RuntimeError('Val inputs only; no labels/Fit/Test observations')
            if path.suffix in ('.npy','.npz'):
                if path not in allowed|schedules:raise RuntimeError('Unapproved observation/artifact array')
                opened.add(str(path))
        sys.addaudithook(audit);return opened
    adapter=SimpleNamespace(__file__=str(Path(__file__).resolve()),original=benchmark.v1.original,
        read_freeze=adapted_freeze,source_seal=source_seal)
    previous=benchmark.reader,benchmark.input_guard
    benchmark.reader=lambda version:adapter
    benchmark.input_guard=input_guard
    remaining=benchmark.DEADLINE-time.time()
    if remaining<=0:raise TimeoutError('13:00 UTC deadline reached')
    def alarm(signum,frame):raise TimeoutError('13:00 UTC process alarm')
    old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        benchmark.run(SimpleNamespace(freeze=args.freeze,output=args.output,freeze_reader='d4_self_distillation_six_selected_val_weights_v2'))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        benchmark.reader,benchmark.input_guard=previous


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','run'))
    parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.command=='check':
        benchmark.check()
        print(json.dumps(dict(adapter_import_pass=True,freeze_opened=False,weights_opened=False,data_opened=False,gpu_used=False)))
    else:
        if args.freeze is None or args.output is None:parser.error('Real six-completion --freeze and new --output required')
        args.freeze=args.freeze.resolve();args.output=args.output.resolve();run(args)
