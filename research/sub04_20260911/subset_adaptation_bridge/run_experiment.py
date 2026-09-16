"""Run the registered matched pair only after real three-seed Val completion."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SUB04 = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(SUB04))
import evaluate_finalist as finalist


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-plan', type=Path, required=True)
    parser.add_argument('--paired-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(20260921, 20260922), default=20260921)
    parser.add_argument('--initialization-selection', type=Path)
    parser.add_argument('--previous-evaluation', type=Path)
    args = parser.parse_args()
    args.paired_plan, args.paired_results, args.output = args.paired_plan.resolve(), args.paired_results.resolve(), args.output.resolve()
    plan, entries = finalist.plan_entries(args.paired_plan)
    result = read(args.paired_results)
    if plan['pipeline'] != 'original' or result.get('pipeline') != 'original' or result.get('split') != 'validation' or result.get('seed_count') != 3 or result.get('seeds') != [20260905, 20260912, 20260913]:
        raise ValueError('Use the real complete original-pipeline three-seed Val comparison')
    prediction_receipt = args.paired_results.parent / 'predictions_complete.json'
    if result['finalist_plan_sha256'] != sha(args.paired_plan) or result['prediction_receipt_sha256'] != sha(prediction_receipt):
        raise ValueError('Paired results are not bound to this completed plan and predictions')
    predicted = read(prediction_receipt)
    identity = lambda rows: [(row['architecture'], row['seed'], row['checkpoint_sha256']) for row in rows]
    if predicted['stage'] != 'all_predictions_sealed' or predicted['split'] != 'validation' or identity(predicted['entries']) != identity(entries):
        raise ValueError('The six evaluated original weights differ from the paired plan')
    architecture = min(('baseline', plan['candidate']), key=lambda name: (result['mean_metrics'][name]['rmse'], name != 'baseline'))
    selected = next(row for row in entries if row['architecture'] == architecture and row['seed'] == 20260905)
    budget_path = SUB04 / 'time_budget_20260912.json'
    deadline = read(budget_path)['hard_deadline_unix']
    # Preserve time for replication after a positive first screen and final Test/report.
    reserve = 170 if args.seed == 20260921 else 110
    if time.time() + reserve * 60 >= deadline:
        raise RuntimeError('Insufficient remaining deadline budget for this pair and required final work')
    if args.seed == 20260922:
        if args.previous_evaluation is None or args.initialization_selection is None:
            raise ValueError('Replication requires the original initialization and successful first screen')
        previous = read(args.previous_evaluation)
        if previous.get('cost_branch_pass') is not True or previous.get('usage_rule_numeric_screen_pass') is not True or previous.get('test_opened') is not False or previous.get('primary_policy') != 'mixed__joint__cover_greedy_0':
            raise ValueError('The first registered cost screen did not pass')
        previous_plan = args.previous_evaluation.resolve().parent.parent / 'source_plan.json'
        original_pair = read(previous_plan)
        first_run = Path(original_pair['runs']['full_control']['path']) / 'run.json'
        if previous['plan_sha256'] != sha(previous_plan) or original_pair['weights']['initialization']['sha256'] != selected['checkpoint_sha256'] or read(first_run)['config']['seed'] != 20260921:
            raise ValueError('Replication requires this initialization\'s actual seed20260921 screen')
    args.output.mkdir(parents=True, exist_ok=False)
    selection = dict(paired_repeats_complete=True, initialization_selection_complete=True,
                     architecture=architecture, checkpoint_sha256=selected['checkpoint_sha256'],
                     checkpoint_seed=20260905,
                     source_completion=dict(path=str(Path(selected['run']) / 'complete.json'), sha256=sha(Path(selected['run']) / 'complete.json')),
                     paired_results=dict(path=str(args.paired_results), sha256=sha(args.paired_results)),
                     selection_rule='Lower original three-seed mean Val RMSE; exact tie favors baseline. Always use seed20260905 original best.pt.',
                     mean_metrics=result['mean_metrics'], test_opened=False)
    if args.initialization_selection is None:
        args.initialization_selection = args.output / 'initialization_selection.json'
        dump(args.initialization_selection, selection)
    else:
        args.initialization_selection = args.initialization_selection.resolve()
        if read(args.initialization_selection) != selection:
            raise ValueError('Replication must retain exactly the original initialization selection')
    files = [Path(__file__), HERE / 'train.py', HERE / 'evaluate.py', HERE / 'benchmark.py',
             args.paired_plan, args.paired_results, args.initialization_selection, budget_path,
             SUB04 / 'user_pause.json', SUB04 / 'user_release.json']
    if args.previous_evaluation is not None:
        files.append(args.previous_evaluation.resolve())
    sources = {str(path): sha(path) for path in files}
    state = dict(status='ready', pid=os.getpid(), seed=args.seed, architecture=architecture,
                 started=time.time(), deadline=deadline, source_sha256=sources, jobs=[], test_opened=False)

    def save():
        dump(args.output / 'state.json', state)

    def job(name, command):
        if time.time() >= deadline or any(sha(path) != digest for path, digest in sources.items()):
            raise RuntimeError('Deadline reached or bound protocol/user stop record changed')
        row = dict(name=name, command=command, status='running', started=time.time())
        state['jobs'].append(row)
        state['status'] = name
        with (args.output / f'{name}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            save()
            print(json.dumps(dict(event='started', **row)), flush=True)
            code = child.wait()
        row.update(exit_code=code, finished=time.time(), status='complete' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(f'{name} exited {code}; no automatic retry')

    try:
        save()
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU is occupied; run only after the paired queue releases it')
        runs = {}
        for arm in ('full_control', 'mixed'):
            run = args.output / arm
            job(arm, [sys.executable, '-I', str(HERE / 'train.py'), 'train', '--architecture', architecture,
                      '--arm', arm, '--checkpoint', selected['checkpoint'], '--initialization-selection',
                      str(args.initialization_selection), '--output', str(run), '--seed', str(args.seed), '--device', 'cuda'])
            runs[arm] = dict(path=str(run), files={name: sha(run / name) for name in ('run.json', 'complete.json', 'validation.json')})
        weights = dict(initialization=dict(path=selected['checkpoint'], sha256=selected['checkpoint_sha256']))
        for arm in runs:
            for label, filename in [('joint', 'best.pt'), ('full9', 'best_full9.pt'), ('greedy3', 'best_greedy3.pt')]:
                path = args.output / arm / filename
                weights[f'{arm}__{label}'] = dict(path=str(path), sha256=sha(path))
        source_plan = args.output / 'source_plan.json'
        dump(source_plan, dict(architecture=architecture, manifest_sha256=plan['manifest_sha256'], runs=runs,
              initialization_selection=dict(path=str(args.initialization_selection), sha256=sha(args.initialization_selection)), weights=weights))
        predictions, timings, scores = (args.output / name for name in ('predictions', 'timings', 'scores'))
        job('source_predict', [sys.executable, '-I', str(HERE / 'evaluate.py'), 'predict', '--plan', str(source_plan),
                              '--output', str(predictions), '--device', 'cuda', '--batch-size', '1'])
        job('source_timing', [sys.executable, '-I', str(HERE / 'benchmark.py'), '--plan', str(source_plan),
                             '--predictions-receipt', str(predictions / 'source_cohort_complete.json'), '--output', str(timings)])
        job('source_score', [sys.executable, '-I', str(HERE / 'evaluate.py'), 'score', '--plan', str(source_plan),
                            '--predictions', str(predictions), '--output', str(scores), '--timing', str(timings / 'benchmark.json')])
        evaluation = read(scores / 'evaluation.json')
        state.update(status='complete_need_scientific_review', evaluation=str(scores / 'evaluation.json'),
                     cost_branch_pass=evaluation['cost_branch_pass'],
                     usage_rule_numeric_screen_pass=evaluation['usage_rule_numeric_screen_pass'], full_goal_complete=False)
    except Exception as error:
        state.update(status='stopped_need_review', error=repr(error))
        raise
    finally:
        state['last_update'] = time.time()
        save()


if __name__ == '__main__':
    main()
