"""Execute only the three explicitly handed-over runs; never change their trainer."""
from pathlib import Path
import hashlib, json, math, subprocess, sys, time

ROOT = Path('/home/jf_xjr/uhi_cdc')
HERE = ROOT / 'research/sub04_20260913/compact_recovery'
LOGS = Path(__file__).resolve().parent
PY = '/home/jf_xjr/miniconda3/envs/mmwsl/bin/python'
TRAIN = HERE / 'train.py'
BUDGET = HERE.parent / 'four_hour_extension.json'
JOBS = [('compact', 20260913), ('baseline', 20260912), ('baseline', 20260913)]
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def require(ok, why):
    if not ok: raise RuntimeError(why)
def write(p, v): Path(p).write_text(json.dumps(v, indent=2) + '\n')
def idle():
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    require(not r.stdout.strip(), 'GPU occupied; stop without interrupting any process')
def sources():
    for p, h in PARENT['source_sha256'].items(): require(sha(p) == h, 'Original source changed: ' + p)
def validate(out, architecture, seed):
    require((out/'complete.json').is_file() and not (out/'interrupted.json').exists(), 'Incomplete training')
    done, run, rows = [read(out/n) for n in ('complete.json', 'run.json', 'validation.json')]
    require(done['status'] == 'complete' and done['architecture'] == architecture and done['original_seed'] == seed, 'Wrong completed job')
    require(done['updates'] == 6000 and done['validation_weight_candidates'] == 26 and run['validation_candidates'] == 26, 'Wrong update/selection budget')
    require(done['config'] == run['config'] and done['run_sha256'] == sha(out/'run.json'), 'Runtime binding differs')
    require(done['source_sha256'] == run['source_sha256'] == PARENT['source_sha256'], 'Original common source seal differs')
    for p, h in done['source_sha256'].items(): require(sha(p) == h, 'Changed source: '+p)
    require(sha(out/'best.pt') == done['selected_checkpoint_sha256'], 'Selected checkpoint hash differs')
    require([(r['step'], r['weights']) for r in rows] == [(s,w) for s in range(0,6001,500) for w in ('raw','ema')], 'Incomplete 26 candidates')
    require(all(math.isfinite(r['rmse']) for r in rows), 'Nonfinite validation')
    selected = min(rows, key=lambda r:r['rmse'])
    require((selected['step'], selected['weights']) == (done['selected_step'], done['selected_weights']), 'Wrong selected candidate')
    require(abs(done['selected_fp32']['macro']['rmse']-selected['rmse']) < 1e-5, 'Selected FP32 verification differs')
    c = done['counters']
    require(c['updates'] == c['finite_updates'] == 6000 and c['forward_calls'] == c['backward_calls'] == 6000+c['amp_backoffs'] and c['validation_forward_calls'] == 621, 'Incomplete actual work')
    require(done['actual_start_unix'] == run['actual_start_unix'] and done['actual_end_unix'] <= run['deadline_unix'] <= STOP and 0 < done['seconds'] <= 2700, 'Execution deadline violated')
    require(abs(done['actual_end_unix']-done['actual_start_unix']-done['seconds']) < .1, 'Invalid actual duration')
    require(done['test_opened'] is False and done['teacher_used_at_inference'] is False and done['goal1_pass'] is False, 'Premature success or changed inference')
    return dict(completion_sha256=sha(out/'complete.json'), checkpoint_sha256=done['selected_checkpoint_sha256'], seconds=done['seconds'], selected_step=done['selected_step'], selected_weights=done['selected_weights'], validation_rmse=done['selected_fp32']['macro']['rmse'], counters=c)

PARENT = read(HERE/'runs/baseline_20260905/complete.json')
STOP = read(BUDGET)['stop_training_search_by_unix']
require(PARENT['status']=='complete' and PARENT['selected_checkpoint_sha256']=='1318079db4ac12a18ace60b45bf0ed514a70a740cf3c54e2a6a97ac65b67046e', 'Wrong root handover completion')
require(STOP == 1789298796 and sha(BUDGET)=='3ffe172c7c6c4967a49f77066174c5f39ee5bad388a5bc6fffc1784d4311c8d1', 'Original budget changed')
require(not (LOGS/'execution.json').exists(), 'One execution only')
for a, s in JOBS: require(not (HERE/f'runs/{a}_{s}').exists(), 'Refuse existing run')
ledger = dict(status='running', started_unix=time.time(), stop_training_unix=STOP, controller_sha256=sha(__file__), trainer_sha256=sha(TRAIN), budget_sha256=sha(BUDGET), jobs=[])
write(LOGS/'execution.json', ledger)
try:
    for architecture, seed in JOBS:
        require(time.time() < STOP, 'Training stop reached')
        sources(); idle()
        out = HERE/f'runs/{architecture}_{seed}'
        command = [PY, '-B', str(TRAIN), '--architecture', architecture, '--original-seed', str(seed), '--output', str(out)]
        entry = dict(architecture=architecture, seed=seed, output=str(out), command=command, started_unix=time.time(), stdout_log=str(LOGS/f'{architecture}_{seed}.log'))
        ledger['jobs'].append(entry)
        with Path(entry['stdout_log']).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            entry['pid']=child.pid; write(LOGS/'execution.json', ledger)
            print(json.dumps(dict(event='actual_child_started', **entry)), flush=True)
            for line in child.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            entry['exit_code']=child.wait(); entry['ended_unix']=time.time()
        write(LOGS/'execution.json', ledger)
        require(entry['exit_code']==0, 'Actual training child failed; no next run')
        entry['validated_completion']=validate(out, architecture, seed)
        idle(); write(LOGS/'execution.json', ledger)
        print(json.dumps(dict(event='actual_child_completed_and_validated', **entry)), flush=True)
    ledger.update(status='complete_three_actual_runs', ended_unix=time.time())
    write(LOGS/'execution.json', ledger)
    print(json.dumps(dict(event='all_three_complete_GPU_released', receipt=str(LOGS/'execution.json'))), flush=True)
except BaseException as exc:
    ledger.update(status='stopped_after_error', error=repr(exc), ended_unix=time.time())
    write(LOGS/'execution.json', ledger)
    raise
