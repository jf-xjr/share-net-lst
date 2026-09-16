"""Run the two prespecified ablation seeds sequentially on the shared GPU."""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
assert json.loads((HERE / 'thermal_bypass/intervention_check.json').read_text())[
    'first_scale_historical_feature_and_moment_injection_unchanged']
for seed in (20260914, 20260915):
    folder = HERE / 'thermal_bypass/runs' / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'complete.json').exists(), 'Completed run already exists'
    with (folder / 'stdout.log').open('w') as stream:
        process = subprocess.Popen([sys.executable, str(HERE / 'train_thermal_bypass.py'),
                                    'train', '--seed', str(seed)], stdout=stream,
                                   stderr=subprocess.STDOUT)
        print(json.dumps({'seed': seed, 'pid': process.pid, 'started': time.time()}), flush=True)
        code = process.wait()
    if code:
        print((folder / 'stdout.log').read_text()[-5000:], flush=True)
        raise SystemExit(code)
    print((folder / 'complete.json').read_text(), flush=True)
print('Both fixed-budget thermal bypass ablations completed.', flush=True)
