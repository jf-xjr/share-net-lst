"""Match the existing learned-weight scratch controls except for the bypass."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import argparse
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / 'resources/historylst246'
CONTROL = ROOT / 'research/near_neighbor_attribution_20260914'
sys.path[:0] = [str(HERE), str(PACKAGE)]
spec = importlib.util.spec_from_file_location('bypass_portable_runner', PACKAGE / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
from thermal_bypass_model import construct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'train'])
    parser.add_argument('--seed', type=int, choices=[20260914, 20260915], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    control = CONTROL / 'attribution' / f'learned_{args.seed}'
    cfg = json.loads((control / 'config.json').read_text())
    cfg['architecture'] = 'without_thermal_bypass'
    args.device = 'cuda'
    args.root = PACKAGE
    args.output = HERE / 'thermal_bypass' / ('smoke' if args.mode == 'smoke' else 'runs') / str(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    args.config = args.output / 'config.json'
    if args.config.exists():
        assert json.loads(args.config.read_text()) == cfg
    else:
        r.dump(args.config, cfg)
    # All historical model/loader sources must still match the old control.
    control_hashes = json.loads((control / 'implementation.json').read_text())
    for path, value in control_hashes.items():
        assert r.digest(Path(path)) == value, path
    r.HistoryUTAE = construct
    original_inference = r.inference
    r.inference = lambda model, data, device, batch_size, amp=False: original_inference(
        model, data, device, batch_size, amp=False)
    original_augment = r.augment

    def crop_augment(batch, code):
        batch = original_augment(batch, code)
        rng = r.np.random.default_rng(r.torch.initial_seed())
        top, left = (int(x) * 8 for x in rng.integers(0, 5, size=2))
        return {k: (v if k == 'context' else
                    v[..., top//4:top//4+32, left//4:left//4+32] if k == 'coarse' else
                    v[..., top:top+128, left:left+128]).contiguous()
                for k, v in batch.items()}

    r.augment = crop_augment
    sources = {**control_hashes, **{str(p): r.digest(p) for p in
               [Path(__file__), HERE / 'thermal_bypass_model.py']}}
    receipt = args.output / 'implementation.json'
    if receipt.exists():
        assert json.loads(receipt.read_text()) == sources
    else:
        r.dump(receipt, sources)
    r.setup('cuda')
    r.torch.backends.cuda.matmul.allow_tf32 = False
    r.torch.backends.cudnn.allow_tf32 = False
    r.train(args)
    assert all(r.digest(Path(p)) == h for p, h in sources.items())


if __name__ == '__main__':
    main()
