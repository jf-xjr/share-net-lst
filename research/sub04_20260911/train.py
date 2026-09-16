"""Sub-0.4 K candidates through the existing, matched 18k training loop."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'resources/historylst246'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PACKAGE))
spec = importlib.util.spec_from_file_location('sub04_original_runner', PACKAGE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'train'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.root = PACKAGE
    args.config, args.output = args.config.resolve(), args.output.resolve()
    cfg = json.loads(args.config.read_text())
    if cfg['architecture'] == 'baseline':
        from dropout_models import DropoutUTAE as model_class
    elif cfg['architecture'] == 'wide':
        from dropout_models import DropoutWideUTAE as model_class
    elif cfg['architecture'] == 'naf_history':
        from naf_history.model import HistoryNAFReconstructor as model_class
    else:
        raise ValueError(cfg['architecture'])
    if not all(0 <= cfg[k] < 1 for k in ('history_dropout', 'emissivity_dropout')):
        raise ValueError('Input dropout must be in [0, 1)')
    runner.HistoryUTAE = lambda: model_class(history_dropout=cfg['history_dropout'],
                                            emissivity_dropout=cfg['emissivity_dropout'])
    model_file = Path(sys.modules[model_class.__module__].__file__)
    sources = [Path(__file__), model_file, PACKAGE / 'run.py', args.config,
               ROOT / 'research/strong_history_20260911/candidates/network_review.py']
    sources += list((PACKAGE / 'historylst').rglob('*.py'))
    seal = {str(p): runner.digest(p) for p in sources}
    receipt = dict(config=cfg, source_sha256=seal, initialization='independent random',
                   objective_macro_rmse_k_lt=.4, test_opened=False,
                   evaluation='Unchanged city split, formal support and scene-city-region macro RMSE',
                   sampling_loss_optimizer_augmentation_ema='Original historylst246/run.py unchanged')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'implementation.json'
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != receipt:
            raise FileExistsError('Use an identical --resume or a new output directory')
    else:
        runner.dump(path, receipt)
    runner.setup(args.device)
    if args.device == 'cpu':
        runner.torch.set_num_threads(1)
    runner.train(args)
    for source, expected in seal.items():
        if runner.digest(source) != expected:
            raise RuntimeError(f'Executed source changed during experiment: {source}')


if __name__ == '__main__':
    main()
