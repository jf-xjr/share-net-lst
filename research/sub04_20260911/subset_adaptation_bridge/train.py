"""Thin model/dropout bridge to the unchanged matched subset-adaptation v1.

No source-selection rule or training loop is added. Formal launch requires an
external initialization-selection receipt with these fields:
paired_repeats_complete=true, initialization_selection_complete=true,
architecture, checkpoint_sha256, checkpoint_seed=20260905, plus
source_completion and paired_results, each containing path and sha256.
The paired-results JSON must declare split=validation, seed_count=3 and the
registered seeds 20260905/20260912/20260913. Relative paths use the receipt's directory.
This program never creates that receipt or chooses the initialization model.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import importlib.util
import json
import math
import sys

HERE = Path(__file__).resolve().parent
SUB04 = HERE.parent
ROOT = HERE.parents[2]
V1 = ROOT / 'research/strong_history_20260911/usage/subset_adaptation_v1/train.py'
V1_SHA256 = '4722b2ddc60e9076f64f7804fec980305a31d79d421587d7388ce0d6ac7e6ac9'
ARCHITECTURES = ('baseline', 'wide', 'current_query', 'naf_history', 'baseline_history0')
sys.path.insert(0, str(SUB04))


def load_v1():
    if hashlib.sha256(V1.read_bytes()).hexdigest() != V1_SHA256:
        raise RuntimeError('The bridge requires the reviewed, unchanged v1 trainer')
    spec = importlib.util.spec_from_file_location('subset_adaptation_original_v1', V1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(name):
    return 'baseline' if name == 'baseline_history0' else name


def model_factories(v1, architecture, dropout):
    from dropout_models import DropoutUTAE, DropoutWideUTAE
    from naf_history.model import HistoryNAFReconstructor
    if architecture == 'current_query':
        if any(value != .25 for value in dropout.values()):
            raise ValueError('The unchanged current_query class has fixed .25/.25 dropout')
        return v1.model_classes(architecture)
    cls = {'baseline': DropoutUTAE, 'wide': DropoutWideUTAE,
           'naf_history': HistoryNAFReconstructor}[architecture]
    dynamic = functools.partial(cls, **dropout)
    original = dynamic if architecture == 'naf_history' else v1.model_classes(architecture)[1]
    return dynamic, original


def selected_configuration(v1, args):
    """Read explicit finalized identities only; never inspect comparative scores."""
    receipt = json.loads(args.initialization_selection.read_text())
    if receipt.get('paired_repeats_complete') is not True or receipt.get('initialization_selection_complete') is not True:
        raise ValueError('Matched repetitions and initialization selection must finish before adaptation')
    if canonical(receipt['architecture']) != args.architecture or receipt.get('checkpoint_seed') != 20260905:
        raise ValueError('Selection receipt architecture or fixed checkpoint seed differs')
    if v1.runner.digest(args.checkpoint) != receipt['checkpoint_sha256']:
        raise ValueError('Explicit checkpoint differs from the finalized initialization selection')
    def bound_file(key):
        record = receipt[key]
        path = Path(record['path'])
        path = (args.initialization_selection.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if v1.runner.digest(path) != record['sha256']:
            raise ValueError(f'The bound {key} file changed')
        return path
    completion_path = bound_file('source_completion')
    if args.source_completion is not None and args.source_completion != completion_path:
        raise ValueError('--source-completion differs from the finalized source-run binding')
    args.source_completion = completion_path
    args.paired_results = bound_file('paired_results')
    paired = json.loads(args.paired_results.read_text())
    seeds = paired.get('seeds', [])
    if paired.get('split') != 'validation' or paired.get('seed_count') != 3 or len(seeds) != 3 or set(seeds) != {20260905, 20260912, 20260913}:
        raise ValueError('The bound evidence must use all three registered Val comparison seeds')
    completion = json.loads(completion_path.read_text())
    ck = v1.torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if ck.get('smoke_only') or cfg.get('architecture', 'baseline') != args.architecture or cfg.get('seed') != 20260905:
        raise ValueError('Expected the selected architecture\'s original seed20260905 non-smoke checkpoint')
    if completion.get('status') != 'complete' or completion.get('updates') != cfg['updates'] or not 0 < ck['step'] <= completion['updates']:
        raise ValueError('Bound original-run completion does not match the selected checkpoint')
    if completion['updates'] != 18000 or completion.get('validation_candidates') != 50:
        raise ValueError('Initialization requires the original complete 18k/50-choice training budget')
    dropout = {}
    for key in ('history_dropout', 'emissivity_dropout'):
        # Original U-TAE/wide/current_query had hard-coded .25. New NAF and
        # history0 training explicitly recorded both values in their config.
        if key not in cfg and args.architecture == 'naf_history':
            raise ValueError('NAF initialization must explicitly record both dropout probabilities')
        value = cfg.get(key, .25)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'Invalid inherited {key}')
        dropout[key] = float(value)
    if args.requested_architecture == 'baseline_history0' and dropout['history_dropout'] != 0.:
        raise ValueError('baseline_history0 must load a checkpoint trained with history_dropout=0')
    return dropout


def run(args, v1):
    # This dependency check precedes v1 setup(), datasets, or any GPU action.
    dropout = selected_configuration(v1, args)
    factories = model_factories(v1, args.architecture, dropout)
    original_config, original_seal, original_classes = v1.config, v1.source_seal, v1.model_classes

    def inherited_config(inner_args):
        cfg = original_config(inner_args)
        cfg.update(dropout)
        cfg.update(initialization_model_label=args.requested_architecture,
                   initialization_selection=str(args.initialization_selection),
                   initialization_selection_sha256=v1.runner.digest(args.initialization_selection),
                   dropout_policy='Both arms inherit the selected original checkpoint; legacy missing values are .25',
                   unchanged_usage_rule='cover_greedy_0')
        return cfg

    def extended_seal(inner_args):
        seal = original_seal(inner_args)
        for path in (Path(__file__).resolve(), SUB04 / 'dropout_models.py',
                     SUB04 / 'naf_history/model.py', args.initialization_selection, args.paired_results):
            seal[str(path)] = v1.runner.digest(path)
        return seal

    def chosen_classes(architecture):
        if architecture != args.architecture:
            raise ValueError('Unexpected architecture request inside the inherited trainer')
        return factories

    # Rebind only this privately imported module, never the original files or
    # shared factory module. All updates, schedules, scoring and saves stay v1.
    v1.config, v1.source_seal, v1.model_classes = inherited_config, extended_seal, chosen_classes
    try:
        v1.run(args)
    finally:
        v1.config, v1.source_seal, v1.model_classes = original_config, original_seal, original_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'train'))
    parser.add_argument('--architecture', choices=ARCHITECTURES)
    parser.add_argument('--arm', choices=('full_control', 'mixed'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source-completion', type=Path)
    parser.add_argument('--initialization-selection', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    v1 = load_v1()
    if args.mode == 'check':
        from dropout_models import DropoutUTAE, DropoutWideUTAE
        from naf_history.model import HistoryNAFReconstructor
        print(json.dumps(dict(status='import_check_pass', architectures=ARCHITECTURES,
                              reused_trainer=str(V1), reused_trainer_sha256=V1_SHA256,
                              reused_functions=['run', 'schedule', 'inference', 'initialization', 'restore_last'],
                              replaced_extension_points=['config', 'model_classes', 'source_seal'],
                              models_instantiated=False, checkpoints_opened=False, data_opened=False, gpu_used=False,
                              initialization_selected=False, formal_launch_dependencies_checked=False)))
        return
    for key in ('architecture', 'arm', 'checkpoint', 'initialization_selection', 'output'):
        if getattr(args, key) is None:
            parser.error(f'--{key.replace("_", "-")} is required for train')
    for key in ('checkpoint', 'source_completion', 'initialization_selection', 'output'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    args.requested_architecture = args.architecture
    args.architecture = canonical(args.architecture)
    run(args, v1)


if __name__ == '__main__':
    main()
