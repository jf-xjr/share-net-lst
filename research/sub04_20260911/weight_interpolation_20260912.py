"""Three frozen parameter interpolations; one model per seed, Val only.

prepare reads the completed original six-run plan and best/last checkpoints,
without opening Val arrays or scores. predict/score reuse the unchanged common
paired evaluator. Buffers are kept from the originally selected best; there is
no BN recalibration and this is not standard SWA training or a prediction ensemble.
"""
from pathlib import Path
import argparse
import copy
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_finalist as original
import torch

paired = original.paired
ALPHAS = (.25, .5, .75)


def seal():
    return dict(original.source_seal(), **{str(Path(__file__).resolve()): paired.sha(__file__)})


def identities(entries):
    return [(row['architecture'], row['seed'], row['checkpoint_sha256']) for row in entries]


def prepare(args):
    plan = json.loads(args.plan.read_text())
    expected = [(seed, role) for seed in paired.SEEDS for role in ('baseline', 'naf_history')]
    if (plan['pipeline'] != 'original' or plan['candidate'] != 'naf_history'
            or [(r['seed'], r['architecture']) for r in plan['runs']] != expected
            or plan['manifest_sha256'] != paired.sha(paired.PACKAGE / 'manifest.json')):
        raise ValueError('Require the original uncalibrated three-pair plan and unchanged manifest')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    output = dict(status='all_three_interpolations_frozen', alphas=list(ALPHAS), manifest_sha256=plan['manifest_sha256'],
        original_plan=dict(path=str(args.plan), sha256=paired.sha(args.plan)), evaluator_source_sha256=seal(),
        formula='theta(alpha)=(1-alpha)*original_selected_best+alpha*last_18000_ema; learned parameters only',
        buffers='All buffers, including BatchNorm statistics, remain exactly the original selected-best buffers',
        caveat='Baseline913 selected best is RAW, not EMA. No best-EMA checkpoint was retained there. '
               'NAF endpoints span steps9000/18000; baseline endpoints span17250/18000. This is parameter interpolation, not SWA-trained weights.',
        ensemble_used=False, bn_recalibration=False, validation_opened=False, test_opened=False, candidates={})
    for alpha in ALPHAS:
        output['candidates'][str(alpha)] = []
    for row in plan['runs']:
        run = (args.plan.parent / row['run']).resolve()
        before_path, after_path = run / 'best.pt', run / 'last.pt'
        before_sha, after_sha = paired.sha(before_path), paired.sha(after_path)
        if before_sha != row['files']['best.pt']:
            raise ValueError('Original selected best changed')
        before = torch.load(before_path, map_location='cpu', weights_only=False)
        after = torch.load(after_path, map_location='cpu', weights_only=False)
        if before['config'] != after['config'] or after['step'] != 18000 or before['config']['updates'] != 18000:
            raise ValueError('Endpoints must come from the same completed original 18k run')
        state, tail = before['state_dict'], after['ema']
        if state.keys() != tail.keys() or any(v.shape != tail[k].shape or v.dtype != tail[k].dtype for k, v in state.items()):
            raise ValueError('Weight endpoints are incompatible')
        cls = original.screens.HistoryUTAE if row['architecture'] == 'baseline' else original.screens.HistoryNAFReconstructor
        model = cls()
        parameters = set(dict(model.named_parameters()))
        if not parameters <= state.keys():
            raise ValueError('Model parameter identities differ from the selected checkpoint')
        for alpha in ALPHAS:
            mixed = {key: value.lerp(tail[key], alpha) if key in parameters else value.clone() for key, value in state.items()}
            model.load_state_dict(mixed, strict=True)
            provenance = dict(alpha=alpha, selected_best=dict(path=str(before_path), sha256=before_sha,
                step=before['step'], weights=before['weights']), last=dict(path=str(after_path), sha256=after_sha, step=18000, weights='ema'),
                parameter_count=sum(state[key].numel() for key in parameters), nonparameter_buffers_unchanged=True)
            filename = f"{row['architecture']}_{row['seed']}_alpha{int(alpha*100):02d}.pt"
            target = args.output / filename
            paired.runner.save(target, dict(state_dict=mixed, config=before['config'], weight_interpolation=provenance))
            entry = dict(architecture=row['architecture'], seed=row['seed'], run=str(run), checkpoint=str(target),
                checkpoint_sha256=paired.sha(target), config=before['config'], interpolation=provenance,
                additional_training_updates=0, additional_inference_model_count=0)
            output['candidates'][str(alpha)].append(entry)
        del before, after, state, tail, model
    paired.dump(args.output / 'interpolation_plan.json', output)
    print(json.dumps(dict(status=output['status'], plan=str(args.output / 'interpolation_plan.json'),
        candidates=3, validation_opened=False, test_opened=False, gpu_used=False)))


def execute(args):
    plan = json.loads(args.plan.read_text()); digest = paired.sha(args.plan)
    if (plan['status'] != 'all_three_interpolations_frozen' or plan['alphas'] != list(ALPHAS)
            or plan['evaluator_source_sha256'] != seal()
            or paired.sha(paired.PACKAGE / 'manifest.json') != plan['manifest_sha256']):
        raise ValueError('Frozen interpolation plan or evaluation source changed')
    for rows in plan['candidates'].values():
        for row in rows:
            if paired.sha(row['checkpoint']) != row['checkpoint_sha256']:
                raise ValueError('A predeclared interpolation checkpoint changed')
    entries = plan['candidates'][str(args.alpha)]
    if args.command == 'predict':
        if args.device != 'cuda':
            raise ValueError('Use common GPU FP32 prediction')
        paired.setup('cuda')
    else:
        receipt = json.loads((args.output / 'predictions_complete.json').read_text())
        if (receipt.get('interpolation_plan_sha256') != digest or receipt.get('alpha') != args.alpha
                or receipt.get('evaluator_source_sha256') != seal() or receipt.get('device') != 'cuda'
                or receipt.get('tf32') is not False or identities(receipt['entries']) != identities(entries)):
            raise ValueError('Prediction identities differ from the frozen candidate')
    previous = paired.pairs, paired.load_model, paired.dump
    def bound_dump(path, value):
        if Path(path).name in ('prediction_started.json', 'predictions_complete.json', 'results.json'):
            value = dict(value, interpolation_plan_sha256=digest, alpha=args.alpha, evaluator_source_sha256=seal(),
                candidate_count=3, buffers='original selected-best buffers retained', bn_recalibration=False,
                scientific_goal_complete=False, original_results_unchanged=True)
            if Path(path).name == 'results.json':
                value['absolute_three_seed_mean_lt_042'] = value['mean_metrics']['naf_history']['rmse'] < .42
        previous[2](path, value)
    paired.pairs = lambda unused: copy.deepcopy(entries)
    paired.load_model, paired.dump = original.load_model, bound_dump
    args.queue, args.split, args.test_freeze = args.plan, 'validation', None
    try:
        (paired.predict if args.command == 'predict' else paired.evaluate)(args)
    finally:
        paired.pairs, paired.load_model, paired.dump = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'prepare', 'predict', 'score'))
    parser.add_argument('--plan', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--alpha', type=float, choices=ALPHAS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.command == 'check':
        a, b = torch.tensor([1., 3.]), torch.tensor([3., 1.])
        assert torch.equal(a.lerp(b, .5), torch.tensor([2., 2.]))
        print(json.dumps(dict(status='syntax_import_interpolation_check_pass', alphas=ALPHAS,
            weights_opened=False, validation_opened=False, test_opened=False, gpu_used=False))); return
    if args.plan is None or args.output is None or (args.command != 'prepare' and args.alpha is None):
        parser.error('--plan/--output are required; predict/score also require --alpha')
    args.plan, args.output = args.plan.resolve(), args.output.resolve()
    torch.set_num_threads(1)
    (prepare if args.command == 'prepare' else execute)(args)


if __name__ == '__main__':
    main()
