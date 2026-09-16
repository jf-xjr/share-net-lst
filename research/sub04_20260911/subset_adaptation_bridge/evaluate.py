"""Seven fixed checkpoints, five fixed source sets, one predeclared main rule.

Plan schema: architecture, manifest_sha256, initialization_selection:{path,
sha256}, runs:{full_control:{path,files:{run.json:sha,complete.json:sha,
validation.json:sha}},mixed:{...}}, weights:{name:{path,sha256},...}.
Names are initialization and {full_control,mixed}__{joint,full9,greedy3}.
Paths are relative to the plan. This entry never chooses initialization, rules,
or new weights. Predict and score must run as separate processes.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SUB04, ROOT = HERE.parent, HERE.parents[2]
USAGE = ROOT / 'research/strong_history_20260911/usage'
PACKAGE = ROOT / 'resources/historylst246'
for path in (SUB04, USAGE):
    sys.path.insert(0, str(path))
import numpy as np
import torch
import predict_candidate_sources as sources
import fit_policy as policies
import policy_acceptance as acceptance
from dropout_models import DropoutUTAE, DropoutWideUTAE
from naf_history.model import HistoryNAFReconstructor

spec = importlib.util.spec_from_file_location('matched_subset_training_bridge', HERE / 'train.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
SELECTIONS = {'joint': ('best.pt', 'mean_full9_greedy3_macro_rmse'),
              'full9': ('best_full9.pt', 'full9_macro_rmse'),
              'greedy3': ('best_greedy3.pt', 'greedy3_macro_rmse')}
NAMES = ('initialization',) + tuple(f'{arm}__{key}' for arm in ('full_control', 'mixed') for key in SELECTIONS)
RULE = dict(name='cover_greedy_0', mode='greedy3', conditions=[['greedy_uncovered_reliable', '<=', 0.]])
SIMPLE_RULES = {config['name']: config for config in policies.candidates()
                if config['name'] in ('coverage_recent_0.95', 'nearest_0.95_0.9_16.5')}
if set(SIMPLE_RULES) != {'coverage_recent_0.95', 'nearest_0.95_0.9_16.5'}:
    raise RuntimeError('The two exact predeclared simple rules are missing')
PRIMARY = 'mixed__joint__cover_greedy_0'
digest, dump, crop, forward = sources.digest, sources.dump, sources.crop, sources.forward


def model_classes(architecture):
    architecture = bridge.canonical(architecture)
    if architecture == 'naf_history':
        return HistoryNAFReconstructor, HistoryNAFReconstructor
    if architecture in ('baseline', 'wide'):
        return {'baseline': DropoutUTAE, 'wide': DropoutWideUTAE}[architecture], sources.model_classes(architecture)[1]
    return sources.model_classes(architecture)


def load_models(architecture, checkpoint):
    """Strict CPU FP32 eval models; native K and the original full9 reference."""
    architecture = bridge.canonical(architecture)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if cfg.get('architecture', 'baseline') != architecture or ck.get('smoke_only'):
        raise ValueError('Checkpoint architecture or non-smoke identity differs')
    models = []
    for cls in model_classes(architecture):
        model = cls().float().eval()
        model.load_state_dict(ck['state_dict'], strict=True)
        for key in ('history_dropout', 'emissivity_dropout'):
            setattr(model, key, cfg.get(key, .25))
        models.append(model)
    return tuple(models)


def read_plan(plan_path):
    """Validate actual completed runs; no labels, model construction or inference."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    def resolve(path):
        return (plan_path.parent / path).resolve()
    def bound(record):
        path = resolve(record['path'])
        if digest(path) != record['sha256']:
            raise ValueError(f'Plan-bound file changed: {path}')
        return path
    if plan['architecture'] not in bridge.ARCHITECTURES or set(plan['weights']) != set(NAMES) or set(plan['runs']) != {'full_control', 'mixed'}:
        raise ValueError('Expected exactly the predeclared architecture, seven weights and two arms')
    if digest(PACKAGE / 'manifest.json') != plan['manifest_sha256']:
        raise ValueError('Common manifest changed')
    selection, initial = bound(plan['initialization_selection']), bound(plan['weights']['initialization'])
    architecture = bridge.canonical(plan['architecture'])
    args = SimpleNamespace(initialization_selection=selection, checkpoint=initial,
        architecture=architecture, requested_architecture=plan['architecture'], source_completion=None)
    v1 = bridge.load_v1()
    dropout = bridge.selected_configuration(v1, args)
    entries = [dict(name='initialization', architecture=architecture, checkpoint=initial,
                    checkpoint_sha256=digest(initial), **dropout)]
    common = None
    for arm in ('full_control', 'mixed'):
        item = plan['runs'][arm]
        folder = resolve(item['path'])
        if set(item['files']) != {'run.json', 'complete.json', 'validation.json'}:
            raise ValueError('Bind exactly run.json, complete.json and validation.json for each arm')
        records = {}
        for name, sha in item['files'].items():
            path = folder / name
            if digest(path) != sha:
                raise ValueError(f'Completed run record changed: {path}')
            records[name] = json.loads(path.read_text())
        run, done, rows = (records[name] for name in ('run.json', 'complete.json', 'validation.json'))
        cfg = run['config']
        if done.get('status') != 'complete' or done.get('updates') != 3000 or done.get('arm') != arm or done.get('test_opened') is not False:
            raise ValueError('Both actual arms must finish 3000 updates without Test')
        if done.get('validation_weight_candidates') != 26 or done.get('validation_action_scores') != 52:
            raise ValueError('Expected 26 weight candidates and 52 action scores')
        expected = dict(architecture=architecture, arm=arm, updates=3000, batch_size=4, lr=1e-4,
            warmup=100, weight_decay=1e-4, validation_interval=250, evaluation_batch=2,
            initialization_sha256=digest(initial), initialization_selection_sha256=digest(selection),
            unchanged_usage_rule='cover_greedy_0', **dropout)
        if any(cfg.get(k) != value for k, value in expected.items()) or run.get('smoke_only'):
            raise ValueError('Completed adaptation differs from the matched protocol or initialization')
        current = ({k: value for k, value in cfg.items() if k != 'arm'}, run['schedule'])
        if common is not None and current != common:
            raise ValueError('Arms must share initialization, dropout, Fit/D4 and source schedules')
        common = current
        if run['schedule']['source_schedule_counts'] != dict(full9=1500, greedy3=1500):
            raise ValueError('The independent mixed schedule must contain 1500 of each action')
        v1.verify_seal(run['source_sha256'], initial, digest(initial))
        if [(r['step'], r['weights']) for r in rows] != [(step, w) for step in range(0, 3001, 250) for w in ('raw', 'ema')]:
            raise ValueError('Validation weight schedule differs from the original 26 choices')
        for key, (filename, metric) in SELECTIONS.items():
            name = f'{arm}__{key}'
            checkpoint = bound(plan['weights'][name])
            if checkpoint != folder / filename:
                raise ValueError('Each explicit weight must be its registered original best file')
            def value(row):
                scores = row['validation_macro_rmse']
                return (scores['full9'] + scores['greedy3']) / 2 if key == 'joint' else scores[key]
            chosen = min(rows, key=value)
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if ck.get('smoke_only') or ck['config'] != cfg or ck['selection_metric'] != metric or ck['selection_score'] != value(chosen) or (ck['step'], ck['weights']) != (chosen['step'], chosen['weights']):
                raise ValueError('Checkpoint differs from its original AMP-selected best; no FP32 reselection allowed')
            if ck['initialization']['checkpoint_sha256'] != digest(initial):
                raise ValueError('Checkpoint initialization differs')
            entries.append(dict(name=name, architecture=architecture, checkpoint=checkpoint,
                                checkpoint_sha256=digest(checkpoint), **dropout))
    return plan, entries


def code_seal():
    paths = {Path(__file__).resolve(), HERE / 'train.py'}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        # Torch exposes synthetic modules (e.g. torch.classes) with relative
        # pseudo-filenames. Only real project source files have bytes to seal.
        if path and Path(path).is_file() and str(Path(path).resolve()).startswith(str(ROOT) + '/') and Path(path).suffix == '.py':
            paths.add(Path(path).resolve())
    return {str(path): digest(path) for path in sorted(paths)}


def predict(args):
    sources.guard_inputs_only()
    _, entries = read_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    seal = dict(stage='seven_weight_predictions_only', plan_sha256=digest(args.plan), code_sha256=code_seal(),
                labels_opened=False, test_opened=False, primary_policy=PRIMARY, weights={},
                entries=[dict(name=e['name'], checkpoint_sha256=e['checkpoint_sha256']) for e in entries])
    dump(args.output / 'prediction_run.json', seal)
    for entry in entries:
        name = entry['name']
        with (args.output / (name + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), 'predict-one',
                '--plan', str(args.plan), '--entry', name, '--output', str(args.output / name),
                '--device', args.device, '--batch-size', str(args.batch_size)], stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = args.output / name / 'source_predictions_complete.json'
        sources.verify(receipt.parent)
        seal['weights'][name] = dict(checkpoint_sha256=entry['checkpoint_sha256'], receipt_sha256=digest(receipt))
    for path, sha in seal['code_sha256'].items():
        if digest(path) != sha:
            raise ValueError('Evaluation source changed while predictions were running')
    if digest(args.plan) != seal['plan_sha256']:
        raise ValueError('Evaluation plan changed while predictions were running')
    dump(args.output / 'source_cohort_complete.json', seal)


def verify(plan_path, folder):
    plan, entries = read_plan(plan_path)
    receipt = json.loads((folder / 'source_cohort_complete.json').read_text())
    if receipt.get('stage') != 'seven_weight_predictions_only' or receipt.get('primary_policy') != PRIMARY or receipt['plan_sha256'] != digest(plan_path) or set(receipt['weights']) != set(NAMES) or receipt['labels_opened'] or receipt['test_opened']:
        raise ValueError('Expected the complete seven-weight input-only prediction seal')
    if receipt['entries'] != [dict(name=e['name'], checkpoint_sha256=e['checkpoint_sha256']) for e in entries]:
        raise ValueError('Explicit prediction-entry identities differ from the plan')
    for path, sha in receipt['code_sha256'].items():
        if digest(path) != sha:
            raise ValueError(f'Evaluation source changed: {path}')
    tables = None
    for entry in entries:
        root = folder / entry['name']
        bound = receipt['weights'][entry['name']]
        if bound['receipt_sha256'] != digest(root / 'source_predictions_complete.json') or bound['checkpoint_sha256'] != entry['checkpoint_sha256']:
            raise ValueError('Weight prediction seal changed')
        child = sources.verify(root)
        if child['checkpoint_sha256'] != entry['checkpoint_sha256'] or child['manifest_sha256'] != plan['manifest_sha256']:
            raise ValueError('Prediction used a different checkpoint or manifest')
        if tables is not None and tables != child['input_table_sha256']:
            raise ValueError('Source rules or input features differ across weights')
        tables = child['input_table_sha256']
    return receipt


def score(args):
    verify(args.plan, args.predictions)  # Every input-only prediction finishes before any label opens.
    args.output.mkdir(parents=True, exist_ok=False)
    all_risks, all_actions, summaries, records_by_split = {}, {}, {}, {}
    for split in ('fit', 'validation'):
        risks, actions, summary = {}, {}, {}
        for name in NAMES:
            source = args.predictions / name
            out = args.output / name
            out.mkdir(exist_ok=True)
            records, arrays = policies.risk_arrays(source / 'predictions', split, out)
            rows, features = policies.read_features(source / (split + '_features.csv'))
            if [r['scene_id'] for r in rows] != [r['scene_id'] for r in records]:
                raise ValueError('Feature and label scene order differs')
            features['greedy_uncovered_reliable'] = np.maximum(0, features['all9_reliable'] - features['greedy3_reliable'])
            applied_rules = dict(cover_greedy_0=RULE)
            if name == 'mixed__joint':
                applied_rules.update(SIMPLE_RULES)
            source_counts = {mode: np.full(len(records), count) for mode, count in sources.COUNTS.items()}
            for rule_name, config in applied_rules.items():
                active = policies.mask(config, features)
                arrays[rule_name] = {k: np.where(active, arrays[config['mode']][k], arrays['full9'][k]) for k in arrays['full9']}
                source_counts[rule_name] = np.where(active, sources.COUNTS[config['mode']], 9)
            for strategy, values in arrays.items():
                method = name + '__' + strategy
                counts = source_counts[strategy]
                identities = [{k: r[k] for k in ('scene_id', 'city', 'region')} for r in records]
                risks[method] = [dict(**identity, **{k: float(v[i]) for k, v in values.items()}) for i, identity in enumerate(identities)]
                actions[method] = [dict(**identity, source_count=int(counts[i])) for i, identity in enumerate(identities)]
                summary[method] = {k: policies.macro(v, records) for k, v in values.items()}
                summary[method]['source_count'] = policies.macro(counts, records)
            if name == 'mixed__joint':
                methods = [name + '__' + rule_name for rule_name in applied_rules]
                dump(out / (split + '_rule_records.json'), dict(
                    risks={m: risks[m] for m in methods}, actions={m: actions[m] for m in methods}))
            records_by_split[split] = records
        all_risks[split], all_actions[split], summaries[split] = risks, actions, summary
    fixed = [name + '__' + strategy for name in NAMES for strategy in sources.MODES]
    simple = ['mixed__joint__' + name for name in SIMPLE_RULES]
    strongest = min(fixed, key=lambda m: summaries['validation'][m]['rmse'])
    comparisons = {}
    for split, risks in all_risks.items():
        records, macro = records_by_split[split], summaries[split]
        regions = {r['city']: r['region'] for r in records}
        rng = np.random.default_rng(20260911)
        sizes = {g: sum(r == g for r in regions.values()) for g in sorted(set(regions.values()))}
        draws = {g: rng.integers(0, n, size=(20000, n)) for g, n in sizes.items()} if min(sizes.values()) >= 2 else None
        city = {m: acceptance._city_values(acceptance._index(rows, 'scene'), 'rmse') for m, rows in risks.items()}
        contrasts = {m: acceptance._paired_interval(city[PRIMARY], city[m], regions, draws) for m in fixed}
        simple_contrasts = {m: dict(acceptance._paired_interval(city[PRIMARY], city[m], regions, draws),
            source_saving_fraction=1 - macro[PRIMARY]['source_count'] / macro[m]['source_count']) for m in simple}
        dominated = [m for m in fixed if all(macro[m][k] <= macro[PRIMARY][k] + 1e-12 for k in ('rmse', 'source_count'))
                     and any(macro[m][k] < macro[PRIMARY][k] - 1e-12 for k in ('rmse', 'source_count'))]
        comparisons[split] = dict(comparisons=contrasts, source_accuracy_dominating_fixed=dominated,
            simple_rule_comparisons=simple_contrasts,
            source_accuracy_dominating_simple_rules=[m for m in simple
                if all(macro[m][k] <= macro[PRIMARY][k] + 1e-12 for k in ('rmse', 'source_count'))
                and any(macro[m][k] < macro[PRIMARY][k] - 1e-12 for k in ('rmse', 'source_count'))],
            necessary_accuracy_gates={m: c['degradation_upper95_k'] is not None and c['degradation_upper95_k'] <= acceptance.RMSE_MARGIN_K + 1e-12 for m, c in contrasts.items()})
    result = dict(status='Fit/Val accuracy and source screen; measured cost acceptance pending', primary_policy=PRIMARY,
        rule=RULE, simple_rules=SIMPLE_RULES, simple_methods=simple, fixed_methods=fixed,
        strongest_fixed_validation=strongest, macro=summaries, **comparisons,
        plan_sha256=digest(args.plan), predictions_receipt_sha256=digest(args.predictions / 'source_cohort_complete.json'),
        cost_branch_pass=False, usage_rule_numeric_screen_pass=False, all_user_conditions_assessed=False,
        test_opened=False, code_sha256=code_seal(),
        caveat='Val selected these weights. City bootstrap is descriptive, pointwise, and excludes training and selection uncertainty. Other weight/rule combinations are descriptive; no rule or weight search. Source-count dominance is not measured-time dominance.')
    if args.timing:
        timing = json.loads(args.timing.read_text())
        if timing.get('status') != 'complete' or timing.get('split') != 'validation' or timing.get('labels_opened') is not False or timing.get('test_opened') is not False:
            raise ValueError('Expected completed input-only Val timing')
        for path, sha in timing['source_sha256'].items():
            if digest(path) != sha:
                raise ValueError('Timing source changed after measurement')
        for key in ('plan_sha256', 'predictions_receipt_sha256'):
            if timing[key] != result[key]:
                raise ValueError('Timing does not bind this complete prediction cohort')
        identity = {r['scene_id']: (r['city'], r['region']) for r in records_by_split['validation']}
        measured = {}
        for method in [PRIMARY] + fixed + simple:
            actual = acceptance._align(timing['actions'][method], identity, 'scene')
            expected = acceptance._index(all_actions['validation'][method], 'scene')
            if any(actual[key]['source_count'] != expected[key]['source_count'] for key in expected):
                raise ValueError('Online timed actions differ from sealed strategy predictions')
            times = acceptance._align(timing['timings'][method], identity, 'scene')
            measured[method] = acceptance._macro(acceptance._city_values(times, 'seconds'), {r['city']: r['region'] for r in records_by_split['validation']})
        for method in simple:
            result['validation']['simple_rule_comparisons'][method]['time_saving_fraction'] = 1 - measured[PRIMARY] / measured[method] if measured[method] > 0 else None
        metrics = {method: dict(summaries['validation'][method], seconds=measured[method]) for method in [PRIMARY] + simple}
        simple_dominating = [method for method in simple
            if all(metrics[method][key] <= metrics[PRIMARY][key] + 1e-12 for key in ('rmse', 'source_count', 'seconds'))
            and any(metrics[method][key] < metrics[PRIMARY][key] - 1e-12 for key in ('rmse', 'source_count', 'seconds'))]
        eligible = [m for m in fixed if summaries['validation'][m]['rmse'] - summaries['validation'][strongest]['rmse'] <= acceptance.RMSE_MARGIN_K]
        refs = dict(selected_on='validation', strongest_fixed=strongest, cheapest_eligible_fixed=min(eligible, key=lambda m: measured[m]))
        records = records_by_split['validation']
        result['cost_acceptance'] = acceptance.evaluate_cost_policy(all_risks['validation'], all_actions['validation'], timing['timings'],
            policy=PRIMARY, fixed_methods=fixed, frozen_references=refs, timing_protocol=timing['timing_protocol'],
            scope=dict(unit_level='scene', expected_unit_count=len(records), expected_city_count=len({r['city'] for r in records}),
                       expected_regions=sorted({r['region'] for r in records}), predictions_sealed=True, split='validation'))
        fixed_pass = result['cost_acceptance']['cost_branch_pass']
        result.update(status='Fit/Val measured cost and simple-rule screen complete',
            cost_branch_pass=fixed_pass, usage_rule_numeric_screen_pass=fixed_pass and not simple_dominating,
            measured_simple_rule_comparators=metrics, measured_dominating_simple_rules=simple_dominating,
            usage_rule_numeric_screen_scope='Original 35 fixed-strategy gates plus no measured simple-rule dominance; Test, repeats and scientific acceptance remain unassessed',
            timing_sha256=digest(args.timing))
    dump(args.output / 'evaluation.json', result)
    print(json.dumps(dict(status=result['status'], primary_policy=PRIMARY, cost_branch_pass=result['cost_branch_pass'],
                          usage_rule_numeric_screen_pass=result['usage_rule_numeric_screen_pass'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'predict', 'predict-one', 'verify', 'score'))
    for key in ('plan', 'output', 'predictions', 'timing'):
        parser.add_argument('--' + key, type=lambda value: Path(value).resolve())
    parser.add_argument('--entry', choices=NAMES)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'check':
        print(json.dumps(dict(status='import_check_pass', weight_names=NAMES, primary=PRIMARY,
            simple_rules=SIMPLE_RULES,
            model_classes=[cls.__name__ for a in ('baseline', 'wide', 'naf_history') for cls in model_classes(a)],
            data_opened=False, checkpoints_opened=False, models_instantiated=False, gpu_used=False)))
        return
    required = ['plan'] + (['predictions'] if args.mode == 'verify' else ['output'])
    if args.mode == 'score':
        required.append('predictions')
    if args.mode == 'predict-one':
        required.append('entry')
    if any(getattr(args, key) is None for key in required) or args.batch_size < 1:
        parser.error('Required arguments: ' + ', '.join('--' + key for key in required) + '; positive batch-size')
    def no_test(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            parts = Path(os.fsdecode(values[0])).parts
            if 'data' in parts and 'test' in parts:
                raise RuntimeError('This evaluator cannot open Test')
    sys.addaudithook(no_test)
    if args.mode == 'predict-one':
        sources.guard_inputs_only()
        _, entries = read_plan(args.plan)
        entry = next(item for item in entries if item['name'] == args.entry)
        # Preserve the original factory used by model_classes; only override its
        # load hook, and use a local factory for source-file discovery.
        original_factory = sources.model_classes
        factories = {a: model_classes(a) for a in ('baseline', 'wide', 'current_query', 'naf_history')}
        sources.model_classes = lambda a: factories[a]
        sources.load_models = load_models
        sources.predict(SimpleNamespace(architecture=entry['architecture'], checkpoint=entry['checkpoint'],
            root=PACKAGE, source_specs=USAGE, output=args.output, device=args.device, batch_size=args.batch_size))
        sources.model_classes = original_factory
    elif args.mode == 'verify':
        verify(args.plan, args.predictions)
        print('All seven Fit/Val source prediction seals verified; no labels opened.')
    else:
        globals()[args.mode](args)


if __name__ == '__main__':
    main()
