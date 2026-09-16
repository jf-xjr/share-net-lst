"""Frozen confirmation on the already consumed Test30 cohort; never selection.

External freeze schema: split='test', cohort_identity='Consumed 30-city
follow-up', model_and_rule_selection_complete=true, test_opened=false,
manifest_sha256, primary_policy, rule, simple_rules, experiments[2]. Each
experiment: seed (20260921/20260922), source_plan:{path,sha256},
validation_evaluation:{path,sha256}, weights:{seven names:sha256}, and
frozen_references copied verbatim from its Val cost_acceptance. Paths are
relative to the freeze. This program never creates a freeze or chooses refs.
"""
from pathlib import Path
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as evaluation
from policy_inputs import choose_sources
from historylst.data import Dataset
from historylst.metrics import repair, score as metric_score
from historylst.hotspots import add_hotspot_metrics
import numpy as np

SEEDS = (20260921, 20260922)
COHORT = 'Consumed 30-city follow-up'
RULES = dict(cover_greedy_0=evaluation.RULE, **evaluation.SIMPLE_RULES)
STRATEGIES = tuple(evaluation.sources.MODES) + tuple(RULES)
FIXED = [name + '__' + strategy for name in evaluation.NAMES for strategy in evaluation.sources.MODES]
SIMPLE = ['mixed__joint__' + name for name in evaluation.SIMPLE_RULES]
METHODS = FIXED + [evaluation.PRIMARY] + SIMPLE
digest, dump = evaluation.digest, evaluation.dump


def read_freeze(path):
    """Validate both real successful Val screens before any Test input access."""
    path = Path(path).resolve()
    freeze = json.loads(path.read_text())
    if freeze.get('split') != 'test' or freeze.get('cohort_identity') != COHORT or freeze.get('model_and_rule_selection_complete') is not True or freeze.get('test_opened') is not False:
        raise ValueError('Require a separate pre-Test freeze acknowledging the consumed cohort')
    if freeze.get('primary_policy') != evaluation.PRIMARY or freeze.get('rule') != evaluation.RULE or freeze.get('simple_rules') != evaluation.SIMPLE_RULES:
        raise ValueError('The main and two simple rules must be frozen without modifications')
    if digest(evaluation.PACKAGE / 'manifest.json') != freeze['manifest_sha256']:
        raise ValueError('Frozen common manifest changed')
    rows = {row['seed']: row for row in freeze['experiments']}
    if len(freeze['experiments']) != 2 or set(rows) != set(SEEDS):
        raise ValueError('Require exactly both registered adaptation seeds')
    def bound(record):
        file = (path.parent / record['path']).resolve()
        if digest(file) != record['sha256']:
            raise ValueError(f'Frozen evidence changed: {file}')
        return file
    experiments, shared = [], None
    for seed in SEEDS:
        row = rows[seed]
        plan_path, val_path = bound(row['source_plan']), bound(row['validation_evaluation'])
        plan, entries = evaluation.read_plan(plan_path)
        val = json.loads(val_path.read_text())
        if val.get('usage_rule_numeric_screen_pass') is not True or val.get('test_opened') is not False or val.get('plan_sha256') != digest(plan_path):
            raise ValueError('Both actual Val usage screens must pass before Test confirmation')
        if val.get('primary_policy') != evaluation.PRIMARY or val.get('rule') != evaluation.RULE or val.get('simple_rules') != evaluation.SIMPLE_RULES or val.get('fixed_methods') != FIXED:
            raise ValueError('Val did not evaluate the frozen methods and rules')
        cost = val['cost_acceptance']
        refs = row['frozen_references']
        if cost.get('cost_branch_pass') is not True or cost['scope'].get('split') != 'validation' or refs != cost['frozen_references'] or refs.get('selected_on') != 'validation':
            raise ValueError('Keep the actual Val-frozen accuracy and measured-cost references')
        if any(refs[key] not in FIXED for key in ('strongest_fixed', 'cheapest_eligible_fixed')):
            raise ValueError('Frozen reference is not one of the 35 fixed comparators')
        if row['weights'] != {e['name']: e['checkpoint_sha256'] for e in entries} or plan['manifest_sha256'] != freeze['manifest_sha256']:
            raise ValueError('The explicit frozen seven weights or manifest differ')
        for arm in ('full_control', 'mixed'):
            run = (plan_path.parent / plan['runs'][arm]['path'] / 'run.json').resolve()
            if json.loads(run.read_text())['config']['seed'] != seed:
                raise ValueError('Adaptation seed differs from its actual completed run')
        common = (plan['architecture'], row['weights']['initialization'])
        if shared is not None and shared != common:
            raise ValueError('Both repeats must retain the same chosen original initialization')
        shared = common
        experiments.append(dict(seed=seed, source_plan=plan_path, validation_evaluation=val_path,
                                frozen_references=refs, entries=entries))
    return freeze, experiments


def select(strategy, sample, record):
    """Same input-only online selector as Val; sample retains batch dimension."""
    if strategy in evaluation.sources.CONSTANTS:
        return evaluation.sources.CONSTANTS[strategy], {}
    config = dict(mode='greedy3', conditions=[['greedy_uncovered_reliable', '<=', 1.]]) if strategy == 'greedy3' else RULES[strategy]
    return choose_sources({key: value[0] for key, value in sample.items()}, record, config)


def identity_records():
    records = json.loads((evaluation.PACKAGE / 'manifest.json').read_text())['roles']['test']['scenes']
    if len(records) != 90 or len({row['city'] for row in records}) != 30:
        raise ValueError('Expected the complete already consumed Test90 scenes / 30 cities')
    return [{key: row[key] for key in ('scene_id', 'city', 'region')} for row in records]


def guard():
    state = dict(test_inputs_allowed=False, labels_allowed=False)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        parts = Path(os.fsdecode(values[0])).parts
        if 'labels' in parts and not state['labels_allowed']:
            raise RuntimeError('No labels before all frozen Test predictions are sealed')
        if 'data' in parts and 'test' in parts and not state['test_inputs_allowed']:
            raise RuntimeError('No Test inputs before the external freeze is validated')
    sys.addaudithook(audit)
    return state


def predict(args):
    access = guard()
    _, experiments = read_freeze(args.freeze)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(stage='predicting', split='test', cohort_identity=COHORT,
        freeze_sha256=digest(args.freeze), code_sha256=evaluation.code_seal(),
        labels_opened=False, test_inputs_opened=True, forward_dtype='float32', output_dtype='float64',
        tf32=False, batch_size=1, methods=METHODS, experiments={}, timing_is_benchmark=False)
    dump(args.output / 'prediction_started.json', receipt)
    access['test_inputs_allowed'] = True
    evaluation.sources.setup(args.device)
    data = Dataset(evaluation.PACKAGE, 'test', labels=False)
    identities = identity_records()
    receipt['records'] = identities
    indices = {strategy: [] for strategy in STRATEGIES}
    for i, record in enumerate(data.records):
        sample = data.batch([i])
        for strategy in STRATEGIES:
            selected, _ = select(strategy, sample, record)
            indices[strategy].append(selected)
    dump(args.output / 'source_indices.json', indices)
    receipt['source_indices_sha256'] = digest(args.output / 'source_indices.json')
    for experiment in experiments:
        items, actions = {}, {}
        for entry in experiment['entries']:
            model, original = evaluation.load_models(entry['architecture'], entry['checkpoint'])
            del original
            model = model.to(args.device).eval()
            strategies = evaluation.sources.MODES + (tuple(RULES) if entry['name'] == 'mixed__joint' else ())
            for strategy in strategies:
                method = entry['name'] + '__' + strategy
                relative = Path(str(experiment['seed'])) / (method + '.npy')
                path = args.output / relative
                path.parent.mkdir(exist_ok=True)
                temporary = path.with_suffix('.partial.npy')
                prediction = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64', shape=(90, 1, 160, 160))
                for i in range(len(data)):
                    sample = evaluation.crop(data.batch([i]), np.asarray([indices[strategy][i]], dtype=np.int64))
                    raw = evaluation.forward(model, sample, args.device)
                    prediction[i:i + 1] = repair(raw, sample['coarse'], sample['support'])
                prediction.flush()
                del prediction
                temporary.replace(path)
                items[method] = dict(path=str(relative), sha256=digest(path), checkpoint_sha256=entry['checkpoint_sha256'])
                actions[method] = [dict(**identity, source_count=len(selected), selected_slots=selected)
                                   for identity, selected in zip(identities, indices[strategy])]
                print(json.dumps(dict(seed=experiment['seed'], method=method, prediction_sealed=True)), flush=True)
            del model
        receipt['experiments'][str(experiment['seed'])] = dict(predictions=items, actions=actions,
            weights={e['name']: e['checkpoint_sha256'] for e in experiment['entries']})
    if digest(args.freeze) != receipt['freeze_sha256'] or any(digest(path) != sha for path, sha in receipt['code_sha256'].items()):
        raise ValueError('Frozen identities or evaluator changed during prediction')
    read_freeze(args.freeze)  # Bound plans, Val evidence and weights must still be intact.
    receipt['stage'] = 'all_76_methods_sealed'
    dump(args.output / 'test_predictions_complete.json', receipt)


def verify_predictions(freeze_path, predictions_dir):
    _, experiments = read_freeze(freeze_path)
    folder = Path(predictions_dir)
    receipt = json.loads((folder / 'test_predictions_complete.json').read_text())
    if receipt.get('stage') != 'all_76_methods_sealed' or receipt.get('split') != 'test' or receipt.get('cohort_identity') != COHORT or receipt.get('labels_opened') is not False or receipt['freeze_sha256'] != digest(freeze_path):
        raise ValueError('Require all frozen Test predictions before any scoring')
    if receipt['records'] != identity_records() or receipt['methods'] != METHODS or set(receipt['experiments']) != {str(seed) for seed in SEEDS}:
        raise ValueError('Test methods or consumed cohort identity changed')
    for path, sha in receipt['code_sha256'].items():
        if digest(path) != sha:
            raise ValueError('Prediction source changed')
    source_path = folder / 'source_indices.json'
    if digest(source_path) != receipt['source_indices_sha256']:
        raise ValueError('Input-only source selections changed')
    indices = json.loads(source_path.read_text())
    if set(indices) != set(STRATEGIES):
        raise ValueError('Source strategy family differs')
    for strategy, rows in indices.items():
        if len(rows) != 90 or any(len(set(row)) != len(row) or not row or any(type(i) is not int or not 0 <= i < 9 for i in row) for row in rows):
            raise ValueError('Invalid physical source indices')
        if strategy in evaluation.sources.CONSTANTS and any(row != evaluation.sources.CONSTANTS[strategy] for row in rows):
            raise ValueError('A fixed strategy changed')
        allowed = {evaluation.sources.COUNTS[strategy]} if strategy in evaluation.sources.MODES else {9, evaluation.sources.COUNTS[RULES[strategy]['mode']]}
        if any(len(row) not in allowed for row in rows):
            raise ValueError('A strategy has an invalid history count')
    for experiment in experiments:
        entry = receipt['experiments'][str(experiment['seed'])]
        if entry['weights'] != {e['name']: e['checkpoint_sha256'] for e in experiment['entries']} or set(entry['predictions']) != set(METHODS) or set(entry['actions']) != set(METHODS):
            raise ValueError('Incomplete frozen weight/method predictions')
        for method, item in entry['predictions'].items():
            weight, strategy = method.rsplit('__', 1)
            path = folder / item['path']
            expected_actions = [dict(**identity, source_count=len(row), selected_slots=row) for identity, row in zip(receipt['records'], indices[strategy])]
            if item['checkpoint_sha256'] != entry['weights'][weight] or digest(path) != item['sha256'] or entry['actions'][method] != expected_actions:
                raise ValueError('Prediction, actions or effective checkpoint changed')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (90, 1, 160, 160) or array.dtype != np.float64:
                raise ValueError('Prediction geometry or FP64 dtype differs')
    return receipt


def score(args):
    access = guard()
    _, experiments = read_freeze(args.freeze)
    receipt = verify_predictions(args.freeze, args.predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    access.update(test_inputs_allowed=True, labels_allowed=True)
    data = Dataset(evaluation.PACKAGE, 'test', labels=True)
    acceptance = evaluation.acceptance
    timing = None
    if args.timing:
        timing = json.loads(args.timing.read_text())
        if timing.get('status') != 'complete' or timing.get('split') != 'test' or timing.get('labels_opened') is not False or timing['freeze_sha256'] != digest(args.freeze) or timing['predictions_receipt_sha256'] != digest(args.predictions / 'test_predictions_complete.json'):
            raise ValueError('Require actual Test timing bound to this complete frozen prediction cohort')
        for path, sha in timing['source_sha256'].items():
            if digest(path) != sha:
                raise ValueError('Timing source changed after measurement')
    results = {}
    regions = {r['city']: r['region'] for r in data.records}
    rng = np.random.default_rng(20260911)
    sizes = {g: sum(r == g for r in regions.values()) for g in sorted(set(regions.values()))}
    draws = {g: rng.integers(0, n, size=(20000, n)) for g, n in sizes.items()} if min(sizes.values()) >= 2 else None
    for experiment in experiments:
        seed = str(experiment['seed'])
        sealed = receipt['experiments'][seed]
        scores, risks, macro = {}, {}, {}
        for method, item in sealed['predictions'].items():
            prediction = np.load(args.predictions / item['path'], mmap_mode='r', allow_pickle=False)
            scores[method] = add_hotspot_metrics(metric_score(prediction, data.arrays['target'], data.arrays['formal'], data.records), prediction, data.arrays['target'], data.arrays['formal'])
            risks[method], macro[method] = scores[method]['scenes'], dict(scores[method]['macro'])
            macro[method]['source_count'] = evaluation.policies.macro(np.asarray([r['source_count'] for r in sealed['actions'][method]]), data.records)
        city = {m: acceptance._city_values(acceptance._index(rows, 'scene'), 'rmse') for m, rows in risks.items()}
        paired = {m: acceptance._paired_interval(city[evaluation.PRIMARY], city[m], regions, draws) for m in FIXED + SIMPLE}
        result = dict(macro=macro, comparisons=paired, frozen_references=experiment['frozen_references'],
            cost_branch_pass=False, usage_rule_numeric_screen_pass=False, measured_timing_available=timing is not None,
            necessary_accuracy_gates={m: paired[m]['degradation_upper95_k'] is not None and paired[m]['degradation_upper95_k'] <= acceptance.RMSE_MARGIN_K + 1e-12 for m in FIXED})
        if timing:
            actual = timing['experiments'][seed]
            identity = {r['scene_id']: (r['city'], r['region']) for r in data.records}
            for method in METHODS:
                actions = acceptance._align(actual['actions'][method], identity, 'scene')
                if any(actions[row['scene_id']]['source_count'] != row['source_count'] for row in sealed['actions'][method]):
                    raise ValueError('Measured online actions differ from frozen predictions')
                macro[method]['seconds'] = acceptance._macro(acceptance._city_values(acceptance._align(actual['timings'][method], identity, 'scene'), 'seconds'), regions)
            cost = acceptance.evaluate_cost_policy(risks, sealed['actions'], actual['timings'], policy=evaluation.PRIMARY,
                fixed_methods=FIXED, frozen_references=experiment['frozen_references'], timing_protocol=timing['timing_protocol'],
                scope=dict(split='test', unit_level='scene', expected_unit_count=90, expected_city_count=30,
                           expected_regions=sorted(set(regions.values())), predictions_sealed=True))
            dominating = [m for m in SIMPLE if all(macro[m][k] <= macro[evaluation.PRIMARY][k] + 1e-12 for k in ('rmse', 'source_count', 'seconds'))
                          and any(macro[m][k] < macro[evaluation.PRIMARY][k] - 1e-12 for k in ('rmse', 'source_count', 'seconds'))]
            result.update(cost_acceptance=cost, cost_branch_pass=cost['cost_branch_pass'],
                measured_dominating_simple_rules=dominating, usage_rule_numeric_screen_pass=cost['cost_branch_pass'] and not dominating)
        results[seed] = result
        dump(args.output / (seed + '_scores.json'), scores)
    aggregate = {method: {metric: float(np.mean([results[str(seed)]['macro'][method][metric] for seed in SEEDS]))
        for metric in ('rmse', 'mae', 'hotspot_iou', 'hotspot_mae', 'source_count')} for method in METHODS}
    dump(args.output / 'evaluation_test.json', dict(status='Frozen consumed-cohort confirmation scored', split='test',
        cohort_identity=COHORT, freeze_sha256=digest(args.freeze), predictions_receipt_sha256=digest(args.predictions / 'test_predictions_complete.json'),
        timing_sha256=digest(args.timing) if args.timing else None, experiments=results, descriptive_two_seed_mean=aggregate,
        both_seed_numeric_screens_pass=all(r['usage_rule_numeric_screen_pass'] for r in results.values()),
        references_reselected=False, thresholds_changed=False, all_user_conditions_assessed=False,
        caveat='Previously consumed Test30; confirmation only. Equal-seed means and within-region paired-city intervals are descriptive, not fresh held-out evidence. No Test reference or rule selection.',
        code_sha256=evaluation.code_seal()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'predict', 'verify', 'score'))
    for key in ('freeze', 'predictions', 'output', 'timing'):
        parser.add_argument('--' + key, type=lambda value: Path(value).resolve())
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.mode == 'check':
        print(json.dumps(dict(status='import_check_pass', seeds=SEEDS, methods_per_seed=len(METHODS),
            primary=evaluation.PRIMARY, simple_rules=evaluation.SIMPLE_RULES, freeze_created=False,
            data_opened=False, checkpoints_opened=False, models_instantiated=False, gpu_used=False)))
        return
    required = ['freeze'] + (['predictions'] if args.mode == 'verify' else ['output'])
    if args.mode == 'score':
        required.append('predictions')
    if any(getattr(args, key) is None for key in required):
        parser.error('Required: ' + ', '.join('--' + key for key in required))
    if args.mode == 'verify':
        verify_predictions(args.freeze, args.predictions)
        print('All 76 frozen Test prediction methods verified; no labels opened.')
    else:
        globals()[args.mode](args)


if __name__ == '__main__':
    main()
