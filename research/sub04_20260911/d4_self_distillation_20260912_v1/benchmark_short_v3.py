"""Short v3: six frozen D4 students, Val45, 2 warmups + 2 paired timing repeats.

Same original common eager FP32 batch1 resident NumPy/H2D/forward/D2H/FP64
repair scope. 540 measured plus540 warmup requests total, not the old10-repeat
benchmark. No speed acceptance gate, labels, Test, training or model selection.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import final_effective_evaluation_20260912 as v1
import numpy as np
import torch

p = v1.p
ROLES, SEEDS = v1.ROLES, p.SEEDS
STAGES = ('h2d_seconds', 'forward_seconds', 'd2h_seconds', 'repair_seconds', 'total_seconds')
DEADLINE = 1789218000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def deadline():
    if time.time() >= DEADLINE:
        raise TimeoutError('Hard deadline 2026-09-12 13:00 UTC reached')


def gpu_idle():
    deadline()
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True)
    others = [line.strip() for line in query.stdout.splitlines() if line.strip() and line.strip() != str(os.getpid())]
    if others:
        raise RuntimeError(f'GPU occupied by {others}; no waiting, interruption or retry')


def reader(version):
    from types import SimpleNamespace
    path = HERE / 'd4_self_distillation_20260912_v1/summarize_matched_v2.py'
    spec = importlib.util.spec_from_file_location('short_cost_strict_d4_freeze', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def adapted_read(freeze_path):
        value = module.read_freeze(freeze_path)
        value['evidence_sha256'] = value['source_and_evidence_sha256']
        value['methods'] = {role: 'd4_self_distillation' for role in ROLES}
        value['training_cost'] = {k: value['validation'][k] for k in
            ('actual_continuation_costs', 'total_process_seconds', 'teacher_generation_cost',
             'total_additional_seconds_including_shared_teacher', 'matched_additional_budget', 'shared_gpu_pair_spans')
            if k in value['validation']}
        value['cost_note'] = 'Actual matched1000/10 continuations plus one shared teacher cache. Original training/search/discarded work are separate; per-run elapsed sums are not physical node span.'
        return value
    def seal():
        return dict(v1.source_seal(), **{str(q.resolve()): p.sha(q) for q in (path, Path(__file__))})
    return SimpleNamespace(__file__=str(Path(__file__).resolve()), original=v1.original,
                           read_freeze=adapted_read, source_seal=seal)


def input_guard(manifest, schedules):
    allowed = {(p.PACKAGE / manifest['roles']['validation']['fields'][name]['path']).resolve() for name in p.runner.INPUTS}
    opened = set()
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if 'labels' in path.parts or ('data' in path.parts and {'fit', 'test'} & set(path.parts)):
                raise RuntimeError('This benchmark allows Val inputs only; no Fit/Test/labels')
            if path.suffix in ('.npy', '.npz'):
                if path not in allowed | schedules:
                    raise RuntimeError('Array outside the six Val input fields')
                opened.add(str(path))
    sys.addaudithook(audit)
    return opened


def macro_weights(records):
    regions = sorted({r['region'] for r in records})
    cities = {g: {r['city'] for r in records if r['region'] == g} for g in regions}
    counts = {(g, city): sum(r['region'] == g and r['city'] == city for r in records) for g in regions for city in cities[g]}
    return np.array([1/(len(regions)*len(cities[r['region']])*counts[r['region'], r['city']]) for r in records])


@torch.inference_mode()
def execute(model, sample):
    torch.cuda.synchronize(); started = time.perf_counter()
    tensors = {key: torch.from_numpy(value).to('cuda') for key, value in sample.items()}
    torch.cuda.synchronize(); transferred = time.perf_counter()
    with torch.autocast('cuda', enabled=False):
        output = model(**tensors)
    torch.cuda.synchronize(); forwarded = time.perf_counter()
    raw = output.float().cpu().numpy()
    returned = time.perf_counter()
    repaired = p.runner.repair(raw, sample['coarse'], sample['support'])
    finished = time.perf_counter()
    support = sample['support'].astype(bool)
    if (output.dtype != torch.float32 or repaired.dtype != np.float64
            or not np.isfinite(repaired[support]).all() or not np.isnan(repaired[~support]).all()):
        raise RuntimeError('Invalid FP32/FP64 output on original observation support')
    return dict(h2d_seconds=transferred-started, forward_seconds=forwarded-transferred,
        d2h_seconds=returned-forwarded, repair_seconds=finished-returned, total_seconds=finished-started)


def summarize(rows, records):
    weights = macro_weights(records)
    per_seed = {}
    for seed in SEEDS:
        table = {}
        for role in ROLES:
            scenes = []
            for record in records:
                values = [r for r in rows if r['seed'] == seed and r['architecture'] == role and r['scene_id'] == record['scene_id']]
                if len(values) != 2 or sorted(r['repeat'] for r in values) != list(range(2)):
                    raise RuntimeError('Require two paired measurements for every seed/architecture/scene')
                scenes.append(dict(**{k: record[k] for k in ('scene_id', 'city', 'region')},
                    stages={key: dict(mean=float(np.mean([r[key] for r in values])),
                                     median=float(np.median([r[key] for r in values]))) for key in STAGES}))
            table[role] = dict(scenes=scenes, macro_seconds={key:
                float(weights @ np.array([r['stages'][key]['mean'] for r in scenes])) for key in STAGES})
        per_seed[str(seed)] = table
    means = {role: {key: float(np.mean([per_seed[str(seed)][role]['macro_seconds'][key] for seed in SEEDS]))
                   for key in STAGES} for role in ROLES}
    return dict(per_seed=per_seed, mean_architecture_macro_seconds=means,
        mean_candidate_minus_baseline_seconds=means['naf_history']['total_seconds']-means['baseline']['total_seconds'],
        candidate_relative_time_change=means['naf_history']['total_seconds']/means['baseline']['total_seconds']-1,
        aggregation='Repetition means per scene; scene means per city; city means per region; equal regions; then equal original training seeds')


def inherited_cost(frozen):
    """Copy only freeze-bound costs; no new training or invented total walltime."""
    rows = []
    for entry in frozen['checkpoints']:
        item = dict(architecture=entry['architecture'], seed=entry['seed'],
            effective_method=entry['effective_method'], frozen_entry_training=entry.get('training'), completions=[])
        for key in ('run', 'original_run'):
            if not entry.get(key):
                continue
            path = str(Path(entry[key]) / 'complete.json')
            digest = frozen['evidence_sha256'].get(path)
            if digest is not None:
                if p.sha(path) != digest:
                    raise RuntimeError('A frozen training completion changed')
                completion = read(path)
                item['completions'].append(dict(role=key, path=path, sha256=digest,
                    retained_updates=completion.get('updates'), completed_schedule_seconds=completion.get('seconds'),
                    counters=completion.get('counters'), status=completion.get('status')))
        rows.append(item)
    return dict(entries=rows, frozen_cost_note=frozen.get('cost_note'),
        discarded_original_work=frozen.get('discarded_original_work'),
        additional_frozen_training_cost={k: frozen[k] for k in ('training_cost', 'training_costs', 'cost') if k in frozen},
        note='Verbatim frozen metadata and its hash-bound completions. No retraining; no claim that completed-schedule time includes all discarded/search work. Concurrent per-run elapsed sums are not node wall-clock.')


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('Explicit new output directory required; no retry or overwrite')
    raw_freeze = read(args.freeze)
    schedules = {(Path(e['run']) / 'schedule.npz').resolve() for e in raw_freeze['checkpoints']}
    if len(schedules) != 6 or any(str(q) not in raw_freeze['source_and_evidence_sha256'] for q in schedules):
        raise ValueError('Six hash-bound training schedules required for provenance only')
    manifest = read(p.PACKAGE / 'manifest.json'); opened = input_guard(manifest, schedules)
    selected_reader = reader(args.freeze_reader)
    frozen = selected_reader.read_freeze(args.freeze)
    entries = frozen['checkpoints']
    if [(e['architecture'], e['seed']) for e in entries] != v1.ORDER:
        raise ValueError('Need the explicit actual six effective frozen weights')
    freeze_sha = p.sha(args.freeze)
    reader_source = selected_reader.source_seal()
    source = dict(reader_source, **{str(Path(__file__).resolve()): p.sha(__file__),
                                                  str(Path(selected_reader.__file__).resolve()): p.sha(selected_reader.__file__)})
    costs = inherited_cost(frozen)
    gpu_idle(); p.setup('cuda')
    torch.set_float32_matmul_precision('highest')
    data_start = time.perf_counter()
    data = p.Dataset(p.PACKAGE, 'validation', labels=False)
    records = data.records
    if len(records) != 45 or len({(r['region'], r['city']) for r in records}) != 15 or len({r['region'] for r in records}) != 3:
        raise ValueError('Expected complete common Val45/15-city/3-region input cohort')
    samples = [data.batch([i]) for i in range(len(data))]
    hashes = [{key: dict(shape=list(v.shape), dtype=str(v.dtype),
        sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for key, v in s.items()} for s in samples]
    input_startup = time.perf_counter()-data_start
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(measurement_protocol='short_v3_two_repeats', old_ten_repeat_protocol_used=False, total_measured_requests=540, total_warmup_requests=540, speed_gate_applied=False, device='cuda', gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
        runtime='common eager FP32, no CUDA Graphs or compiler', tf32=False, autocast=False, batch_size=1,
        cpu_threads=torch.get_num_threads(), phases=list(STAGES), warmups_per_scene_architecture_seed=2,
        paired_repetitions_per_scene_seed=2, random_method_order_seed=20260911,
        scope='Resident six NumPy inputs -> tensor construction/H2D -> synchronized eager forward -> D2H -> original FP64 repair',
        excluded='Input disk I/O, model loading, startup, warmup, checks and logging; all reported separately where measured',
        residency='Only one original training seed pair (two effective models) resident at a time; freed before next seed',
        memory_scope='Loaded-pair peak, including BOTH models and request allocations. It is not per-model peak memory.',
        labels_opened=False, test_opened=False, method_selection=False)
    write(args.output / 'started.json', dict(status='running', protocol=protocol,
        freeze=dict(path=str(args.freeze), sha256=freeze_sha, reader=args.freeze_reader), methods=frozen['methods'],
        checkpoints=[{k: e[k] for k in ('architecture', 'seed', 'checkpoint', 'checkpoint_sha256', 'effective_method')} for e in entries],
        source_sha256=source, manifest_sha256=p.sha(p.PACKAGE / 'manifest.json'),
        input_loading_and_hashing_seconds=input_startup, resident_input_hashes=hashes,
        scene_ids=[r['scene_id'] for r in records], inherited_training_cost=costs,
        labels_opened=False, test_opened=False, model_selection=False))
    rows, pairs, models = [], [], {}
    rng = np.random.default_rng(20260911)
    try:
        for seed in SEEDS:
            gpu_idle()
            pair_start = time.perf_counter()
            selected = [e for e in entries if e['seed'] == seed]
            for entry in selected:
                deadline()
                # Exactly the effective evaluator's original loader; no weight,
                # BN, architecture, dropout, or precision override is introduced.
                models[entry['architecture']] = selected_reader.original.load_model(entry, 'cuda')
            torch.cuda.synchronize()
            loading_seconds = time.perf_counter()-pair_start
            params = {role: sum(x.numel() for x in models[role].parameters()) for role in ROLES}
            torch.cuda.reset_peak_memory_stats()
            allocated_loaded = torch.cuda.memory_allocated()
            warmup_seconds, loop_start = 0., time.perf_counter()
            for index, (sample, record) in enumerate(zip(samples, records)):
                warm_start = time.perf_counter()
                for _ in range(2):
                    for role in ROLES:
                        deadline(); execute(models[role], sample)
                warmup_seconds += time.perf_counter()-warm_start
                for repeat in range(2):
                    for role in rng.permutation(ROLES).tolist():
                        deadline(); timing = execute(models[role], sample)
                        rows.append(dict(**timing, seed=seed, architecture=role, repeat=repeat,
                            **{key: record[key] for key in ('scene_id', 'city', 'region')}))
                if (index+1) % 15 == 0:
                    print(json.dumps(dict(event='paired_seed_scene_progress', seed=seed, completed=index+1, total=45)), flush=True)
            pairs.append(dict(seed=seed, parameters=params, model_loading_seconds=loading_seconds,
                warmup_elapsed_seconds=warmup_seconds, request_loop_seconds_including_warmup=time.perf_counter()-loop_start,
                loaded_pair_allocated_gpu_bytes=allocated_loaded, loaded_pair_peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                loaded_pair_peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
                memory_is_per_model=False, measured_requests=180, warmup_requests=180))
            models.clear(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        if p.sha(args.freeze) != freeze_sha or selected_reader.source_seal() != reader_source:
            raise RuntimeError('Effective freeze or reader source changed')
        for path, digest in source.items():
            if p.sha(path) != digest:
                raise RuntimeError('Benchmark source changed')
        selected_reader.read_freeze(args.freeze)  # Rebind every effective checkpoint at finish.
        result = dict(status='complete', total_measured_requests=len(rows), total_warmup_requests=sum(x['warmup_requests'] for x in pairs), split='validation', labels_opened=False, test_opened=False,
            scenes=45, cities=15, regions=3, seeds=list(SEEDS), protocol=protocol, summaries=summarize(rows, records),
            observations=rows, pair_costs=pairs, input_loading_and_hashing_seconds=input_startup,
            job_elapsed_seconds=time.perf_counter()-started, opened_arrays=sorted(opened),
            freeze_sha256=freeze_sha, freeze_reader=args.freeze_reader, source_sha256=source,
            effective_checkpoints=[{k: e[k] for k in ('architecture', 'seed', 'checkpoint_sha256', 'effective_method')} for e in entries],
            inherited_training_cost=costs, model_selection=False, retrained=False, accuracy_evaluated=False,
            scientific_goal_complete=False, usage_cost_branch_evaluated=False,
            caveat='SHORT v3: only two paired repeats per scene, not a passed10-repeat measurement; lower timing precision. Common eager effective-network latency only. No labels or source-policy cost acceptance. Loaded-pair memory must not be reported as either model alone.')
        write(args.output / 'benchmark.json', result)
        print(json.dumps(dict(status='complete', result=str(args.output / 'benchmark.json'),
            mean_architecture_macro_seconds=result['summaries']['mean_architecture_macro_seconds'])))
    except BaseException as exc:
        write(args.output / 'failed.json', dict(status='failed', error=repr(exc),
            completed_measured_requests=len(rows), seconds=time.perf_counter()-started, automatic_retry=False))
        raise
    finally:
        models.clear(); gc.collect(); torch.cuda.empty_cache()


def check():
    records = [dict(region=g, city=g, scene_id=g+str(i)) for g in ('a', 'b', 'c') for i in range(3)]
    rows = [dict(seed=seed, architecture=role, scene_id=r['scene_id'], repeat=repeat,
                 **{k: 2. if role == 'baseline' else 1. for k in STAGES})
            for seed in SEEDS for role in ROLES for r in records for repeat in range(2)]
    result = summarize(rows, records)
    assert result['mean_architecture_macro_seconds']['naf_history']['total_seconds'] == 1.
    assert result['candidate_relative_time_change'] == -.5
    print(json.dumps(dict(status='syntax_import_and_synthetic_paired_macro_check_pass',
        data_opened=False, weights_opened=False, freeze_created=False, gpu_used=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--freeze', type=Path)
    parser.set_defaults(freeze_reader='d4_self_distillation_six_selected_val_weights_v2_short_cost_v3')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        check()
    else:
        if args.freeze is None or args.output is None:
            parser.error('Actual --freeze and explicit new --output are required')
        args.freeze, args.output = args.freeze.resolve(), args.output.resolve()
        import signal
        remaining = DEADLINE - __import__('time').time()
        if remaining <= 0: raise TimeoutError('13:00 UTC hard deadline')
        def timeout(signum, frame): raise TimeoutError('13:00 UTC process alarm')
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, remaining)
        try: run(args)
        finally: signal.setitimer(signal.ITIMER_REAL, 0)
