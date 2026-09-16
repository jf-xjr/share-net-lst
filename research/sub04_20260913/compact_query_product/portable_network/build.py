"""Package actual frozen QP weights and completed evaluation; no training or selection."""
from pathlib import Path
import argparse, importlib.util, json, math, shutil, zipfile

HERE = Path(__file__).resolve().parent
QP = HERE.parent
NEW = QP.parent
ROOT = NEW.parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


original = module(NEW / 'compact_recovery/portable_network/builder.py', 'original_portable_code_stager')
sha = original.sha


def stage_code(output):
    provenance = original.stage_code(output)
    for source, relative in ((QP / 'model.py', Path('research/sub04_20260913/compact_query_product/model.py')),
                             (HERE / 'predict.py', Path('predict.py'))):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert sha(source) == sha(destination)
        provenance['copied_unmodified'][str(relative)] = sha(source)
    return provenance


def verify_evaluation(reader, frozen, args, results, timing):
    """Check sealed bytes/counts and identities only; never rescore or infer."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    expected = [(e['architecture'], e['seed'], e['checkpoint_sha256']) for e in frozen['checkpoints']]
    require(len(expected) == 6 and len(set(expected)) == 6, 'Exactly six frozen model identities required')
    expected_sources = reader.source_seal()
    def sources(actual):
        require(all(actual.get(p) == h for p, h in expected_sources.items()), 'Evaluation omits a frozen evaluator source')
        for path, digest in actual.items():
            require(sha(path) == digest, 'Evaluated source changed: ' + path)
            if path in frozen['source_and_evidence_sha256']:
                require(digest == frozen['source_and_evidence_sha256'][path], 'Evaluation source differs from freeze')
    manifest = json.loads((reader.PACKAGE / 'manifest.json').read_text())
    receipt_path = args.test_results.parent / 'predictions_complete.json'
    require(sha(receipt_path) == results['prediction_receipt_sha256'], 'Test scores reference different predictions')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['stage'] == 'all_predictions_sealed' and receipt['freeze_sha256'] == sha(args.freeze)
            and receipt['labels_opened'] is False and receipt['device'] == 'cuda'
            and receipt['dtype'] == 'float32' and receipt['tf32'] is False and receipt['batch'] == 1
            and receipt['repair_dtype'] == 'float64' and receipt['student_inference_views'] == 1
            and receipt['total_image_forwards'] == 270 and receipt['total_reused_images'] == 270,
            'Complete270-new/270-reused original-protocol Test receipt required')
    require(receipt['scene_ids'] == [r['scene_id'] for r in manifest['roles']['test']['scenes']]
            and len(receipt['scene_ids']) == 90, 'Original ordered Test90 cohort required')
    require([(e['architecture'], e['seed'], e['checkpoint_sha256']) for e in receipt['entries']] == expected,
            'Test prediction model identities differ from freeze')
    require(receipt['evaluator_source_sha256'] == results['evaluator_source_sha256'], 'Test predict/score source seals differ')
    sources(receipt['evaluator_source_sha256'])
    require(receipt['explicit_test_authorization'] == results['explicit_test_authorization'], 'Test authorization binding differs')
    authorization = receipt['explicit_test_authorization']
    require(sha(authorization['path']) == authorization['sha256'], 'Actual post-freeze Test authorization changed')
    authority = json.loads(Path(authorization['path']).read_text())
    require(authority['status'] == 'explicit_root_authorization_after_QP_three_seed_freeze'
            and authority['freeze_sha256'] == sha(args.freeze) and authority['source_sha256'] == expected_sources
            and frozen['frozen_at_unix'] <= authority['authorized_at_unix']
            and authority['test_previously_consumed'] is True and authority['weight_or_method_selection'] is False
            and authority['matched_training_budget'] is False and authority['single_forward_views'] == 1,
            'Explicit unchanged-method consumed-Test authorization required')
    legacy = json.loads(reader.LEGACY_RECEIPT.read_text())
    reused = {e['seed']: e for e in legacy['entries'] if e['architecture'] == 'baseline'}
    for entry, frozen_entry in zip(receipt['entries'], frozen['checkpoints']):
        name = f"{entry['architecture']}_{entry['seed']}.npy"
        require(entry['prediction'] == name and entry['config'] == frozen_entry['config']
                and entry['shape'] == [90, 1, 160, 160] and entry['output_dtype'] == 'float64'
                and sha(args.test_results.parent / name) == entry['prediction_sha256'],
                'Complete frozen Test prediction bytes/configuration differ')
        if entry['architecture'] == 'baseline':
            old = reused[entry['seed']]
            require(entry['image_forwards'] == 0 and entry['reused_images'] == 90 and entry['prediction_reused'] is True
                    and entry['reused_receipt'] == reader.binding(reader.LEGACY_RECEIPT)
                    and entry['reused_prediction'] == reader.binding(reader.LEGACY_RECEIPT.parent / old['prediction'])
                    and old['checkpoint_sha256'] == entry['checkpoint_sha256']
                    and old['prediction_sha256'] == entry['prediction_sha256'], 'Baseline reuse provenance/count differs')
        else:
            require(entry['image_forwards'] == 90 and entry.get('reused_images', 0) == 0,
                    'Every QP seed must have90 new actual forwards')
    require(sum(e['image_forwards'] for e in receipt['entries']) == 270
            and sum(e.get('reused_images', 0) for e in receipt['entries']) == 270, 'Test aggregate counters differ')
    require(timing['split'] == 'validation' and timing['scenes'] == 45 and timing['cities'] == 15
            and timing['regions'] == 3 and timing['seeds'] == [20260905, 20260912, 20260913]
            and timing['labels_opened'] is False and timing['test_opened'] is False
            and timing['model_selection'] is False, 'Complete unchanged Val-only timing required')
    require([(e['architecture'], e['seed'], e['checkpoint_sha256']) for e in timing['effective_checkpoints']] == expected
            and [e['effective_method'] for e in timing['effective_checkpoints']]
            == [e['effective_method'] for e in frozen['checkpoints']], 'Measured six effective checkpoints differ')
    sources(timing['source_sha256'])
    protocol = timing['protocol']
    require(protocol['device'] == 'cuda' and protocol['runtime'] == 'common eager FP32, no CUDA Graphs or compiler'
            and protocol['tf32'] is False and protocol['autocast'] is False and protocol['batch_size'] == 1
            and protocol['warmups_per_scene_architecture_seed'] == 2
            and protocol['paired_repetitions_per_scene_seed'] == 10, 'Actual paired eager FP32 timing protocol differs')
    records = manifest['roles']['validation']['scenes']
    scene_meta = {r['scene_id']: (r['city'], r['region']) for r in records}
    wanted = {(role, seed, scene, repeat) for role, seed, _ in expected
              for scene in scene_meta for repeat in range(10)}
    observed = set()
    for row in timing['observations']:
        key = (row['architecture'], row['seed'], row['scene_id'], row['repeat'])
        require(key in wanted and key not in observed
                and (row['city'], row['region']) == scene_meta[row['scene_id']], 'Missing, duplicate or foreign paired timing row')
        observed.add(key)
        stages = [row[k] for k in ('h2d_seconds', 'forward_seconds', 'd2h_seconds', 'repair_seconds')]
        require(all(math.isfinite(v) and v >= 0 for v in stages)
                and math.isfinite(row['total_seconds']) and row['total_seconds'] > 0
                and abs(sum(stages) - row['total_seconds']) < 1e-7, 'Incomplete/nonfinite online timing scope')
    require(len(timing['observations']) == 2700 and observed == wanted, 'All six models need45 scenes times10 measurements')
    pairs = timing['pair_costs']
    require([p['seed'] for p in pairs] == [20260905, 20260912, 20260913], 'Three actual seed pairs required')
    for pair in pairs:
        parameters = {e['architecture']: (e['parameters'] if 'parameters' in e else
                      json.loads((Path(e['run']) / 'complete.json').read_text())['parameters'])
                      for e in frozen['checkpoints'] if e['seed'] == pair['seed']}
        require(pair['measured_requests'] == 900 and pair['warmup_requests'] == 180
                and pair['parameters'] == parameters, 'Actual model-pair warmup/count/parameter receipt differs')
    return dict(status='six_predictions_and2700_cost_rows_verified_without_rescoring',
                predictions_complete_sha256=sha(receipt_path), new_QP_image_forwards=270,
                reused_baseline_images=270, actual_timing_rows=2700, source_hashes_verified=True)


def build(args):
    reader = module(QP / 'final_evaluation/reader.py', 'actual_final_qp_reader')
    frozen = reader.read_freeze(args.freeze)
    results = json.loads(args.test_results.read_text())
    timing = json.loads(args.cost.read_text())
    if ('mean_metrics' not in results or 'summaries' not in timing
            or results.get('freeze_sha256') != sha(args.freeze)
            or timing.get('freeze_sha256') != sha(args.freeze)
            or results.get('test_previously_consumed') is not True
            or results.get('model_or_method_reselected') is not False
            or timing.get('status') != 'complete'):
        raise ValueError('Completed original Test and measured cost are required')
    verification = verify_evaluation(reader, frozen, args, results, timing)
    entries = [e for e in frozen['checkpoints'] if e['architecture'] == 'naf_history']
    if [e['seed'] for e in entries] != [20260905, 20260912, 20260913]:
        raise ValueError('All three actually completed selected seeds required')
    if args.output.exists() or args.zip.exists():
        raise FileExistsError('Use new package and archive paths')
    import torch
    torch.set_num_threads(1)
    for entry in entries:
        model = reader.load_model(entry, 'cpu')
        assert sum(p.numel() for p in model.parameters()) == 5977045
        del model
    provenance = stage_code(args.output)
    weights = []
    for entry in entries:
        destination = args.output / 'weights' / f"query_product_{entry['seed']}.pt"
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(entry['checkpoint'], destination)
        assert sha(destination) == entry['checkpoint_sha256']
        weights.append(dict(seed=entry['seed'], path=str(destination.relative_to(args.output)),
                            sha256=sha(destination), original_checkpoint_sha256=entry['checkpoint_sha256']))
    evaluation = dict(mean_metrics=results['mean_metrics'], evidence_verification=verification,
                      test_results=results, measured_cost=timing,
                      consumed_Test=True, independent_holdout=False,
                      matched_additional_training_budget=False,
                      goal2_not_established_by_this_network_bundle=True)
    (args.output / 'evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False) + '\n')
    manifest = dict(protocol='portable_query_product_three_seed_inference_v1',
                    status='complete_actual_three_weights_after_six_model_evaluation',
                    parameters=5977045, inference_views=1, teacher_used_at_inference=False,
                    ensemble_used=False, example_seed=20260905, example_seed_selected_before_Test=True,
                    checkpoints=weights, provenance=provenance,
                    recorded_environment=original.runtime_versions(),
                    original_evidence_sha256={str(p): sha(p) for p in
                        (args.freeze, args.test_results, args.cost, Path(__file__), HERE / 'predict.py')})
    score = results['mean_metrics']['naf_history']['rmse']
    baseline = results['mean_metrics']['baseline']['rmse']
    (args.output / 'requirements.txt').write_text('numpy\ntorch\n')
    (args.output / 'README.md').write_text(f'''# 查询匹配历史热影像网络

包含三个独立种子的选中权重，每次推理仅用其中一个；参数量 5,977,045。
使用当前影像与原九份历史输入，四尺度历史融合增加当前查询与历史特征的乘积项。
教师仅用于训练，推理不使用集成、增强或教师。

```bash
python predict.py --seed 20260905 --inputs-npz encoded_six_fields.npz --output prediction.npy
```

默认 CPU；可添加 `--device cuda`。示例种子在 Test 前固定，未按 Test 选种子。
运行依赖 NumPy、PyTorch；不依赖原工作区、训练数据、教师、父权重或训练日志。

输入为原预处理后的六个字段，保留 batch 维：fine [1,52,160,160]、coarse
[1,1,40,40]、support [1,1,160,160]、context [1,15]、emissivity
[1,4,160,160]、history [1,9,9,160,160]。support 为 bool，其余为 float32。
仅缺失的粗观测允许 NaN。精确通道、单位和归一化见 input_contract.json 及
resources/historylst246/PREPROCESSING.md。本入口接收编码后的 NPZ，不是原始 TIFF。

输出 [1,1,H,W]、单位 K，原有效支持外为 NaN。FP32 单次前向后使用原 FP64
粗格一致性修复；支持域是输入预测支持，不是目标评分掩膜。

实际三种子 Test macro RMSE 为 {score:.9f} K；已有三种子 U-TAE 为 {baseline:.9f} K。
追加训练与选模预算不同。完整指标、热点和实测时间见 evaluation.json。
Test30 已被消费，不能称作新的独立测试。此包不声称目标二的使用规则已成立。

manifest.json 校验包内源代码与权重。模型代码按原相对布局逐字复制；
NAFNet 和 U-TAE 的来源、固定版本与许可证保留在对应 upstream/third_party 目录。
''')
    manifest['files_sha256'] = {str(p.relative_to(args.output)): sha(p)
                                for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=str(Path(args.output.name) / path.relative_to(args.output)))
    args.zip.with_suffix(args.zip.suffix + '.sha256').write_text(sha(args.zip) + '  ' + args.zip.name + '\n')
    print(json.dumps(dict(status='actual_three_seed_QP_bundle_complete', output=str(args.output),
                          archive=str(args.zip), sha256=sha(args.zip), bytes=args.zip.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'test-results', 'cost', 'output', 'zip'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('freeze', 'test_results', 'cost', 'output', 'zip'):
        setattr(args, name, getattr(args, name).resolve())
    build(args)
