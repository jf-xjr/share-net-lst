"""Summarize actual tiered six-model evidence; missing evidence remains pending.

Reads evidence JSON and verifies artifact hashes; never runs inference, scoring
against observations, training, or model selection. Goal2 is explicitly pending
in this adapter until a separate final acceptance implementation is supplied.
"""
from pathlib import Path
from collections import Counter
import argparse
import importlib.util
import json
import math
import statistics
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('delivery_recovery_reader', HERE / 'final_reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
require = reader.require
read = reader.read
sha = reader.sha
STAGES = ('h2d_seconds','forward_seconds','d2h_seconds','repair_seconds','total_seconds')


def finite(value):
    return isinstance(value, (float,int)) and not isinstance(value,bool) and math.isfinite(value)


def close(left, right, tolerance=1e-12):
    require(finite(left) and finite(right) and abs(left-right) <= tolerance, 'Evidence numeric mismatch')


def sources(value):
    require(isinstance(value,dict) and value, 'Nonempty source bindings required')
    for path, digest in value.items():
        require(sha(path) == digest, 'Changed bound source: ' + path)


def identity(entries):
    return [(e['architecture'],e['seed'],e['checkpoint_sha256']) for e in entries]


def numeric_outcome(parameters, means, paired):
    threshold = reader.tier(parameters)
    require(len(paired)==3 and all(finite(v) for v in paired), 'Three finite paired differences required')
    gain = means['baseline']['rmse'] - means['naf_history']['rmse']
    return dict(parameters=parameters, user_absolute_threshold_k=threshold,
        absolute_pass=means['naf_history']['rmse'] < threshold,
        mean_paired_gain_k=gain, mean_gain_at_least_001=gain >= .01,
        paired_seed_differences_k=paired, every_seed_positive=all(v > 0 for v in paired),
        numeric_pass=bool(means['naf_history']['rmse'] < threshold and gain >= .01 and all(v > 0 for v in paired)))


def validate_test_score(score, records):
    # The inherited completion validator is deliberately Val45-only. Validate
    # Test90 aggregation here without evaluating any prediction against labels.
    require([(r['scene_id'],r['city'],r['region']) for r in score['scenes']]
        ==[(r['scene_id'],r['city'],r['region']) for r in records], 'Complete unchanged Test90 score identities required')
    expected={(r['region'],r['city']) for r in records}
    require(len(score['cities'])==30 and {(r['region'],r['city']) for r in score['cities']}==expected
        and set(score['regions'])=={r['region'] for r in records}, 'Complete Test30 city/region identities required')
    for city in score['cities']:
        rows=[r for r in score['scenes'] if (r['region'],r['city'])==(city['region'],city['city'])]
        require(len(rows)==3, 'Three real dates per Test city required')
        for key in reader.p.METRICS:
            require(all(finite(r[key]) for r in rows), 'Nonfinite per-scene score')
            close(city[key],statistics.mean(r[key] for r in rows))
    for region,metrics in score['regions'].items():
        rows=[r for r in score['cities'] if r['region']==region]
        require(len(rows)==10, 'Ten Test cities per original region required')
        for key in reader.p.METRICS:close(metrics[key],statistics.mean(r[key] for r in rows))
    for key in reader.p.METRICS:close(score['macro'][key],statistics.mean(r[key] for r in score['regions'].values()))


def verify_test(path, freeze_path, frozen):
    result = read(path)
    receipt_path = path.parent / 'predictions_complete.json'
    receipt = read(receipt_path)
    digest = sha(freeze_path)
    require(result['freeze_sha256']==receipt['freeze_sha256']==digest, 'Test/freezing binding differs')
    require(result['prediction_receipt_sha256']==sha(receipt_path), 'Test scores lack exact six-prediction receipt')
    for obj in (result, receipt):
        require(obj['split']=='test' and obj['test_previously_consumed'] is True
            and obj['independent_holdout'] is False and obj['student_inference_views']==1
            and obj['teacher_eight_views_used_at_inference'] is False
            and obj['teacher_twenty_four_views_used_at_inference'] is False
            and obj['model_or_method_reselected'] is False
            and obj['selected_family']==frozen['selected_family']
            and obj['effective_training_method']==reader.METHOD, 'Wrong cohort/method identity')
        sources(obj['evaluator_source_sha256'])
        auth = obj['explicit_test_authorization']
        require(sha(auth['path'])==auth['sha256'], 'Test authorization changed')
        authorization = read(auth['path'])
        require(authorization['status']=='explicit_root_authorization_after_six_weight_freeze'
            and authorization['freeze_sha256']==digest
            and frozen['frozen_at_unix']<=authorization['authorized_at_unix']<=time.time()
            and authorization['test_previously_consumed'] is True
            and authorization['weight_or_method_selection'] is False
            and authorization['single_forward_views']==1
            and authorization['source_sha256']==reader.source_seal(), 'Missing genuine post-freeze authorization')
        require(obj['user_tier_rule']==reader.binding(reader.RULE), 'User tier evidence changed')
    require(result['explicit_test_authorization']==receipt['explicit_test_authorization'], 'Different predict/score authorizations')
    require(result['ensemble_used'] is False,'Single-model per-seed results required')
    records = read(reader.PACKAGE/'manifest.json')['roles']['test']['scenes']
    require(len(records)==90 and len({r['city'] for r in records})==30 and len({r['region'] for r in records})==3, 'Full consumed cohort required')
    require(receipt['stage']=='all_predictions_sealed' and receipt['labels_opened'] is False
        and receipt['scene_ids']==[r['scene_id'] for r in records]
        and receipt['total_image_forwards']==540 and receipt['dtype']=='float32'
        and receipt['tf32'] is False and receipt['batch']==1 and receipt['repair_dtype']=='float64'
        and identity(receipt['entries'])==identity(frozen['checkpoints']), 'Complete six-model single-view prediction seal required')
    scores = {}
    bindings = {str(path.resolve()):sha(path),str(receipt_path.resolve()):sha(receipt_path)}
    for entry in receipt['entries']:
        prediction = (path.parent / entry['prediction']).resolve()
        require(prediction.parent==path.parent.resolve(), 'Prediction must be inside its sealed output directory')
        require(entry['image_forwards']==90 and entry['shape']==[90,1,160,160]
            and entry['output_dtype']=='float64' and sha(prediction)==entry['prediction_sha256'], 'Incomplete or changed sealed predictions')
        score_path=path.parent/f"{entry['architecture']}_{entry['seed']}_scores.json"
        score=read(score_path)
        validate_test_score(score,records)
        for key in ('rmse','mae','hotspot_iou','hotspot_mae'):
            require(finite(score['macro'][key]), 'Missing finite actual hotspot/error metric')
        scores[(entry['architecture'],entry['seed'])]=score
        bindings[str(prediction)]=entry['prediction_sha256'];bindings[str(score_path.resolve())]=sha(score_path)
    recomputed=reader.p.compare_scores(scores,'naf_history')
    for key in ('mean_metrics','effects','criteria'):
        require(result[key]==recomputed[key], 'Summary differs from unchanged paired city/seed metric core: '+key)
    parameters=reader.loader.trainer.PARAMETERS[frozen['selected_family']]
    require(result['actual_parameters']==receipt['actual_parameters']==parameters, 'Parameter count differs from selected factory')
    outcome=numeric_outcome(parameters,recomputed['mean_metrics'],recomputed['effects']['rmse']['paired_seed_differences'])
    close(result['user_absolute_threshold_k'],outcome['user_absolute_threshold_k'])
    require(result['tier_absolute_pass']==outcome['absolute_pass']
        and result['tier_and_paired_numeric_pass']==outcome['numeric_pass'], 'Original scored tier gate differs')
    return dict(status='complete', **outcome, mean_metrics=recomputed['mean_metrics'],
        paired_effects=recomputed['effects'], per_seed_metrics=[dict(architecture=r,seed=s,macro=scores[(r,s)]['macro']) for r,s in reader.ORDER],
        prediction_seal_verified=True, evidence_sha256=bindings,
        cohort='Already consumed Test90 / 30 cities / 3 regions', independent_holdout=False)


def verify_cost(path,freeze_path,frozen):
    value=read(path)
    require(value['status']=='complete' and value['freeze_sha256']==sha(freeze_path)
        and value['split']=='validation' and value['labels_opened'] is False
        and value['test_opened'] is False and value['model_selection'] is False
        and value['retrained'] is False and value['accuracy_evaluated'] is False,
        'Actual common input-only timing receipt required')
    require(value['scenes']==45 and value['cities']==15 and value['regions']==3
        and value['seeds']==list(reader.p.SEEDS), 'Full original validation cost cohort required')
    require(identity(value['effective_checkpoints'])==identity(frozen['checkpoints']), 'Timing used different weights')
    require(all(e['effective_method']==reader.METHOD for e in value['effective_checkpoints']), 'Timing method identity differs')
    protocol=value['protocol']
    require(protocol['device']=='cuda' and protocol['batch_size']==1 and protocol['tf32'] is False
        and protocol['autocast'] is False and protocol['warmups_per_scene_architecture_seed']==2
        and protocol['paired_repetitions_per_scene_seed']==10
        and protocol['runtime']=='common eager FP32, no CUDA Graphs or compiler'
        and protocol['phases']==list(STAGES), 'Same complete actual10-repeat cost protocol required')
    sources(value['source_sha256'])
    started_path=path.parent/'started.json';started=read(started_path)
    require(started['freeze']['sha256']==sha(freeze_path) and started['protocol']==protocol
        and started['source_sha256']==value['source_sha256']
        and started['manifest_sha256']==sha(reader.PACKAGE/'manifest.json')
        and started['labels_opened'] is False and started['test_opened'] is False,
        'Actual timing start/model/input protocol binding differs')
    records=read(reader.PACKAGE/'manifest.json')['roles']['validation']['scenes']
    require(started['scene_ids']==[r['scene_id'] for r in records], 'Actual complete Val timing order differs')
    byid={r['scene_id']:r for r in records}
    expected=Counter((s,r,row['scene_id'],i) for s in reader.p.SEEDS for r in ('baseline','naf_history') for row in records for i in range(10))
    actual=Counter()
    for row in value['observations']:
        actual[(row['seed'],row['architecture'],row['scene_id'],row['repeat'])]+=1
        require(row['scene_id'] in byid and row['city']==byid[row['scene_id']]['city']
            and row['region']==byid[row['scene_id']]['region'], 'Cost scene metadata differs')
        require(all(finite(row[k]) and row[k]>=0 for k in STAGES), 'Invalid actual timing observation')
    require(actual==expected and len(value['observations'])==2700, 'Missing, duplicated or partial paired repetitions')
    core=reader.module(reader.OLD/'benchmark_final_frozen_20260912.py','readout_original_cost_aggregation')
    summary=core.summarize(value['observations'],records)
    require(summary==value['summaries'], 'Reported macro cost differs from actual45-scene10-repeat observations')
    require([q['seed'] for q in value['pair_costs']]==list(reader.p.SEEDS), 'Missing seed-pair cost metadata')
    for item in value['pair_costs']:
        require(item['measured_requests']==900 and item['warmup_requests']==180
            and item['memory_is_per_model'] is False, 'Actual measured/warmup counts or memory scope differs')
        for role in ('baseline','naf_history'):
            entry,=[e for e in frozen['checkpoints'] if (e['architecture'],e['seed'])==(role,item['seed'])]
            require(item['parameters'][role]==entry['parameters'], 'Timing model parameters differ')
    return dict(status='complete',receipt=reader.binding(path),protocol=protocol,
        start_receipt=reader.binding(started_path),
        summaries=summary,pair_costs=value['pair_costs'],actual_measured_requests=2700,
        actual_warmup_requests=540,memory_scope='Loaded seed pair; not separate per-model peak memory',
        elapsed_seconds=value['job_elapsed_seconds'],evidence_sha256=value['source_sha256'])


def build(args):
    result=dict(status='pending',goal1_status='pending',goal1_pass=False,
        goal2_status='pending',goal2_pass=False,both_goals_pass=False,
        missing_evidence=[],user_tiers=read(reader.RULE)['tiers'],source=reader.binding(__file__),
        inference_entry=reader.binding(HERE/'predict_single.py'),no_inference_or_training_performed=True)
    if args.goal2_receipt:
        external=read(args.goal2_receipt)
        result['goal2_external_stage_evidence']=dict(receipt=reader.binding(args.goal2_receipt),
            reported_status=external.get('status'),reported_split=external.get('split'),
            final_acceptance_verified=False,
            note='Optional stage evidence only; this adapter cannot promote a Fit/Val result or unverified external pass flag to final Goal2 success.')
    if args.freeze is None or not args.freeze.is_file():
        result['missing_evidence'].append('actual selected-family six-weight freeze')
        return result
    frozen=reader.read_freeze(args.freeze)
    result.update(freeze=reader.binding(args.freeze),selected_family=frozen['selected_family'],
        checkpoints=frozen['checkpoints'],matched_training_cost=frozen['validation']['actual_continuation_costs'],
        total_six_training_process_seconds=frozen['validation']['total_process_seconds'],
        historical_teacher_generation=frozen['validation']['teacher_generation_cost'],
        new_teacher_generation_seconds=0,historical_search_costs_equal=False)
    if args.test_results is not None and args.test_results.is_file():
        result['test']=verify_test(args.test_results,args.freeze,frozen)
    else:result['missing_evidence'].append('sealed six-model consumed-Test predictions and actual paired scores')
    if args.cost is not None and args.cost.is_file():
        result['cost']=verify_cost(args.cost,args.freeze,frozen)
    else:result['missing_evidence'].append('complete original Val45 ten-repeat actual cost receipt')
    if 'test' in result and not result['test']['numeric_pass']:
        result['goal1_status']='failed'
    elif 'test' in result and 'cost' in result:
        result['goal1_status']='passed';result['goal1_pass']=True
    result['status']='goal1_'+result['goal1_status']+'_goal2_pending'
    return result


def markdown(result):
    lines=['目标一：'+result['goal1_status']+'；目标二：'+result['goal2_status']+'。','']
    if 'test' in result:
        t=result['test'];lines += ['| 指标 | 网络 | 配对 U-TAE |','|---|---:|---:|']
        for key in ('rmse','mae','hotspot_iou','hotspot_mae'):
            lines.append(f"| {key} | {t['mean_metrics']['naf_history'][key]:.9f} | {t['mean_metrics']['baseline'][key]:.9f} |")
        lines += ['',f"参数 {t['parameters']:,}；适用 RMSE 严格门槛 < {t['user_absolute_threshold_k']} K。三种子平均提升 {t['mean_paired_gain_k']:.9f} K；各种子提升 {t['paired_seed_differences_k']}。",
                  '','Test 已消费，此处为冻结权重后的完整复核，不能称为新的独立留出验证。']
    if 'cost' in result:
        timing=result['cost']['summaries']['mean_architecture_macro_seconds']
        lines += ['',f"实际单景总耗时：网络 {timing['naf_history']['total_seconds']:.6f} 秒，配对 U-TAE {timing['baseline']['total_seconds']:.6f} 秒。完整 Val45 × 10 次配对重复，含输入传输、前向、输出回传和原支持修复。"]
    if result['missing_evidence']:
        lines += ['','尚缺：'+'；'.join(result['missing_evidence'])+'。']
    lines += ['','目标二等待单独完成的正式最终验收；Fit、Val 或外部未复核的成功标记不会令它通过。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--test-results',type=Path)
    parser.add_argument('--cost',type=Path)
    parser.add_argument('--goal2-receipt',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve prior readouts; use a new output directory')
    result=build(args)
    args.output.mkdir(parents=True,exist_ok=False)
    reader.write(args.output/'results.json',result)
    with (args.output/'readout.md').open('x') as stream:stream.write(markdown(result))
    print(json.dumps({k:result[k] for k in ('status','goal1_status','goal1_pass','goal2_status','both_goals_pass','missing_evidence')}))


if __name__=='__main__':main()
