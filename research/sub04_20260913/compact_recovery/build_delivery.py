"""Thin final delivery: unchanged Goal1 verification plus a verified stopped Fit gate.

The Goal2 check reads existing metadata, feature rows and scene-score JSON only.
It neither reopens observation/prediction arrays nor fits a policy or a network.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,importlib.util,json,os,shlex,sys
import numpy as np

HERE=Path(__file__).resolve().parent;NEW=HERE.parent
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
base=module(HERE/'build_readout.py','unchanged_tiered_goal1_delivery')
reader=base.reader;read=reader.read;sha=reader.sha;require=reader.require
GOAL2=NEW/'compact_history_actions/policy/fit_v1/results.json'
GOAL2_SHA='1f98fa78c1bbad60dbf6ffb66f0e21fac53b0a0424ad443fe183b8e12da9b643'
ACTION_SHA='5caf8f8076ab270f114cec616d634316c8f88ddc5086d40ad08ecb540917c1df'
METHODS=('compact_full9','recovery_full9','compact_merge3','adapted_full9','legacy_full9','legacy_fourhead')
PROTOCOL='compact_fourhead_merged_history_risk6_q85_v1'


def goal2_failure():
    evidence={}
    def verified(item):
        path=Path(item['path']).resolve()
        require(path.suffix in ('.json','.py','.csv'), 'Goal2 verification reads metadata/score/features only')
        require(sha(path)==item['sha256'],'Changed bound Goal2 metadata: '+str(path))
        evidence[str(path)]=item['sha256'];return path
    def artifact(path):
        return verified(reader.binding(path))
    def source_map(mapping):
        require(bool(mapping),'Missing original Goal2 source seal')
        for path,digest in mapping.items():verified(dict(path=path,sha256=digest))
    result=read(verified(dict(path=str(GOAL2),sha256=GOAL2_SHA)))
    require(result['status']=='complete_compact_q85_nested_fit_risk' and result['protocol']==PROTOCOL
        and result['split']=='fit' and result['quantile']==.85 and result['risk_fits']==25
        and result['validation_opened'] is False and result['test_opened'] is False
        and result['actual_online_time_measured'] is False and result['goal2_pass'] is False,
        'Exact completed Fit-only failed policy family required')
    source_map(result['source_sha256'])
    core_path=NEW/'compact_history_actions/policy/cached_fit_core.py'
    # The unchanged numerical module contains no automatic fitting or inference.
    original_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:core=module(core_path,'delivery_original_cached_fit_summary')
    finally:
        if original_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=original_cuda
    manifest=read(reader.PACKAGE/'manifest.json');records=manifest['roles']['fit']['scenes'];ids=[r['scene_id'] for r in records]
    require(len(ids)==603 and len(set(ids))==603 and len({(r['region'],r['city']) for r in records})==201,'Complete original Fit603/201city required')
    controls=read(verified(result['controls']));pred=read(verified(result['predictions_complete']))
    scored=read(verified(result['fixed_scores_complete']));rows=read(verified(result['training_rows']))
    actions=read(verified(dict(path=str(GOAL2.parent/'oof_actions.json'),sha256=ACTION_SHA)))
    started=read(artifact(GOAL2.parent/'started.json'));pred_start_path=Path(result['predictions_complete']['path']).parent/'started.json'
    pred_started=read(artifact(pred_start_path))
    for obj in (controls,pred,scored,rows,actions):require(obj['scene_ids']==ids,'Goal2 scene order changed')
    for obj in (controls,pred,pred_started,started):source_map(obj['source_sha256'])
    require(started['status']=='six_predictions_verified_before_label_open' and started['labels_opened'] is False
        and started['predictions_complete']==result['predictions_complete'] and started['controls']==result['controls']
        and started['source_sha256']==result['source_sha256'],'Score-start prediction/control binding differs')
    require(controls['status']=='compact_q85_input_controls_frozen_before_new_predictions'
        and controls['new_action_scores_opened'] is False and controls['labels_opened'] is False
        and controls['created_unix']<=pred_start_path.stat().st_mtime<=started['fit_started_unix'],
        'Actual controls-before-prediction-before-scoring chronology differs')
    require(pred['status']=='complete_six_fixed_predictions_before_labels'
        and pred['protocol']=='compact_history_actions_six_fixed_fit_cache_v1' and pred['split']=='fit'
        and pred['labels_opened'] is False and pred['validation_opened'] is False and pred['test_opened'] is False
        and pred['fp32'] is True and pred['tf32'] is False and pred['batch']==2 and pred['repair_dtype']=='float64'
        and pred['manifest_sha256']==sha(reader.PACKAGE/'manifest.json')
        and set(pred['entries'])==set(METHODS),'Actual six fixed prediction seal differs')
    require(scored['status']=='complete_six_fixed_scores' and scored['split']=='fit'
        and scored['predictions_complete']==result['predictions_complete']
        and set(scored['entries'])==set(METHODS),'Complete fixed scene-score receipt required')
    require(actions['protocol']==PROTOCOL and actions['split']=='fit' and actions['no_held_rank_or_forced_budget'] is True
        and actions['actions']==rows['actions'],'Frozen OOF action chain differs')
    for obj in (pred_started,pred,scored,actions):require(obj['model_bindings']==result['model_bindings'],'Fixed model binding chain differs')
    for name,item in pred['input_bindings'].items():
        expected=manifest['roles']['fit']['fields'][name]
        require(item['sha256']==expected['sha256'] and Path(item['path']).resolve()==(reader.PACKAGE/expected['path']).resolve(),'Original input metadata differs')
    for name,item in controls['input_bindings'].items():require(item['sha256']==pred['input_bindings'][name]['sha256'],'Feature support/input source differs')
    for name,item in scored['label_bindings'].items():
        require(name in ('target','formal') and item['sha256']==manifest['roles']['fit']['fields'][name]['sha256'], 'Original label binding differs')
    require(set(scored['label_bindings'])=={'target','formal'},'Original fixed scoring labels required')
    legacy=read(verified(pred['legacy_receipt']))
    require(legacy['scene_ids']==ids and legacy['labels_opened'] is False and legacy['test_opened'] is False,'Legacy reused prediction cohort differs')
    for method,old_name in [('legacy_full9','full9'),('legacy_fourhead','full4head')]:
        item=pred['entries'][method];old_item=legacy['entries'][old_name]
        require(item['path']==old_item['path'] and item['sha256']==old_item['sha256']
            and item['new_image_forwards']==0 and pred['counters'][method]['image_forwards']==0,'Legacy prediction reuse differs')
    for name in METHODS:
        item=pred['entries'][name]
        require(item['shape']==[603,1,160,160] and item['dtype']=='float64','Sealed prediction schema differs')
        binding=result['model_bindings'][name];done=read(verified(binding['completion']))
        require(done['status']=='complete' and done['selected_checkpoint_sha256']==binding['checkpoint']['sha256'],'Selected completed model SHA differs')
        if 'training_run' in binding:
            runtime=read(verified(binding['training_run']))
            require(done['run_sha256']==binding['training_run']['sha256'] and runtime['parameters']==binding['parameters'],'Real model completion/runtime binding differs')
        if name in METHODS[:4]:
            require(pred['counters'][name]==dict(network_calls=302,image_forwards=603,stem_source_calls=5427,
                deep_block_tokens=603*binding['history_block_tokens']),'Actual new fixed-action counts differ')
    require(pred['new_image_forwards']==2412 and pred['reused_image_predictions']==1206 and pred['new_network_calls']==1208,'Actual new/reused work differs')
    feature_csv=verified(controls['feature_csv']);verified(controls['feature_source']);previous=read(verified(controls['previous_input_receipt']))
    require(previous['scene_ids']==ids and previous['feature_csv']==controls['feature_csv'],'Original feature-cache chain differs')
    csv_rows=list(csv.DictReader(feature_csv.open()));require([r['scene_id'] for r in csv_rows]==ids,'Cached feature scene order differs')
    X=np.asarray(rows['X'],dtype=np.float64);expected_X=np.array([[float(r[k]) for k in core.fixed.FEATURES] for r in csv_rows])
    require(X.shape==(603,6) and np.array_equal(X,expected_X) and np.isfinite(X).all(),'Frozen six features differ')
    require(rows['fits']==25 and rows['quantile']==.85 and rows['fallback']=='compact_full9' and rows['cheap']=='compact_merge3'
        and rows['neural_models_fit_trained'] is True and rows['only_risk_is_nested_city_OOF'] is True,'Actual fixed OOF scope differs')
    folds=np.asarray(rows['folds']);weights=core.fixed.macro_weights(records)
    require(np.array_equal(folds,core.fixed.assign_folds(records)) and np.allclose(weights,rows['weights'],rtol=0,atol=1e-15),'Original city folds/macro weights differ')
    require([f['fold'] for f in rows['fold_models']]==list(range(5)),'Exactly five stored outer models required')
    for city in {(r['region'],r['city']) for r in records}:
        require(len({int(folds[i]) for i,r in enumerate(records) if (r['region'],r['city'])==city})==1,'City split across folds')
    for fold in rows['fold_models']:
        held=folds==fold['fold'];model=fold['model'];mean=np.array(model['scaler_mean']);scale=np.array(model['scaler_scale'])
        coefficients=np.array([model['standardized_coefficients'][k] for k in core.fixed.FEATURES])
        predicted=((X[held]-mean)/scale)@coefficients+model['intercept_standardized']
        require(np.allclose(predicted,np.asarray(rows['oof_risk_k'])[held],rtol=0,atol=1e-12)
            and np.array_equal(predicted<=fold['threshold_k'],np.asarray(actions['actions']['risk6'])[held])
            and int(held.sum())==fold['held_scenes'],'Stored outer-model risk/actions differ')
    require(controls['controls']==core.input_controls(records,X),'Fixed q85 simple controls differ')
    for name in core.POLICIES[1:]:require(actions['actions'][name]==controls['controls'][name]['actions'],'Simple actions changed')
    errors={}
    for method,item in scored['entries'].items():
        score=read(verified(item));require([(r['scene_id'],r['city'],r['region']) for r in score['scenes']]
            ==[(r['scene_id'],r['city'],r['region']) for r in records],'Complete original fixed score order differs')
        values=np.array([r['rmse'] for r in score['scenes']]);require(np.isfinite(values).all() and (values>=0).all(),'Invalid fixed scene errors')
        macro,_,_=core.fixed.aggregate(values,records);base.close(macro,score['macro']['rmse']);errors[method]=values
    require(np.array_equal(errors['compact_merge3']-errors['compact_full9'],np.asarray(rows['target_risk_k'])),'Stored risk training targets differ')
    summary=core.summarize(records,errors,actions['actions'],'compact_full9','compact_merge3')
    require(all(result[k]==v for k,v in summary.items()),'Original cached summary/CI/gates do not reproduce')
    require(result['fit_may_advance'] is False and summary['necessary_precision_and_token_gate'] is False
        and summary['structurally_possible_cost_success'] is False and result['next_action'].startswith('STOP '),'Actual necessary failure and stop required')
    return dict(status='not_achieved_stopped_after_fit_gate_failure',goal2_pass=False,fit_gate_recomputed=True,
        summary=summary,results=reader.binding(GOAL2),source_and_evidence_sha256=evidence,
        model_bindings=result['model_bindings'],registered_quantile=.85,risk_fits=25,city_folds=5,
        neural_models_fit_trained=True,only_risk_is_nested_city_OOF=True,
        policy_validation_evaluated=False,policy_Test_evaluated=False,final_online_policy_cost_measured=False,
        prediction_array_bytes_reopened=False,observation_arrays_reopened=False,models_refitted=False,
        verification_scope='Exact existing result and metadata/scene-score hashes, sealed input/label/prediction identifiers; no raw arrays or checkpoint bytes reopened',
        next_action=result['next_action'])


def build(args):
    # Preserve the already-reviewed Goal1 verifier and its pending semantics.
    result=base.build(SimpleNamespace(freeze=args.freeze,test_results=args.test_results,cost=args.cost,goal2_receipt=None))
    failure=goal2_failure()
    result.update(goal2=failure,goal2_status=failure['status'],goal2_pass=False,both_goals_pass=False,
        status='goal1_'+result['goal1_status']+'_goal2_not_achieved_stopped_after_fit_gate_failure',
        delivery_wrapper=reader.binding(__file__),goal1_verifier=reader.binding(HERE/'build_readout.py'))
    return result


def markdown(result):
    lines=[f"目标一：{result['goal1_status']}。目标二：未达成，Fit 必要门失败后已停止。两项目标未同时达成。",'']
    if 'test' in result:
        t=result['test'];lines+=['| 指标 | 最终网络三 seed 均值 | 配对 U-TAE |','|---|---:|---:|']
        for k in ('rmse','mae','hotspot_iou','hotspot_mae'):lines.append(f"| {k} | {t['mean_metrics']['naf_history'][k]:.9f} | {t['mean_metrics']['baseline'][k]:.9f} |")
        lines+=['',f"参数 {t['parameters']:,}；适用 RMSE < {t['user_absolute_threshold_k']} K。平均配对提升 {t['mean_paired_gain_k']:.9f} K；三 seed 提升 {t['paired_seed_differences_k']}。",
            '','此 Test 为已消费30城的冻结权重复核；不是新的独立留出确认。']
    if 'cost' in result:
        t=result['cost']['summaries']['mean_architecture_macro_seconds']
        lines+=['',f"原共同 Val45×10 实测单景总耗时：网络 {t['naf_history']['total_seconds']:.6f} s，U-TAE {t['baseline']['total_seconds']:.6f} s；包含 H2D、单次前向、D2H 与原 FP64 支持修复。"]
    if 'checkpoints' in result:
        source=HERE/'model.py' if result['selected_family']=='compact' else NEW/'multihead_fusion/model.py'
        lines+=['',f"模型源码：[定义]({source})；单模型入口：[predict_single.py]({HERE/'predict_single.py'})。",'']
        for entry in result['checkpoints']:
            if entry['architecture']=='naf_history':
                lines.append(f"- seed {entry['seed']}：[实际选中权重]({entry['checkpoint']})；SHA256 `{entry['checkpoint_sha256']}`。")
        command=[sys.executable,'-B',str(HERE/'predict_single.py'),'--freeze',result['freeze']['path'],
            '--architecture','naf_history','--seed','20260905','--inputs-npz',"user's_encoded_six_fields.npz",'--output','new.npy']
        lines+=['','seed 20260905 是预先指定的调用示例，不是按 Test 挑选的最佳 seed。以下默认使用 CPU；输入须含原编码的六个字段，输出路径须尚不存在。',
            '','```bash',' '.join(shlex.quote(x) for x in command),'```']
    if result['missing_evidence']:lines+=['','目标一尚缺：'+'；'.join(result['missing_evidence'])+'。']
    g=result['goal2']['summary'];reference=g['strongest_fixed']
    lines+=['',f"目标二固定 q=0.85 的规则在 Fit603 / 201 城筛查中失败。最强固定参照 {reference} 为 {g['macro_rmse_k'][reference]:.9f} K。",'',
        '| 规则 | Fit macro RMSE / K | 相对最强固定参照差值 / K | 城市配对95%区间 / K |','|---|---:|---:|---|']
    for name in ('risk6','coverage','age_discounted_coverage'):
        c=g['comparisons'][name][reference];lo,hi=c['city_paired_ci95_k']
        lines.append(f"| {name} | {g['macro_rmse_k'][name]:.9f} | {c['delta_rmse_k']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
    lines+=['',f"risk6 的误差差值上界 {g['comparisons']['risk6'][reference]['upper95_delta_k']:.9f} K 超过允许退化 0.005 K。计算的历史深层处理量减少 {g['token_saving_vs_full']*100:.2f}% 不能弥补该精度门失败，也不是已测在线时间收益。",
        '','这里只对风险规则进行了嵌套城市 OOF；底层神经网络已在 Fit 城市训练。该规则未进入 Val/Test 验收，也没有最终在线规则成本测量。原失败记录保持不变。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',type=Path);parser.add_argument('--test-results',type=Path)
    parser.add_argument('--cost',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'Preserve existing deliveries; use a new output directory')
    result=build(args);args.output.mkdir(parents=True,exist_ok=False);reader.write(args.output/'results.json',result)
    with (args.output/'readout.md').open('x') as stream:stream.write(markdown(result))
    print(json.dumps({k:result[k] for k in ('status','goal1_status','goal1_pass','goal2_status','goal2_pass','both_goals_pass')}))


if __name__=='__main__':main()
