"""Export the completed Test90 statistical comparison while the deep runs finish."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_comparisons as e

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / 'statistics_test_comparison'
    out.mkdir(exist_ok=True)
    data = e.Dataset(e.PACKAGE, 'test', labels=True)
    folder = e.ROOT / 'research/sub04_20260913/compact_query_product/final_evaluation/test'
    receipt = json.loads((folder / 'predictions_complete.json').read_text())
    assert receipt['scene_ids'] == [q['scene_id'] for q in data.records]
    members, sources = [], {}
    for row in receipt['entries']:
        if row['architecture'] != 'naf_history':
            continue
        path = folder / row['prediction']
        assert e.digest(path) == row['prediction_sha256']
        members.append(e.evaluate(np.load(path), data))
        sources[str(path)] = row['prediction_sha256']
    assert len(members) == 3
    final = e.mean_scores(members)
    assert abs(final['macro']['rmse'] - .42613199570471244) < 1e-9
    stats = json.loads((HERE / 'statistics_v2/test/scores.json').read_text())
    selection = json.loads((HERE / 'statistics_v2/selection.json').read_text())
    assert selection['scoring_source_sha256'] == e.digest(HERE / 'evaluate_comparisons.py')
    assert selection['analysis_protocol_sha256'] == e.digest(HERE / 'ANALYSIS_PROTOCOL.md')
    ridge = selection['selected_eofrv']
    methods = {k: stats[k] for k in ('yoo_llf', 'allinputs_llf', ridge)}
    methods['final_network'] = final
    effects = {k: e.bootstrap(v, final) for k, v in methods.items() if k != 'final_network'}
    names = ['Yoo LASSO＋LLF', '完整历史字段 LLF', 'EOFRV＋Val所选ridge', '原0.426 K网络']
    lines = ['| 方法 | Test90 RMSE (K) | MAE (K) | 热点IoU | 热点MAE (K) |',
             '|---|---:|---:|---:|---:|']
    for name, row in zip(names, methods.values()):
        m = row['macro']
        lines.append(f"| {name} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['hotspot_iou']:.4f} | {m['hotspot_mae']:.4f} |")
    paired = effects['allinputs_llf']
    strong, f = methods['allinputs_llf']['macro'], final['macro']
    reduction = 1 - f['rmse'] / strong['rmse']
    lo, hi = paired['confidence_interval_95']
    report = f'''**完整专业统计近邻比较：Test90，2026-09-14**

原网络在30个测试城市、90景上的RMSE为{f['rmse']:.4f} K，相对本轮最强统计历史近邻降低{reduction:.1%}；热点IoU提高{f['hotspot_iou']-strong['hotspot_iou']:.4f}，热点MAE降低{1-f['hotspot_mae']/strong['hotspot_mae']:.1%}。

{chr(10).join(lines)}

相对完整历史字段LLF，30个城市中{paired['positive_cities']}个城市改善。城市配对、地区分层bootstrap的平均RMSE差为{paired['delta']:.4f} K，95%区间为[{lo:.4f}, {hi:.4f}] K。三个地区的平均改善分别为：{'; '.join(f'{k} {v:.4f} K' for k,v in paired['regional_delta'].items())}。

共同任务是480→120 m城市热场重建，使用相同历史输入、当前粗温度、预测支持和评分掩码；逐景→城市→地区等权。原网络为三个独立种子的指标平均，使用原有Fit603训练成果。LLF按每景当前已知粗温度拟合历史关系；EOFRV的正则强度lambda=1在Val45选定。预测完成、顺序及文件哈希核验后评分；共同粗温度均值约束误差小于1e-8 K。

本表展示系统相对专业统计方法的工程收益。同骨干结构归因和THSTNet共同损失校准正在独立完成，完整报告将合并这两项结果。

[文献与方法对应](../LITERATURE_COMPARISON.md) · [图PDF](statistics_test_comparison.pdf) · [城市配对数据](paired.json)
'''
    (out / 'RESULTS.md').write_text(report)
    e.dump(out / 'metrics.json', methods)
    e.dump(out / 'paired.json', effects)
    e.dump(out / 'receipt.json', dict(split='test', queries=90, original_sources=sources,
        statistical_scores_sha256=e.digest(HERE / 'statistics_v2/test/scores.json'),
        selection_sha256=e.digest(HERE / 'statistics_v2/selection.json'), source_sha256=e.digest(__file__)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    labels = ['Yoo LASSO + LLF', 'LLF: all history fields', 'EOFRV + ridge', 'Existing final network']
    for ax, metric, title in zip(axes, ('rmse', 'hotspot_iou'),
                                ('RMSE (K), lower is better', 'Hotspot IoU, higher is better')):
        values = [v['macro'][metric] for v in methods.values()]
        ax.barh(range(4), values, color=['#8b9daa'] * 3 + ['#176d9c'], height=.6)
        for i, v in enumerate(values):
            ax.text(v + .012 * max(values), i, f'{v:.3f}', va='center')
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_xlabel(title)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(4), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Completed Test90 comparison | 30 cities, 3 regions')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / ('statistics_test_comparison.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(reduction=reduction, paired=paired)), flush=True)


if __name__ == '__main__':
    main()
