"""Publish the completed statistical Val45 comparison while training continues."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import evaluate_comparisons as e
def main():
    out=HERE/'validation_comparison';out.mkdir(exist_ok=True)
    data=e.Dataset(e.PACKAGE,'validation',labels=False)
    stats=json.loads((HERE/'statistics_v2/validation/scores.json').read_text())
    original=e.ROOT/'research/sub04_20260913/compact_query_product/runs'
    members=[];sources={}
    for seed in (20260905,20260912,20260913):
        path=original/f'compact_{seed}'/'complete.json';c=json.loads(path.read_text());s=c['selected_fp32']
        assert [q['scene_id'] for q in s['scenes']]==[q['scene_id'] for q in data.records]
        assert c['source_sha256'][str(e.PACKAGE/'manifest.json')]==e.digest(e.PACKAGE/'manifest.json')
        members.append(s);sources[str(path)]=e.digest(path)
    final=e.mean_scores(members);assert abs(final['macro']['rmse']-.4112458805131804)<1e-9
    ridge=json.loads((HERE/'statistics_v2/selection.json').read_text())['selected_eofrv']
    methods={k:stats[k] for k in ('yoo_llf','allinputs_llf',ridge)};methods['final_network']=final
    effects={k:e.bootstrap(v,final) for k,v in methods.items() if k!='final_network'}
    names={'yoo_llf':'Yoo型 LASSO＋LLF','allinputs_llf':'全部历史字段 LLF',ridge:'EOFRV＋验证集所选ridge','final_network':'原0.426 K网络'}
    rows=['| 方法 | Val45 RMSE (K) | MAE (K) | 热点IoU |','|---|---:|---:|---:|']
    for k,v in methods.items():
        m=v['macro'];rows.append(f"| {names[k]} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['hotspot_iou']:.4f} |")
    strong=methods['allinputs_llf']['macro'];f=final['macro'];gain=1-f['rmse']/strong['rmse']
    paired=effects['allinputs_llf'];lo,hi=paired['confidence_interval_95']
    report=f'''**首批完整专业比较：Val45，2026-09-14**

原0.426 K网络在共同Val45上的RMSE为{f['rmse']:.4f} K，较本轮最强统计历史近邻降低{gain:.1%}，热点IoU提高{f['hotspot_iou']-strong['hotspot_iou']:.4f}。这组结果支持把该网络作为具有实测竞争力的工程成果保留在论文中。

{chr(10).join(rows)}

相对全部历史字段LLF，15个城市中{paired['positive_cities']}个城市的RMSE更低。地区分层、城市配对bootstrap的平均差为{paired['delta']:.4f} K，95%区间为[{lo:.4f}, {hi:.4f}] K。网络的数值为三个独立种子的指标平均。

专业方法在本轮输入和支持上重算，原网络验证指标来自已经封存的三个FP32逐景结果；已核对相同45景顺序、数据manifest及原三种子均值。LLF保留作者机制并适配当前变量，EOFRV采用完整Val45从五个正则强度中选出的lambda=1。评价任务为共同480→120 m，逐景→城市→地区宏平均。原网络接受Fit603监督训练，统计方法在每景已知coarse上拟合。

当前Test90统计预测、THSTNet两阶段训练和同骨干两种子结构归因仍在运行。结构贡献将由固定历史汇聚与学习历史汇聚的完整配对结果确定。

数据见metrics.json、paired.json；图见validation_comparison.png和同名PDF。完整文献机制比较见上级目录LITERATURE_COMPARISON.md。
'''
    (out/'RESULTS.md').write_text(report)
    e.dump(out/'metrics.json',methods);e.dump(out/'paired.json',effects)
    e.dump(out/'receipt.json',dict(split='validation',queries=45,scene_order_verified=True,
        original_sources=sources,statistical_scores_sha256=e.digest(HERE/'statistics_v2/validation/scores.json'),
        source_sha256=e.digest(__file__),original_three_seed_rmse=[v['macro']['rmse'] for v in members]))
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharey=True)
    labels=['Yoo LASSO + LLF','LLF: all history fields','EOFRV + ridge','Existing final network']
    for ax,metric,title in zip(axes,('rmse','hotspot_iou'),('RMSE (K), lower is better','Hotspot IoU, higher is better')):
        values=[v['macro'][metric] for v in methods.values()]
        ax.barh(range(4),values,color=['#8b9daa']*3+['#176d9c'],height=.6)
        for i,v in enumerate(values):ax.text(v+.012*max(values),i,f'{v:.3f}',va='center')
        ax.set_xlim(0,max(values)*1.2);ax.set_xlabel(title);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(4),labels);axes[0].invert_yaxis()
    fig.suptitle('Completed Val45 comparison | 15 cities, 3 regions');fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/('validation_comparison.'+suffix),dpi=180,bbox_inches='tight')
    print(json.dumps(dict(rmse_reduction=gain,paired=paired,metrics={k:v['macro'] for k,v in methods.items()})),flush=True)
if __name__=='__main__':main()
