"""Export final comparison, paired attribution and training curves."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'analysis'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def read(path):return json.loads(path.read_text())
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
    scores=read(OUT/'scores.json');pairs=read(OUT/'paired.json');strata=read(OUT/'strata.json')
    ridge=read(HERE/'statistics_v2/selection.json')['selected_eofrv']
    methods=['yoo_llf','allinputs_llf',ridge,'THST_selected','original_UTAE','final_0.426']
    names=['Yoo LASSO + LLF','LLF: all input fields','EOFRV + Val-selected ridge',
           'THSTNet: common inputs','Original history U-TAE','Existing final NAF']
    colors=['#8496a5']*4+['#cda265','#176d9c']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.4),sharey=True)
    for ax,metric,label in zip(axes,['rmse','hotspot_iou'],['RMSE (K), lower is better','Hotspot IoU, higher is better']):
        values=[scores[k]['macro'][metric] for k in methods]
        ax.barh(np.arange(len(methods)),values,color=colors,height=.62)
        for i,(k,value) in enumerate(zip(methods,values)):
            ax.text(value+.015*max(values),i,f'{value:.3f}',va='center')
            if len(scores[k]['members'])>1:
                ax.scatter([m[metric] for m in scores[k]['members']],np.full(len(scores[k]['members']),i),
                           s=17,c='white',edgecolors='#243846',linewidths=.7,zorder=3)
        ax.set_xlim(0,max(values)*1.18);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(names)),names);axes[0].invert_yaxis()
    fig.suptitle('Professional historical-LST comparison | 90 scenes, 30 cities, 3 regions')
    fig.text(.5,-.025,'Common inputs and scoring support; statistical and THST methods are documented task adaptations. Dots are individual seeds.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'professional_comparison')

    fig,axes=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[.9,1.1]})
    a=axes[0];rows=read(OUT/'attribution_members.json')
    for seed,color in zip((20260914,20260915),('#176d9c','#c36f28')):
        y=[next(r['scores']['macro']['rmse'] for r in rows if r['seed']==seed and r['variant']==v)
           for v in ('coverage','learned')]
        a.plot([0,1],y,'o-',label=str(seed),color=color)
        for x,v in enumerate(y):a.annotate(f'{v:.4f}',(x,v),xytext=(5,6),textcoords='offset points')
    a.set_xticks([0,1],['Fixed coverage weights','Learned source weights']);a.set_ylabel('RMSE (K)')
    a.set_xlim(-.3,1.5);a.legend(title='Initialization seed');a.grid(axis='y',alpha=.2)
    a.set_title('Same backbone, data schedule and training budget')
    pair=pairs['coverage_minus_learned'];rows=sorted(pair['city_deltas'],key=lambda r:r['delta'])
    palette={'us':'#4b84a6','china':'#ba7440','europe':'#5e9974'}
    a=axes[1];a.barh(np.arange(len(rows)),[r['delta'] for r in rows],
        color=[palette.get(r['region'],'#4b84a6') for r in rows])
    a.set_yticks(np.arange(len(rows)),[r['city'] for r in rows],fontsize=7)
    a.axvline(0,color='#333333',lw=.8);a.set_xlabel('Fixed minus learned RMSE (K)')
    lo,hi=pair['confidence_interval_95']
    a.set_title(f"City-paired gain: {pair['delta']:.4f} K\n95% stratified bootstrap interval [{lo:.4f}, {hi:.4f}]")
    fig.tight_layout();save(fig,'structural_attribution')

    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    selections=[(['history_0','history_1_2','history_3_5','history_6_9'],['0','1-2','3-5','6-9'],'Visible historical observations'),
                (['state_below','state_inside','state_above'],['Below','Inside','Above'],'Current coarse temperature vs historical envelope (+/-1 K)')]
    for ax,(keys,labels,xlabel) in zip(axes,selections):
        for method,color in [('coverage','#8496a5'),('learned','#176d9c'),('final_0.426','#c36f28')]:
            values=[strata[k]['methods'].get(method,{}).get('rmse',np.nan) for k in keys]
            ax.plot(np.arange(len(keys)),values,'o-',color=color,label=method)
        ax.set_xticks(np.arange(len(keys)),[f"{l}\n{strata[k]['cities']} cities" for l,k in zip(labels,keys)])
        ax.set_xlabel(xlabel);ax.set_ylabel('RMSE (K)');ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Input-defined subsets, common formal support');fig.tight_layout();save(fig,'input_stratification')

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for seed,ax in zip((20260914,20260915),axes[:2]):
        for variant,color in [('coverage','#8496a5'),('learned','#176d9c')]:
            rows=read(HERE/'attribution'/f'{variant}_{seed}'/'validation.json')
            rows=[r for r in rows if r['weights']=='ema' and r['step']>0]
            ax.plot([r['step'] for r in rows],[r['rmse'] for r in rows],'o-',ms=3,color=color,label=variant)
        ax.set_title(f'Matched seed {seed}');ax.legend()
    ax=axes[2]
    for stage,color in [(1,'#5e9974'),(2,'#c36f28')]:
        rows=[r for r in read(HERE/'thst'/f'stage{stage}'/'validation.json') if r['step']>0]
        ax.plot([r['step'] for r in rows],[r['rmse'] for r in rows],'o-',ms=3,color=color,label=f'THST stage {stage}')
    ax.set_title('THSTNet two-stage convergence');ax.legend()
    tail=HERE/'thst_task_loss/validation.json'
    if OUT.name=='analysis_augmented' and tail.exists():
        rows=[v for v in read(tail) if v['weights']=='ema']
        ax.plot([6000+v['step'] for v in rows],[v['rmse'] for v in rows],'o--',ms=3,color='#755a95',label='THST task-loss EMA')
        ax.legend()
    for ax in axes:ax.set_xlabel('Optimizer updates');ax.set_ylabel('Full Val45 RMSE (K)');ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'validation_curves')
if __name__=='__main__':main()
