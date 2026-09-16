"""Export the measured accuracy/cost comparison after all GPU runs finish."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis_augmented'


def main():
    read = lambda p: json.loads(p.read_text())
    scores = read(OUT / 'scores.json')
    timing = read(OUT / 'original_inference_benchmark.json')['results']
    timing.update(read(OUT / 'task_loss_inference_benchmark.json')['results'])
    methods = ['final_0.426', 'THST_task_loss_matched', 'THST_task_loss_all']
    labels = ['Final NAF', 'THST: matched reference', 'THST: all references']
    colors = ['#176d9c', '#ba7440', '#8496a5']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={'width_ratios': [1.6, 1]})
    for name, label, color in zip(methods, labels, colors):
        t = timing[name]
        x = 1000 * t['median_across_case_medians']
        y = scores[name]['macro']['rmse']
        low = 1000 * min(c['median_seconds'] for c in t['cases'])
        high = 1000 * max(c['median_seconds'] for c in t['cases'])
        axes[0].errorbar(x, y, xerr=[[x-low], [high-x]], fmt='o', color=color,
                         capsize=4, markersize=8, label=label)
        axes[0].annotate(f'{x:.1f} ms; {y:.3f} K', (x, y), xytext=(0, 12),
                         textcoords='offset points', ha='center', fontsize=9)
    axes[0].set_xscale('log')
    axes[0].set_xticks([30, 100, 300, 1000])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_xlim(25, 1350)
    axes[0].set_ylim(.39, .72)
    axes[0].set_xlabel('Complete 160 x 160 scene inference (ms; logarithmic axis)')
    axes[0].set_ylabel('Test90 RMSE (K)')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=.2)
    values = [timing[k]['parameters']/1e6 for k in ('final_0.426', 'THST_task_loss_all')]
    axes[1].bar([0, 1], values, color=[colors[0], colors[2]], width=.55)
    for i, v in enumerate(values):
        axes[1].text(i, v+.6, f'{v:.2f} M', ha='center')
    axes[1].set_xticks([0, 1], ['Final NAF', 'THSTNet'])
    axes[1].set_ylim(0, 40)
    axes[1].set_ylabel('Parameters (millions)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Measured accuracy and inference cost | RTX 5060 Ti, FP32')
    fig.text(.5, -.035, 'Timing: three Val scenes, 3 warmups + 10 repetitions each; bars span case medians. Both THST variants include task-loss calibration.',
             ha='center', fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('accuracy_efficiency.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
