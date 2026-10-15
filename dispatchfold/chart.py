"""Render docs/before_after.png from a benchmark run."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#dddcd6'
# categorical slots in fixed order: entity 1 the compiler baseline,
# entity 2 this tool, entity 3 the fold everywhere policy
C_TABLE = '#2a78d6'
C_PICK = '#eb6834'
C_FOLD = '#1baf7a'


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_chart(results_path='docs/results.json', out_path='docs/before_after.png'):
    with open(results_path) as fh:
        data = json.load(fh)
    rows = sorted(data['workloads'], key=lambda r: -r['speedup_vs_baseline'])
    totals = data['totals']

    fig = plt.figure(figsize=(13.2, 6.2), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.28,
                          left=0.185, right=0.975, top=0.80, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)

    ys = range(len(rows))
    h = 0.33
    gap = 0.035
    before = [r['baseline_ns'] for r in rows]
    after = [r['chosen_ns'] for r in rows]
    ax.barh([y - (h + gap) / 2 for y in ys], before, height=h,
            color=C_TABLE, label='compiler jump table', zorder=3)
    ax.barh([y + (h + gap) / 2 for y in ys], after, height=h,
            color=C_PICK, label='dispatchfold', zorder=3)

    span = max(before)
    for y, r in zip(ys, rows):
        ax.text(r['baseline_ns'] + span * 0.012, y - (h + gap) / 2,
                '%.2f' % r['baseline_ns'], va='center', ha='left',
                fontsize=8.5, color=INK_2)
        tag = '%.2f' % r['chosen_ns']
        if r['speedup_vs_baseline'] > 1.01:
            tag += '   %.2fx faster' % r['speedup_vs_baseline']
        else:
            tag += '   kept the jump table'
        ax.text(r['chosen_ns'] + span * 0.012, y + (h + gap) / 2, tag,
                va='center', ha='left', fontsize=8.5, color=INK)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(['%s\n%d cases, %.0f%% of keys mispredicted'
                        % (r['workload'], r['cases'], r['miss_rate'] * 100)
                        for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, span * 1.34)
    ax.set_xlabel('nanoseconds per dispatch, lower is better', fontsize=9,
                  color=INK_2)
    ax.set_title('Per site, on six real key streams from this machine',
                 fontsize=10.5, color=INK, loc='left', pad=10)
    ax.legend(frameon=False, fontsize=9, loc='lower right',
              labelcolor=INK_2)
    ax.invert_yaxis()

    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2)
    names = ['jump table\neverywhere', 'fold\neverywhere',
             'dispatchfold\ndecides per site']
    vals = [totals['always_table_ns'], totals['always_fold_ns'],
            totals['dispatchfold_ns']]
    colors = [C_TABLE, C_FOLD, C_PICK]
    ax2.bar(range(3), vals, width=0.52, color=colors, zorder=3)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax2.xaxis.grid(False)
    for i, v in enumerate(vals):
        ax2.text(i, v + max(vals) * 0.02, '%.1f ns' % v, ha='center',
                 fontsize=9, color=INK)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(names, fontsize=8.5, color=INK)
    ax2.set_ylim(0, max(vals) * 1.16)
    ax2.set_ylabel('sum of ns per dispatch over the six sites', fontsize=9,
                   color=INK_2)
    ax2.set_title('Policy, summed over all six sites', fontsize=10.5,
                  color=INK, loc='left', pad=10)

    fig.suptitle('Switch dispatch cost is set by branch mispredicts, so the '
                 'right lowering depends on the key stream',
                 fontsize=13.5, color=INK, x=0.013, ha='left', y=0.955)
    fig.text(0.013, 0.885,
             'Measured by JIT emitting each lowering as x86-64 and timing it '
             'on %s. Every kernel returns a bit identical result.'
             % data['host']['cpu'],
             fontsize=9, color=INK_2, ha='left')
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
