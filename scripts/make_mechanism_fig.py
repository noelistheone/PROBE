#!/usr/bin/env python
"""Figure: where a uniform dose does its damage, and where degree-adaptive weighting repairs it."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from per_user_significance import user_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'SAC2027', 'fig')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 6.5, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'figure.dpi': 300,
                     'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'serif', 'axes.grid': True, 'grid.alpha': .25,
                     'grid.linewidth': .4, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01})

ARMS = [('uniform', 'XSimGCLg_w10', '#B2182B', 'o'),
        (r'adaptive $\beta$=0.5', 'AdaG_b05', '#E08214', 's'),
        (r'adaptive $\beta$=1', 'AdaG_b10', '#2166AC', '^')]
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.12), sharey=True)
for ax, ds, title in zip(axes, ['ml-1M', 'douban-book'], ['ML-1M', 'Douban-Book']):
    sc = {}
    enc, deg, _ = user_scores('XSimGCLg_w00', ds, '2024', 'XSimGCLg')
    for name, tag, _, _ in ARMS:
        s, d, _ = user_scores(tag, ds, '2024', 'XSimGCLg')
        if s:
            sc[name] = s
            deg.update(d)
    users = sorted(set(enc).intersection(*[set(v) for v in sc.values()]), key=lambda u: (deg.get(u, 0), int(u)))
    q = len(users) // 5
    groups = [users[i*q:(i+1)*q] if i < 4 else users[4*q:] for i in range(5)]
    base = [sum(enc[u] for u in g) / len(g) for g in groups]
    for name, tag, col, mk in ARMS:
        if name not in sc: continue
        y = [100 * (sum(sc[name][u] for u in g) / len(g) - b) / b for g, b in zip(groups, base)]
        ax.plot(range(5), y, marker=mk, ms=2.6, lw=1.1, color=col, label=name)
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels([str(deg.get(g[len(g)//2], 0)) for g in groups])
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('median user degree')
axes[0].set_ylabel(r'$\Delta$NDCG@20 (%)')
axes[1].legend(frameon=False, loc='lower left')  # Douban's lower half is empty; on ML-1M the legend overprinted the curves
fig.savefig(os.path.join(OUT, 'mechanism.pdf'))
print('wrote', os.path.join(OUT, 'mechanism.pdf'))
