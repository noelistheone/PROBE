#!/usr/bin/env python
r"""Per-degree-quintile table: where a uniform dose does its damage, and where the gate repairs it."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_user_significance import find_dump
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def degrees(ds):
    d = collections.Counter()
    for line in open(os.path.join(ROOT, 'dataset', ds, 'train.txt')):
        p = line.split()
        if len(p) >= 2:
            d[p[0]] += 1
    return d

ARMS = [('encoder', 'XSimGCLg_w00'), (r'\;$+$uniform', 'XSimGCLg_w10'),
        (r'\;$+$gated $\beta{=}0.5$', 'AdaG_b05'), (r'\;$+$gated $\beta{=}1$', 'AdaG_b10')]

print(r"""\begin{table}[t]
\centering
\caption{Where a \emph{uniform} dose does its damage, and where the gate repairs it: NDCG@20 by user
degree quintile on ML-1M (seed 2024, $\mu_g{=}1$ throughout, $\Delta$ against the encoder). The damage
grows monotonically with degree---exactly the nodes Proposition~\ref{prop:degree} says need least
regularization---and the gate removes it monotonically, recovering $95\%$ of it in the top quintile.
This is the mechanism the aggregate numbers only imply.}
\label{tab:quintile}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
& Q1 & Q2 & Q3 & Q4 & Q5 \\
\midrule""")
ds = 'ml-1M'
scores = {}
for name, tag in ARMS:
    s, _ = find_dump(tag, ds, '2024', 'XSimGCLg')
    if s:
        scores[name] = s
deg = degrees(ds)
users = sorted(set.intersection(*[set(v) for v in scores.values()]), key=lambda u: deg.get(u, 0))
q = len(users) // 5
groups = [users[i * q:(i + 1) * q] if i < 4 else users[4 * q:] for i in range(5)]
print('median degree & ' + ' & '.join(str(deg.get(g[len(g) // 2], 0)) for g in groups) + r' \\')
base = None
for name, _ in ARMS:
    cells = [sum(scores[name][u] for u in g) / len(g) for g in groups]
    if base is None:
        base = cells
        print(f'{name} & ' + ' & '.join(f'${c:.3f}$' for c in cells) + r' \\')
        print(r'\midrule')
    else:
        print(f'{name} & ' + ' & '.join(f'${100*(c-b)/b:+.1f}\\%$' for c, b in zip(cells, base)) + r' \\')
print(r"""\bottomrule
\end{tabular}
\end{table}""")
