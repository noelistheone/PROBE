#!/usr/bin/env python
r"""Transfer table: the geometric regularizer applied directly to the pre-trained encoder.

All arms are dose-matched at mu_g=1 so the comparison isolates the exponent, and the last column
reports what validation selects over the full six-candidate (beta, mu_g) grid.
"""
import json, math, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'wm')
SEEDS = ['2024', '2025', '2026']
GRID = [(r'$(0,1)$', 'XSimGCLg_w10'), (r'$(0,2)$', 'XSimGCLg_w20'), (r'$(0.5,1)$', 'AdaG_b05'),
        (r'$(1,1)$', 'AdaG_b10'), (r'$(0.5,2)$', 'AdaG_b05w2'), (r'$(1,2)$', 'AdaG_b10w2')]
DS = [('amazon-kindle', 'Amazon-Kindle', '0.014\\%'), ('yelp2018', 'Yelp2018', '0.130\\%'),
      ('douban-book', 'Douban-Book', '0.209\\%'), ('ml-1M', 'ML-1M', '2.697\\%')]

def vt(tag, ds):
    vs, ts = [], []
    for s in SEEDS:
        f = f'{R}/{tag}__{ds}__seed{s}.json'
        if os.path.exists(f):
            r = json.load(open(f)); vs.append(r['val_metrics']['NDCG']); ts.append(r['metrics']['NDCG@20'])
    return (sum(vs) / len(vs), sum(ts) / len(ts)) if vs else (None, None)

def cell(t, base):
    return '--' if t is None else f'${t:.4f}$ \\,(${100*(t-base)/base:+.1f}\\%$)'

print(r"""\begin{table}[t]
\centering
\caption{The geometric regularizer applied \emph{directly to the pre-trained encoder}, with the
prompt-tuning stage removed; $\Delta$ is against that encoder and all three regularized columns are
dose-matched at $\mu_g{=}1$, so the comparison isolates the exponent. The last column is what
validation selects over the full six-candidate $(\beta,\mu_g)$ grid of Table~\ref{tab:selection},
including the option of not regularizing at all. Three seeds throughout.}
\label{tab:transfer}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrccccl}
\toprule
Dataset & Density & Encoder & +uniform & +DA $\beta{=}0.5$ & +DA $\beta{=}1$ & validation selects \\
\midrule""")
for ds, name, dens in DS:
    ev, et = vt('XSimGCLg_w00', ds)
    if et is None:
        continue
    cells = [cell(vt(t, ds)[1], et) for t in ['XSimGCLg_w10', 'AdaG_b05', 'AdaG_b10']]
    cands = [(lab, *vt(t, ds)) for lab, t in GRID]
    cands = [c for c in cands if c[1] is not None]
    best = max(cands, key=lambda c: c[1])
    pick = best[0] if best[1] > ev else 'encoder (no reg.)'
    picked_test = best[2] if best[1] > ev else et
    print(f'{name} & {dens} & ${et:.4f}$ & ' + ' & '.join(cells) +
          f' & {pick}, ${picked_test:.4f}$ \\\\')
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
