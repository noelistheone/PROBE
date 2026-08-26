#!/usr/bin/env python
"""Transfer table: the geometric regularizer applied directly to the pre-trained encoder.

Uses the three canonical seeds (2024-2026) wherever they exist; cells backed by fewer runs are
marked, as are cells produced under the earlier negative sampler.
"""
import json, math, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
SEEDS = ['2024', '2025', '2026']

def ms(tag, ds):
    """-> (mean, n_seeds, 'cur'|'old') or None."""
    for sub, src in (('wm', 'cur'), ('prev_sampler', 'old'), ('prev_sampler', 'old')):
        fs = [os.path.join(R, sub, f'{tag}__{ds}__seed{s}.json') for s in SEEDS]
        v = [json.load(open(f))['metrics']['NDCG@20'] for f in fs if os.path.exists(f)]
        if v:
            return sum(v) / len(v), len(v), src
    return None

def f4(x):
    return f'{math.floor(x * 1e4 + 0.5) / 1e4:.4f}'

def marks(r):
    return ('$^{\\ddagger}$' if r[2] == 'old' else '') + ('$^{*}$' if r[1] < 3 else '')

def cell(r, base):
    if not r:
        return '--'
    return f"${f4(r[0])}${marks(r)}\\,(${100*(r[0]-base[0])/base[0]:+.1f}\\%$)"

DS = [('amazon-kindle', 'Amazon-Kindle', '0.014\\%'), ('yelp2018', 'Yelp2018', '0.130\\%'),
      ('douban-book', 'Douban-Book', '0.209\\%'), ('ml-1M', 'ML-1M', '2.697\\%')]

print(r"""\begin{table}[t]
\centering
\caption{The geometric regularizer of \eqref{eq:gate}--\eqref{eq:dagr} applied \emph{directly to the
pre-trained encoder}, with the prompt-tuning stage removed. $\Delta$ is against that encoder, and the
uniform and adaptive columns are dose-matched at $\mu_g{=}2$. Uniform weighting helps on Douban-Book
and is destructive on the dense benchmark; degree adaptation bounds that damage without turning it
into a gain, and no exponent is best everywhere. Three seeds throughout, all under the same negative
sampler; $\beta{=}1$ was not run on Amazon-Kindle.}
\label{tab:transfer}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrcccc}
\toprule
Dataset & Density & Encoder & +uniform & +DA ($\beta{=}0.5$) & +DA ($\beta{=}1$) \\
\midrule""")
for ds, name, dens in DS:
    b = ms('XSimGCLg_w00', ds)
    if not b:
        continue
    row = f"{name} & {dens} & ${f4(b[0])}${marks(b)} & " + " & ".join(
        cell(ms(t, ds), b) for t in ['XSimGCLg_w20', 'AdaG_b05', 'AdaG_b10'])
    print(row + r" \\")
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
