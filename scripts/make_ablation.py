#!/usr/bin/env python
r"""Leave-one-out ablation table, both benchmarks, each judged against its OWN measured noise floor."""
import glob, json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'wm')
S = ['2024', '2025', '2026']
FLOOR = {'douban-book': 0.00065, 'ml-1M': 0.00219}   # measured, see the noise-floor subsection

def series(tag, ds):
    return [json.load(open(f'{R}/{tag}__{ds}__seed{s}.json'))['metrics']['NDCG@20']
            for s in S if glob.glob(f'{R}/{tag}__{ds}__seed{s}.json')]

def cell(tag, ds, base):
    v = series(tag, ds)
    if not v:
        return '--', '--'
    m, sd = sum(v) / len(v), (st.stdev(v) if len(v) > 1 else 0.0)
    if tag == 'AB_full':
        return '$%.5f$ {\\scriptsize$\\pm$%.5f}' % (m, sd), '---'
    d = m - base
    return ('$%.5f$ {\\scriptsize$\\pm$%.5f}' % (m, sd),
            '$%+.5f$ (%s)' % (d, ('%.0f$\\times$' % (abs(d)/FLOOR[ds])) if abs(d) >= FLOOR[ds] else 'below'))

ROWS = [('Full system', 'AB_full'), (r'\;$-$ geometric module', 'AB_nogeom'),
        (r'\;$-$ signed router', 'AB_nodual'), (r'\;$-$ popularity residual', 'AB_nopop'),
        (r'\;$-$ popularity, $\gamma{=}0$ control', 'AB_popg0'),
        (r'\;$-$ hard negatives', 'AB_nohn'), ('Geometric module only', 'AB_geomonly')]
base = {ds: sum(series('AB_full', ds)) / len(series('AB_full', ds)) for ds in FLOOR}

print(r"""\begin{table}[t]
\centering
\caption{Leave-one-out ablation (three seeds, NDCG@20). $\Delta$ is against the full system, and the
multiple is of \emph{that dataset's} measured noise floor ($0.00065$ Douban-Book, $0.00219$ ML-1M).
Only the popularity residual is inert on both; the router and the hard negatives are large and
opposite in sign across the two, while removing all three together is harmless on one and beneficial
on the other. The ML-1M arm is leave-one-out from the geometry-\emph{on} configuration.}
\label{tab:ablation}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Douban-Book} & \multicolumn{2}{c}{ML-1M} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Variant & NDCG@20 & $\Delta$ & NDCG@20 & $\Delta$ \\
\midrule""")
for name, tag in ROWS:
    d_v, d_d = cell(tag, 'douban-book', base['douban-book'])
    m_v, m_d = cell(tag, 'ml-1M', base['ml-1M'])
    print(f'{name} & {d_v} & {d_d} & {m_v} & {m_d} \\\\')
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
