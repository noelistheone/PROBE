#!/usr/bin/env python
r"""Exposure of the top-20 lists on every benchmark: who gets recommended, not only how accurately."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'wm')
S = ['2024', '2025', '2026']
DATASETS = [('Yelp2018', 'yelp2018'), ('Douban-Book', 'douban-book'), ('ML-1M', 'ml-1M')]
MODELS = [('XSimGCL (backbone)', 'XSimGCLg_w00'), ('DirectAU', 'DirectAU'), ('PT4Rec', 'PTbase_XSim_nofz'),
          (r'Ours$_{-g}$', 'OURS_XSim_nofz'), ('Ours', 'OURSgeom_w2')]
# (metric key, decimals); lower ARP and Gini mean less concentrated exposure
COLS = [('NDCG@20', 4), ('TailRecall@20', 4), ('ItemCoverage@20', 3), ('ARP@20', 3), ('Novelty@20', 2),
        ('Gini@20', 3)]


def cell(tag, ds, key, nd):
    xs = [json.load(open(f'{R}/{tag}__{ds}__seed{s}.json'))['metrics'][key] for s in S]
    # exact decimal mean, rounded half-up (float sum() differs across Python versions)
    m = sum(Decimal(repr(x)) for x in xs) / len(xs)
    return str(m.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


print(r"""\begin{table}[t]
\centering
\caption{Exposure of top-20 lists (three seeds). Tail: recall on the $80\%$ least popular items (by
training degree); Cov.: item coverage; ARP: mean item degree over the maximum degree; Nov.: mean
self-information ($-\log_2$ popularity share). Lower ARP and Gini mean less concentrated exposure.}
\label{tab:exposure}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
System & NDCG & Tail & Cov. & ARP & Nov. & Gini \\
\midrule""")
for i, (name, ds) in enumerate(DATASETS):
    if i:
        print(r'\addlinespace')
    print(r'\multicolumn{7}{l}{\emph{%s}} \\' % name)
    for label, tag in MODELS:
        print(f'{label} & ' + ' & '.join(cell(tag, ds, k, nd) for k, nd in COLS) + r' \\')
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
