#!/usr/bin/env python
r"""Every per-dataset choice, the validation score that decided it, and the test score that followed."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'wm')
S = ['2024', '2025', '2026']

def vt(tag, ds):
    vs, ts = [], []
    for s in S:
        f = f'{R}/{tag}__{ds}__seed{s}.json'
        if os.path.exists(f):
            r = json.load(open(f)); vs.append(r['val_metrics']['NDCG']); ts.append(r['metrics']['NDCG@20'])
    # exact decimal means: float sum() differs across Python versions (3.12+ compensates), which flips
    # half-way cases such as 0.15285 under '%.4f'; the paper rounds half-up on the exact mean
    mean = lambda xs: sum(Decimal(repr(x)) for x in xs) / len(xs)
    return (mean(vs), mean(ts)) if vs else (None, None)

MODULE = [('Yelp2018', 'yelp2018'), ('Douban-Book', 'douban-book'), ('ML-1M', 'ml-1M')]
# no geometry-disabled pipeline run exists for Amazon-Kindle, so it appears only in the encoder grid
ENCODER = [('Kindle', 'amazon-kindle')] + MODULE
GRID = [(r'none', 'XSimGCLg_w00'), (r'$(0,1)$', 'XSimGCLg_w10'), (r'$(0,2)$', 'XSimGCLg_w20'),
        (r'$(0.5,1)$', 'AdaG_b05'), (r'$(1,1)$', 'AdaG_b10'), (r'$(0.5,2)$', 'AdaG_b05w2'),
        (r'$(1,2)$', 'AdaG_b10w2')]

def bold(x, on):
    q = x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    return r'\textbf{%s}' % q if on else '%s' % q

print(r"""\begin{table}[t]
\centering
\caption{Validation and test NDCG@20 behind each selection decision (selected in \textbf{bold}). Top: enabling \textsc{DAGR} in the pipeline. Bottom: encoder-only grid over exponent $\beta$ and dose $\mu_g$ ($\beta{=}0$ is uniform). Validation picks the test-best option except on Yelp2018 (\S\ref{sec:selection}).}
\label{tab:selection}
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\multicolumn{3}{l}{\emph{Geometric module}} & \multicolumn{2}{c}{enabled} & \multicolumn{2}{c}{disabled} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\multicolumn{3}{l}{} & valid. & test & valid. & test \\
\midrule""")
for name, ds in MODULE:
    on_v, on_t = vt('OURSgeom_w2', ds)
    off_v, off_t = vt('OURS_XSim_nofz', ds)
    pick_on = on_v > off_v
    print(f'\\multicolumn{{3}}{{l}}{{{name}}} & {bold(on_v,pick_on)} & {bold(on_t,pick_on)} '
          f'& {bold(off_v,not pick_on)} & {bold(off_t,not pick_on)} \\\\')
print(r'\midrule')
print(r'\emph{Encoder-only} $(\beta,\mu_g)$ & ' + ' & '.join(lab for lab, _ in GRID) + r' \\')
print(r'\midrule')
for name, ds in ENCODER:
    vs = [vt(t, ds) for _, t in GRID]
    bi = max(range(len(vs)), key=lambda k: vs[k][0] if vs[k][0] is not None else -1)
    print(f'{name} valid. & ' + ' & '.join(bold(v, k == bi) if v is not None else '--'
                                                for k, (v, _) in enumerate(vs)) + r' \\')
    print(f'{name} test & ' + ' & '.join(bold(t, k == bi) if t is not None else '--'
                                          for k, (_, t) in enumerate(vs)) + r' \\')
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
