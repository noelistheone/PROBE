#!/usr/bin/env python
r"""Every per-dataset choice, the validation score that decided it, and the test score that followed."""
import json, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'wm')
S = ['2024', '2025', '2026']

def vt(tag, ds):
    vs, ts = [], []
    for s in S:
        f = f'{R}/{tag}__{ds}__seed{s}.json'
        if os.path.exists(f):
            r = json.load(open(f)); vs.append(r['val_metrics']['NDCG']); ts.append(r['metrics']['NDCG@20'])
    return (sum(vs) / len(vs), sum(ts) / len(ts)) if vs else (None, None)

MODULE = [('Douban-Book', 'douban-book'), ('ML-1M', 'ml-1M')]
GRID = [(r'none', 'XSimGCLg_w00'), (r'$(0,1)$', 'XSimGCLg_w10'), (r'$(0,2)$', 'XSimGCLg_w20'),
        (r'$(0.5,1)$', 'AdaG_b05'), (r'$(1,1)$', 'AdaG_b10'), (r'$(0.5,2)$', 'AdaG_b05w2'),
        (r'$(1,2)$', 'AdaG_b10w2')]

def bold(x, on):
    return r'\textbf{%.4f}' % x if on else '%.4f' % x

print(r"""\begin{table}[t]
\centering
\caption{Every per-dataset choice, the validation score that decided it and the test score that
followed; the chosen option is in \textbf{bold}. Top: whether the geometric module is enabled. Bottom:
the exponent $\beta$ and dose $\mu_g$ of the encoder-only regularizer, with every candidate listed --
including \emph{none}, i.e.\ not regularizing at all ($\beta{=}0$ is uniform weighting). In all four
decisions the validation ordering agrees with the test ordering, so none would have changed had we
selected on test.}
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
for name, ds in MODULE:
    vs = [vt(t, ds) for _, t in GRID]
    bi = max(range(len(vs)), key=lambda k: vs[k][0] if vs[k][0] is not None else -1)
    print(f'{name}, validation & ' + ' & '.join(bold(v, k == bi) if v is not None else '--'
                                                for k, (v, _) in enumerate(vs)) + r' \\')
    print(f'{name}, test & ' + ' & '.join(bold(t, k == bi) if t is not None else '--'
                                          for k, (_, t) in enumerate(vs)) + r' \\')
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
