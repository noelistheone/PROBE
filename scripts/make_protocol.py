#!/usr/bin/env python
r"""Protocol-decomposition table: each shortcut isolated as one binary condition, on both benchmarks."""
import json, math, os, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
WM = os.path.join(R, "wm")
S = ['2024', '2025', '2026']
# frozen counterparts differ from their joint arm ONLY in -freeze_encoder
FROZEN = {'PTbase_XSim_nofz': 'PTbase_XSim', 'OURSgeom_w2': 'OURSgeom_w2_fz'}
ROWS = [('XSimGCL (backbone)', 'XSimGCLg_w00'), ('PT4Rec', 'PTbase_XSim_nofz'), ('Ours', 'OURSgeom_w2')]

def ser(tag, ds, d=WM):
    return [json.load(open(f'{d}/{tag}__{ds}__seed{s}.json'))['metrics']['NDCG@20'] for s in S
            if os.path.exists(f'{d}/{tag}__{ds}__seed{s}.json')]

def block(ds):
    out = []
    for name, tag in ROWS:
        base = ser(tag, ds)
        if not base:
            continue
        b = sum(base) / len(base)
        cells = [f'${b:.4f}$']
        for arm in ['_SELTEST', '_VR0']:
            v = ser(tag + arm, ds)
            cells.append(f'${sum(v)/len(v):.4f}$ \\,(${100*(sum(v)/len(v)-b)/b:+.1f}\\%$)' if v else '--')
        fz = FROZEN.get(tag)
        v = ser(fz, ds) if fz else []
        cells.append(f'${sum(v)/len(v):.4f}$ \\,($\\times{sum(v)/len(v)/b:.2f}$)' if v else '---')
        out.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    return out

print(r"""\begin{table}[t]
\centering
\caption{Each shortcut isolated as one binary condition, on both benchmarks (three seeds, NDCG@20).
``Selection on test'' keeps the validation split carved out and changes only which set picks the
reported epoch; ``no validation split'' is the common implementation, in which the held-out
interactions return to training \emph{and} selection sees the test set. Isolated, the selection
shortcut stays below each dataset's noise floor in five of the six arms---the exception is our own
model on ML-1M, at $2.1\times$ the floor. Almost all of the inflation comes from the data the split
would have removed, and the frozen/joint discrepancy is larger than either. Each arm differs from its
controlled counterpart in exactly one setting.}
\label{tab:protocol}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
System & controlled & selection on test & no val.\ split & encoder frozen \\
\midrule
\multicolumn{5}{l}{\emph{Douban-Book}} \\""")
for r in block('douban-book'):
    print(r)
print(r'\addlinespace')
print(r'\multicolumn{5}{l}{\emph{ML-1M}} \\')
for r in block('ml-1M'):
    print(r)
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
