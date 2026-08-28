#!/usr/bin/env python
"""Generate the main IEEE results table straight from results/wm/*.json.
Columns with no data on any dataset are dropped automatically."""
import glob, json, math, sys
# SAC version: NDCG and Recall only. Hit Ratio and Precision are near-monotone transforms of these
# (the harness's HR is micro-averaged recall, and P@N is R@N rescaled per user); both are released
# per seed. Set FULL=True to restore all eight.
FULL = False
METS=[('HitRatio@10','HR@10'),('HitRatio@20','HR@20'),('Precision@10','P@10'),('Precision@20','P@20'),
      ('NDCG@10','N@10'),('NDCG@20','N@20'),('Recall@10','R@10'),('Recall@20','R@20')]
if not FULL: METS=[m for m in METS if m[1].startswith(('N@','R@'))]
BASE=[('MF','MF'),('LightGCN','LGCN'),('SGL','SGL'),('NCL','NCL'),('SSL4Rec','SSL4Rec'),
      ('DirectAU','DirectAU'),('BUIR','BUIR'),('SelfCF','SelfCF'),('CPTPP','CPTPP'),('LightGCL','LightGCL'),
      ('SimGCL','SimGCL')]
BACK=[('XSimGCLg_w00','XSimGCL$^{\\dagger}$')]
OURS=[('PTbase_XSim_nofz','PT4Rec'),('OURS_XSim_nofz','Ours$_{-g}$'),('OURSgeom_w2','\\textbf{Ours}')]
DS=[('douban-book','Douban-Book'),('ml-1M','ML-1M'),('yelp2018','Yelp2018')]
SEEDS=['2024','2025','2026']   # canonical seed set: every table in the paper uses exactly these
def f4(x):
    import math
    return f"{math.floor(x*1e4+0.5)/1e4:.4f}"
def st(t,ds,m):
    fs=[f'./results/wm/{t}__{ds}__seed{s}.json' for s in SEEDS]
    v=[json.load(open(f))['metrics'][m] for f in fs if glob.glob(f)]
    if not v: return None
    return sum(v)/len(v), len(v)
def has(t): return any(st(t,ds,'NDCG@20') for ds,_ in DS)
base=[x for x in BASE if has(x[0])]; tags=base+BACK+OURS
nb=len(base)
spec='ll '+'c'*nb+'|c|'+'c'*len(OURS)
print(r"""\begin{table*}[t]
\centering
\caption{Overall comparison under the leakage-controlled protocol of Section~\ref{sec:setup}
(validation-only model selection, full ranking over the catalogue, mean over three seeds). Seed
variation is small: over all released cells the seed standard deviation has median $0.0004$ and
$90$th percentile $0.0026$ ($0.0019$ excluding the unstable LightGCL rows); across both of our columns
the maximum is $0.0012$. Per-seed values are released. LightGCL is stable on Douban-Book and Yelp2018 but varies by a factor of two across seeds on ML-1M in our implementation ($0.095$, $0.188$, $0.205$); its ML-1M mean should be read with that in mind. Best per row in
\textbf{bold}. $\dagger$~XSimGCL is the pre-trained backbone that PT4Rec and our model adapt, shown
as the reference a prompt-tuning method must improve upon. Ours$_{-g}$ disables the degree-adaptive geometric
module (retaining only the small fixed-weight anchoring and uniformity terms); its on/off state is
selected per dataset on validation data.}
\label{tab:main}
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{%
\begin{tabular}{"""+spec+r"""}
\toprule
Dataset & Metric & """ + " & ".join(n for _,n in tags) + r" \\"+"\n\\midrule")
for di,(ds,dn) in enumerate(DS):
    for r,(mk,mn) in enumerate(METS):
        vals=[(n,st(t,ds,mk)) for t,n in tags]
        pres=[v[0] for _,v in vals if v]; best=max(pres) if pres else None
        cells=['--' if v is None else (f"\\textbf{{{f4(v[0])}}}" if abs(v[0]-best)<1e-12 else f4(v[0])) for _,v in vals]
        lead=f"\\multirow{{{len(METS)}}}{{*}}{{{dn}}}" if r==0 else ""
        print(f"{lead} & {mn} & "+" & ".join(cells)+r" \\")
    if di<len(DS)-1: print(r"\midrule")
print(r"""\bottomrule
\end{tabular}}
\end{table*}""")
