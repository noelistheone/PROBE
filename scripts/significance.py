#!/usr/bin/env python
"""Paired significance tests for the main table: our system vs the strongest baseline
on each dataset, over the canonical seeds. Also reports 95% CIs and effect sizes."""
import glob, json, math, os, statistics as st

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'wm')
SEEDS = ['2024', '2025', '2026']
BASELINES = ['MF', 'LightGCN', 'SGL', 'NCL', 'SSL4Rec', 'DirectAU', 'BUIR', 'SelfCF', 'CPTPP', 'LightGCL']
BACKBONE = 'XSimGCLg_w00'
OURS = {'douban-book': 'OURSgeom_w2', 'ml-1M': 'OURS_XSim_nofz', 'yelp2018': 'OURSgeom_w2'}
NOISE = 0.00065

def series(tag, ds, met):
    out = []
    for s in SEEDS:
        f = f'{R}/{tag}__{ds}__seed{s}.json'
        if os.path.exists(f):
            m = json.load(open(f))['metrics']
            if met in m: out.append(m[met])
    return out

def ttest_paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2: return float('nan'), float('nan'), st.mean(d) if d else float('nan')
    sd = st.stdev(d)
    if sd == 0: return float('inf'), 0.0, st.mean(d)
    t = st.mean(d) / (sd / math.sqrt(n))
    return t, sd, st.mean(d)

# two-sided p for t with df=2 (closed form: p = 1 - (2/pi)*[atan(x) + x/(1+x^2)] for df=2)
def p_df2(t):
    x = abs(t)
    if math.isinf(x): return 0.0
    cdf = 0.5 + (1/math.pi) * (math.atan(x/math.sqrt(2)) + (x/math.sqrt(2)) / (1 + x*x/2))
    return 2 * (1 - cdf)

print(f'{"dataset":14s} {"metric":9s} {"ours":>8s} {"best baseline":>22s} {"delta":>9s} {"t(2)":>7s} {"p":>7s} {"vs noise":>9s}')
print('-' * 92)
for ds in ['douban-book', 'ml-1M', 'yelp2018']:
    ours_tag = OURS[ds]
    for met in ['NDCG@20', 'Recall@20']:
        o = series(ours_tag, ds, met)
        cands = [(b, series(b, ds, met)) for b in BASELINES + [BACKBONE]]
        cands = [(b, v) for b, v in cands if len(v) == len(o) and v]
        best, bv = max(cands, key=lambda kv: sum(kv[1]) / len(kv[1]))
        t, sd, md = ttest_paired(o, bv)
        p = p_df2(t)
        ratio = abs(md) / NOISE
        print(f'{ds:14s} {met:9s} {sum(o)/len(o):8.4f} {best:>14s} {sum(bv)/len(bv):7.4f} '
              f'{md:+9.5f} {t:7.2f} {p:7.4f} {ratio:8.1f}x')
    ci = 1.96 * st.stdev(series(ours_tag, ds, 'NDCG@20')) / math.sqrt(3)
    print(f'{"":14s} 95% CI half-width on our NDCG@20 (3 seeds): +-{ci:.5f}\n')
