#!/usr/bin/env python
"""Recompute every number quoted in the paper from the released per-seed records.

Each block prints the value, the tag(s) it comes from and the seeds used, so any table
cell or in-text figure can be traced back to the JSON records in results/.
Run from the repository root:  python scripts/report_paper_numbers.py
"""
import glob, json, math, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUR = os.path.join(ROOT, 'results', 'wm')
OLD = os.path.join(ROOT, 'results', 'prev_sampler')
NF = os.path.join(ROOT, 'results', 'wm_noisefloor')
SEEDS = ['2024', '2025', '2026']          # the canonical seed set used by every table


def rec(tag, ds, seed, d=CUR):
    f = os.path.join(d, f'{tag}__{ds}__seed{seed}.json')
    return json.load(open(f)) if os.path.exists(f) else None


def series(tag, ds, met='NDCG@20', d=CUR, seeds=SEEDS):
    out = []
    for s in seeds:
        r = rec(tag, ds, s, d)
        if r and met in r['metrics']:
            out.append(r['metrics'][met])
    return out


def mean(tag, ds, met='NDCG@20', d=CUR):
    v = series(tag, ds, met, d)
    return sum(v) / len(v) if v else None


def f4(x):
    return f'{math.floor(x * 1e4 + 0.5) / 1e4:.4f}' if x is not None else '--'


def pct(x, base):
    return f'{100 * (x - base) / base:+.1f}%'


def head(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


head('Table I  Overall comparison (3 seeds, NDCG@20 only; run make_table.py for all metrics)')
TAGS = [('MF', 'MF'), ('LightGCN', 'LightGCN'), ('SGL', 'SGL'), ('NCL', 'NCL'),
        ('SSL4Rec', 'SSL4Rec'), ('DirectAU', 'DirectAU'), ('BUIR', 'BUIR'),
        ('SelfCF', 'SelfCF'), ('CPTPP', 'CPTPP'), ('LightGCL', 'LightGCL'),
        ('XSimGCL (backbone)', 'XSimGCLg_w00'), ('PT4Rec', 'PTbase_XSim_nofz'),
        ('Ours -g', 'OURS_XSim_nofz'), ('Ours', 'OURSgeom_w2')]
for ds in ['douban-book', 'ml-1M', 'yelp2018']:
    print(f'\n{ds}')
    for name, tag in TAGS:
        v = series(tag, ds)
        if v:
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f'  {name:20s} {f4(sum(v)/len(v))}  sd={sd:.5f}  n={len(v)}  [{tag}]')

head('Table II  Protocol decomposition on Douban-Book (paired, 3 seeds)')
FROZEN = {'PTbase_XSim_nofz': 'PTbase_XSim', 'OURSgeom_w2': 'OURS_XSim'}
for name, tag in [('XSimGCL (backbone)', 'XSimGCLg_w00'), ('PT4Rec', 'PTbase_XSim_nofz'),
                  ('Ours', 'OURSgeom_w2')]:
    base = series(tag, 'douban-book')
    b = sum(base) / len(base)
    line = f'  {name:20s} controlled {f4(b)}'
    for arm, lab in [('_SELTEST', 'selection-on-test'), ('_VR0', 'no-val-split')]:
        v = series(tag + arm, 'douban-book')
        if v:
            m = sum(v) / len(v)
            d = [y - x for x, y in zip(base, v)]
            t = st.mean(d) / (st.stdev(d) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
            line += f' | {lab} {f4(m)} ({pct(m, b)}, paired d={st.mean(d):+.5f}, t={t:.2f})'
    fz = FROZEN.get(tag)
    if fz:
        v = series(fz, 'douban-book', d=OLD)
        if v:
            m = sum(v) / len(v)
            line += f' | encoder-frozen {f4(m)} (x{m/b:.2f})'
    print(line)
    print(f'      selected epochs: controlled={[rec(tag, "douban-book", s)["best_val_epoch"] for s in SEEDS]}'
          f' selection-on-test={[rec(tag+"_SELTEST", "douban-book", s)["best_val_epoch"] for s in SEEDS]}')

head('Table III  Leave-one-out ablation on Douban-Book (3 seeds)')
full = series('AB_full', 'douban-book')
mf = sum(full) / len(full)
for name, tag in [('full system', 'AB_full'), ('- geometric module', 'AB_nogeom'),
                  ('- signed router', 'AB_nodual'), ('- popularity residual', 'AB_nopop'),
                  ('- hard negatives', 'AB_nohn'), ('geometric module only', 'AB_geomonly')]:
    v = series(tag, 'douban-book')
    if v:
        m = sum(v) / len(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {name:22s} {m:.5f} +- {sd:.5f}   delta={m-mf:+.5f}  ({abs(m-mf)/0.00065:.1f}x noise floor)')

head('Table IV  Beyond-accuracy on Douban-Book (3 seeds)')
METS = ['NDCG@20', 'TailRecall@20', 'ItemCoverage@20', 'ARP@20', 'Novelty@20', 'Gini@20']
for name, tag in [('XSimGCL (backbone)', 'XSimGCLg_w00'), ('PT4Rec', 'PTbase_XSim_nofz'),
                  ('+ routing, popularity', 'OURS_XSim_nofz'), ('+ geometric (ours)', 'OURSgeom_w2'),
                  ('DirectAU', 'DirectAU')]:
    cells = []
    for m in METS:
        v = series(tag, 'douban-book', m)
        cells.append(f'{sum(v)/len(v):.4f}' if v else '--')
    print(f'  {name:22s} ' + '  '.join(f'{m.split("@")[0]}={c}' for m, c in zip(METS, cells)))

head('Table V  Per-dataset selection decisions (validation vs test, 3 seeds)')
def val(tag, ds):
    v = [rec(tag, ds, s)['val_metrics']['NDCG'] for s in SEEDS if rec(tag, ds, s)]
    return sum(v) / len(v) if v else None
for label, ds, opts in [
        ('Douban-Book: module', 'douban-book', [('enabled', 'OURSgeom_w2'), ('disabled', 'OURS_XSim_nofz')]),
        ('ML-1M: module', 'ml-1M', [('enabled', 'OURSgeom_w2'), ('disabled', 'OURS_XSim_nofz')]),
        ('Douban-Book: exponent', 'douban-book', [('beta=0.5', 'AdaG_b05'), ('beta=1.0', 'AdaG_b10')]),
        ('ML-1M: exponent', 'ml-1M', [('beta=0.5', 'AdaG_b05'), ('beta=1.0', 'AdaG_b10')])]:
    for o, tag in opts:
        print(f'  {label:24s} {o:9s} val={f4(val(tag, ds))}  test={f4(mean(tag, ds))}  [{tag}]')

head('Table VI  Regularizer applied to the encoder alone (dagger = earlier sampler)')
for ds in ['amazon-kindle', 'yelp2018', 'douban-book', 'ml-1M']:
    b = mean('XSimGCLg_w00', ds) or mean('XSimGCLg_w00', ds, d=OLD)
    row = f'  {ds:14s} encoder {f4(b)}'
    for tag, lab in [('XSimGCLg_w20', 'uniform'), ('AdaG_b05', 'DA b=0.5'), ('AdaG_b10', 'DA b=1')]:
        m = mean(tag, ds)
        src = ''
        if m is None:
            m, src = mean(tag, ds, d=OLD), '(earlier sampler)'
        if m is not None:
            row += f' | {lab} {f4(m)} ({pct(m, b)}){src}'
    print(row)

head('In-text numbers')
nf = [json.load(open(f))['metrics']['NDCG@20'] for f in glob.glob(os.path.join(NF, '*.json'))]
print(f'  noise floor: n={len(nf)} identical runs, min={min(nf):.5f} max={max(nf):.5f} '
      f'spread={max(nf)-min(nf):.5f} sd={st.stdev(nf):.5f}')
p = json.load(open(os.path.join(CUR, 'param_counts.json')))
for k, v in p.items():
    print(f'  params {k:20s} encoder={v["encoder_params"]:,} added={v["added_params"]:,} (+{v["pct_added"]:.2f}%)')
c = json.load(open(os.path.join(CUR, 'cost_benchmark.json')))
for k, v in c.items():
    print(f'  peak GPU {k:20s} {v["peak_gpu_gb"]:.3f} GB')
th = json.load(open(os.path.join(CUR, 'theory_prop2_verification.json')))
print('  Proposition 2 grid search: ' + ', '.join(
    f'd={r["degree"]}: predicted {r["theory_mu_star"]:.2f} / measured {r["empirical_mu_star"]:.2f}' for r in th))
print('  effective rank (one run per variant, Douban-Book):')
for name, tag in [('XSimGCL', 'XSimGCLg_w00'), ('PT4Rec', 'PTbase_XSim_nofz'),
                  ('Ours -g', 'OURS_XSim_nofz'), ('Ours', 'OURSgeom_w2')]:
    for f in sorted(glob.glob(os.path.join(CUR, f'{tag}__douban-book__seed*.json'))):
        g = json.load(open(f)).get('geometry', {})
        if 'user_eff_rank' in g:
            print(f'    {name:10s} user={g["user_eff_rank"]:.2f} item={g["item_eff_rank"]:.2f} '
                  f'[{os.path.basename(f)}]')
            break
print('  ML-1M temporal split (per-user chronological 80/20, 3 seeds):')
for name, tag in [('SGL', 'SGL'), ('NCL', 'NCL'), ('XSimGCL', 'XSimGCLg_w00'),
                  ('PT4Rec', 'PTbase_XSim_nofz'), ('Ours -g', 'OURS_XSim_nofz'), ('Ours', 'OURSgeom_w2')]:
    m = mean(tag, 'ml-1M-temporal')
    if m: print(f'    {name:10s} {f4(m)}')
print('  Yelp2018 configuration sweep (all variants tried):')
lo, hi = 1, 0
for f in sorted(glob.glob(os.path.join(CUR, 'OURS*__yelp2018__seed*.json'))):
    v = json.load(open(f))['metrics']['NDCG@20']
    lo, hi = min(lo, v), max(hi, v)
print(f'    range {lo:.4f} - {hi:.4f}; DirectAU {f4(mean("DirectAU","yelp2018"))}, '
      f'backbone {f4(mean("XSimGCLg_w00","yelp2018"))}')
