#!/usr/bin/env python
"""Per-user paired significance tests (Wilcoxon signed-rank) on the delivered ranking lists.

The harness writes one `<Model>@<timestamp>-top-20items.txt` dump per run, which records each test
user's top-20 with `*` marking hits, but the dump filename carries no tag/seed.  We therefore map
each released record (`results/wm/<tag>__<ds>__seed<s>.json`) to a dump by write-time proximity and
then VERIFY the mapping: the per-user NDCG@20 recomputed from the dump must reproduce the NDCG@20
stored in the record to 5 decimals (the harness's own rounding).  Unverified mappings are discarded
rather than used, so a wrong pairing cannot silently enter a significance test.
"""
import glob, json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, WM = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'results', 'wm')
SEEDS = ['2024', '2025', '2026']


def train_vocab(ds, ratio=0.1, split_seed=2024):
    """User/item vocabulary AFTER the validation carve-out, exactly as data/ui_graph.py builds it."""
    import random as _random
    from collections import defaultdict
    rows = [l.split()[:3] for l in open(os.path.join(ROOT, 'dataset', ds, 'train.txt'))]
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    rng, keep = _random.Random(split_seed), []
    for u, recs in by.items():
        n = len(recs)
        if n < 2:
            keep.extend(recs); continue
        n_val = min(max(1, int(round(ratio * n))), n - 1)
        idx = list(range(n)); rng.shuffle(idx); val = set(idx[:n_val])
        keep.extend(r for j, r in enumerate(recs) if j not in val)
    return {r[0] for r in keep}, {r[1] for r in keep}


def test_set(ds):
    """user -> test items, restricted to the training vocabulary, as the harness does."""
    users, items = train_vocab(ds)
    out = {}
    for line in open(os.path.join(ROOT, 'dataset', ds, 'test.txt')):
        p = line.split()
        if len(p) >= 2 and p[0] in users and p[1] in items:
            out.setdefault(p[0], {})[p[1]] = 1
    return out


def per_user_ndcg(dump_path, origin, N=20):
    """Replicates util.evaluation.Metric.NDCG, but per user instead of averaged."""
    scores = {}
    with open(dump_path) as f:
        f.readline()                                    # header
        for line in f:
            if ':' not in line:
                continue
            user, rest = line.split(':', 1)
            user = user.strip()
            if user not in origin:
                continue
            hits = [bool(m) for m in re.findall(r'\)(\*)?', rest)][:N]
            dcg = sum(1.0 / math.log(n + 2, 2) for n, h in enumerate(hits) if h)
            idcg = sum(1.0 / math.log(n + 2, 2) for n in range(min(len(origin[user]), N)))
            scores[user] = dcg / idcg if idcg else 0.0
    return scores


def find_dump(tag, ds, seed, model_hint):
    rec_path = os.path.join(WM, f'{tag}__{ds}__seed{seed}.json')
    if not os.path.exists(rec_path):
        return None, None
    rec = json.load(open(rec_path))
    t_rec = os.path.getmtime(rec_path)
    origin = test_set(ds)
    cands = sorted(glob.glob(os.path.join(RES, f'{model_hint}@*-top-20items.txt')),
                   key=lambda f: abs(os.path.getmtime(f) - t_rec))
    for c in cands[:6]:                                  # nearest few by write time
        s = per_user_ndcg(c, origin)
        if not s:
            continue
        avg = round(sum(s.values()) / len(s), 5)
        if abs(avg - rec['metrics']['NDCG@20']) < 1e-5:  # mapping verified against the record
            return s, c
    return None, None


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank with a normal approximation and tie correction."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n < 10:
        return float('nan'), n
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p, n


if __name__ == '__main__':
    PAIRS = [('douban-book', 'OURSgeom_w2', 'PT4Rec_Enhanced', 'SGL', 'SGL'),
             ('ml-1M',       'OURS_XSim_nofz', 'PT4Rec_Enhanced', 'XSimGCLg_w00', 'XSimGCLg'),
             ('yelp2018',    'OURSgeom_w2', 'PT4Rec_Enhanced', 'XSimGCLg_w00', 'XSimGCLg')]
    
    print('Per-user paired Wilcoxon signed-rank on NDCG@20 (seed 2024 lists unless noted)\n')
    for ds, ours_tag, ours_model, base_tag, base_model in PAIRS:
        for seed in SEEDS:
            a, fa = find_dump(ours_tag, ds, seed, ours_model)
            b, fb = find_dump(base_tag, ds, seed, base_model)
            if not a or not b:
                print(f'{ds:12s} seed{seed}: mapping unverified, skipped')
                continue
            users = sorted(set(a) & set(b))
            d = [a[u] - b[u] for u in users]
            z, p, n = wilcoxon(d)
            mean_d = sum(d) / len(d)
            wins = sum(1 for x in d if x > 0)
            print(f'{ds:12s} seed{seed}: ours {ours_tag} vs {base_tag} | users={len(users)} '
                  f'mean dNDCG@20={mean_d:+.5f} | users better={wins} ({100*wins/len(users):.1f}%) '
                  f'| nonzero={n} z={z:.2f} p={p:.3g}')
            break                                            # one verified seed per dataset is enough
