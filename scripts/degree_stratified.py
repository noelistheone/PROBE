#!/usr/bin/env python
"""Is the geometric regularizer's effect explained by DENSITY or by degree HETEROGENEITY?

If a graph were uniformly well observed, the unit-mean gate w_n = (1+d_n)^-beta / E[(1+d)^-beta]
would equal 1 everywhere and degree adaptation would be a no-op. So the mechanism must be
heterogeneity. This script tests that directly: it splits users into degree quintiles and measures,
per quintile, what a uniform dose costs and what adaptation gives back.
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_user_significance import find_dump, test_set   # verified dump->record mapping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def degrees(ds):
    d = collections.Counter()
    for line in open(os.path.join(ROOT, 'dataset', ds, 'train.txt')):
        p = line.split()
        if len(p) >= 2:
            d[p[0]] += 1
    return d

def quintile_report(ds, seed='2024'):
    arms = [('encoder', 'XSimGCLg_w00', 'XSimGCLg'), ('uniform mu=1', 'XSimGCLg_w10', 'XSimGCLg'),
            ('DA beta=0.5', 'AdaG_b05', 'XSimGCLg'), ('DA beta=1', 'AdaG_b10', 'XSimGCLg')]
    scores = {}
    used = {}
    for name, tag, model in arms:
        s, f = find_dump(tag, ds, seed, model)
        if s:
            scores[name] = s
            used[name] = os.path.basename(f) if f else '?'
    if 'encoder' not in scores:
        print(f'{ds}: no verified dump for the encoder arm'); return
    deg = degrees(ds)
    users = sorted(set.intersection(*[set(v) for v in scores.values()]), key=lambda u: deg.get(u, 0))
    q = len(users) // 5
    print(f'\n{ds} (seed {seed}, {len(users)} users, quintiles by training degree)')
    for k, v in used.items():
        print(f'    [{k}] <- {v}')
    header = f'  {"arm":14s}' + ''.join(f'{"Q%d" % (i+1):>12s}' for i in range(5))
    print(header)
    meds = [deg.get(users[min(i*q + q//2, len(users)-1)], 0) for i in range(5)]
    print(f'  {"median degree":14s}' + ''.join(f'{m:12d}' for m in meds))
    base = None
    for name, _, _ in arms:
        if name not in scores: continue
        cells = []
        for i in range(5):
            grp = users[i*q:(i+1)*q] if i < 4 else users[4*q:]
            cells.append(sum(scores[name][u] for u in grp) / len(grp))
        if name == 'encoder':
            base = cells
            print(f'  {name:14s}' + ''.join(f'{c:12.4f}' for c in cells))
        else:
            print(f'  {name:14s}' + ''.join(f'{100*(c-b)/b:11.1f}%' for c, b in zip(cells, base)))

for ds in ['ml-1M', 'douban-book']:
    quintile_report(ds)
