#!/usr/bin/env python
"""Export the per-user NDCG@20 behind every per-user analysis in the paper.

The per-user Wilcoxon tests (per_user_significance.py) and the degree quintiles (degree_stratified.py,
make_quintile.py, make_mechanism_fig.py) need each test user's NDCG@20, which only the raw top-20 dumps
and the dataset's test split can give. Neither is released, so this script writes the scores those
analyses consume to results/per_user/<tag>__<ds>__seed<s>.tsv (user, training-file degree, NDCG@20).
A dump is used only when its mean reproduces the NDCG@20 of the released record (find_dump), and the
analyses re-check every file against its record when they load it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_user_significance import PU, find_dump, per_user_path, train_degrees

SEED = '2024'
# (tag, dataset, model name in the dump filename)
NEEDED = [('OURSgeom_w2', 'douban-book', 'PT4Rec_Enhanced'), ('SGL', 'douban-book', 'SGL'),
          ('OURS_XSim_nofz', 'ml-1M', 'PT4Rec_Enhanced'),
          ('OURSgeom_w2', 'yelp2018', 'PT4Rec_Enhanced'), ('XSimGCLg_w00', 'yelp2018', 'XSimGCLg')]
NEEDED += [(tag, ds, 'XSimGCLg') for ds in ('ml-1M', 'douban-book')
           for tag in ('XSimGCLg_w00', 'XSimGCLg_w10', 'AdaG_b05', 'AdaG_b10')]

if __name__ == '__main__':
    os.makedirs(PU, exist_ok=True)
    for tag, ds, hint in NEEDED:
        scores, dump = find_dump(tag, ds, SEED, hint)
        if scores is None:
            sys.exit(f'{tag} {ds} seed{SEED}: no dump reproduces the record; nothing written')
        deg = train_degrees(ds)
        path = per_user_path(tag, ds, SEED)
        with open(path, 'w') as f:
            f.write('user\tdegree\tndcg20\n')
            for u in sorted(scores, key=int):
                f.write(f'{u}\t{deg.get(u, 0)}\t{scores[u]!r}\n')      # repr round-trips the float exactly
        print(f'wrote {os.path.relpath(path)}  ({len(scores)} users, from {os.path.basename(dump)})')
