#!/usr/bin/env python
"""Build the per-user chronological (temporal) split used for the ML-1M robustness check.

Each user's interactions are sorted by timestamp and the most recent 20% are held out as
the test set, so no future interaction of a user is ever visible while predicting that
user's later ones. A user with a single interaction stays entirely in the training set.

Input : ratings with timestamps, one per line: "user item rating timestamp".
Output: train.txt / test.txt in the same "user item rating" format the loader expects.

Usage: python scripts/make_temporal_split.py dataset/ml-1M/ratings_ts.txt dataset/ml-1M-temporal
"""
import os, sys, collections

def main(src, out_dir, test_ratio=0.2):
    rows = collections.defaultdict(list)
    for line in open(src):
        p = line.split()
        if len(p) < 4:
            continue
        # ties on the timestamp are broken by item id, so the split is deterministic
        rows[p[0]].append((int(p[3]), p[1], p[2]))

    os.makedirs(out_dir, exist_ok=True)
    n_tr = n_te = 0
    with open(os.path.join(out_dir, 'train.txt'), 'w') as ftr, \
         open(os.path.join(out_dir, 'test.txt'), 'w') as fte:
        for user in sorted(rows, key=int):
            items = sorted(rows[user])
            n = len(items)
            k = int(round(n * (1 - test_ratio)))
            if n > 1 and k >= n:            # every user with >1 interaction keeps a test item
                k = n - 1
            for j, (_, item, rating) in enumerate(items):
                (ftr if j < k else fte).write(f'{user} {item} {rating}\n')
            n_tr += k
            n_te += n - k
    print(f'{out_dir}: {n_tr} train / {n_te} test interactions over {len(rows)} users')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.2)
