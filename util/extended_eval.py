"""Extended, beyond-accuracy evaluation metrics for top-N recommendation.

Directly addresses reviewer requests to VALIDATE the popularity-debiasing claim with
behavioural evidence (long-tail recall, catalog/item coverage, average recommendation
popularity, novelty, Gini exposure concentration, head/tail split) rather than an
internal cosine diagnostic. All statistics are computed from the model's ranked lists,
the held-out test set, and the TRAINING-item popularity (no test leakage).
"""
import math
import numpy as np


def item_train_degree(data):
    """Per-item training interaction count, keyed by original item id."""
    deg = {}
    for item, users in data.training_set_i.items():
        deg[item] = len(users)
    return deg


def _tail_items(deg, tail_frac=0.8):
    """The `tail_frac` least popular items (by training degree) = long tail."""
    if not deg:
        return set()
    items_sorted = sorted(deg.keys(), key=lambda it: deg[it])  # ascending popularity
    cut = int(round(tail_frac * len(items_sorted)))
    return set(items_sorted[:cut])


def extended_metrics(test_set, rec_list, data, N, tail_frac=0.8):
    """Compute beyond-accuracy metrics at cutoff N.

    Returns a dict: tail_recall@N, item_coverage@N, arp@N (avg rec popularity, in
    [0,1]), novelty@N (mean self-information, bits), gini@N (exposure concentration),
    head_recall@N, tail_frac_recs@N (share of recommended items that are long-tail).
    """
    deg = item_train_degree(data)
    max_deg = max(deg.values()) if deg else 1
    total_deg = sum(deg.values()) if deg else 1
    n_items = data.item_num
    tail = _tail_items(deg, tail_frac)

    exposure = np.zeros(n_items, dtype=np.float64)  # #times each item recommended
    recommended_items = set()
    arp_sum, nov_sum, rec_count = 0.0, 0.0, 0
    tail_rec_count = 0

    # accuracy split into head vs tail test items
    tail_hit, tail_tot, head_hit, head_tot = 0, 0, 0, 0

    for user in test_set:
        gt = test_set[user]
        topn = [it for it, _ in rec_list[user][:N]]
        for it in topn:
            iid = data.item.get(it)
            if iid is not None:
                exposure[iid] += 1
            recommended_items.add(it)
            d = deg.get(it, 0)
            arp_sum += d / max_deg
            p = d / total_deg if total_deg > 0 else 0.0
            nov_sum += -math.log2(p) if p > 0 else 0.0
            rec_count += 1
            if it in tail:
                tail_rec_count += 1
        hitset = set(topn)
        for it in gt:
            if it in tail:
                tail_tot += 1
                tail_hit += 1 if it in hitset else 0
            else:
                head_tot += 1
                head_hit += 1 if it in hitset else 0

    # Gini of the exposure distribution over all items (0 = uniform, 1 = concentrated)
    exp_sorted = np.sort(exposure)
    n = len(exp_sorted)
    cum = np.cumsum(exp_sorted)
    gini = (n + 1 - 2 * (cum.sum() / cum[-1])) / n if cum[-1] > 0 else 0.0

    return {
        f'TailRecall@{N}': round(tail_hit / tail_tot, 5) if tail_tot else 0.0,
        f'HeadRecall@{N}': round(head_hit / head_tot, 5) if head_tot else 0.0,
        f'ItemCoverage@{N}': round(len(recommended_items) / n_items, 5) if n_items else 0.0,
        f'ARP@{N}': round(arp_sum / rec_count, 5) if rec_count else 0.0,
        f'Novelty@{N}': round(nov_sum / rec_count, 5) if rec_count else 0.0,
        f'Gini@{N}': round(float(gini), 5),
        f'TailShareRecs@{N}': round(tail_rec_count / rec_count, 5) if rec_count else 0.0,
    }
