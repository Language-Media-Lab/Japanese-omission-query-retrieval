import itertools
from collections import Counter
import numpy as np

def agreement(items):
    # Krippendorff coincidence weighting: each unit contributes m ratings,
    # not m*(m-1) ordered pairs. Ordinal distances use marginal frequencies.
    coincidence = np.zeros((3, 3))
    pairs = []
    for xs in items:
        counts = np.bincount(xs, minlength=4)[1:].astype(float)
        m = len(xs)
        assert m >= 2
        coincidence += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
        pairs.extend(itertools.combinations(xs, 2))
    marginal = coincidence.sum(axis=0)
    expected = (np.outer(marginal, marginal) - np.diag(marginal)) / (marginal.sum() - 1)
    distance = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            distance[i, j] = distance[j, i] = (marginal[i:j+1].sum() - (marginal[i]+marginal[j])/2)**2
    den = (expected * distance).sum()
    return dict(n=len(items), ratings=sum(map(len, items)),
                ordinal_alpha=float(1-(coincidence*distance).sum()/den) if den else None,
                pair_agreement=sum(a == b for a, b in pairs)/len(pairs),
                unanimous=sum(len(set(xs)) == 1 for xs in items)/len(items),
                both_extremes=sum(1 in xs and 3 in xs for xs in items),
                score_counts=dict(Counter(x for xs in items for x in xs)))
