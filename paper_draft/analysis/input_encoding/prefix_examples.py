#!/usr/bin/env python
"""Descriptive (no null): which shared 3-letter prefixes make up the within-feature pairs that are
'named alike but in different HGNC gene groups' (all layers pooled, tie tails removed), with the
most frequent such pairs, per model.  -> prefix_examples.json"""
import json
from collections import Counter
import numpy as np
from ie_common import *

H, _, HVER = hgnc()
out = {"hgnc": HVER, "models": {}}
for m in ORDER:
    d = load_model(m); raw2u, un = upper_map(d["names"])
    rank = universe_ranks(d["names"])
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = np.arange(20)[None, :] < (20 - tail)[:, None]
    X, _ = filtered_top10(d["T20"], raw2u, un, keep)
    A, B = pairs_of(X)
    grp = [set(H[u][1]) if u in H else set() for u in un]
    c3 = {}; p3 = np.array([c3.setdefault(u[:3], len(c3)) for u in un])
    sel = np.where(p3[A] == p3[B])[0]
    pref = Counter(); pairs = Counter(); n_diff = 0
    for i in sel:
        a, b = A[i], B[i]
        if grp[a] and grp[b] and not (grp[a] & grp[b]):
            n_diff += 1; pref[un[a][:3]] += 1; pairs[tuple(sorted((un[a], un[b])))] += 1
    out["models"][m] = {"n_pairs": int(len(A)), "same_p3_pairs": int(len(sel)), "same_p3_diff_group_pairs": n_diff,
                        "top_prefixes": pref.most_common(12),
                        "top_pairs": [["-".join(k), v] for k, v in pairs.most_common(12)]}
    print(m, n_diff, pref.most_common(8), flush=True)
json.dump(out, open(f"{OUTDIR}/prefix_examples.json", "w"), indent=1)
