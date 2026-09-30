#!/usr/bin/env python
"""Review check for notes/11_input_encoding.md T-IE-12 / section 0 item 6.

(1) Which 3-letter prefixes make up ALL same-prefix within-feature pairs (pooled, tie tails removed),
    per model - the source for "MT-, HLA, RPL/RPS, KRT, S100A" in the notes (no null; descriptive).
(2) Is C2S's "same prefix, different HGNC group" excess carried by a few big superfamilies?
    EXCL = the 20 prefixes with the most 'same prefix, different group' pairs summed over all ten
    models (prefix_examples.py logic, pooled, clean) - chosen once, data-driven, before testing.
    Metric p3_diffgrp_excl: among pairs whose genes both have >= 1 HGNC group and share none, the
    share with the same first 3 characters AND that prefix not in EXCL. Mid layer, clean lists,
    slot-permutation null (ie_common.slot_permute), R perms, seed 0 - the same null as the notes.
    ATLAS_BASE=<copy> python prefix_robust.py [R] -> prefix_robust.json
"""
import json, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "input_encoding"))
from ie_common import *   # noqa

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
H, _, HVER = hgnc()
ex = json.load(open(os.path.join(os.path.dirname(HERE), "input_encoding", "prefix_examples.json")))
tot = Counter()
for m, r in ex["models"].items():
    for p, c in r["top_prefixes"]:
        tot[p] += c
EXCL = [p for p, _ in tot.most_common(20)]
print("EXCL", EXCL, flush=True)
out = {"R_mid": R, "seed": 0, "excluded_prefixes": EXCL,
       "exclusion_rule": "top-20 prefixes by summed 'same prefix, different HGNC group' pair counts over the ten models (prefix_examples.json top_prefixes, which lists each model's top 12)",
       "models": {}}


def prep(m):
    d = load_model(m); raw2u, un = upper_map(d["names"])
    c3 = {}; p3 = np.array([c3.setdefault(u[:3], len(c3)) for u in un], np.int64)
    inv3 = {v: k for k, v in c3.items()}
    excl = np.array([inv3[p3[i]] in EXCL for i in range(len(un))])
    K = 4; Gk = np.full((len(un), K), -1, np.int64)
    for i, u in enumerate(un):
        h = H.get(u)
        if h and h[1]:
            g = sorted(h[1])[:K]; Gk[i, :len(g)] = g
    rank = universe_ranks(d["names"])
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = np.arange(20)[None, :] < (20 - tail)[:, None]
    Xc, kc = filtered_top10(d["T20"], raw2u, un, keep)
    return d, un, p3, excl, Gk, Xc, d["layer"][kc]


def met(A, B, p3, excl, Gk):
    s3 = p3[A] == p3[B]
    Ga, Gb = Gk[A], Gk[B]; same = np.zeros(len(A), bool)
    for i in range(Ga.shape[1]):
        v = Ga[:, i] >= 0
        for j in range(Gb.shape[1]): same |= v & (Ga[:, i] == Gb[:, j])
    both = (Ga[:, 0] >= 0) & (Gb[:, 0] >= 0); diff = both & ~same
    s3x = s3 & ~excl[A]
    return {"p3_diffgrp": float(s3[diff].mean()), "p3_diffgrp_excl": float(s3x[diff].mean()),
            "p3_all_excl": float(s3x.mean())}


for m in ORDER:
    rng = np.random.default_rng(0)
    d, un, p3, excl, Gk, Xc, Lc = prep(m)
    A, B = pairs_of(Xc); same = p3[A] == p3[B]
    pc = Counter(un[a][:3] for a in A[same])
    M = {"pooled_clean_same_p3_pairs": int(same.sum()), "pooled_clean_top_prefixes_all_same_p3": pc.most_common(12)}
    X = Xc[Lc == mid_layer(m)]
    a, b = pairs_of(X); obs = met(a, b, p3, excl, Gk); nul = {k: [] for k in obs}
    for _ in range(R):
        aa, bb = pairs_of(slot_permute(X, rng)); mm = met(aa, bb, p3, excl, Gk)
        for k in nul: nul[k].append(mm[k])
    M["mid"] = {"layer": mid_layer(m), "n_features": int(len(X)), "n_pairs": int(len(a)),
                "metrics": {k: empirical(v, nul[k]) for k, v in obs.items()}}
    out["models"][m] = M
    print(m, pc.most_common(8), {k: (v["fold"], v["z"], v["p_emp"]) for k, v in M["mid"]["metrics"].items()}, flush=True)
    json.dump(out, open(f"{HERE}/prefix_robust.json", "w"), indent=1)
    del d, Xc
