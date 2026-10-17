#!/usr/bin/env python
"""Review check for notes/11_input_encoding.md T-IE-8 (equal-budget columns).

The top-8,631-by-weight budget (hypothesis_sizematched.py) sorts pairs by weight and cuts at 8,631.
Many pairs share the cutoff weight, so which tied pairs get in depends on dict order. The published
graph (cache) keeps insertion order; the clean graph in trrust_lexical.py is built from np.unique
(alphabetical key order). This script measures how much the hit count depends on that tie order:
  * n pairs strictly above the cutoff weight, n at the cutoff weight, hits in each group
  * hits under the dict order (published: cache insertion order = hypothesis_sizematched.py), under
    alphabetical tie order (= trrust_lexical.py's clean graph, built from np.unique over alphabetical
    GIDs), and under 1,000 random tie orders
  * the tie-free expected hits = hits above cutoff + hits_at_cutoff * slots_left / n_at_cutoff
Graph construction = ie_common.filtered_top10 + W >= 5 (same as trrust_lexical.py:graph_from).
    ATLAS_BASE=<copy> python budget_tiebreak.py -> budget_tiebreak.json
"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "input_encoding"))
from ie_common import *   # noqa

BUDGET = 8631
TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
PRED_PUB, _ = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))


def graph(m, clean):
    d = load_model(m); names = d["names"]; rank = universe_ranks(names); raw2u, un = upper_map(names)
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = (np.arange(20)[None, :] < (20 - tail)[:, None]) if clean else None
    X, _ = filtered_top10(d["T20"], raw2u, un, keep)
    A, B = pairs_of(X)
    lo = np.minimum(A, B); hi = np.maximum(A, B)
    # order by name so (a, b) is name-sorted, like the repo
    sw = np.array([un[x] for x in range(len(un))], dtype=object)
    k = lo * (1 << 21) + hi
    uk, c = np.unique(k, return_counts=True); sel = c >= W0
    out = {}
    for x, n in zip(uk[sel], c[sel]):
        a, b = sw[int(x >> 21)], sw[int(x & ((1 << 21) - 1))]
        out[tuple(sorted((a, b)))] = int(n)
    return out


def tie_stats(g, rng, order_label):
    items = list(g.items())
    w = np.array([v for _, v in items]); ist = np.array([e in TRUTH for e, _ in items])
    ws = np.sort(w)[::-1]; cut = ws[BUDGET - 1]
    above = w > cut; at = w == cut
    slots = BUDGET - int(above.sum())
    ins = sorted(range(len(items)), key=lambda i: -w[i])[:BUDGET]          # dict order (as given)
    alpha = sorted(range(len(items)), key=lambda i: (-w[i], items[i][0]))[:BUDGET]
    rand = []
    at_idx = np.where(at)[0]
    for _ in range(1000):
        pick = rng.choice(at_idx, slots, replace=False)
        rand.append(int(ist[above].sum() + ist[pick].sum()))
    rand = np.array(rand)
    return {"dict_order": order_label, "cutoff_weight": int(cut), "n_above_cutoff": int(above.sum()),
            "n_at_cutoff": int(at.sum()), "slots_for_tied": slots,
            "hits_above_cutoff": int(ist[above].sum()), "hits_at_cutoff": int(ist[at].sum()),
            "hits_dict_order": int(ist[ins].sum()), "hits_alphabetical_ties": int(ist[alpha].sum()),
            "hits_random_ties_mean": round(float(rand.mean()), 2), "hits_random_ties_min_max": [int(rand.min()), int(rand.max())],
            "hits_expected_tie_free": round(float(ist[above].sum() + ist[at].sum() * slots / max(at.sum(), 1)), 2)}


out = {"budget": BUDGET, "random_tie_orders": 1000, "seed": 0, "models": {}}
for m in ["C2S", "UCE", "Tahoe", "Geneformer", "MaxToki", "tGPT", "GeneCompass"]:
    rng = np.random.default_rng(0)
    pub = PRED_PUB[m]
    if len(pub) < BUDGET: continue
    raw_alpha = graph(m, False); assert raw_alpha == pub
    cln = graph(m, True)
    out["models"][m] = {"published": tie_stats(pub, rng, "cache insertion order"),
                        "clean": tie_stats(cln, rng, "this script's internal uid order (NOT trrust_lexical.py); hits_alphabetical_ties reproduces trrust_lexical.py")}
    print(m, json.dumps(out["models"][m]), flush=True)
json.dump(out, open(f"{HERE}/budget_tiebreak.json", "w"), indent=1)
