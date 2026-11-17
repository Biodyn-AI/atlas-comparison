#!/usr/bin/env python
"""Step 3 (job items 2 and 5): hub concentration, leave-one-TF-out, and direction / TF-TF checks.

Graphs: the repo's pooled W>=5 co-firing graph per model (identical to .cofire_cache.pkl, s01).
Null: the repo's stub-matching configuration model (config_shuffle), NPERM draws, seed 0 per test.

(2) Hub concentration. A hit is "from TF X" when X regulates the other gene in TRRUST (direction
    known from the TRRUST file). Distinct TFs among hits; share of hits involving the top-1 TF and
    any of the top-3 TFs (union, so a hit is counted once).
    Leave-one-TF-out: for every TF with >= 1 hit, remove that gene from the model's graph (all its
    pairs) and every TRRUST edge that touches it (as regulator or target), then re-score with a fresh
    null. Report the minimum fold and maximum p over TFs. Also drop the top-3 TFs together.
(5) Direction. TRRUST regulators ("TF" below = any of the 795 TRRUST regulators, which include some
    co-factors such as NFKBIA, CREBBP, EP300). For observed and null graphs count: TF-TF pairs in the
    graph; hits that are TF-TF; hits that are TF-nonTF. Fold and p for each class (the TF-nonTF class
    is the one where TRRUST direction is unambiguous). Also the share of hits whose two directions
    are both in TRRUST.
Output: OUT/s03_hub_direction.json
"""
import json, os, sys, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, directed, tfs = load_truth()
tkey = truth_keys(truth, gid)
tf_ids = np.array(sorted(gid[t] for t in tfs if t in gid), np.int64)
edges = json.load(open(f"{OUT}/s02_edges.json"))["per_model"]

out = {"n_perm": NPERM, "null": "stub-matching configuration model (repo config_shuffle)", "seed": 0,
       "tf_definition": "TRRUST v2 regulators (795 symbols, keys of trrust_edges.json)", "models": {}}
t0 = time.time()
for m in SEVEN:
    feat, lay, L = load_model(m)
    keys, w = graph_from(feat, lay, None, 5)
    del feat, lay
    hits = edges[m]["hits"]
    n = len(hits)
    # ---------------- (2) hub concentration
    tfc = Counter(t for h in hits for t in h["tf"])
    ranked = [t for t, _ in tfc.most_common()]
    top1 = ranked[:1]; top3 = ranked[:3]
    share1 = sum(any(t in top1 for t in h["tf"]) for h in hits) / n if n else None
    share3 = sum(any(t in top3 for t in h["tf"]) for h in hits) / n if n else None
    base, base_null = score(keys, tkey, NPERM, seed=0, return_null=True)

    def drop(tf_list):
        ids = np.array([gid[t] for t in tf_list], np.int64)
        a, b = decode(keys)
        keep = ~(np.isin(a, ids) | np.isin(b, ids))
        ta, tb = decode(tkey)
        tk = tkey[~(np.isin(ta, ids) | np.isin(tb, ids))]
        return score(keys[keep], tk, NPERM, seed=0)

    loto = {}
    for t in ranked:
        loto[t] = drop([t])
    worst_fold = min(loto, key=lambda t: (loto[t]["fold"] if loto[t]["fold"] is not None else -1)) if loto else None
    worst_p = max(loto, key=lambda t: loto[t]["p_emp"]) if loto else None
    top3_drop = drop(top3) if top3 else None

    # ---------------- (5) direction / TF-TF
    a, b = decode(keys)
    tt_obs = int((np.isin(a, tf_ids) & np.isin(b, tf_ids)).sum())
    ish = np.isin(keys, tkey)
    tt_hit_obs = int((ish & np.isin(a, tf_ids) & np.isin(b, tf_ids)).sum())
    tn_hit_obs = int(ish.sum()) - tt_hit_obs
    rng = np.random.default_rng(1)
    tt_null, tth_null, tnh_null = [], [], []
    for _ in range(NPERM):
        k2 = config_shuffle_keys(keys, rng)
        a2, b2 = decode(k2)
        both = np.isin(a2, tf_ids) & np.isin(b2, tf_ids)
        h2 = np.isin(k2, tkey)
        tt_null.append(int(both.sum())); tth_null.append(int((h2 & both).sum())); tnh_null.append(int((h2 & ~both).sum()))
    tt_null, tth_null, tnh_null = map(lambda x: np.array(x, float), (tt_null, tth_null, tnh_null))
    n_bi = sum(h["bidirectional"] for h in hits)
    n_tt = sum(h["both_trrust_regulators"] for h in hits)
    out["models"][m] = {
        "n_pairs": int(len(keys)), "hits": n, "baseline_this_run": base,
        "hub": {"n_distinct_tfs_among_hits": len(ranked), "top_tfs": tfc.most_common(10),
                "share_hits_top1_tf": round(share1, 4) if n else None,
                "share_hits_top3_tfs": round(share3, 4) if n else None, "top3": top3},
        "leave_one_tf_out": {"per_tf": loto,
                             "min_fold": loto[worst_fold]["fold"] if loto else None, "min_fold_tf": worst_fold,
                             "max_p": loto[worst_p]["p_emp"] if loto else None, "max_p_tf": worst_p,
                             "drop_top3_together": top3_drop},
        "direction": {"hits_tf_tf": n_tt, "hits_bidirectional_in_trrust": n_bi,
                      "share_hits_tf_tf": round(n_tt / n, 4) if n else None,
                      "share_hits_bidirectional": round(n_bi / n, 4) if n else None,
                      "graph_tf_tf_pairs": summarize(tt_obs, tt_null),
                      "hits_tf_tf_vs_null": summarize(tt_hit_obs, tth_null),
                      "hits_tf_nontf_vs_null": summarize(tn_hit_obs, tnh_null)}}
    r = out["models"][m]
    print(f"{m}: {n} hits, base fold {base['fold']} p {base['p_emp']}; {len(ranked)} TFs; top1 {top1} {r['hub']['share_hits_top1_tf']}"
          f" top3 {top3} {r['hub']['share_hits_top3_tfs']}; LOTO min fold {r['leave_one_tf_out']['min_fold']} ({worst_fold})"
          f" max p {r['leave_one_tf_out']['max_p']} ({worst_p}); drop top3 {top3_drop}", flush=True)
    print(f"   TF-TF graph pairs obs {tt_obs} vs null {tt_null.mean():.1f}; TF-TF hits {tt_hit_obs} vs {tth_null.mean():.2f};"
          f" TF-nonTF hits {tn_hit_obs} vs {tnh_null.mean():.2f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"{OUT}/s03_hub_direction.json", "w"), indent=1)
