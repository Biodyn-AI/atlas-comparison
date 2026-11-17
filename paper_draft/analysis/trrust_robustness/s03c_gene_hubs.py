#!/usr/bin/env python
"""Step 3c (job item 2, extension): hub concentration by GENE in any role (TF or target).

s03 counts hits per regulating TF. A target gene can also be a hub (e.g. ATF3 is the target in
4 of Geneformer's 12 hits). Here: count hits per gene in either position; share of hits touching
the top-1 and top-3 genes; then remove the top-1 gene, and the top-3 genes together, from the graph
and from TRRUST (node removal, as in s03) and re-score with the repo's stub-matching null.
Output: OUT/s03c_gene_hubs.json
"""
import json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, _, _ = load_truth()
tkey = truth_keys(truth, gid)
edges = json.load(open(f"{OUT}/s02_edges.json"))["per_model"]
out = {"n_perm": NPERM, "seed": 0, "models": {}}
for m in SEVEN:
    feat, lay, L = load_model(m)
    keys, w = graph_from(feat, lay, None, 5)
    del feat, lay
    hits = edges[m]["hits"]; n = len(hits)
    gc = Counter(g for h in hits for g in h["pair"])
    ranked = [g for g, _ in gc.most_common()]

    def drop(gl):
        ids = np.array([gid[g] for g in gl], np.int64)
        a, b = decode(keys); keep = ~(np.isin(a, ids) | np.isin(b, ids))
        ta, tb = decode(tkey); tk = tkey[~(np.isin(ta, ids) | np.isin(tb, ids))]
        return score(keys[keep], tk, NPERM, seed=0)

    top1, top3 = ranked[:1], ranked[:3]
    out["models"][m] = {"hits": n, "n_distinct_genes_in_hits": len(gc), "top_genes": gc.most_common(8),
                        "share_hits_top1_gene": round(sum(any(g in top1 for g in h["pair"]) for h in hits) / n, 4) if n else None,
                        "share_hits_top3_genes": round(sum(any(g in top3 for g in h["pair"]) for h in hits) / n, 4) if n else None,
                        "top1": top1, "top3": top3,
                        "drop_top1_gene": drop(top1) if top1 else None, "drop_top3_genes": drop(top3) if top3 else None}
    r = out["models"][m]
    print(m, r["top_genes"][:5], r["share_hits_top1_gene"], r["share_hits_top3_genes"],
          {k: (r[k]["hits"], r[k]["fold"], r[k]["p_emp"]) for k in ("drop_top1_gene", "drop_top3_genes") if r[k]}, flush=True)
json.dump(out, open(f"{OUT}/s03c_gene_hubs.json", "w"), indent=1)
