#!/usr/bin/env python
"""Direct test of the tie-fill mechanism.

pipeline/atlas_h100/common/pipeline.py:build_catalog does
    uniq  = np.array(sorted(set(syms)))              # alphabetical gene index
    G     = per-gene mean SAE activation (exactly 0.0 for genes on which the feature never fired)
    order = np.argsort(G[:, f])[::-1][:20]           # top-20 genes
For a feature that fired on fewer genes than 20, the remaining slots are exact ties at 0.0 and are
filled in whatever order argsort leaves tied elements, which depends only on index positions.

Test: for features that fired on < 10 token positions (so at most that many genes can be non-zero),
rebuild a G column with the observed head genes (the slots before the detected tie tail) set to
decreasing positive values and every other gene at 0.0, run the same argsort line with this
process's NumPy, and compare the 20 genes with the published top-20 list.
Needs a complete universe: used only for models whose reconstructed universe size equals the
catalogue's n_genes.  Run under NumPy 1.26.4 (the version the GPU host used, per requirements.txt)
and, as a check, under NumPy 2.3.4.
    python tie_reproduce.py  -> tie_reproduce_np<version>.json
"""
import json, platform, sys
import numpy as np
from ie_common import *

rng = np.random.default_rng(0)
out = {"numpy": np.__version__, "machine": platform.machine(), "models": {}}
for m in ORDER:
    d = load_model(m, cache=False); names = d["names"]; rank = universe_ranks(names); inv = np.argsort(rank)
    ng = json.load(open(catalog_paths(m)[0]))["n_genes"]
    if len(names) != ng or (d["nfire"] < 0).any():
        out["models"][m] = {"skipped": f"universe {len(names)} vs n_genes {ng} or firing counts unknown"}; continue
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    cand = np.where((d["nfire"] > 0) & (d["nfire"] < 10))[0]
    pick = rng.choice(cand, min(300, len(cand)), replace=False)
    full = setm = 0; ex = None; ov = []
    for i in pick:
        head = 20 - tail[i]
        g = np.zeros(ng, np.float32)
        hid = rank[d["T20"][i, :head]]                         # universe index of the head genes
        g[hid] = np.linspace(1.0, 0.5, head, dtype=np.float32) if head > 0 else g[hid]
        order = np.argsort(g)[::-1][:20]
        obs = rank[d["T20"][i]]
        pt = set(obs[head:].tolist()); ov.append(len(pt & set(order[head:].tolist())) / max(len(pt), 1))
        full += int(np.array_equal(order, obs)); setm += int(set(order.tolist()) == set(obs.tolist()))
        if ex is None and head > 0:
            ex = {"feature_row": int(i), "layer": int(d["layer"][i]), "fired_positions": int(d["nfire"][i]),
                  "published_top20": [names[k] for k in d["T20"][i]],
                  "reproduced_top20": [names[inv[k]] for k in order]}
    out["models"][m] = {"n_genes": ng, "n_tested": len(pick), "exact_order_match": full,
                        "same_20_genes": setm,
                        "mean_share_of_published_tie_genes_reproduced": round(float(np.mean(ov)), 4),
                        "share_features_ge80pct_tie_genes_reproduced": round(float(np.mean(np.array(ov) >= 0.8)), 4),
                        "example": ex}
    print(m, out["models"][m]["n_tested"], "exact order", full, "same set", setm, "tie-gene overlap", round(float(np.mean(ov)), 3) if ov else None, flush=True)
json.dump(out, open(f"{OUTDIR}/tie_reproduce_np{np.__version__}.json", "w"), indent=1)
