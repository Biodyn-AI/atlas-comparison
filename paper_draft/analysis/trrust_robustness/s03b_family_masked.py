#!/usr/bin/env python
"""Step 3b (job items 1/2): does recovery survive when family-like pairs are masked?

Two masks, each applied to BOTH the co-firing graph and TRRUST (the "masked null" design of the
reproduction check, 08 section 6.2), then re-scored with the repo's stub-matching null, NPERM draws:
  M1 name mask: drop every pair whose two symbols share >= 2 leading characters (all genes; not
     informed by the hits). Stricter than the 3-letter rule of 08 (prefix_rescore.py).
  M2 structural-family mask: drop every pair whose two genes are in the same group of the hand-made
     DNA-binding-domain / protein-family map of s02b (FLI1/ERG, RORA/NR1D1, STAT1/STAT3 ...).
     The map only covers genes that appear in hits, so this mask is informed by the hits: it can only
     remove hits, and is a deliberately pessimistic stress test.
  M1+M2 together.
Output: OUT/s03b_family_masked.json
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, _, _ = load_truth()
tkey = truth_keys(truth, gid)
fam = json.load(open(f"{OUT}/s02b_family.json"))["family_map"]
G2F = {g: f for f, gs in fam.items() for g in gs.split()}
fam_id = np.full(len(genes), -1, np.int64)
for g, f in G2F.items():
    if g in gid: fam_id[gid[g]] = list(fam).index(f)
pre2 = np.array([hash(g[:2]) for g in genes], np.int64)


def masks(keys):
    a, b = decode(keys)
    m1 = pre2[a] == pre2[b]
    m2 = (fam_id[a] >= 0) & (fam_id[a] == fam_id[b])
    return m1, m2


tm1, tm2 = masks(tkey)
out = {"n_perm": NPERM, "seed": 0, "models": {}}
t0 = time.time()
for m in SEVEN:
    feat, lay, L = load_model(m)
    keys, w = graph_from(feat, lay, None, 5)
    del feat, lay
    g1, g2 = masks(keys)
    out["models"][m] = {"full": score(keys, tkey, NPERM, seed=0)}
    for lab, gm, tm in (("M1_prefix2", g1, tm1), ("M2_struct_family", g2, tm2), ("M1_plus_M2", g1 | g2, tm1 | tm2)):
        out["models"][m][lab] = score(keys[~gm], tkey[~tm], NPERM, seed=0)
        out["models"][m][lab]["pairs_removed"] = int(gm.sum())
    r = out["models"][m]
    print(m, {k: (v.get("n_pairs"), v.get("hits"), v.get("fold"), v.get("p_emp")) for k, v in r.items()}, f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"{OUT}/s03b_family_masked.json", "w"), indent=1)
