"""Is our rebuilt corpus the same ordered 6,000 cells the authors used?
Replays the labels() subsample of pipeline/atlas_h100/celltype_difficulty.py:18-28 and
hardcell_agreement.py:18-28 (default_rng(0).choice(6000, 2000), cell types with >=25 cells)
and compares with the stored outputs celltype_difficulty.json:classes and hardcell_agreement.json:n_cells."""
import json, sys, os
from collections import Counter
import numpy as np, anndata as ad
ORIG = os.path.join(os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".."))), "outputs/atlas/comparative")
a = ad.read_h5ad(sys.argv[1], backed="r")
idx = np.sort(np.random.default_rng(0).choice(a.n_obs, 2000, replace=False))
y = a.obs["cell_type"].astype(str).values[idx]
vc = Counter(y); keep = sorted(c for c, n in vc.items() if n >= 25)
n = int(sum(v in keep for v in y))
pub_classes = json.load(open(f"{ORIG}/celltype_difficulty.json"))["classes"]
pub_n = json.load(open(f"{ORIG}/hardcell_agreement.json"))["n_cells"]
out = {"our_classes": keep, "published_classes": pub_classes, "classes_identical": keep == sorted(pub_classes),
       "our_n_cells_after_mask": n, "published_n_cells": pub_n, "n_identical": n == pub_n,
       "our_class_counts_in_2000_subsample": {c: vc[c] for c in keep}}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "same_cells_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
