#!/usr/bin/env python
"""Step 5 (job item 4): how much does the TRRUST result depend on pooling many layers, and on W?

Layer sets (per model; layer index = position of feature_catalog_L*.json in sorted order):
  all       - every layer (the published test)
  noL0      - every layer except the first (L00)
  even8     - 8 layers evenly spaced by depth, exactly the rule of controls.py:129-133
              (idx = linspace(0, L-1, 8).round(); models with <= 8 layers keep all layers)
  even8noL0 - the same rule applied to layers 1..L-1
Thresholds W in {2, 3, 5, 10} (a pair is predicted if >= W features contain both genes).
Per model (all ten): repo's stub-matching configuration null, NPERM draws, seed 0.
Cross-model curve: all ten models and the five passing models; precision at >= 1 and >= 2 models;
NCURVE draws, each rewiring every graph independently (as hypothesis_trrust2.py:124-146).
Output: OUT/s05_pooling.json
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NCURVE = int(sys.argv[2]) if len(sys.argv) > 2 else 100
WS = [2, 3, 5, 10]
MINL = 8

genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, _, _ = load_truth()
tkey = truth_keys(truth, gid)


def evenly(Ls):
    if len(Ls) <= MINL:
        return list(Ls)
    idx = np.linspace(0, len(Ls) - 1, MINL).round().astype(int)
    return [Ls[i] for i in sorted(set(idx))]


GR = {}          # (layerset, W) -> {model: keys}
out = {"n_perm": NPERM, "n_perm_curve": NCURVE, "W": WS, "layer_sets": {}, "per_model": {}, "curve": {}}
t0 = time.time()
for m in ORDER:
    feat, lay, L = load_model(m)
    Ls = list(range(L))
    sets = {"all": Ls, "noL0": Ls[1:], "even8": evenly(Ls), "even8noL0": evenly(Ls[1:])}
    out["layer_sets"][m] = {k: v for k, v in sets.items()}
    out["per_model"][m] = {}
    for sname, keep in sets.items():
        sub = feat[np.isin(lay, np.array(keep))]
        u, c = np.unique(pair_keys(sub), return_counts=True)
        out["per_model"][m][sname] = {"n_layers": len(keep), "n_features": int(len(sub))}
        for W in WS:
            keys = u[c >= W]
            GR.setdefault((sname, W), {})[m] = keys
            r = score(keys, tkey, NPERM, seed=0) if len(keys) else {"n_pairs": 0, "hits": 0}
            out["per_model"][m][sname][f"W{W}"] = r
            print(f"{m:<12} {sname:<10} W{W:<3} {r.get('n_pairs'):>9} pairs {r.get('hits'):>4} hits null {r.get('null_mean')} "
                  f"fold {r.get('fold')} p {r.get('p_emp')}  [{time.time()-t0:.0f}s]", flush=True)
        del sub, u, c
    del feat, lay


def curve(gmap, models):
    allk = np.concatenate([gmap[m] for m in models if len(gmap[m])])
    kk, vv = np.unique(allk, return_counts=True)
    ist = np.isin(kk, tkey)
    obs = {k: (int((vv >= k).sum()), int(ist[vv >= k].sum())) for k in (1, 2)}
    rng = np.random.default_rng(0)
    prec = {k: [] for k in obs}; nh = {k: [] for k in obs}
    for _ in range(NCURVE):
        allr = np.concatenate([config_shuffle_keys(gmap[m], rng) for m in models if len(gmap[m])])
        k3, v3 = np.unique(allr, return_counts=True)
        i3 = np.isin(k3, tkey)
        for k in obs:
            sel = v3 >= k
            prec[k].append(float(i3[sel].sum()) / max(int(sel.sum()), 1)); nh[k].append(int(i3[sel].sum()))
    res = {}
    for k, (n, h) in obs.items():
        pc = np.array(prec[k]); po = h / n if n else 0.0
        res[k] = {"n_pairs": n, "hits": h, "precision": round(po, 6), "null_precision_mean": round(float(pc.mean()), 8),
                  "null_precision_sd": round(float(pc.std()), 8), "null_hits_mean": round(float(np.mean(nh[k])), 3),
                  "null_draws_with_zero_hits": int((np.array(nh[k]) == 0).sum()),
                  "fold": round(po / pc.mean(), 2) if pc.mean() > 0 else None,
                  "p_emp": round((int((pc >= po).sum()) + 1) / (NCURVE + 1), 5), "n_perm": NCURVE}
    return res


for (sname, W), gmap in GR.items():
    out["curve"][f"{sname}_W{W}"] = {"all_ten": curve(gmap, ORDER), "five_passing": curve(gmap, PASSED)}
    c = out["curve"][f"{sname}_W{W}"]
    print(f"curve {sname} W{W}: ten k>=2 {c['all_ten'][2]['hits']}/{c['all_ten'][2]['n_pairs']} fold {c['all_ten'][2]['fold']} p {c['all_ten'][2]['p_emp']};"
          f" five k>=2 {c['five_passing'][2]['hits']}/{c['five_passing'][2]['n_pairs']} fold {c['five_passing'][2]['fold']} p {c['five_passing'][2]['p_emp']}"
          f"  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"{OUT}/s05_pooling.json", "w"), indent=1)
print("done", time.time() - t0)
