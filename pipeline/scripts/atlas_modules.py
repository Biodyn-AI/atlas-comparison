#!/usr/bin/env python
"""bio-sae-style views for OUR models from TRUE co-activation graphs (coactivation.py on the
cluster): (1) feature-module network — force-directed layout of the real SAE feature
co-activation graph (Pearson corr across gene-token positions) + modularity communities,
colored by module, sized by co-activation degree; nodes carry their top concept / genes from
the aligned catalog. (2) per-layer annotation profile (rate + semantic richness).

    python scripts/atlas_modules.py -> outputs/atlas/comparative/modules_data.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os, glob
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from collections import Counter

BASE = _B
CL = f"{BASE}/outputs/atlas/cluster/out"
OUT = f"{BASE}/outputs/atlas/comparative/modules_data.json"
# (model, layer, catalog path, annotations path, coact path)
SPEC = {
    "AIDO": (8, f"{CL}/AIDO/feature_catalog_L08.json", f"{CL}/annotations/AIDO_L08_annotations.json", f"{CL}/coact/coact_AIDO_L08.json"),
    "UCE":  (3, f"{CL}/UCE/feature_catalog_L03.json",  f"{CL}/annotations/UCE_L03_annotations.json",  f"{CL}/coact/coact_UCE_L03.json"),
    "tGPT": (8, f"{CL}/tGPT/feature_catalog_L08.json", f"{CL}/annotations/tGPT_L08_annotations.json", f"{CL}/coact/coact_tGPT_L08.json"),
}
MAX_NODES = 260
SEED = 7


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "")
            .replace("TRRUST:", "TF ").split(" (GO:")[0].split(" (R-HSA")[0]
            .split(" (hsa")[0].replace("Homo sapiens ", "").strip())


def build(model):
    layer, cat_p, ann_p, co_p = SPEC[model]
    cat = json.load(open(cat_p))["features"]          # {fid: {top_genes,...}}
    ann = json.load(open(ann_p))["annotations"]       # {fid: [{source,term,...}]}
    co = json.load(open(co_p))
    edges_all = co["edges"]                            # [[i,j,w],...] ids are catalog feature ids
    def concept(fid):
        a = ann.get(str(fid))
        if a:
            t = a[0]; return f"{t['source']}:{t['term']}"
        g = cat.get(str(fid), {}).get("top_genes", [])
        return "gene:" + (g[0] if g else "?")
    def is_tf(fid):
        a = ann.get(str(fid), [])
        return any(x["source"] == "TRRUST" for x in a)

    G = nx.Graph()
    for a, b, w in edges_all:
        G.add_edge(a, b, weight=w)
    n_connected = G.number_of_nodes()
    # cap to top-degree nodes for a legible render
    if G.number_of_nodes() > MAX_NODES:
        keep = [n for n, _ in sorted(G.degree(), key=lambda x: -x[1])[:MAX_NODES]]
        G = G.subgraph(keep).copy()
    if G.number_of_edges() == 0:
        return {"nodes": [], "edges": [], "n_mod": 0, "mod_labels": {},
                "n_alive": len(co["features"]), "n_edges": len(edges_all),
                "n_connected": n_connected, "layer": layer}
    comms = list(greedy_modularity_communities(G, weight="weight"))
    mod = {}
    for ci, c in enumerate(comms):
        for n in c:
            mod[n] = ci
    sz = Counter(mod.values())
    keep_m = {c for c, s in sz.items() if s >= 3}
    remap = {c: i for i, c in enumerate(sorted(keep_m, key=lambda c: -sz[c]))}
    n_mod = len(remap)
    for n in list(mod):
        mod[n] = remap.get(mod[n], n_mod)             # overflow bucket
    pos = nx.spring_layout(G, weight="weight", seed=SEED,
                           k=2.2 / np.sqrt(max(G.number_of_nodes(), 1)), iterations=200)
    ids = list(G.nodes())
    xs = np.array([pos[n][0] for n in ids]); ys = np.array([pos[n][1] for n in ids])
    xs = (xs - xs.min()) / max(float(np.ptp(xs)), 1e-9)
    ys = (ys - ys.min()) / max(float(np.ptp(ys)), 1e-9)
    deg = dict(G.degree())
    # module label = dominant top concept
    lab = {}
    for m in range(n_mod + 1):
        c = Counter(clean(concept(n)) for n in ids if mod[n] == m)
        lab[m] = c.most_common(1)[0][0] if c else f"module {m}"
    idx = {n: k for k, n in enumerate(ids)}
    def anns_of(fid):
        a = ann.get(str(fid), [])[:6]
        return [[x["source"], clean(f"{x['source']}:{x['term']}"), round(float(x.get("q", 1)), 4)] for x in a]
    nodes = [{"id": int(n), "x": round(float(xs[k]), 4), "y": round(float(ys[k]), 4),
              "m": int(mod[n]), "deg": int(deg[n]),
              "lab": clean(concept(n)),
              "genes": cat.get(str(n), {}).get("top_genes", [])[:5],
              "g": cat.get(str(n), {}).get("top_genes", [])[:10],
              "fr": round(float(cat.get(str(n), {}).get("activation_frequency", 0)), 5),
              "an": anns_of(n),
              "tf": bool(is_tf(n))} for k, n in enumerate(ids)]
    ed = [[idx[a], idx[b], w] for a, b, w in
          [(a, b, w) for a, b, w in edges_all if a in idx and b in idx]]
    return {"nodes": nodes, "edges": ed, "n_mod": n_mod,
            "mod_labels": {str(k): v for k, v in lab.items()},
            "n_alive": len(co["features"]), "n_edges": len(edges_all),
            "n_connected": n_connected, "layer": layer}


def layer_profile(model):
    pts = []
    for p in sorted(glob.glob(f"{CL}/annotations/{model}_L*_annotations.json")):
        d = json.load(open(p))
        na, n = d["n_annotated"], d["n_features"]
        tot = sum(len(v) for v in d["annotations"].values())
        pts.append({"layer": d["layer"], "rate": round(100 * na / max(n, 1), 1),
                    "rich": round(tot / max(na, 1), 1), "n_feat": n})
    return pts


data = {"models": list(SPEC), "graphs": {}, "profiles": {}}
for m in SPEC:
    g = build(m)
    data["graphs"][m] = g
    data["profiles"][m] = layer_profile(m)
    print(f"{m:5s} L{g['layer']}: {g['n_alive']} alive, {g['n_edges']} coact-edges, "
          f"{g['n_connected']} connected nodes -> rendered {len(g['nodes'])} nodes / "
          f"{len(g['edges'])} edges / {g['n_mod']} modules; profile pts={len(data['profiles'][m])}")
json.dump(data, open(OUT, "w"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
