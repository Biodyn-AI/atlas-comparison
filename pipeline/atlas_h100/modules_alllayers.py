#!/usr/bin/env python
"""Co-activation MODULE graphs for ALL five depth-matched layers (the Modules section currently shows one).
Reuses the per-layer co-activation edges already computed by layer_explorer (out_ts3/explorer/<M>_explorer.json)
+ the local catalogs/annotations for node genes/concepts. Per layer: build the strong-edge graph, find
modularity communities, spring-layout the top-degree nodes, attach genes + top concept + TF flag. Output
matches modules_ts3.json's node/edge schema (edges index into nodes; x,y in 0..1) so the existing renderer
just gains a layer selector.

    python modules_alllayers.py --all --out out_ts3   -> out_ts3/modules_alllayers.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from collections import Counter

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
MAX_NODES = 260
SEED = 7


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "")
            .replace("TRRUST:", "TF ").replace("STRING:", "PPI ").split(" (GO:")[0].split(" (R-HSA")[0]
            .split(" (hsa")[0].replace("Homo sapiens ", "").strip())


def load_ann(path):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return d.get("annotations", d)


def build_layer(L, feats_alive, edges, cat, ann):
    # graph from strong edges (ids)
    G = nx.Graph()
    G.add_nodes_from(feats_alive)
    for e in edges:
        a, b, w = int(e[0]), int(e[1]), float(e[2])
        if a in G and b in G:
            G.add_edge(a, b, weight=w)
    deg = dict(G.degree())
    n_connected = sum(1 for v in deg.values() if v > 0)
    # keep the top-degree nodes (the co-activating core) for a legible layout
    keep = sorted(feats_alive, key=lambda i: -deg.get(i, 0))[:MAX_NODES]
    keep = [i for i in keep if deg.get(i, 0) > 0] or keep[:MAX_NODES]
    H = G.subgraph(keep).copy()
    comms = list(greedy_modularity_communities(H)) if H.number_of_edges() else [set(H.nodes())]
    mod = {}
    for mi, cset in enumerate(comms):
        for n in cset:
            mod[n] = mi
    pos = nx.spring_layout(H, seed=SEED, k=1.6 / max(np.sqrt(max(H.number_of_nodes(), 1)), 1), iterations=120)

    def concept(fid):
        a = ann.get(str(fid))
        if a:
            t = a[0]; return clean(f"{t.get('source','')}:{t.get('term','')}")
        g = cat.get(str(fid), {}).get("top_genes", [])
        return g[0] if g else "?"

    def is_tf(fid):
        return any(x.get("source") == "TRRUST" for x in ann.get(str(fid), []))

    nodes = list(H.nodes())
    xs = np.array([pos[n][0] for n in nodes]); ys = np.array([pos[n][1] for n in nodes])
    xs = (xs - xs.min()) / max(np.ptp(xs), 1e-9); ys = (ys - ys.min()) / max(np.ptp(ys), 1e-9)
    idx = {n: i for i, n in enumerate(nodes)}
    nlist = [{"id": int(n), "x": round(float(xs[i]), 4), "y": round(float(ys[i]), 4), "m": int(mod.get(n, 0)),
              "deg": int(deg.get(n, 0)), "lab": concept(n),
              "genes": [str(g).upper() for g in cat.get(str(n), {}).get("top_genes", [])[:6]], "tf": bool(is_tf(n))}
             for i, n in enumerate(nodes)]
    elist = []
    for a, b, d in H.edges(data=True):
        elist.append([idx[a], idx[b], round(float(d.get("weight", 0)), 3)])
    elist.sort(key=lambda e: -e[2]); elist = elist[:600]
    # module labels = dominant concept per community
    lab = {}
    for mi in set(mod.values()):
        c = Counter(clean(concept(n)) for n in nodes if mod.get(n) == mi)
        lab[str(mi)] = c.most_common(1)[0][0] if c else f"module {mi}"
    return {"nodes": nlist, "edges": elist, "n_mod": len(comms), "mod_labels": lab,
            "n_alive": len(feats_alive), "n_edges": len(edges), "n_connected": n_connected, "layer": int(L)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_ts3")
    args = ap.parse_args()
    targets = MODELS if args.all else [args.model]
    graphs = {}
    for m in targets:
        expp = os.path.join(args.out, "explorer", f"{m}_explorer.json")
        if not os.path.exists(expp):
            print(f"{m}: no {expp}, skip", flush=True); continue
        exp = json.load(open(expp))
        per = {}
        for L, ld in exp["layers"].items():
            LL = f"{int(L):02d}"
            cat = json.load(open(f"{args.out}/{m}/feature_catalog_L{LL}.json"))["features"]
            ann = load_ann(f"{args.out}/annotations/{m}_L{LL}_annotations.json")
            feats = [f["id"] for f in ld["features"]]
            g = build_layer(L, feats, ld.get("edges", []), cat, ann)
            per[str(L)] = g
            print(f"  {m} L{L}: {len(g['nodes'])} nodes, {g['n_mod']} modules, {len(g['edges'])} edges", flush=True)
        graphs[m] = per
    out = {"models": [m for m in MODELS if m in graphs], "graphs": graphs}
    od = args.out; p = os.path.join(od, "modules_alllayers.json"); json.dump(out, open(p, "w"))
    print(f"==> {p} ({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
