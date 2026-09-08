#!/usr/bin/env python
"""Cross-layer feature CIRCUITS from the saved SAEs (no forward, CPU only). A feature f in layer L
writes its decoder direction W_dec_L[:,f] into the residual; a feature g in the next depth-matched
layer reads via its encoder row W_enc_L2[g,:]. The first-order residual-skip connection strength is the
cosine between them (signed: + promotes g, - suppresses). This gives a weighted, directed feature->feature
circuit graph across depth — the causal-ish counterpart to the gene-Jaccard 'flow' (persistence).

Per model, exports: (a) global connectivity stats per adjacent layer pair, (b) a compact 5-column circuit
diagram (top circuit-participating features per layer + signed edges) for rendering.

    python circuits.py --model UCE --out out_ts3          # one model
    python circuits.py --all  --out out_ts3               # all models -> circuits_ts3.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT"]
STRONG = 0.30          # |cos| above which a connection counts as a real circuit link
EDGE_THR = 0.22        # edge kept for the diagram
NODES_PER_LAYER = 34   # features shown per column in the diagram
TOPK_IN = 2            # incoming edges kept per shown late feature


def load_sae_dirs(path):
    import torch
    from common.sae import TopKSAE
    ck = torch.load(path, map_location="cpu"); cfg = ck["cfg"]
    d_model = int(cfg["d_model"]); d_sae = int(cfg["expansion"]) * d_model
    sae = TopKSAE(d_model, d_sae, int(cfg["k"])); sae.load_state_dict(ck["state_dict"])
    Wd = sae.W_dec.weight.detach().numpy()                       # [d_model, d_sae]
    We = sae.W_enc.weight.detach().numpy()                       # [d_sae, d_model]
    Wd = Wd / (np.linalg.norm(Wd, axis=0, keepdims=True) + 1e-9)
    We = We / (np.linalg.norm(We, axis=1, keepdims=True) + 1e-9)
    return Wd, We


def model_circuits(model, out, freq_all):
    mdir = os.path.join(out, model)
    layers = sorted(int(re.search(r"sae_L(\d+)\.pt", p).group(1)) for p in glob.glob(f"{mdir}/sae_L*.pt"))
    if len(layers) < 2:
        print(f"{model}: <2 SAE layers, skip", flush=True); return None
    # alive masks per absolute layer, from freq_ts3 (rel index -> abs via its 'layers')
    alive = {}
    fm = freq_all.get(model)
    if fm:
        for ri, L in enumerate(fm["layers"]):
            f = np.asarray(fm["freq"].get(str(ri), []), float)
            if f.size:
                alive[L] = np.where(f > 0)[0]
    dirs = {L: load_sae_dirs(f"{mdir}/sae_L{L:02d}.pt") for L in layers}
    pairs, cols, edges = [], [], []
    # node id lists per layer for the diagram (filled as we go; first layer seeded from first pair)
    col_ids = [None] * len(layers)
    for i in range(len(layers) - 1):
        L, L2 = layers[i], layers[i + 1]
        Wd_L = dirs[L][0]; We_L2 = dirs[L2][1]
        aL = alive.get(L, np.arange(Wd_L.shape[1])); aL2 = alive.get(L2, np.arange(We_L2.shape[0]))
        Wd_a = Wd_L[:, aL]                                        # [d_model, nL]
        We_a = We_L2[aL2]                                         # [nL2, d_model]
        M = We_a @ Wd_a                                           # [nL2, nL] signed cosine
        Amax = np.abs(M)
        top1 = Amax.max(1)                                        # best incoming |cos| per late feature
        arg = Amax.argmax(1)
        sign_at = np.sign(M[np.arange(M.shape[0]), arg])
        pairs.append({"L": int(L), "L2": int(L2), "n_late": int(len(aL2)), "n_early": int(len(aL)),
                      "connectivity": round(float((top1 > STRONG).mean()), 3),
                      "mean_top1": round(float(top1.mean()), 3),
                      "frac_excit": round(float((sign_at[top1 > STRONG] > 0).mean()) if (top1 > STRONG).any() else 0.0, 3)})
        # participation score per feature (sum of strong |cos|) to pick diagram nodes
        strong = Amax * (Amax > EDGE_THR)
        part_late = strong.sum(1); part_early = strong.sum(0)
        if col_ids[i] is None:
            sel_e = np.argsort(-part_early)[:NODES_PER_LAYER]
            col_ids[i] = sel_e
        sel_l = np.argsort(-part_late)[:NODES_PER_LAYER]
        col_ids[i + 1] = sel_l
        # edges among selected: for each shown late node, top-k incoming from shown early nodes
        se = col_ids[i]; sl = col_ids[i + 1]
        sub = M[np.ix_(sl, se)]                                   # [nodes_l, nodes_e]
        for gi in range(sub.shape[0]):
            order = np.argsort(-np.abs(sub[gi]))[:TOPK_IN]
            for fi in order:
                w = float(sub[gi, fi])
                if abs(w) >= EDGE_THR:
                    edges.append([i, int(fi), int(gi), round(w, 3)])
    # materialise columns as global feature ids (aL indices back to real feature ids)
    for i, L in enumerate(layers):
        a = alive.get(L, None)
        ids = col_ids[i]
        if a is not None:
            cols.append([int(a[j]) for j in ids])
        else:
            cols.append([int(j) for j in ids])
    return {"layers": [int(x) for x in layers], "pairs": pairs, "cols": cols, "edges": edges}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_ts3")
    args = ap.parse_args()
    fp = os.path.join(args.out, "cka", "freq_ts3.json")
    freq_all = json.load(open(fp)) if os.path.exists(fp) else {}
    if not freq_all:
        print("WARN: no freq_ts3.json -> using ALL features as alive (dead features add noise)", flush=True)
    targets = MODELS if args.all else [args.model]
    res = {}
    for m in targets:
        r = model_circuits(m, args.out, freq_all)
        if r:
            res[m] = r
            cn = np.mean([p["connectivity"] for p in r["pairs"]])
            print(f"{m}: layers={r['layers']} mean_connectivity={cn:.2f} edges={len(r['edges'])}", flush=True)
    od = os.path.join(args.out, "circuits"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "circuits_ts3.json"); json.dump({"models": [m for m in MODELS if m in res], **{"data": res}}, open(p, "w"))
    print(f"==> {p} ({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
