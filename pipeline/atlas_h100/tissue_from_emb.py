#!/usr/bin/env python
"""Per-layer tissue specificity from the all-layers cell embeddings (cell_cka --all-layers). Each sae_L is
[n_cells, d_sae] mean feature activation per cell; grouping cells by tissue (obs['compartment']) gives each
feature's mean activation per tissue. A feature is tissue-specific if one tissue holds >= THRESH of its
total. Output: per model, the fraction of (alive) features that are tissue-specific at every layer + the
split by tissue. Reads out/cka/<M>_emb.npz. CPU, tiny output.
    python tissue_from_emb.py --all --out out_alllayers   -> out_alllayers/tissue_alllayers.json
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THRESH = 0.5


def labels(corpus, max_cells, key):
    import scanpy as sc
    ad = sc.read_h5ad(corpus)
    if max_cells and ad.n_obs > max_cells:
        idx = np.sort(np.random.default_rng(0).choice(ad.n_obs, max_cells, replace=False))
    else:
        idx = np.arange(ad.n_obs)
    return ad.obs[key].astype(str).values[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--tissue-key", default="compartment")
    ap.add_argument("--max-cells", type=int, default=2000)
    args = ap.parse_args()
    y = labels(args.corpus, args.max_cells, args.tissue_key)
    tissues = sorted(set(y)); tidx = {t: i for i, t in enumerate(tissues)}
    yi = np.array([tidx[v] for v in y])
    print(f"tissues: {tissues} ({len(y)} cells)", flush=True)
    targets = MODELS if args.all else [args.model]
    res = {"tissues": tissues, "models": {}}
    for m in targets:
        p = os.path.join(args.out, "cka", f"{m}_emb.npz")
        if not os.path.exists(p):
            print(f"{m}: no {p}, skip", flush=True); continue
        e = dict(np.load(p))
        lay = e["layers"].tolist() if "layers" in e else list(range(sum(1 for k in e if k.startswith("sae_"))))
        prof = []
        rels = sorted(int(k.split("_")[1]) for k in e if k.startswith("sae_"))
        for r in rels:
            H = e[f"sae_{r}"]                                     # [cells, d_sae]
            means = np.stack([H[yi == ti].mean(0) for ti in range(len(tissues))], 1)  # [d_sae, n_tissue]
            alive = means.sum(1) > 1e-9
            share = means[alive] / (means[alive].sum(1, keepdims=True) + 1e-12)
            spec = share.max(1); pref = share.argmax(1)
            is_spec = spec >= THRESH
            by = {tissues[ti]: int(((pref == ti) & is_spec).sum()) for ti in range(len(tissues))}
            prof.append({"layer": int(lay[r]) if r < len(lay) else r,
                         "frac_specific": round(float(is_spec.mean()), 3),
                         "n_alive": int(alive.sum()), "by_tissue": by})
        res["models"][m] = prof
        print(f"{m}: {len(prof)} layers, frac_specific {prof[0]['frac_specific']}->{prof[-1]['frac_specific']}", flush=True)
    fp = os.path.join(args.out, "tissue_alllayers.json"); json.dump(res, open(fp, "w"))
    print(f"==> {fp}", flush=True)


if __name__ == "__main__":
    main()
