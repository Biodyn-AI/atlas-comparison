#!/usr/bin/env python
"""Reviewer control #3 — SAE-seed variance / CIs on the headline numbers. The whole atlas rests on ONE SAE per
model per layer; SAE training is stochastic, so which features emerge varies. Here we retrain the mid-layer SAE
with several seeds per model (from the SAVED activations — no re-extraction), rebuild the top-5 catalog, annotate
(Fisher+BH, 5 DBs), and record per (model, seed): n_alive, annotation rate, n_concepts, and the concept SET. From
the per-seed concept sets we also get a CI on the UNIVERSAL CORE (intersection across models, per seed). Checkpoints
after every (model, seed) and resumes, because the shared pod OOM-cycles.

    python seed_variance.py --out_root out_alllayers --genesets data/genesets --bg data/genesets/background.json \
        --models AIDO C2S Geneformer MaxToki UCE scGPT tGPT scFoundation GeneCompass Tahoe \
        --seeds 0 1 2 --dst seed_variance.json --device cuda
"""
from __future__ import annotations
import argparse, glob, json, os, sys, gc
import numpy as np
from scipy.stats import fisher_exact
from collections import defaultdict, Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.sae import SAECfg, train_sae, feature_activations

MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def load_sets(gdir):
    gs = {}
    for nm in ("GO_BP", "KEGG", "Reactome"):
        p = os.path.join(gdir, f"{nm}_gene_sets.json")
        if os.path.exists(p):
            for t, g in json.load(open(p)).items():
                s = set(x.upper() for x in g)
                if MIN <= len(s) <= MAX:
                    gs[f"{nm}:{t}"] = s
    for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
        p = os.path.join(gdir, fn)
        if os.path.exists(p):
            for t, g in json.load(open(p)).items():
                s = set(x.upper() for x in g)
                if MIN <= len(s) <= MAX:
                    gs[f"{nm}:{t}"] = s
    return gs


def annotate(top_genes, idx, size, bgn):
    recs = []
    for fid, genes in top_genes.items():
        g = set(genes[:TOP])
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < 2:
                continue
            tg = size[k]; b = len(g) - a; c = tg - a; dd = bgn - a - b - c
            _, p = fisher_exact([[a, b], [c, dd]], alternative="greater")
            recs.append((fid, k, float(p)))
    if not recs:
        return set(), 0
    q = bh([r[2] for r in recs]); ann = defaultdict(list)
    for i, (fid, k, p) in enumerate(recs):
        if q[i] <= ALPHA:
            ann[fid].append(k)
    terms = set(t for v in ann.values() for t in v)
    return terms, len(ann)


def mid_paths(d):
    cats = sorted(glob.glob(os.path.join(d, "feature_catalog_L*.json")), key=lambda p: json.load(open(p))["layer"])
    if not cats:
        return None
    cp = cats[len(cats) // 2]; L = json.load(open(cp))["layer"]
    return L, os.path.join(d, f"layer_{L:02d}_activations.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_alllayers")
    ap.add_argument("--genesets", default="data/genesets")
    ap.add_argument("--bg", default="data/genesets/background.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dst", default="seed_variance.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    lockf = open(a.dst + ".lock", "w")
    import fcntl
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another seed_variance holds the lock — exiting", flush=True); sys.exit(0)

    gs = load_sets(a.genesets)
    idx = defaultdict(list); size = {}
    for t, genes in gs.items():
        size[t] = len(genes)
        for g in genes:
            idx[g].append(t)
    bg = set(x.upper() for x in json.load(open(a.bg))); bg |= set().union(*gs.values())
    bgn = len(bg)
    print(f"{len(gs)} sets, bg {bgn}, device {a.device}", flush=True)

    out = json.load(open(a.dst)) if os.path.exists(a.dst) else {"models": {}, "concept_sets": {}}
    out.setdefault("models", {}); out.setdefault("concept_sets", {})
    for m in a.models:
        d = os.path.join(a.out_root, m)
        mp = mid_paths(d)
        if not mp:
            print(f"  skip {m}"); continue
        L, act = mp
        if not os.path.exists(act):
            print(f"  {m}: missing {act}"); continue
        syms = np.load(os.path.join(d, "gene_symbols.npy"), allow_pickle=True).astype(str)
        syms = np.char.upper(syms)
        uniq = np.array(sorted(set(syms.tolist()))); gi = {g: i for i, g in enumerate(uniq)}
        sym_idx = np.array([gi[g] for g in syms])
        A = np.load(act, mmap_mode="r")
        for seed in a.seeds:
            key = f"{m}#{seed}"
            if key in out["concept_sets"]:
                print(f"  {m} seed {seed} cached"); continue
            torch.manual_seed(seed); np.random.seed(seed)
            cfg = SAECfg(d_model=A.shape[1], expansion=4, k=32, epochs=4)
            sae, stats = train_sae(np.asarray(A), cfg, device=a.device, log=lambda *x: None)
            F = feature_activations(sae, np.asarray(A), device=a.device)
            Gm = np.zeros((len(uniq), F.shape[1]), np.float64); c = np.zeros(len(uniq))
            np.add.at(Gm, sym_idx, F); np.add.at(c, sym_idx, 1); Gm /= np.maximum(c[:, None], 1)
            alive = np.where((F > 0).mean(0) > 0)[0]
            topg = {int(f): [str(uniq[i]) for i in np.argsort(Gm[:, f])[::-1][:20] if uniq[i] in bg][:TOP] for f in alive}
            terms, nann = annotate(topg, idx, size, bgn)
            rec = {"seed": seed, "n_alive": int(len(alive)), "n_annot": int(nann),
                   "annot_rate": round(100 * nann / max(len(alive), 1), 1), "n_concepts": len(terms),
                   "var_explained": round(float(stats.get("var_explained", 0)), 3)}
            out["models"].setdefault(m, []).append(rec)
            out["concept_sets"][key] = sorted(terms)
            gc.collect()
            json.dump(out, open(a.dst, "w"))
            print(f"  {m} seed {seed}: rate {rec['annot_rate']}% · {len(terms)} concepts · {len(alive)} alive [saved]", flush=True)

    # universal-core CI across seeds (intersection across models, per seed)
    seeds = sorted(set(int(k.split('#')[1]) for k in out["concept_sets"]))
    cores = []
    for s in seeds:
        sets = [set(out["concept_sets"][f"{m}#{s}"]) for m in a.models if f"{m}#{s}" in out["concept_sets"]]
        if len(sets) == len(a.models):
            cores.append(len(set.intersection(*sets)))
    if cores:
        out["core_by_seed"] = cores
        out["core_mean"] = round(float(np.mean(cores)), 1)
        out["core_sd"] = round(float(np.std(cores)), 1)
        print(f"\nMID-LAYER universal core across seeds: {cores} -> {out['core_mean']} ± {out['core_sd']}", flush=True)
    json.dump(out, open(a.dst, "w"))
    print(f"==> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
