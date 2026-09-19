#!/usr/bin/env python
"""All-layer SAE-seed CI — control #3 at full depth. Retrain the SAE with several seeds on EVERY layer (from the
saved frozen-model activations; only the SAE varies), annotate top-5 per layer, union across layers per seed ->
each model's all-layer concept repertoire per seed -> intersection across models = the ALL-LAYER universal core
per seed -> CI. GPU (fast); checkpoints after every (model, seed, layer) and resumes, because the pod is shared.

    python seed_variance_alllayer.py --out_root out_alllayers --genesets data/genesets --bg data/genesets/background.json \
        --models AIDO C2S Geneformer MaxToki UCE scGPT tGPT scFoundation GeneCompass Tahoe \
        --seeds 0 1 2 --dst seed_variance_alllayer.json --device cuda
"""
from __future__ import annotations
import argparse, glob, json, os, sys, gc, fcntl
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.sae import SAECfg, train_sae, feature_activations

MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5


@torch.no_grad()
def gpu_gene_scores(sae, Anp, sym_idx, n_uniq, device, batch=65536):
    """Per-gene mean SAE-feature activation accumulated ON the GPU (index_add_) — avoids materialising the
    huge [n_pos x d_sae] activation matrix on CPU. Returns G[n_uniq x d_sae] (cpu) + alive mask [d_sae]."""
    d_sae = sae.d_sae
    G = torch.zeros(n_uniq, d_sae, device=device, dtype=torch.float32)
    cnt = torch.zeros(n_uniq, device=device, dtype=torch.float32)
    nz = torch.zeros(d_sae, device=device, dtype=torch.float32)
    si = torch.as_tensor(sym_idx, device=device, dtype=torch.long)
    ones = None
    for i in range(0, len(Anp), batch):
        xb = torch.as_tensor(Anp[i:i + batch], dtype=torch.float32, device=device)
        fb = sae.encode(xb)                       # [b, d_sae]
        idxb = si[i:i + batch]
        G.index_add_(0, idxb, fb)
        if ones is None or ones.shape[0] != xb.shape[0]:
            ones = torch.ones(xb.shape[0], device=device)
        cnt.index_add_(0, idxb, ones[:xb.shape[0]])
        nz += (fb > 0).sum(0)
    G /= cnt.clamp(min=1).unsqueeze(1)
    return G.cpu().numpy(), (nz > 0).cpu().numpy()


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


def annotate(top_genes, idx, size, bgn, bg):
    """Vectorised: Fisher's exact one-sided ('greater') == hypergeometric upper tail. Same p-values as
    scipy.fisher_exact(alternative='greater') but computed in one hypergeom.sf call over all candidates."""
    A = []; TG = []; LG = []; TERM = []
    for fid, genes in top_genes.items():
        g = set(genes[:TOP]) & bg      # canonical: raw top-5, then intersect background
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        lg = len(g)
        for k, a in cand.items():
            if a < 2:
                continue
            A.append(a); TG.append(size[k]); LG.append(lg); TERM.append(k)
    if not A:
        return set()
    A = np.array(A); TG = np.array(TG); LG = np.array(LG)
    p = hypergeom.sf(A - 1, bgn, TG, LG)   # P(X >= a): pop=bgn, successes=set size, draws=feature genes
    q = bh(p)
    return set(TERM[i] for i in np.where(q <= ALPHA)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_alllayers")
    ap.add_argument("--genesets", default="data/genesets")
    ap.add_argument("--bg", default="data/genesets/background.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dst", default="seed_variance_alllayer.json")
    ap.add_argument("--max_positions", type=int, default=0, help="0 = use all saved positions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    lockf = open(a.dst + ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another seed_variance_alllayer holds the lock — exiting", flush=True); sys.exit(0)

    gs = load_sets(a.genesets)
    idx = defaultdict(list); size = {}
    for t, genes in gs.items():
        size[t] = len(genes)
        for g in genes:
            idx[g].append(t)
    # background EXACTLY as canonical alllayer_matrix: background.json + STRING neighbour genes + STRING hub keys
    bg = set(x.upper() for x in json.load(open(a.bg)))
    st_vals = set().union(*[genes for t, genes in gs.items() if t.startswith("STRING:")]) if any(t.startswith("STRING:") for t in gs) else set()
    st_keys = set(t.split(":", 1)[1].upper() for t in gs if t.startswith("STRING:"))
    bg |= st_vals | st_keys; bgn = len(bg)
    print(f"{len(gs)} sets, bg {bgn} (canonical: background+STRING), device {a.device}", flush=True)

    # checkpoint: layer-level concept sets keyed "model#seed#Lnn"
    out = json.load(open(a.dst)) if os.path.exists(a.dst) else {"layer_sets": {}}
    out.setdefault("layer_sets", {})
    for m in a.models:
        d = os.path.join(a.out_root, m)
        cats = sorted(glob.glob(os.path.join(d, "feature_catalog_L*.json")), key=lambda p: json.load(open(p))["layer"])
        if not cats:
            print(f"  skip {m}: no catalogs"); continue
        syms = np.char.upper(np.load(os.path.join(d, "gene_symbols.npy"), allow_pickle=True).astype(str))
        uniq = np.array(sorted(set(syms.tolist()))); gi = {g: i for i, g in enumerate(uniq)}
        sym_idx = np.array([gi[g] for g in syms])
        for cp in cats:
            L = json.load(open(cp))["layer"]
            act = os.path.join(d, f"layer_{L:02d}_activations.npy")
            if not os.path.exists(act):
                continue
            todo = [s for s in a.seeds if f"{m}#{s}#L{L:02d}" not in out["layer_sets"]]
            if not todo:
                continue
            A = np.load(act, mmap_mode="r")
            if a.max_positions and len(A) > a.max_positions:
                A = A[:a.max_positions]
            Anp = np.ascontiguousarray(A)               # materialise ONCE per layer (was 6x: train+feat x 3 seeds)
            si = sym_idx[:len(Anp)] if a.max_positions else sym_idx
            for seed in todo:
                key = f"{m}#{seed}#L{L:02d}"
                torch.manual_seed(seed); np.random.seed(seed)
                cfg = SAECfg(d_model=Anp.shape[1], expansion=4, k=32, epochs=4)
                try:
                    sae, stats = train_sae(Anp, cfg, device=a.device, log=lambda *x: None)
                    Gm, alive_mask = gpu_gene_scores(sae, Anp, si, len(uniq), a.device)   # G on GPU, not CPU np.add.at
                except RuntimeError as e:
                    print(f"  {key} FAILED: {e}", flush=True); gc.collect(); (torch.cuda.empty_cache() if a.device == 'cuda' else None); continue
                alive = np.where(alive_mask)[0]
                topg = {int(f): [str(uniq[i]) for i in np.argsort(Gm[:, f])[::-1][:20]] for f in alive}  # raw top-20
                terms = annotate(topg, idx, size, bgn, bg)
                out["layer_sets"][key] = sorted(terms)
                del Gm; gc.collect()
                if a.device == "cuda":
                    torch.cuda.empty_cache()
                json.dump(out, open(a.dst, "w"))
                print(f"  {key}: {len(terms)} concepts · {len(alive)} alive [saved {len(out['layer_sets'])}]", flush=True)
            del A, Anp; gc.collect()

    # all-layer union per (model, seed) -> core (intersection across models) per seed
    seeds = sorted(set(int(k.split('#')[1]) for k in out["layer_sets"]))
    per_model = {}
    for m in a.models:
        for s in seeds:
            u = set()
            for k, v in out["layer_sets"].items():
                mm, ss, _ = k.split('#')
                if mm == m and int(ss) == s:
                    u.update(v)
            per_model[f"{m}#{s}"] = u
    cores = []
    for s in seeds:
        sets = [per_model[f"{m}#{s}"] for m in a.models if per_model.get(f"{m}#{s}")]
        if len(sets) == len(a.models):
            cores.append(len(set.intersection(*sets)))
    if cores:
        out["core_by_seed"] = cores
        out["core_mean"] = round(float(np.mean(cores)), 1)
        out["core_sd"] = round(float(np.std(cores)), 1)
        out["nconc_by_model_seed"] = {k: len(v) for k, v in per_model.items()}
        print(f"\nALL-LAYER universal core across seeds: {cores} -> {out['core_mean']} ± {out['core_sd']}", flush=True)
    json.dump(out, open(a.dst, "w"))
    print(f"==> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
