#!/usr/bin/env python
"""Second annotation method — a real rank-based (GSEA-family) test over the FULL gene ranking, run on the cluster
where the SAE weights + activations live. Fisher (the atlas default) asks 'do a feature's top-5 genes overlap a
set more than chance' (hard cutoff, counts). This asks the position-aware question GSEA asks — 'are a set's genes
shifted toward the TOP of the feature's whole gene ranking' — but using ALL ~19k genes (not just the stored 20),
so it is properly powered. Per feature we rebuild the full per-gene score vector G[:,f] exactly as build_catalog
does (mean feature activation per gene), then run a competitive Wilcoxon rank-sum of in-set vs the rest across
every gene; BH<0.05 across all (feature,set). Reports per model: annotation rate, distinct concepts, median
top-shift (AUC-0.5), and Jaccard of concepts vs the Fisher(top-5) call — does the method change the picture?

    python gsea_prerank.py --out_root out_alllayers --genesets genesets --bg genesets/background.json \
        --models AIDO C2S Geneformer MaxToki UCE scGPT tGPT scFoundation GeneCompass Tahoe \
        --fisher_matrix matrix_ts3_string.json --dst gsea_annot.json
"""
from __future__ import annotations
import argparse, glob, json, os, gc, sys, fcntl
import numpy as np
import torch
from scipy.stats import rankdata

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.sae import TopKSAE, SAECfg, feature_activations

MIN, MAX, ALPHA = 5, 500, 0.05


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def load_sets(gdir, bg):
    """Gene sets + edge sets, restricted to [MIN,MAX] and to genes in bg (the tested universe)."""
    sets = {}
    for nm in ("GO_BP", "KEGG", "Reactome"):
        p = os.path.join(gdir, f"{nm}_gene_sets.json")
        if os.path.exists(p):
            for t, g in json.load(open(p)).items():
                s = set(x.upper() for x in g) & bg
                if MIN <= len(s) <= MAX:
                    sets[f"{nm}:{t}"] = s
    for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
        p = os.path.join(gdir, fn)
        if os.path.exists(p):
            for t, g in json.load(open(p)).items():
                s = set(x.upper() for x in g) & bg
                if MIN <= len(s) <= MAX:
                    sets[f"{nm}:{t}"] = s
    return sets


def mid_layer_paths(out_dir):
    cats = sorted(glob.glob(os.path.join(out_dir, "feature_catalog_L*.json")),
                  key=lambda p: json.load(open(p))["layer"])
    if not cats:
        return None
    cp = cats[len(cats) // 2]; L = json.load(open(cp))["layer"]
    return cp, L, os.path.join(out_dir, f"sae_L{L:02d}.pt"), os.path.join(out_dir, f"layer_{L:02d}_activations.npy")


@torch.no_grad()
def per_gene_scores(out_dir, sae_pt, act_npy, device, batch=8192):
    """Rebuild G[genes x features] = mean feature activation per gene (as build_catalog does), streaming the
    SAE encode batch-by-batch so the full [positions x d_sae] activation matrix is never materialised."""
    syms = np.load(os.path.join(out_dir, "gene_symbols.npy"), allow_pickle=True).astype(str)
    syms = np.char.upper(syms)
    uniq = np.array(sorted(set(syms.tolist()))); gi = {g: i for i, g in enumerate(uniq)}
    sym_idx = np.array([gi[g] for g in syms])
    ck = torch.load(sae_pt, map_location=device); cfg = SAECfg(**ck["cfg"])
    sae = TopKSAE(cfg.d_model, cfg.d_sae, cfg.k).to(device); sae.load_state_dict(ck["state_dict"]); sae.eval()
    A = np.load(act_npy, mmap_mode="r")
    G = np.zeros((len(uniq), cfg.d_sae), np.float64); c = np.zeros(len(uniq))
    nz = np.zeros(cfg.d_sae)                       # nonzero-count per feature -> alive/freq
    for i in range(0, len(A), batch):
        xb = torch.as_tensor(np.ascontiguousarray(A[i:i + batch]), dtype=torch.float32, device=device)
        Fb = sae.encode(xb).cpu().numpy()
        si = sym_idx[i:i + batch]
        np.add.at(G, si, Fb); np.add.at(c, si, 1); nz += (Fb > 0).sum(0)
    G /= np.maximum(c[:, None], 1); G = G.astype(np.float32)
    alive = np.where(nz > 0)[0]
    return uniq, G, alive


def gsea_model(out_dir, sets, device, log=print):
    mp = mid_layer_paths(out_dir)
    if not mp:
        return None
    cp, L, sae_pt, act_npy = mp
    if not (os.path.exists(sae_pt) and os.path.exists(act_npy)):
        log(f"    !! missing {sae_pt} or {act_npy}"); return None
    uniq, G, alive = per_gene_scores(out_dir, sae_pt, act_npy, device)
    ngene = len(uniq); pos = {g: i for i, g in enumerate(uniq)}
    # set -> integer gene indices present in this model's universe (+ inverted gene->sets index)
    set_idx = {k: np.array([pos[g] for g in s if g in pos]) for k, s in sets.items()}
    set_idx = {k: v for k, v in set_idx.items() if len(v) >= MIN}
    from collections import defaultdict as _dd
    g2sets = _dd(list)
    for k, idx in set_idx.items():
        for gi_ in idx:
            g2sets[gi_].append(k)
    TOPK = 100                                       # only test sets present in a feature's top region (else z~0)
    from math import erfc, sqrt
    recs_feat = []; recs_term = []; recs_p = []; recs_shift = []
    for f in alive:
        col = G[:, f]
        if np.count_nonzero(col) < 3:
            continue
        topk = np.argpartition(col, -TOPK)[-TOPK:] if ngene > TOPK else np.arange(ngene)
        cand = {}
        for gi_ in topk:
            for k in g2sets.get(gi_, ()):
                cand[k] = cand.get(k, 0) + 1
        cand = [k for k, h in cand.items() if h >= 2]     # >=2 of the feature's top genes fall in the set
        if not cand:
            continue
        ranks = rankdata(col)                       # 1..ngene, higher score -> higher rank (nearer top)
        for k in cand:
            idx = set_idx[k]; nin = len(idx); nout = ngene - nin
            R = ranks[idx].sum()
            mu = nin * (ngene + 1) / 2.0
            var = nin * nout * (ngene + 1) / 12.0
            if var <= 0:
                continue
            z = (R - mu) / sqrt(var)                # positive -> set genes ranked high (enriched at top)
            if z <= 0:
                continue
            recs_feat.append(int(f)); recs_term.append(k); recs_p.append(0.5 * erfc(z / sqrt(2)))
            recs_shift.append(float(R / nin / ngene - 0.5))   # AUC-0.5, mean-rank top-shift
    if not recs_p:
        return {"annot_rate": 0.0, "n_concepts": 0, "med_shift": 0.0, "layer": int(L), "n_alive": int(len(alive))}
    q = bh(recs_p); ann = {}; terms = set(); shifts = []
    for i in range(len(recs_p)):
        if q[i] <= ALPHA:
            ann.setdefault(recs_feat[i], []).append(recs_term[i]); terms.add(recs_term[i]); shifts.append(recs_shift[i])
    return {"annot_rate": round(100 * len(ann) / max(len(alive), 1), 1), "n_concepts": len(terms),
            "med_shift": round(float(np.median(shifts)), 3) if shifts else 0.0,
            "layer": int(L), "n_alive": int(len(alive)), "_terms": sorted(terms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_alllayers")
    ap.add_argument("--genesets", default="genesets")
    ap.add_argument("--bg", default="genesets/background.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fisher_matrix", default=None, help="matrix_ts3_string.json for concept-Jaccard vs Fisher")
    ap.add_argument("--dst", default="gsea_annot.json")
    ap.add_argument("--force", action="store_true", help="recompute even models already cached in --dst")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    # single-instance lock: a duplicate relaunch (watcher racing on flaky ssh) exits instead of clobbering
    # the shared checkpoint. The lock frees automatically when an OOM-killed process's fd closes.
    lockf = open(a.dst + ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another gsea_prerank holds the lock — exiting (no duplicate run)", flush=True); sys.exit(0)

    bg = set(x.upper() for x in json.load(open(a.bg)))
    sets = load_sets(a.genesets, bg)
    print(f"loaded {len(sets)} gene/edge sets; background {len(bg)} genes; device {a.device}", flush=True)
    fisher = json.load(open(a.fisher_matrix)) if a.fisher_matrix and os.path.exists(a.fisher_matrix) else {}

    # resume: keep any models already computed in dst (survives an OOM/node kill; rerun to finish the rest)
    out = json.load(open(a.dst)) if os.path.exists(a.dst) else {"models": {}}
    out.setdefault("models", {})
    for m in a.models:
        if m in out["models"] and not a.force:
            print(f"== {m} == (cached, skip)", flush=True); continue
        d = os.path.join(a.out_root, m)
        if not os.path.isdir(d):
            d = os.path.join(a.out_root, f"out_{m}", m)
        if not os.path.isdir(d):
            print(f"  (skip {m}: no dir under {a.out_root})", flush=True); continue
        print(f"== {m} ==", flush=True)
        try:
            r = gsea_model(d, sets, a.device)
        except Exception as e:
            print(f"   !! {m} FAILED: {type(e).__name__}: {e}", flush=True); continue
        if r is None:
            continue
        gt = set(r.pop("_terms", []))
        ft = set(fisher.get(m, {}).get("term_count", {})) if fisher else set()
        r["concept_jaccard_vs_fisher"] = round(len(gt & ft) / max(len(gt | ft), 1), 3) if (gt or ft) else 0.0
        gc.collect()
        # merge-on-write: reload disk first so a concurrent/earlier writer's models are never clobbered
        if os.path.exists(a.dst):
            try:
                disk = json.load(open(a.dst)).get("models", {})
                out["models"].update({k: v for k, v in disk.items() if k not in out["models"]})
            except Exception:
                pass
        out["models"][m] = r
        json.dump(out, open(a.dst, "w"))          # checkpoint after every model
        print(f"   GSEA {r['annot_rate']}% · {r['n_concepts']} concepts · top-shift {r['med_shift']} · "
              f"Jaccard vs Fisher {r['concept_jaccard_vs_fisher']}  [saved {len(out['models'])}/{len(a.models)}]", flush=True)
    print(f"==> {a.dst} ({len(out['models'])} models)", flush=True)


if __name__ == "__main__":
    main()
