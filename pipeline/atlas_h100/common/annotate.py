"""Unified annotator — the JOIN KEY of the comparative atlas. Runs the SAME test on
every model's feature_catalog (ours, Igor's ingested, new H100 models), so features
are comparable in a shared concept vocabulary regardless of model.

Method (matches bio-sae src/04_annotate_features.py): for each feature's top-20 genes,
Fisher's exact ('greater') vs GO_BP/KEGG/Reactome gene sets and edge-enrichment vs
STRING(>=700)/TRRUST; Benjamini-Hochberg FDR < 0.05; term size in [5, 500].
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from scipy.stats import fisher_exact

MIN_TERM, MAX_TERM, BH_ALPHA = 5, 500, 0.05


def bh_fdr(pvals: np.ndarray, alpha: float = BH_ALPHA):
    """Benjamini-Hochberg. Returns (reject_mask, qvalues) aligned to input order."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranks = np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    q = np.empty(n); q[order] = np.clip(q_sorted, 0, 1)
    return q <= alpha, q


def load_genesets(geneset_dir: str):
    """Expect {GO_BP,KEGG,Reactome}_gene_sets.json (term -> [genes]) built by data/build_genesets.py,
    plus trrust_edges.json (TF -> [targets]) and optionally string_edges.json."""
    gs = {}
    for name in ("GO_BP", "KEGG", "Reactome"):
        p = os.path.join(geneset_dir, f"{name}_gene_sets.json")
        if os.path.exists(p):
            raw = json.load(open(p))
            gs[name] = {t: set(g) for t, g in raw.items() if MIN_TERM <= len(g) <= MAX_TERM}
    trr = os.path.join(geneset_dir, "trrust_edges.json")
    tr = {t: set(v) for t, v in json.load(open(trr)).items()} if os.path.exists(trr) else {}
    return gs, tr


def annotate_catalog(catalog_path: str, gs, tr, background: set, out_path: str, top: int = 5, log=print):
    cat = json.load(open(catalog_path))
    feats = cat["features"]
    bg = len(background)
    recs = []  # (feat, source, term, OR, p, overlap)
    for fid, f in feats.items():
        # top-5 by default: empirically sharper/more specific concepts (higher odds-ratio, no saturation)
        genes = set(g.upper() for g in f["top_genes"][:top]) & background
        if len(genes) < 3:
            continue
        for src, terms in gs.items():
            for term, tg in terms.items():
                ov = genes & tg
                if len(ov) < 2:
                    continue
                a = len(ov); b = len(genes) - a; c = len(tg) - a; d = bg - a - b - c
                orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                recs.append((fid, src, term, float(orr), float(p), sorted(ov)))
        # TRRUST regulon membership (feature's genes enriched among a TF's targets)
        for tf, tgts in tr.items():
            ov = genes & tgts
            if len(ov) < 2:
                continue
            a = len(ov); b = len(genes) - a; c = len(tgts) - a; d = bg - a - b - c
            orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            recs.append((fid, "TRRUST", tf, float(orr), float(p), sorted(ov)))

    if recs:
        keep, q = bh_fdr(np.array([r[4] for r in recs]))
    else:
        keep, q = np.array([]), np.array([])

    ann = {}
    for i, (fid, src, term, orr, p, ov) in enumerate(recs):
        if keep[i]:
            ann.setdefault(fid, []).append(
                {"source": src, "term": term, "odds_ratio": orr, "q": float(q[i]),
                 "overlap": ov})
    for fid in ann:
        ann[fid].sort(key=lambda t: t["q"])
    out = {"model": cat["model"], "layer": cat["layer"], "n_features": len(feats),
           "n_annotated": len(ann), "annotations": ann}
    json.dump(out, open(out_path, "w"))
    log(f"    {cat['model']} L{cat['layer']}: {len(ann)}/{len(feats)} features annotated "
        f"({100*len(ann)/max(len(feats),1):.0f}%)")
    return out


def annotate_all(catalog_glob: str, geneset_dir: str, out_dir: str, background_json: str = None, top: int = 5, log=print):
    os.makedirs(out_dir, exist_ok=True)
    gs, tr = load_genesets(geneset_dir)
    if background_json and os.path.exists(background_json):
        background = set(json.load(open(background_json)))
    else:
        background = set().union(*[s for terms in gs.values() for s in terms.values()]) | \
                     set().union(*tr.values()) if tr else set()
    log(f"  background {len(background)} genes; GO_BP/KEGG/Reactome + TRRUST loaded")
    for cp in sorted(glob.glob(catalog_glob)):
        model = os.path.basename(os.path.dirname(cp)) or "model"
        L = json.load(open(cp))["layer"]
        op = os.path.join(out_dir, f"{model}_L{L:02d}_annotations.json")
        annotate_catalog(cp, gs, tr, background, op, top=top, log=log)
