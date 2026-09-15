#!/usr/bin/env python
"""Comparative atlas — the payoff. Annotate every model's catalog uniformly (same
top-N genes, same vocabulary), build the concept x model matrix, and run the
cross-model analyses:
  (1) universality spectrum of concepts   (2) per-model blind spots
  (3) regulatory-logic proxy (TRRUST regulon coverage per model)
  (5) feature-orthologs (annotation-Jaccard matches across models)

    conda activate scprint
    python scripts/atlas_compare.py --topn 5
"""
import argparse, glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas_h100"))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogs", default=f"{BASE}/outputs/atlas/*/feature_catalog_L*.json")
    ap.add_argument("--genesets", default=f"{BASE}/outputs/atlas/genesets")
    ap.add_argument("--topn", type=int, default=5, help="top genes/feature (fair common depth = the prior atlases' 5)")
    ap.add_argument("--out", default=f"{BASE}/outputs/atlas/comparative")
    args = ap.parse_args()
    import numpy as np
    from common.annotate import load_genesets, bh_fdr
    from scipy.stats import fisher_exact

    os.makedirs(args.out, exist_ok=True)
    gs, tr = load_genesets(args.genesets)
    background = set(json.load(open(f"{args.genesets}/background.json")))
    bg = len(background)

    def annotate(catalog, topn):
        """-> {feature_id: set(term)} and per-term best OR."""
        ann = {}
        recs = []
        for fid, f in catalog["features"].items():
            genes = set(g.upper() for g in f["top_genes"][:topn]) & background
            if len(genes) < 2:
                continue
            for src, terms in gs.items():
                for term, tgset in terms.items():
                    ov = genes & tgset
                    if len(ov) < 2:
                        continue
                    a = len(ov); b = len(genes)-a; c = len(tgset)-a; d = bg-a-b-c
                    orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                    recs.append((fid, f"{src}:{term}", p, orr))
            for tf, tgt in tr.items():
                ov = genes & tgt
                if len(ov) < 2:
                    continue
                a = len(ov); b = len(genes)-a; c = len(tgt)-a; d = bg-a-b-c
                orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                recs.append((fid, f"TRRUST:{tf}", p, orr))
        if not recs:
            return {}, {}
        keep, q = bh_fdr(np.array([r[2] for r in recs]))
        feat2terms = defaultdict(set); term_best = {}
        for i, (fid, term, p, orr) in enumerate(recs):
            if keep[i]:
                feat2terms[fid].add(term)
                term_best[term] = max(term_best.get(term, 0), orr)
        return dict(feat2terms), term_best

    # ---- annotate every model ------------------------------------------
    models = {}
    for cp in sorted(glob.glob(args.catalogs)):
        if "/genesets/" in cp or "/comparative/" in cp:
            continue
        cat = json.load(open(cp))
        name = cat["model"]
        f2t, tbest = annotate(cat, args.topn)
        # concept -> # features capturing it
        term_count = defaultdict(int)
        for terms in f2t.values():
            for t in terms:
                term_count[t] += 1
        models[name] = {"n_feat": len(cat["features"]), "n_annot": len(f2t),
                        "feat2terms": f2t, "term_count": dict(term_count), "term_best": tbest}
        print(f"  {name:<12} {len(cat['features']):>5} feats | {len(f2t):>5} annotated "
              f"({100*len(f2t)/max(len(cat['features']),1):.0f}%) | {len(term_count):>4} distinct concepts")

    names = list(models)
    allterms = set().union(*[set(m["term_count"]) for m in models.values()])

    # ---- (1) universality spectrum -------------------------------------
    univ = {t: sum(t in models[n]["term_count"] for n in names) for t in allterms}
    from collections import Counter
    spec = Counter(univ.values())
    print(f"\n[1] universality across {len(names)} models ({', '.join(names)}):")
    for k in sorted(spec, reverse=True):
        print(f"    in {k}/{len(names)} models: {spec[k]} concepts")
    core = sorted([t for t, u in univ.items() if u == len(names)])
    print(f"    CORE concepts (all models): {len(core)} — e.g. {core[:8]}")

    # ---- (2) per-model blind spots (concepts >=all-but-one others have, this one lacks) --
    print(f"\n[2] blind spots (concept in all OTHER models but missing here):")
    for n in names:
        miss = [t for t in allterms if t not in models[n]["term_count"]
                and sum(t in models[o]["term_count"] for o in names if o != n) == len(names)-1]
        print(f"    {n:<12} misses {len(miss)}: e.g. {sorted(miss)[:6]}")

    # ---- (3) regulatory-logic proxy: TRRUST regulon coverage per model --
    print(f"\n[3] TRRUST regulon coverage (proxy for the prior TF-logic test; its headline 6.2% TF-specific):")
    for n in names:
        tfs = {t.split(":")[1] for t in models[n]["term_count"] if t.startswith("TRRUST:")}
        print(f"    {n:<12} detects {len(tfs)} TF regulons ({100*len(tfs)/len(tr):.0f}% of {len(tr)})")

    # ---- (5) feature-orthologs across models (annotation-Jaccard) -------
    print(f"\n[5] feature-orthologs (best cross-model annotation-Jaccard match, sampled):")
    if len(names) >= 2:
        a, b = names[0], names[1]
        A = models[a]["feat2terms"]; B = models[b]["feat2terms"]
        pairs = 0; strong = 0
        import random; random.seed(0)
        sample = random.sample(list(A), min(200, len(A)))
        for fa in sample:
            ta = A[fa]
            best = max((len(ta & tb) / len(ta | tb) for tb in B.values()), default=0)
            pairs += 1; strong += best >= 0.5
        print(f"    {a} vs {b}: {strong}/{pairs} sampled features have a Jaccard>=0.5 twin")

    json.dump({n: {"n_feat": m["n_feat"], "n_annot": m["n_annot"],
                   "term_count": m["term_count"]} for n, m in models.items()},
              open(f"{args.out}/matrix.json", "w"))
    print(f"\n==> saved concept x model matrix to {args.out}/matrix.json")


if __name__ == "__main__":
    main()
