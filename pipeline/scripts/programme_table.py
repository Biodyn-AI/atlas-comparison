#!/usr/bin/env python
"""Programme table with baselines: which biological programmes each model's SAE features cover,
compared with what chance, an untrained SAE and co-expression give.

Unit: the features of a model's middle layer (the layer of the headline analyses; bb_common.mid_path).
Each feature is matched to gene sets by the strict annotation rule of the paper (top 10 genes, >= 3 in a
GO_BP / KEGG / Reactome set of 5..200 genes, hypergeometric, BH over all candidate feature-set pairs of
the dictionary, q <= 0.05). A labelled feature gets the programme of its best (lowest p) gene set;
programmes come from keyword rules on the gene-set name (RULES below, first match wins); a gene set
that matches no rule is "other pathway" (named as such, never folded into a biological programme).

Per model and programme, the share of all middle-layer features in that programme, for
  trained      the model's SAE (the atlas catalogue);
  untrained    an SAE that was never trained, on the same activations (paper Section 2, untrained
               controls; catalogues in _paper_rerun/untrained/random_sae);
  coexpr       a co-expression dictionary with the model's gene choice: one feature per live feature,
               the feature's top gene plus its 9 most correlated genes in the same 6,000 cells
               (seed-matched variant of coexpr_core_baseline.py);
  random       random 10-gene lists, one per feature, through the same rule (NRAND draws; mean).
Tests: trained vs untrained and trained vs coexpr, two-sided Fisher exact test on features in the
programme / not, BH over all model x programme cells of each comparison.

    python programme_table.py   (ATLAS_BASE = _fixed_base) -> programme_table.json (this folder)
"""
from __future__ import annotations
import glob, json, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backbone_composition"))
from bb_common import Annotator, BG, BGN, BG_ARR, ORDER, mid_path, bh, ALPHA, TOP, AMIN  # noqa: E402

BASE = os.environ.get("ATLAS_BASE")
RS_ROOT = "/Users/annaantipova/Desktop/biomech/_paper_rerun/untrained/random_sae"
CORPUS = f"{BASE}/external/perturb/tabula_3tissue_6k.h5ad"
NRAND = int(os.environ.get("NRAND", 20))

PROGRAMMES = ["translation", "immune", "mito/OXPHOS", "cell cycle/DNA", "RNA processing",
              "metabolism", "membrane/traffic", "signalling", "development/tissue", "other pathway"]
RULES = [
    ("translation", r"translat|ribosom|peptide chain elong|aminoacyl|rrna process|eukaryotic translation|"
                    r"cotranslational|protein targeting to (er|membrane)|srp-dependent|nonsense-mediated|"
                    r"peptide biosynthetic|ribonucleoprotein complex"),
    ("cell cycle/DNA", r"cell cycle|mitotic|dna replicat|dna repair|chromosom|spindle|kinetochore|s phase|"
                       r"g2/m|cell division|dna metabolic|telomere|nucleosome|chromatin"),
    ("RNA processing", r"splic|spliceosome|mrna|rna processing|snrna|capping|polyadenyl|rna polymerase|"
                       r"rna metabolic|rna stability|rna catabolic"),
    ("mito/OXPHOS", r"respiratory electron|atp synth|oxidative phosph|mitochond|tca|citric acid|"
                    r"electron transport|aerobic respiration|cellular respiration"),
    ("immune", r"immune|neutrophil|interferon|cytokine|\bmhc\b|antigen|complement|inflamm|lymphocyte|"
               r"t cell|b cell|interleukin|degranulation|leukocyte|nf-kappa|defen[cs]e response|"
               r"bacteri|viral|virus|toll-like|chemokine|myeloid|natural killer|immunoglobulin"),
    ("membrane/traffic", r"transport|endocytos|golgi|vesicle|slc-mediated|transmembrane|secretion|traffick|"
                         r"exocytos|endosom|lysosom|clathrin|protein localization"),
    ("signalling", r"signal|mapk|kinase|receptor|\bwnt\b|notch|gpcr|phosphoryl|erk|akt|calcium|"
                   r"rho gtpase|apoptot|cell death"),
    ("metabolism", r"metabol|biosynth|catabol|glycol|lipid|fatty acid|cholesterol|sterol|amino acid|"
                   r"nucleotide|steroid|glucose|heme|oxidored|detoxification"),
    ("development/tissue", r"development|differentiation|morphogenesis|muscle|contraction|epitheli|"
                           r"angiogenesis|vasculature|extracellular matrix|collagen|cell adhesion|"
                           r"keratin|cornification|surfactant|kidney|lung"),
]


def programme_of(term):
    name = term.split(":", 1)[-1].lower()
    for p, rx in RULES:
        if re.search(rx, name):
            return p
    return "other pathway"


class BestAnnotator(Annotator):
    """the strict rule of the paper, returning each labelled feature's best gene set"""

    def best(self, feat_lists):
        A, TG, LG, TERM, FI = [], [], [], [], []
        for fi, genes in enumerate(feat_lists):
            g = set(str(x).upper() for x in genes[:self.top]) & BG
            if len(g) < 3:
                continue
            cand = Counter()
            for gene in g:
                for k in self.idx.get(gene, ()):
                    cand[k] += 1
            for k, a in cand.items():
                if a < self.amin:
                    continue
                A.append(a); TG.append(self.size[k]); LG.append(len(g)); TERM.append(k); FI.append(fi)
        best = {}
        if not A:
            return best
        p = hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG))
        q = bh(p)
        for i in np.where(q <= ALPHA)[0]:
            fi = FI[i]
            if fi not in best or p[i] < best[fi][1]:
                best[fi] = (TERM[i], p[i])
        return {fi: t for fi, (t, _) in best.items()}


def counts(best):
    c = Counter(programme_of(t) for t in best.values())
    return {p: c.get(p, 0) for p in PROGRAMMES}


def load(path):
    d = json.load(open(path))
    feats = list(d["features"].values())
    return feats, d.get("layer")


def rs_mid(m):
    ps = glob.glob(f"{RS_ROOT}/{m}/feature_catalog_L*.json")
    ps = sorted(ps, key=lambda p: int(re.search(r"_L(\d+)\.json$", p).group(1)))
    return ps[len(ps) // 2]


def coexpr_builder():
    import anndata as ad
    a = ad.read_h5ad(CORPUS)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    X.data = np.log1p(X.data)
    det = np.asarray((X > 0).sum(0)).ravel() >= 10
    names = np.array([str(s).upper() for s in a.var["feature_name"].values])[det]
    Xd = X[:, det].toarray().astype(np.float32)
    Z = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-8); del Xd
    n = Z.shape[0]
    pos = {}
    for i, g in enumerate(names):
        pos.setdefault(g, i)

    def build(seeds):
        seeds = [s for s in seeds if s in pos]
        cols = np.array([pos[s] for s in seeds])
        out = []
        for c0 in range(0, len(cols), 1000):
            cc = cols[c0:c0 + 1000]
            R = (Z.T @ Z[:, cc]) / n
            R[cc, np.arange(len(cc))] = -np.inf
            top = np.argpartition(-R, TOP - 1, axis=0)[:TOP - 1]
            for j, c in enumerate(cc):
                nb = top[:, j][np.argsort(-R[top[:, j], j])]
                out.append([names[c]] + list(names[nb]))
        return out
    return build


def main():
    t0 = time.time()
    ann = BestAnnotator(mx=200)
    build = coexpr_builder()
    print(f"corpus loaded [{time.time() - t0:.0f}s]", flush=True)
    res = {}
    for mi, m in enumerate(ORDER):
        feats, layer = load(mid_path(m))
        n = len(feats)
        tr = counts(ann.best([f["top_genes"] for f in feats]))
        rfeats, rlayer = load(rs_mid(m))
        un = counts(ann.best([f["top_genes"] for f in rfeats]))
        live = [f for f in feats if f.get("max_activation", 0) > 0 and f.get("top_genes")]
        cx_feats = build([str(f["top_genes"][0]).upper() for f in live])
        cx = counts(ann.best(cx_feats))
        rng = np.random.default_rng(1000 + mi)
        rnd = Counter()
        for _ in range(NRAND):
            rl = [BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(n)]
            for p, v in counts(ann.best(rl)).items():
                rnd[p] += v
        res[m] = {"layer": layer, "untrained_layer": rlayer, "n_features": n, "n_untrained": len(rfeats),
                  "n_coexpr": len(cx_feats), "trained": tr, "untrained": un, "coexpr": cx,
                  "random_mean": {p: rnd[p] / NRAND for p in PROGRAMMES}}
        print(m, n, "trained", sum(tr.values()), "untrained", sum(un.values()), "coexpr", sum(cx.values()),
              f"[{time.time() - t0:.0f}s]", flush=True)

    # tests: trained vs untrained, trained vs coexpr (shares of all features / all dictionary entries)
    for key, nkey in (("untrained", "n_untrained"), ("coexpr", "n_coexpr")):
        cells, ps = [], []
        for m in ORDER:
            r = res[m]
            for p in PROGRAMMES:
                a, b = r["trained"][p], r[key][p]
                _, pv = fisher_exact([[a, r["n_features"] - a], [b, r[nkey] - b]])
                cells.append((m, p)); ps.append(pv)
        q = bh(ps)
        for (m, p), pv, qv in zip(cells, ps, q):
            r = res[m]
            st = r["trained"][p] / r["n_features"]; sb = r[key][p] / r[nkey]
            r.setdefault("tests", {}).setdefault(p, {})[key] = {
                "share_trained": round(st, 5), f"share_{key}": round(sb, 5),
                "ratio": round(st / sb, 3) if sb > 0 else None, "p": float(pv), "q": float(qv)}
    out = {"unit": "features of the middle layer; programme of each feature's best strict-rule gene set",
           "programmes": PROGRAMMES, "rules": [(p, rx) for p, rx in RULES], "n_random_draws": NRAND,
           "models": res}
    json.dump(out, open(os.path.join(HERE, "programme_table.json"), "w"), indent=1)
    print(f"==> programme_table.json [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
