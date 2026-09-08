#!/usr/bin/env python
"""Re-assemble atlas_full_notf.json for ALL models present in the component JSONs (7 or 8 with Tahoe).
Derives coverage / universality / similarity / specific / totals / lineage / firing from the per-model
STRING matrix, and pulls depth / tissue / cka / cka_layers / nonlinearity / module_themes / svd from their
own files if present. Robust to missing pieces (keeps a block absent rather than crashing). Run after
reannotate_string.py (8-model matrix) + the all-layers downstream (with Tahoe added).
    python scripts/atlas_assemble.py
"""
from __future__ import annotations
import json, os
import numpy as np
from collections import defaultdict, Counter

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
PARAMS = {"AIDO": "10M", "UCE": "650M", "tGPT": "~50M", "Geneformer": "316M", "scGPT": "~50M",
          "C2S": "2B", "MaxToki": "217M", "Tahoe": "3B", "scFoundation": "100M", "GeneCompass": "104M"}
PARAMS_M = {"AIDO": 10, "tGPT": 50, "scGPT": 50, "MaxToki": 217, "Geneformer": 316, "UCE": 650, "C2S": 2000, "Tahoe": 3000, "scFoundation": 100, "GeneCompass": 104}
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def load(name):
    p = os.path.join(C, name)
    return json.load(open(p)) if os.path.exists(p) else None


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "").replace("STRING:", "PPI: ")
            .replace("TRRUST:", "TF ").split(" (GO:")[0].split(" R-HSA")[0].split(" (hsa")[0]
            .replace("Homo sapiens ", "").strip())


import re
TS_OUT = "/Users/annaantipova/Desktop/biomech/outputs/atlas/ts3_out"
# inductive-axis taxonomy: tokenization (must match the atlas TCOL keys) / objective / prior
TAX = {
    "AIDO": ("expression", "MLM", "none"), "scGPT": ("expression", "MLM", "none"),
    "tGPT": ("rank", "autoregressive", "none"), "scFoundation": ("expression", "MAE", "read-depth"),
    "GeneCompass": ("rank", "MLM", "knowledge/GRN"), "MaxToki": ("expression", "autoregressive", "none"),
    "Geneformer": ("rank", "MLM", "none"), "UCE": ("protein-token", "masked", "ESM"),
    "C2S": ("cell-sentence", "autoregressive", "text"), "Tahoe": ("expression", "MLM", "none"),
}
_CATRULES = [("translation", "translat|ribosom|peptide chain|aminoacyl|rrna|elongation"),
             ("DNA/cell-cycle", "cell cycle|mitotic|dna replicat|dna repair|chromosom|spindle|dna metabolic|telomere|nucleosome"),
             ("RNA-processing", "splic|mrna|rna processing|snrna|transcription|polymerase|nonsense-mediated"),
             ("mito/OXPHOS", "respiratory electron|atp synth|oxidative phosph|mitochond|electron transport|tca"),
             ("immune", "immune|neutrophil|interferon|cytokine|mhc|antigen|complement|inflamm|lymphocyte|interleukin|degranulation|leukocyte"),
             ("membrane/transport", "transport|endocytos|golgi|vesicle|slc|transmembrane|secretion|traffick|lysosom|endosom"),
             ("signaling", "signal|mapk|kinase|receptor|wnt|notch|gpcr|phosphoryl|pathway|rho gtpase"),
             ("metabolism", "metabol|biosynth|catabol|glycol|lipid|fatty acid|cholesterol|amino acid|nucleotide|heme")]


def catof(term):
    s = term.replace("STRING:", "PPI ").replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "").lower()
    for nm, pat in _CATRULES:
        if re.search(pat, s):
            return nm
    return "other"


def main():
    mat = load("matrix_ts3_string.json")
    assert mat, "need matrix_ts3_string.json (run reannotate_string.py with 8 models first)"
    models = [m for m in ORDER if m in mat]
    n = len(models)
    print(f"assembling {n} models: {models}", flush=True)
    d = {"models": models, "n_models": n}

    # per-model concept sets + coverage
    concepts = {m: set(mat[m]["term_count"]) for m in models}
    SRC = ["GO_BP", "Reactome", "KEGG", "STRING"]
    cov = {}
    for m in models:
        mm = mat[m]; tc = mm["term_count"]
        by = {s: sum(1 for t in tc if t.startswith(s + ":")) for s in SRC}
        cov[m] = {"axis": mm.get("axis", ""), "layer": mm["layer"], "n_feat": mm["n_feat"], "n_annot": mm["n_annot"],
                  "annot_rate": round(100 * mm["n_annot"] / max(mm["n_feat"], 1), 1),
                  "n_concepts": len(tc), "by_source": by}
    d["coverage"] = cov

    # universality spectrum
    concept_models = defaultdict(set)
    for m in models:
        for t in concepts[m]:
            concept_models[t].add(m)
    uni = {str(k): 0 for k in range(1, n + 1)}
    for t, ms in concept_models.items():
        uni[str(len(ms))] += 1
    d["universality"] = uni
    core = sorted(t for t, ms in concept_models.items() if len(ms) == n)
    d["universal_core_terms"] = [clean(t) if False else t for t in core[:120]]

    # specific (exclusive concepts per model)
    spec = {}
    for m in models:
        excl = [t for t in concepts[m] if len(concept_models[t]) == 1]
        cnt = mat[m]["term_count"]
        top = sorted(({"term": t, "count": cnt[t]} for t in excl), key=lambda x: -x["count"])[:12]
        spec[m] = {"n_unique": len(excl), "top": top}
    d["specific"] = spec

    # similarity (Jaccard)
    J = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                J[i][j] = 1.0
            else:
                inter = len(concepts[a] & concepts[b]); uni2 = len(concepts[a] | concepts[b]) or 1
                J[i][j] = round(inter / uni2, 3)
    d["similarity"] = {"models": models, "J": J}

    allc = set().union(*concepts.values())
    d["totals"] = {"total_features": sum(mat[m]["n_feat"] for m in models),
                   "total_annotated": sum(mat[m]["n_annot"] for m in models),
                   "total_concepts": len(allc), "universal_core": uni[str(n)]}
    d["params"] = {m: PARAMS.get(m, "?") for m in models}

    # firing (annotated vs unannotated firing rate at mid layer)
    freq = load("freq_ts3.json")
    if freq:
        firing = []
        for m in models:
            if m not in freq:
                continue
            lays = freq[m]["layers"]; midabs = mat[m]["layer"]
            ri = lays.index(midabs) if midabs in lays else len(lays) // 2
            f = np.array(freq[m]["freq"][str(ri)]); ann = set(int(k) for k in mat[m]["feat_terms"])
            ids = np.arange(len(f)); isann = np.array([i in ann for i in ids]); alive = f > 0
            a = f[isann & alive]; u = f[(~isann) & alive]
            if len(a) and len(u):
                firing.append({"model": m, "annot_pct": round(100 * isann[alive].mean(), 1),
                               "med_ann": round(float(np.median(a)), 5), "med_un": round(float(np.median(u)), 5),
                               "mean_ann": round(float(a.mean()), 5), "mean_un": round(float(u.mean()), 5),
                               "ratio": round(float(np.median(a) / max(np.median(u), 1e-12)), 2),
                               "n_un": int((~isann & alive).sum()), "n_ann": int((isann & alive).sum())})
        d["firing"] = firing

    # lineage (scaling + dendrogram + signature)
    al = load("alllayer_concepts.json")
    scaling = []
    for m in models:
        s = {"model": m, "params_M": PARAMS_M.get(m, 0), "annot_rate": cov[m]["annot_rate"],
             "n_concepts": cov[m]["n_concepts"], "n_feat": cov[m]["n_feat"]}
        if al and m in al:
            s["n_concepts_all"] = al[m]["all_layer_union"]
        scaling.append(s)
    lp = np.log10([max(PARAMS_M.get(m, 1), 1) for m in models])
    corr = {"params_annot": round(float(np.corrcoef(lp, [cov[m]["annot_rate"] for m in models])[0, 1]), 3),
            "params_concepts_mid": round(float(np.corrcoef(lp, [cov[m]["n_concepts"] for m in models])[0, 1]), 3)}
    if al:
        corr["params_concepts_all"] = round(float(np.corrcoef(lp, [al[m]["all_layer_union"] for m in models])[0, 1]), 3)
    # dendrogram from 1-J
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        Dd = 1 - np.array(J); np.fill_diagonal(Dd, 0)
        Z = linkage(squareform(Dd, checks=False), method="average")
        order = [int(i) for i in leaves_list(Z)]
        Zl = [[float(x) for x in row] for row in Z]
    except Exception as e:
        print("dendrogram skipped:", e); order = list(range(n)); Zl = []
    sig = {}
    for m in models:
        tops = spec[m]["top"]
        named = [x for x in tops if not x["term"].startswith("STRING:")]
        pick = (named or tops or [{"term": "?", "count": 0}])[0]
        sig[m] = {"term": clean(pick["term"]), "count": pick["count"], "n_unique": spec[m]["n_unique"]}
    d["lineage"] = {"models": models, "order": order, "Z": Zl, "scaling": scaling, "corr": corr, "signature": sig}

    # inductive axis -> findings (Analysis 13): taxonomy + per-model metrics + category universality
    import glob as _glob
    tl = load("tissue_alllayers.json"); ckal = load("cka_layers.json")
    axmet = {}
    for m in models:
        tdeep = None
        if tl and m in tl.get("models", {}) and tl["models"][m]:
            tdeep = round(tl["models"][m][-1].get("frac_specific", 0), 3)
        drift = None
        if ckal and m in ckal and ckal[m].get("cka"):
            drift = round(ckal[m]["cka"][0][-1], 3)
        axmet[m] = {"annot": cov[m]["annot_rate"], "tissue_deep": tdeep,
                    "cka_drift": drift, "concepts": cov[m]["n_concepts"]}
    progs = defaultdict(list)
    for t, ms in concept_models.items():
        progs[catof(t)].append(len(ms))
    programs = []
    for cat_, ks in progs.items():
        if cat_ in ("other", "PPI-hub"):
            continue
        arr = np.array(ks)
        programs.append({"prog": cat_, "n": len(ks), "mean_k": round(float(arr.mean()), 2),
                         "pct_univ": round(100 * float((arr == n).mean()), 1),
                         "pct_excl": round(100 * float((arr == 1).mean()), 1)})
    programs.sort(key=lambda x: -x["mean_k"])
    d["axes"] = {"models": models,
                 "tax": {m: {"tok": TAX[m][0], "obj": TAX[m][1], "prior": TAX[m][2]} for m in models if m in TAX},
                 "metrics": axmet, "cat_universality": {"n_models": n, "programs": programs}}

    # new biology (Analysis 14): genes that lead UNannotated features across many models
    THRESH = max(4, n // 2)
    gene_models = defaultdict(set); gene_co = defaultdict(Counter)
    for m in models:
        mid = mat[m]["layer"]; ann = set(str(k) for k in mat[m]["feat_terms"])
        cat = None
        for cp in _glob.glob(f"{TS_OUT}/{m}/feature_catalog_L*.json"):
            j = json.load(open(cp))
            if j.get("layer") == mid:
                cat = j; break
        if not cat:
            continue
        for fid, f in cat["features"].items():
            if str(fid) in ann:
                continue
            tg = [str(x).upper() for x in f.get("top_genes", [])[:6] if not str(x).upper().startswith("ENSG")]
            if not tg:
                continue
            gene_models[tg[0]].add(m)
            for g in tg[1:5]:
                gene_co[tg[0]][g] += 1
    cands = [{"gene": g, "n_models": len(ms), "co": [x for x, _ in gene_co[g].most_common(5)]}
             for g, ms in gene_models.items() if len(ms) >= THRESH]
    cands.sort(key=lambda c: (-c["n_models"], -len(c["co"])))
    d["novel_biology"] = {"thresh": THRESH, "n_models": n, "candidates": cands[:14]}

    # ---- extra findings (Analysis 16): feature economy + frontier ----
    extra = {}
    # (1) economy: resolution (features/concept) + polysemanticity (terms/feature) + geometry vs biology
    res = {m: round(float(np.mean(list(mat[m]["term_count"].values()))), 2) for m in models}
    poly = {}
    for m in models:
        vals = [len(v) for v in mat[m]["feat_terms"].values() if v]
        poly[m] = round(float(np.mean(vals)), 2) if vals else 0.0
    lp = np.log10([max(PARAMS_M.get(m, 1), 1) for m in models])
    geom_bio = None
    ckaf = load("cka_ts3.json")
    ck = ckaf.get("residual", {}) if ckaf else {}
    if ck:
        key = "2" if "2" in ck else sorted(ck)[len(ck) // 2]
        cm2 = ckaf["models"]; Jm = d["similarity"]["models"]; Jmat = d["similarity"]["J"]
        ci = {x: i for i, x in enumerate(cm2)}; ji = {x: i for i, x in enumerate(Jm)}
        aa, bb = [], []
        com = [m for m in cm2 if m in ji]
        for i in range(len(com)):
            for j in range(i + 1, len(com)):
                aa.append(ck[key][ci[com[i]]][ci[com[j]]]); bb.append(Jmat[ji[com[i]]][ji[com[j]]])
        if len(aa) > 2:
            geom_bio = round(float(np.corrcoef(aa, bb)[0, 1]), 3)
    extra["economy"] = {"models": models, "resolution": res, "polysemanticity": poly, "geom_bio_corr": geom_bio,
                        "corr_res_size": round(float(np.corrcoef(lp, [res[m] for m in models])[0, 1]), 3),
                        "corr_poly_size": round(float(np.corrcoef(lp, [poly[m] for m in models])[0, 1]), 3)}
    # (2) rare-concept frontier: named pathways encoded by only 1-2 of N models
    rare = [(t, len(ms), sorted(ms)) for t, ms in concept_models.items()
            if len(ms) <= 2 and not t.startswith("STRING:")]
    rare.sort(key=lambda x: (-x[1], x[0]))
    extra["rare"] = {"n_rare_named": len(rare),
                     "examples": [{"term": clean(t), "k": k, "models": ms} for t, k, ms in rare[:24]]}
    # (3) gene universality: genes represented (top-gene of some feature) across models
    gts = load("genes_ts3.json")
    if gts:
        hist = Counter(); allg = []
        for g, rows in gts["index"].items():
            k = len({r[0] for r in rows})
            hist[k] += 1
            if k == len(gts["models"]):
                allg.append(g)
        extra["gene_univ"] = {"n_models": len(gts["models"]), "n_genes": gts["n_genes"],
                              "hist": {str(k): hist[k] for k in sorted(hist)},
                              "n_all": len(allg), "universal_genes": sorted(allg)[:40]}
    # (4) tissue-binding vs universal coverage tradeoff
    core = set(t for t, ms in concept_models.items() if len(ms) == n)
    tl2 = load("tissue_alllayers.json")
    pts = []
    for m in models:
        td = None
        if tl2 and m in tl2.get("models", {}) and tl2["models"][m]:
            td = round(tl2["models"][m][-1].get("frac_specific", 0), 3)
        cshare = round(100 * len(concepts[m] & core) / max(len(concepts[m]), 1), 1)
        pts.append({"model": m, "tissue_deep": td, "core_share": cshare})
    tv = [(p["tissue_deep"], p["core_share"]) for p in pts if p["tissue_deep"] is not None]
    tcorr = round(float(np.corrcoef([x[0] for x in tv], [x[1] for x in tv])[0, 1]), 3) if len(tv) > 2 else None
    extra["tissue_tradeoff"] = {"points": pts, "corr": tcorr}
    # (5) adjacent-layer circuits (weight-based) + (6) hard-cell agreement — cluster outputs, pass through
    circ = load("circuits_adjacent.json")
    if circ:
        extra["circuits"] = {m: {"transitions": circ[m]["transitions"]} for m in models if m in circ}
    hc = load("hardcell_agreement.json")
    if hc:
        extra["hardcells"] = hc
    d["extra"] = extra

    # pull-through blocks from their own files (already 8-model if downstream re-ran)
    for key, fname, xform in [
        ("depth", "depth_alllayers.json", lambda x: x),
        ("tissue_layers", "tissue_alllayers.json", lambda x: x),
        ("cka", "cka_ts3.json", lambda x: {"models": x["models"], "depths": x.get("depths"), "residual": x["residual"], "sae": x["sae"]}),
        ("cka_layers", "cka_layers.json", lambda x: x),
        ("nonlinearity", "nonlinearity_alllayers.json", lambda x: x),
        ("module_themes", "module_themes.json", lambda x: x),
        ("flow", "flow_alllayers.json", lambda x: x),
        ("celltype", "celltype_difficulty.json", lambda x: x),
        ("findings", "findings.json", lambda x: x),
    ]:
        v = load(fname)
        if v is not None:
            d[key] = xform(v)
        else:
            print(f"  (missing {fname} -> {key} block skipped)")

    old = load("atlas_full_notf.json") or {}
    # svd: start from the old block (7 models), update with any per-model svd files (adds Tahoe)
    import glob
    svd = dict(old.get("svd", {}))
    for m in models:
        fs = glob.glob(f"{C}/svd/{m}_L*_svd.json") or glob.glob(f"/Users/annaantipova/Desktop/biomech/outputs/atlas/svd/{m}_L*_svd.json")
        if fs:
            j = json.load(open(sorted(fs)[len(fs) // 2]))
            svd[m] = {"novel": round(100 * j.get("pct_novel", 1), 1), "svd_var": j.get("svd_var_at_k", 0), "sae_var": j.get("sae_var", 0)}
    if svd:
        d["svd"] = {m: svd[m] for m in models if m in svd}
    # preserve flow if we didn't recompute it
    if "flow" in old and "flow" not in d:
        d["flow"] = old["flow"]

    json.dump(d, open(f"{C}/atlas_full_notf.json", "w"))
    print(f"\n==> atlas_full_notf.json ({n} models). keys: {sorted(d)}", flush=True)
    print(f"    universality: {uni} | universal core (all {n}): {uni[str(n)]} | total concepts {len(allc)}", flush=True)


if __name__ == "__main__":
    main()
