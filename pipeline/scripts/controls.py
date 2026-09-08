#!/usr/bin/env python
"""Reviewer-demanded controls for the comparative atlas. All local, from per-layer concept sets + the matrix.
  (1) permutation null for the universality spectrum / universal core / mean pairwise Jaccard — uniform AND
      gene-set-size-weighted (tests 'the core is just big, ubiquitous, well-annotated sets'). observed vs null.
  (2) equal-layer control — recompute core/spectrum giving every model the SAME number of layers (min across
      models, evenly spaced), so the all-layer core isn't just 'more layers = more chances'.
  (4) capacity/vocabulary confound — n_concepts vs d_sae and vs reachable gene vocabulary (corr + per-1000-feat).
  (5) concentration axis — leave-one-out correlations + partial correlation controlling d_sae/vocab (is the
      'orthogonal to design -> emergent' claim robust, or a GeneCompass leverage point?).
  (6) STRING split — concept counts and core with vs without STRING (STRING = PPI, not pathway function).
    python scripts/controls.py   -> outputs/atlas/comparative/controls.json
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
NPERM = 1000
rng = np.random.default_rng(0)

pl = json.load(open(f"{C}/perlayer_concepts.json"))
mat = json.load(open(f"{C}/matrix_alllayer.json"))
models = [m for m in ORDER if m in mat]; N = len(models)

# ---------- gene-set sizes (for the size-weighted null 'ease' weight) ----------
MIN, MAX = 5, 500
size = {}
for nm in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items():
        if MIN <= len(g) <= MAX:
            size[f"{nm}:{t}"] = len(g)
for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
    for t, g in json.load(open(f"{G}/{fn}")).items():
        if MIN <= len(g) <= MAX:
            size[f"{nm}:{t}"] = len(g)


def concept_sets(pick_layers=None):
    """per-model all-layer concept set; pick_layers(model)->list of layer keys to restrict to (equal-layer)."""
    cs = {}
    for m in models:
        Ls = sorted(pl[m].keys(), key=lambda x: int(x))
        if pick_layers is not None:
            Ls = pick_layers(m, Ls)
        s = set()
        for L in Ls:
            s.update(pl[m][L]["concepts"])
        cs[m] = s
    return cs


def spectrum(cs):
    cm = defaultdict(int)
    for m in cs:
        for t in cs[m]:
            cm[t] += 1
    spec = Counter(cm.values())
    return {str(k): int(spec.get(k, 0)) for k in range(1, N + 1)}, cm


def mean_jaccard(cs):
    ms = list(cs); v = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = cs[ms[i]], cs[ms[j]]
            v.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(v))


# ================= (1) permutation null =================
cs_all = concept_sets()
spec_obs, cm_obs = spectrum(cs_all)
core_obs = spec_obs[str(N)]
jac_obs = mean_jaccard(cs_all)
universe = sorted(set().union(*cs_all.values()))
U = len(universe); uidx = {t: i for i, t in enumerate(universe)}
Km = {m: len(cs_all[m]) for m in models}
w = np.array([size.get(t, 1) for t in universe], float); w = w / w.sum()   # size-weighted 'ease'


def null_run(weighted):
    cores = np.empty(NPERM); jacs = np.empty(NPERM); specs = np.zeros((NPERM, N + 1))
    for it in range(NPERM):
        cnt = np.zeros(U, int); sets = []
        for m in models:
            pick = rng.choice(U, size=Km[m], replace=False, p=w if weighted else None)
            cnt[pick] += 1; sets.append(pick)
        cores[it] = int((cnt == N).sum())
        h = np.bincount(cnt[cnt > 0], minlength=N + 1)
        specs[it] = h
        # mean jaccard from counts is not exact; approximate via set overlaps
        S = [set(s.tolist()) for s in sets]; vv = []
        for i in range(N):
            for j in range(i + 1, N):
                vv.append(len(S[i] & S[j]) / max(len(S[i] | S[j]), 1))
        jacs[it] = np.mean(vv)
    return cores, jacs, specs


def summ(obs, null):
    mu, sd = float(null.mean()), float(null.std())
    z = (obs - mu) / sd if sd > 0 else float("inf")
    p = float((null >= obs).mean()) if obs >= mu else float((null <= obs).mean())
    return {"observed": float(obs), "null_mean": round(mu, 1), "null_sd": round(sd, 1),
            "z": round(z, 1), "p_emp": max(p, 1.0 / NPERM)}


print("permutation null (uniform)...", flush=True)
cu, ju, su = null_run(False)
print("permutation null (size-weighted)...", flush=True)
cw, jw, sw = null_run(True)
null_block = {
    "core_observed": core_obs, "jaccard_observed": round(jac_obs, 3), "universe": U,
    "core_vs_uniform": summ(core_obs, cu), "core_vs_sizeweighted": summ(core_obs, cw),
    "jaccard_vs_uniform": summ(jac_obs, ju), "jaccard_vs_sizeweighted": summ(jac_obs, jw),
}

# ================= (2) equal-layer control =================
minL = min(len(pl[m]) for m in models)


def evenly(m, Ls):
    if len(Ls) <= minL:
        return Ls
    idx = np.linspace(0, len(Ls) - 1, minL).round().astype(int)
    return [Ls[i] for i in sorted(set(idx))]


cs_eq = concept_sets(evenly)
spec_eq, cm_eq = spectrum(cs_eq)
mid_core = None
# mid-layer core (single mid layer per model) for the 3-way comparison
cs_mid = {}
for m in models:
    Ls = sorted(pl[m].keys(), key=lambda x: int(x)); midL = Ls[len(Ls) // 2]
    cs_mid[m] = set(pl[m][midL]["concepts"])
spec_mid, _ = spectrum(cs_mid)
equal_block = {"min_layers": minL,
               "core_mid": spec_mid[str(N)], "spectrum_mid": spec_mid,
               "core_equal_layer": spec_eq[str(N)], "spectrum_equal": spec_eq,
               "core_all_layer": core_obs, "spectrum_all": spec_obs,
               "n_layers": {m: len(pl[m]) for m in models}}

# ================= (4) capacity / vocabulary confound =================
dsae = {m: max(pl[m][L]["d_sae"] for L in pl[m]) for m in models}
# reachable-vocabulary proxy = union of all top genes surfaced across the model's catalogs
vocab = {}
for m in models:
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    gv = set()
    for cp in glob.glob(f"{d}/feature_catalog_L*.json"):
        for f in json.load(open(cp))["features"].values():
            gv.update(str(x).upper() for x in f["top_genes"])
    vocab[m] = len(gv)
nconc = {m: len(cs_all[m]) for m in models}
def corr(a, b): return round(float(np.corrcoef(a, b)[0, 1]), 3)
mv = models
cap_block = {
    "n_concepts": nconc, "d_sae": dsae, "vocab_surfaced": vocab,
    "corr_nconcepts_dsae": corr([nconc[m] for m in mv], [dsae[m] for m in mv]),
    "corr_nconcepts_vocab": corr([nconc[m] for m in mv], [vocab[m] for m in mv]),
    "corr_nconcepts_nlayers": corr([nconc[m] for m in mv], [len(pl[m]) for m in mv]),
    "concepts_per_1k_feat": {m: round(1000 * nconc[m] / max(dsae[m], 1), 1) for m in mv},
}

# ================= (5) concentration axis robustness =================
topn = json.load(open(f"{C}/topn_sweep.json"))["models"]
gse = json.load(open(f"{C}/gsea_annot.json"))["models"]
f5 = {m: next(r["annot_rate"] for r in topn[m] if r["top"] == 5) for m in mv if m in topn}
conc = {m: f5[m] - gse[m]["annot_rate"] for m in mv if m in f5 and m in gse}
cm_ = list(conc)
tax = json.load(open(f"{C}/atlas_full_notf.json"))["axes"]["tax"]
isrank = {m: 1.0 if tax[m]["tok"] == "rank" else 0.0 for m in cm_}
cvec = np.array([conc[m] for m in cm_]); rvec = np.array([isrank[m] for m in cm_])
dvec = np.array([dsae[m] for m in cm_], float); vvec = np.array([vocab[m] for m in cm_], float)
f5v = np.array([f5[m] for m in cm_]); gsv = np.array([gse[m]["annot_rate"] for m in cm_])


def partial_corr(x, y, z):
    def resid(a, ctrl):
        A = np.vstack([np.ones_like(ctrl), ctrl]).T
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta
    return round(float(np.corrcoef(resid(x, z), resid(y, z))[0, 1]), 3)


loo = []
for k, m in enumerate(cm_):
    keep = [i for i in range(len(cm_)) if i != k]
    loo.append(round(float(np.corrcoef(f5v[keep], gsv[keep])[0, 1]), 3))
conc_block = {
    "conc": {m: round(conc[m], 1) for m in cm_},
    "corr_fisher5_gsea_full": corr(f5v, gsv),
    "corr_fisher5_gsea_LOO_range": [min(loo), max(loo)],
    "corr_fisher5_gsea_LOO_dropGeneCompass": (loo[cm_.index("GeneCompass")] if "GeneCompass" in cm_ else None),
    "corr_conc_rank": corr(cvec, rvec),
    "corr_conc_dsae": corr(cvec, np.log10(dvec)),
    "corr_conc_vocab": corr(cvec, np.log10(vvec)),
    "partial_corr_conc_rank_given_dsae": partial_corr(cvec, rvec, np.log10(dvec)),
    "partial_corr_conc_rank_given_vocab": partial_corr(cvec, rvec, np.log10(vvec)),
}

# ================= (6) STRING split =================
def by_src(s):
    d = Counter(t.split(":")[0] for t in s); return dict(d)
cs_nostring = {m: {t for t in cs_all[m] if not t.startswith("STRING:")} for m in mv}
spec_ns, cm_ns = spectrum(cs_nostring)
core_ns = spec_ns[str(N)]
core_terms = [t for t, k in cm_obs.items() if k == N]
string_block = {
    "concept_sources_total": by_src(set().union(*cs_all.values())),
    "core_all": core_obs, "core_no_string": core_ns,
    "pct_core_string": round(100 * sum(1 for t in core_terms if t.startswith("STRING:")) / max(len(core_terms), 1), 1),
    "n_concepts_with_string": {m: len(cs_all[m]) for m in mv},
    "n_concepts_no_string": {m: len(cs_nostring[m]) for m in mv},
}

out = {"null": null_block, "equal_layer": equal_block, "capacity": cap_block,
       "concentration": conc_block, "string": string_block, "n_perm": NPERM, "models": models}
json.dump(out, open(f"{C}/controls.json", "w"), indent=1)

print("\n===== CONTROLS SUMMARY =====")
print(f"(1) core observed {core_obs} | uniform null {null_block['core_vs_uniform']['null_mean']}±{null_block['core_vs_uniform']['null_sd']} (z={null_block['core_vs_uniform']['z']}) | size-weighted null {null_block['core_vs_sizeweighted']['null_mean']}±{null_block['core_vs_sizeweighted']['null_sd']} (z={null_block['core_vs_sizeweighted']['z']})")
print(f"    jaccard observed {jac_obs:.3f} | uniform null {null_block['jaccard_vs_uniform']['null_mean']} | size-weighted {null_block['jaccard_vs_sizeweighted']['null_mean']}")
print(f"(2) core: mid {spec_mid[str(N)]} | equal-{minL}-layer {spec_eq[str(N)]} | all-layer {core_obs}")
print(f"(4) corr n_concepts~d_sae {cap_block['corr_nconcepts_dsae']} | ~vocab {cap_block['corr_nconcepts_vocab']} | ~n_layers {cap_block['corr_nconcepts_nlayers']}")
print(f"(5) conc~rank {conc_block['corr_conc_rank']} | ~d_sae {conc_block['corr_conc_dsae']} | partial(rank|d_sae) {conc_block['partial_corr_conc_rank_given_dsae']} | LOO r range {conc_block['corr_fisher5_gsea_LOO_range']} | drop-GeneCompass {conc_block['corr_fisher5_gsea_LOO_dropGeneCompass']}")
print(f"(6) core all {core_obs} vs no-STRING {core_ns} | %core STRING {string_block['pct_core_string']}")
print("==> controls.json")
