#!/usr/bin/env python
"""Hostile review of notes/10_coexpression_baseline.md -- cheap checks that need no corpus.

(1) Re-derive numbers quoted in notes/10 from the JSON files in analysis/coexpression_baseline/ and
    compare with the value printed in the notes (hard-coded below, with the notes line).
(2) Paired (discordant-hit) tests for the node-matched and degree-matched comparisons. The notes use
    a two-sided Fisher test that treats the two graphs as independent samples, but the graphs share
    pairs and hits. Conditioning on the pairs that only one graph has: hits_c_only ~ Binomial(
    hits_c_only + hits_x_only, n_c_only / (n_c_only + n_x_only)) under equal precision. Exact
    two-sided binomial test (scipy.stats.binomtest).
(3) Pre-declared main kind (Pearson) at each passing model's own size: pooled hits, Fisher per model.
(4) Monte Carlo spread of the configuration null for the Tahoe full graph (5 seeds x 200 rewirings),
    same config_shuffle logic as pipeline/scripts/hypothesis_trrust2.py:60-69.
(5) TF-TF share of TRRUST edges inside each model's own node set (the right base rate for the
    'residual hits are TF-TF pairs' claim), vs the union-wide 22.6 % used in the notes.
(6) TRRUST edges touching the salivary genes (notes 10 section 6.2 claims only VDR-CST5).
Writes review_cheap.json next to this script.
"""
from __future__ import annotations
import json, os, pickle
import numpy as np
from scipy.stats import binomtest, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "coexpression_baseline")
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
T = json.load(open(f"{SRC}/coexpr_trrust.json"))
D = json.load(open(f"{SRC}/coexpr_degmatched.json"))
R = json.load(open(f"{SRC}/coexpr_rmatched.json"))
S = json.load(open(f"{SRC}/salivary_check.json"))
P = json.load(open(f"{SRC}/residual_hits_prefix.json"))
PASS = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]
out = {}

# ------------------------------------------------------------------ (1) re-derive numbers
pe = T["kinds"]["pearson"]
checks = [
    # (label, notes value, derived value, notes location)
    ("U genes", 18250, T["gene_universe"]["expressed_ge_min_cells"], "10:136"),
    ("TRRUST edges in U", 6805, T["gene_universe"]["trrust_edges_in_U"], "10:137"),
    ("global top-8631 Pearson hits", 7, pe["global"]["8631"]["hits"], "10:247"),
    ("global top-41952 Pearson hits", 24, pe["global"]["41952"]["hits"], "10:218"),
    ("global top-147728 within-comp hits", 76, T["kinds"]["withincomp"]["global"]["147728"]["hits"], "10:220"),
    ("global top-4149 Pearson fold", 75.0, pe["global"]["4149"]["fold"], "10:212"),
    ("global top-8631 Pearson r cut-off", 0.651, round(pe["global"]["8631"]["r_cutoff"], 3), "10:213"),
    ("node-matched Tahoe Pearson hits", 24, pe["node_matched"]["Tahoe"]["N_cofire_in_U"]["hits"], "10:262"),
    ("node-matched Tahoe Fisher p", 0.00015, round(pe["node_matched"]["Tahoe"]["N_cofire_in_U"]["fisher_cofire_vs_coexpr"]["p_two_sided"], 5), "10:262"),
    ("node-matched scGPT Fisher p", 0.039, round(pe["node_matched"]["scGPT"]["N_cofire_in_U"]["fisher_cofire_vs_coexpr"]["p_two_sided"], 3), "10:263"),
    ("degmatched Tahoe coexpr hits", 41, D["models"]["Tahoe"]["coexpr_degmatched"]["hits"], "10:285"),
    ("degmatched Tahoe Fisher p", 0.088, round(D["models"]["Tahoe"]["fisher_cofire_vs_coexpr"]["p_two_sided"], 3), "10:285"),
    ("degmatched C2S fill", 0.913, round(D["models"]["C2S"]["fill_share"], 3), "10:288"),
    ("degmatched sum coexpr hits (5 passing)", 187, sum(D["models"][m]["coexpr_degmatched"]["hits"] for m in PASS), "10:29"),
    ("degmatched sum cofire hits (5 passing)", 207, sum(D["models"][m]["cofire_in_U"]["hits"] for m in PASS), "10:29"),
    ("Tahoe share of pairs in top 5%", 0.588, round(pe["overlap"]["Tahoe"]["share_in_global_top5pct"], 3), "10:311"),
    ("top5% r threshold", 0.0867, pe["thresholds"]["top5pct_r"], "10:305"),
    ("Tahoe hits inside top 5%", 53, pe["overlap"]["Tahoe"]["cofire_in_global_top5"]["hits"], "10:311"),
    ("C2S hits outside top 5%", 45, pe["overlap"]["C2S"]["cofire_not_global_top5"]["hits"], "10:340"),
    ("C2S outside-top-5% config fold", 3.75, pe["overlap"]["C2S"]["cofire_not_global_top5"]["fold"], "10:340"),
    ("UCE r-level ratio outside top 5%", 9.2, round(R["kinds"]["pearson"]["models"]["UCE"]["not_top5pct"]["ratio"], 1), "10:339"),
    ("Tahoe r-level expected (all pairs)", 7.9, round(R["kinds"]["pearson"]["models"]["Tahoe"]["all_in_U"]["expected_from_r"], 1), "10:378"),
    ("scGPT within-comp outside-top-5% p", 0.16, round(T["kinds"]["withincomp"]["overlap"]["scGPT"]["cofire_not_global_top5"]["p_emp"], 2), "10:349"),
    ("UCE residual TF-TF hits", 19, P["models"]["UCE"]["hits_not_top5_tf_tf"], "10:356"),
    ("C2S residual TF-TF hits", 17, P["models"]["C2S"]["hits_not_top5_tf_tf"], "10:359"),
    ("TRRUST TF-TF share in union", 0.226, round(P["trrust_share_tf_tf"], 3), "10:357"),
    ("parotid cells with core read", "37/38", f"{[r for r in S['by_tissue'] if r['tissue']=='parotid gland'][0]['n_with_core']}/{[r for r in S['by_tissue'] if r['tissue']=='parotid gland'][0]['n_cells']}", "10:41"),
    ("parotid median core UMIs", 523, [r for r in S["by_tissue"] if r["tissue"] == "parotid gland"][0]["median_core_among_expr"], "10:42"),
    ("core UMIs from salivary tissue (no PRH1)", 0.984, round(S["core_without_PRH1"]["umi_share_from_salivary_tissue"], 3), "10:43"),
    ("salivary-tissue cells", 67, S["cells_from_salivary_tissue"], "10:43"),
    ("max core fraction of library", 0.25, round(S["core_frac_of_library_among_expr"]["max"], 2), "10:455"),
    ("PRB3-PRH2 r (all cells)", 0.96, round(S["top_pairs_pearson"]["PRB3-PRH2"]["all"], 2), "10:461"),
]
res1 = []
for lab, nv, dv, loc in checks:
    ok = (nv == dv) if isinstance(nv, str) else bool(abs(float(nv) - float(dv)) <= max(1e-9, 0.006 * abs(float(nv)) + 1e-4))
    res1.append({"item": lab, "notes": nv, "json": dv, "notes_line": loc, "match": ok})
out["rederived_numbers"] = res1
out["n_checked"] = len(res1); out["n_mismatch"] = sum(not r["match"] for r in res1)

# ------------------------------------------------------------------ (2) paired tests on discordant hits
def paired(hc, hx, nc, nx, shared_pairs, shared_hits):
    a, b = hc - shared_hits, hx - shared_hits          # hits found by only one graph
    na, nb = nc - shared_pairs, nx - shared_pairs      # pairs only in one graph
    p0 = na / (na + nb)
    bt = binomtest(a, a + b, p0) if a + b > 0 else None
    return {"cofire_only_hits": a, "coexpr_only_hits": b, "cofire_only_pairs": na, "coexpr_only_pairs": nb,
            "p0": round(p0, 4), "binom_p_two_sided": round(bt.pvalue, 5) if bt else None,
            "fisher_p_independent_samples": round(float(fisher_exact([[hc, nc - hc], [hx, nx - hx]])[1]), 5)}

out["paired_degmatched"] = {}
for m, o in D["models"].items():
    out["paired_degmatched"][m] = paired(o["cofire_in_U"]["hits"], o["coexpr_degmatched"]["hits"],
                                         o["N_cofire_in_U"], o["N_coexpr_accepted"], o["shared_pairs"], o["hits_shared"])
tot = {k: sum(out["paired_degmatched"][m][k] for m in PASS) for k in ("cofire_only_hits", "coexpr_only_hits", "cofire_only_pairs", "coexpr_only_pairs")}
tot["binom_p_two_sided"] = round(binomtest(tot["cofire_only_hits"], tot["cofire_only_hits"] + tot["coexpr_only_hits"],
                                           tot["cofire_only_pairs"] / (tot["cofire_only_pairs"] + tot["coexpr_only_pairs"])).pvalue, 5)
out["paired_degmatched_pooled_5_passing"] = tot
out["paired_nodematched"] = {}
for kind in ("pearson", "spearman", "withincomp"):
    out["paired_nodematched"][kind] = {}
    for m in PASS + ["MaxToki", "tGPT"]:
        s = T["kinds"][kind]["node_matched"][m]["N_cofire_in_U"]; N = s["n_pairs"]
        out["paired_nodematched"][kind][m] = paired(T["cofire"][m]["in_U"]["hits"], s["hits"], N, N,
                                                    s["shared_pairs_with_cofire"], s["hits_shared_with_cofire"])

# ------------------------------------------------------------------ (3) Pearson (main kind) at each model's own size
PUB = T["published"]; SZ = T["graph_sizes_published"]
rows = {}
for m in PASS:
    N = SZ[m]; hc = PUB[m]["full"]["hits"]
    row = {"N": N, "cofire_hits": hc}
    for kind in ("pearson", "spearman", "withincomp"):
        hx = T["kinds"][kind]["global"][str(N)]["hits"]
        row[kind] = {"hits": hx, "fisher_p": round(float(fisher_exact([[hc, N - hc], [hx, N - hx]])[1]), 5)}
    rows[m] = row
pool = {"cofire": sum(rows[m]["cofire_hits"] for m in PASS)}
for kind in ("pearson", "spearman", "withincomp"):
    hx = sum(rows[m][kind]["hits"] for m in PASS); pool[kind] = hx
    pool[f"{kind}_binom_p_equalN"] = round(binomtest(pool["cofire"], pool["cofire"] + hx, 0.5).pvalue, 5)
    wo = [m for m in PASS if m != "Tahoe"]
    hc2 = sum(rows[m]["cofire_hits"] for m in wo); hx2 = sum(rows[m][kind]["hits"] for m in wo)
    pool[f"{kind}_without_Tahoe"] = {"cofire": hc2, "coexpr": hx2, "binom_p": round(binomtest(hc2, hc2 + hx2, 0.5).pvalue, 5)}
out["global_topN_own_size"] = {"per_model": rows, "pooled_5_passing": pool,
                               "note": "binomial test on pooled hits at equal total N; ignores pair overlap and gene dependence"}

# ------------------------------------------------------------------ (4) MC spread of the Tahoe null
PRED, _ = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TRUTH = set()
TRJ = json.load(open(f"{G}/trrust_edges.json"))
for a, bs in TRJ.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
GENES = sorted({g for m in ORDER for e in PRED[m] for g in e}); GID = {g: i for i, g in enumerate(GENES)}
TKEY = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((GID[a], GID[b]))) for a, b in TRUTH if a in GID and b in GID}), np.int64)


def config_shuffle(a, b, rng):          # hypothesis_trrust2.py:60-69
    stubs = np.concatenate([a, b]); rng.shuffle(stubs)
    h = stubs[::2]; t = stubs[1::2]; ok = h != t; h, t = h[ok], t[ok]
    lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
    return np.unique(lo * (1 << 21) + hi)


E = list(PRED["Tahoe"])
a = np.fromiter((GID[e[0]] for e in E), np.int32, len(E)); b = np.fromiter((GID[e[1]] for e in E), np.int32, len(E))
mc = []
for seed in (11, 12, 13, 14, 15):
    rng = np.random.default_rng(seed)
    nul = np.array([np.isin(config_shuffle(a, b, rng), TKEY).sum() for _ in range(200)], float)
    mc.append({"seed": seed, "null_mean": round(float(nul.mean()), 3), "null_sd": round(float(nul.std()), 3),
               "fold": round(59 / nul.mean(), 2)})
nm = np.array([x["null_mean"] for x in mc])
out["tahoe_full_null_mc"] = {"runs": mc, "spread_null_mean": [float(nm.min()), float(nm.max())],
                             "notes_T1_value": 3.93, "published_500perm": 4.44,
                             "se_of_mean_200perm": round(float(np.mean([x["null_sd"] for x in mc])) / np.sqrt(200), 3)}

# ------------------------------------------------------------------ (5) TF-TF base rate inside each model's node set
TF = {str(k).upper() for k in TRJ}
U = set(pe["overlap"].keys())  # placeholder to keep names; node sets from PRED restricted to union genes
tfb = {}
for m in PASS:
    nodes = {g for e in PRED[m] for g in e}
    tr = [(u, v) for u, v in TRUTH if u in nodes and v in nodes]
    tftf = sum(u in TF and v in TF for u, v in tr)
    pm = P["models"][m]
    tfb[m] = {"trrust_edges_in_node_set": len(tr), "tf_tf_share_in_node_set": round(tftf / max(len(tr), 1), 4),
              "residual_hits": pm["hits_not_top5"], "residual_tf_tf": pm["hits_not_top5_tf_tf"],
              "inside_top5_hits": pm["hits_top5"], "inside_top5_tf_tf": pm["hits_top5_tf_tf"],
              "residual_tf_tf_or_same_prefix": sum(((x in TF and y in TF) or x[:3] == y[:3]) for x, y, _ in pm["not_top5_hits"])}
    if pm["hits_not_top5"]:
        tfb[m]["binom_p_residual_tf_tf_vs_node_set_rate"] = round(binomtest(pm["hits_not_top5_tf_tf"], pm["hits_not_top5"], tfb[m]["tf_tf_share_in_node_set"]).pvalue, 5)
out["tf_tf_base_rate"] = tfb
out["residual_pooled_4_models"] = {"hits": sum(P["models"][m]["hits_not_top5"] for m in ["Tahoe", "UCE", "C2S", "scGPT"]),
                                   "tf_tf": sum(P["models"][m]["hits_not_top5_tf_tf"] for m in ["Tahoe", "UCE", "C2S", "scGPT"])}

# ------------------------------------------------------------------ (6) TRRUST edges touching salivary genes
SAL = set(S["core_genes"]) | set(S["extra_genes_checked"]) - {"LYZ"}
out["trrust_edges_touching_salivary"] = sorted([list(e) for e in TRUTH if e[0] in SAL or e[1] in SAL])
out["n_salivary_genes_checked"] = len(SAL)

json.dump(out, open(os.path.join(HERE, "review_cheap.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rederived_numbers"}, indent=1)[:6000])
print("mismatches:", [r for r in res1 if not r["match"]])
