#!/usr/bin/env python
"""Steps (1)-(3) at the headline mid layer: membership, redundancy (programmes), gene-content classes.

(1) per-concept model membership for every concept in >=7 of 10 models; which models miss which concepts;
    checks against backbone_terms.json and recalibrate_final.json (original repo, read-only).
(2) cluster the 78 concepts by Jaccard of (a) their full gene sets and (b) the genes actually hit (union over all
    passing features of all models of top-10 genes in the set); complete linkage at J>=0.3 and J>=0.5, plus
    average and single linkage as sensitivity.
(3) classify each concept by gene content of its hit genes (T = protein-synthesis genes, O = other housekeeping
    families; bb_common.py defines both). Compare with the repo's keyword rule (recalibrate_robust.py:123-132).
Output: s2_mid_composition.json
"""
import json
from collections import Counter, defaultdict

import numpy as np
import bb_common as b

SCR = b.CACHE + "/bb_cache"
HK = ["translat", "ribosom", "rrna", "trna", "srp", "cotranslational", "targeting to membrane", "targeting to er",
      "protein targeting", "nonsense-mediated", "mrna catabolic", "mrna processing", "mrna splic", "spliceosom",
      "rna processing", "proteasom", "oxidative phosphoryl", "respiratory", "electron transport", "mitochond",
      "dna replication", "dna repair", "cell cycle", "mitotic", "chromatin", "histone", "ubiquitin", "chaperone",
      "peptide biosynth", "amide biosynth", "gene expression", "rrna processing", "ribonucleoprotein"]  # recalibrate_robust.py:123-127


def is_hk_keyword(term):  # recalibrate_robust.py:130-132
    return any(h in term.split(":", 1)[1].lower() for h in HK)


def load_slice(name):
    return json.load(open(f"{SCR}/slice_{name}.json"))


def concept_table(per, models=b.ORDER):
    """per-concept: models present, features per model, hit-gene counts pooled over features and models"""
    cm = Counter(); nfeat = defaultdict(dict); hits = defaultdict(Counter); fsets = defaultdict(set)
    for m in models:
        for t, lst in per[m]["terms"].items():
            cm[t] += 1; nfeat[t][m] = len(lst)
            for fi, hs in lst:
                hits[t].update(hs); fsets[t].add((m, fi))
    concept_table.fsets = fsets
    return cm, nfeat, hits


def content(hitcounter, term_genes):
    """gene-content fractions for one concept"""
    u = set(hitcounter); tot = sum(hitcounter.values())
    fam = Counter()
    for g, c in hitcounter.items():
        fam[b.gene_family(g)] += c
    o_hk = b.T_GENES | b.O_ALL
    return {
        "n_hit_genes": len(u), "n_hits_weighted": tot,
        "frac_T_union": round(len(u & b.T_GENES) / max(len(u), 1), 3),
        "frac_T_weighted": round(sum(c for g, c in hitcounter.items() if g in b.T_GENES) / max(tot, 1), 3),
        "frac_Tstrict_union": round(len(u & b.T_STRICT) / max(len(u), 1), 3),
        "frac_TO_union": round(len(u & o_hk) / max(len(u), 1), 3),
        "frac_TO_weighted": round(sum(c for g, c in hitcounter.items() if g in o_hk) / max(tot, 1), 3),
        "frac_T_fullset": round(len(term_genes & b.T_GENES) / max(len(term_genes), 1), 3),
        "family_weighted": {k: round(v / max(tot, 1), 3) for k, v in fam.most_common()},
        "top_hit_genes": [[g, c] for g, c in hitcounter.most_common(12)],
        "dominant_label": b.dominant_label(hitcounter)[0], "dominant_label_share": b.dominant_label(hitcounter)[1],
        "unlabelled_share": b.dominant_label(hitcounter)[2],
    }


def classify(c, thr=0.5, key_T="frac_T_union", key_TO="frac_TO_union"):
    if c[key_T] >= thr: return "protein_synthesis"
    if c[key_TO] >= thr: return "other_housekeeping"
    return "specific"


def programmes(terms, simfun, cm, nfeat, thr, method):
    cls = b.cluster_matrix(sorted(terms), simfun, thr, method)
    out = []
    for c in cls:
        # representative: most models, then most annotated features in total, then alphabetical
        rep = sorted(c, key=lambda t: (-cm[t], -sum(nfeat[t].values()), t))[0]
        out.append({"representative": rep, "n_members": len(c), "members": sorted(c, key=lambda t: (-cm[t], t))})
    return out


def run_slice(name, per, models=b.ORDER, k=8, gold=None):
    cm, nfeat, hits = concept_table(per, models)
    sel = sorted(t for t, v in cm.items() if v >= k)
    full = {t: b.RAW[t.split(":", 1)[0]][t.split(":", 1)[1]] for t in sel}
    hitg = {t: set(hits[t]) for t in sel}
    concepts = {}
    for t in sel:
        c = content(hits[t], full[t])
        c["class"] = classify(c)
        c["class_weighted"] = classify(c, key_T="frac_T_weighted", key_TO="frac_TO_weighted")
        c["class_Tstrict"] = classify(c, key_T="frac_Tstrict_union")
        c["class_thr0.3"] = classify(c, 0.3); c["class_thr0.7"] = classify(c, 0.7)
        c["keyword_rule"] = "housekeeping" if is_hk_keyword(t) else "specific"
        c["n_models"] = cm[t]; c["models"] = [m for m in models if m in nfeat[t]]
        c["missing"] = [m for m in models if m not in nfeat[t]]
        c["n_features_per_model"] = nfeat[t]; c["set_size"] = len(full[t])
        concepts[t] = c
    res = {"n_models_used": len(models), "k": k, "n_concepts": len(sel), "concepts": concepts,
           "class_counts": {}, "programmes": {}}
    for key in ("class", "class_weighted", "class_Tstrict", "class_thr0.3", "class_thr0.7", "keyword_rule"):
        res["class_counts"][key] = dict(Counter(concepts[t][key] for t in sel))
    lab = Counter(concepts[t]["dominant_label"] for t in sel)
    res["dominant_label_counts"] = dict(lab.most_common())
    res["n_distinct_dominant_labels"] = len(lab)
    res["dominant_label_members"] = {k: sorted(t for t in sel if concepts[t]["dominant_label"] == k) for k in lab}
    fsets = concept_table.fsets
    top20 = {t: set(g for g, _ in hits[t].most_common(20)) for t in sel}
    bases = {
        "full_sets": lambda a, c: b.jaccard(full[a], full[c]),                 # Jaccard of the database gene sets
        "hit_genes": lambda a, c: b.jaccard(hitg[a], hitg[c]),                 # Jaccard of the union of hit genes
        "hit_weighted": lambda a, c: b.weighted_jaccard(hits[a], hits[c]),     # Ruzicka on hit counts
        "top20_hits": lambda a, c: b.jaccard(top20[a], top20[c]),              # Jaccard of the 20 most-hit genes
        "features": lambda a, c: b.jaccard(fsets[a], fsets[c]),                # Jaccard of the (model, feature) sets
    }
    res["pairwise_similarity_summary"] = {}
    for basis, sf in bases.items():
        vals = [sf(sel[i], sel[j]) for i in range(len(sel)) for j in range(i + 1, len(sel))]
        res["pairwise_similarity_summary"][basis] = {
            "n_pairs": len(vals), "mean": round(float(np.mean(vals)), 4) if vals else None,
            "frac_pairs_ge_0.3": round(float(np.mean(np.array(vals) >= 0.3)), 4) if vals else None,
            "frac_pairs_ge_0.5": round(float(np.mean(np.array(vals) >= 0.5)), 4) if vals else None}
        for method in ("complete", "average", "single"):
            for thr in (0.3, 0.5):
                P = programmes(sel, sf, cm, nfeat, thr, method)
                for p in P:
                    p["classes"] = dict(Counter(concepts[t]["class"] for t in p["members"]))
                    pooled = Counter()
                    for t in p["members"]: pooled.update(hits[t])
                    tot = max(sum(pooled.values()), 1)
                    p["top_hit_genes"] = [g for g, _ in pooled.most_common(10)]
                    p["frac_T_weighted"] = round(sum(c for g, c in pooled.items() if g in b.T_GENES) / tot, 3)
                    p["frac_TO_weighted"] = round(sum(c for g, c in pooled.items() if g in (b.T_GENES | b.O_ALL)) / tot, 3)
                    p["class"] = ("protein_synthesis" if p["frac_T_weighted"] >= 0.5 else
                                  "other_housekeeping" if p["frac_TO_weighted"] >= 0.5 else "specific")
                    p["dominant_label"], p["dominant_label_share"], _ = b.dominant_label(pooled)
                res["programmes"][f"{basis}|{method}|J>={thr}"] = {
                    "n_programmes": len(P),
                    "n_by_class": dict(Counter(p["class"] for p in P)),
                    "clusters": P}
    return res, cm, nfeat, hits


if __name__ == "__main__":
    per = load_slice("mid")
    res, cm, nfeat, hits = run_slice("mid", per)
    out = {"slice": "mid (cats[len//2])", "layers": {m: per[m]["layer"] for m in b.ORDER},
           "T_definition": {"n_T_genes": len(b.T_GENES), "n_T_strict": len(b.T_STRICT),
                            "rule": "RP regex (RPL/RPS/RPLP/RPSA, +FAU, UBA52, minus RPS6K*) + ^EIF\\d/^EEF\\d + GO_BP ribosome biogenesis "
                                    "(GO:0042254) + GO_BP NMD (GO:0000184)"},
           "O_definition": {k: len(v) for k, v in b.O_GENES.items()}}

    # ---- (1) reproduction checks ----
    orig_bt = json.load(open(f"{b.ORIG_COMPARATIVE}/backbone_terms.json"))["tiers"]
    orig_rf = json.load(open(f"{b.ORIG_COMPARATIVE}/recalibrate_final.json"))
    orig_cp = json.load(open(f"{b.ORIG_COMPARATIVE}/controls.json")).get("calibrated_panels", {}).get("n_concepts")
    spec = {str(k): sum(1 for v in cm.values() if v == k) for k in range(1, 11)}
    checks = {"tiers": b.tiers(cm), "spectrum": spec,
              "spectrum_equals_recalibrate_final": spec == orig_rf["real_spectrum"],
              "per_model_n_concepts": {m: len(per[m]["terms"]) for m in b.ORDER},
              "per_model_equals_controls_calibrated_panels": ({m: len(per[m]["terms"]) for m in b.ORDER} == orig_cp)
              if isinstance(orig_cp, dict) else f"not comparable: {orig_cp}"}
    for k in (7, 8, 9, 10):
        mine = sorted(t.split(":", 1)[1] + "|" + t.split(":", 1)[0] for t, v in cm.items() if v >= k)
        theirs = sorted(x["term"] + "|" + x["source"] for x in orig_bt[f">={k}"])
        checks[f"names_equal_backbone_terms_>={k}"] = mine == theirs
    out["reproduction"] = checks

    # ---- (1) membership ----
    ge7 = sorted((t for t, v in cm.items() if v >= 7), key=lambda t: (-cm[t], t))
    out["membership_ge7"] = {t: {"n_models": cm[t], "models": [m for m in b.ORDER if m in nfeat[t]],
                                 "missing": [m for m in b.ORDER if m not in nfeat[t]],
                                 "n_features_per_model": nfeat[t]} for t in ge7}
    ge8 = [t for t in ge7 if cm[t] >= 8]
    miss_count = Counter(m for t in ge8 for m in b.ORDER if m not in nfeat[t])
    pattern = Counter(tuple(m for m in b.ORDER if m not in nfeat[t]) for t in ge8)
    out["missing_summary_ge8"] = {
        "times_each_model_missing": {m: miss_count.get(m, 0) for m in b.ORDER},
        "missing_patterns": [{"missing": list(k), "n_concepts": v,
                              "concepts": sorted(t for t in ge8 if tuple(m for m in b.ORDER if m not in nfeat[t]) == k)}
                             for k, v in pattern.most_common()],
        "nine_of_ten": {t: [m for m in b.ORDER if m not in nfeat[t]] for t in ge8 if cm[t] == 9},
        "ten_of_ten": [t for t in ge8 if cm[t] == 10]}
    # which models miss which gene-content class
    cls = {t: res["concepts"][t]["class"] for t in ge8}
    out["missing_by_class"] = {c: {m: sum(1 for t in ge8 if cls[t] == c and m not in nfeat[t]) for m in b.ORDER}
                               for c in sorted(set(cls.values()))}
    out["n_by_class"] = dict(Counter(cls.values()))
    # models that have ANY protein-synthesis concept at the mid layer (from their own concept list)
    out["per_model_protein_synthesis_concepts_in_78"] = {
        m: sum(1 for t in ge8 if cls[t] == "protein_synthesis" and m in nfeat[t]) for m in b.ORDER}
    out["backbone"] = res
    print("==>", b.dump(out, "s2_mid_composition.json"))

    # console summary
    print("tiers", checks["tiers"], "checks", {k: v for k, v in checks.items() if k.startswith("names") or k.endswith("final") or k.startswith("per_model_equals")})
    print("missing counts", out["missing_summary_ge8"]["times_each_model_missing"])
    for p in out["missing_summary_ge8"]["missing_patterns"][:8]: print("  ", p["missing"], p["n_concepts"])
    print("9/10", out["missing_summary_ge8"]["nine_of_ten"]); print("10/10", out["missing_summary_ge8"]["ten_of_ten"])
    print("class counts", res["class_counts"])
    for key, v in res["programmes"].items(): print(key, v["n_programmes"], v["n_by_class"])
