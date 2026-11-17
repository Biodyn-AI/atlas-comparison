#!/usr/bin/env python
"""Step 6 (job item 6): Benjamini-Hochberg across families of TRRUST tests.

Families (p-values read from the ORIGINAL repo's result files, or from this folder's JSON):
  F1  the ten per-model tests (hypothesis_trrust2.json:per_model.<m>.p_emp; 500 rewirings, p floor 0.002)
  F2  the robustness variants (hypothesis_robust.json:variants.<v>.<k>.p_emp; 5 variants x k in {1,2}; 200 rewirings)
  F1+F2 together (20 tests)
  F3  the equal-budget tests (hypothesis_sizematched.json:models.<m>.topN.p_emp; 8 scoreable models; 100 rewirings)
  F4  this folder's per-model tests under every pooling variant (s05_pooling.json; 10 models x 4 layer sets x 4 W)
  F5  this folder's Maslov-Sneppen per-model tests (s04_null_choice.json; 7 models)
BH q = min over ranks >= r of p * n / rank. A test "survives" at q < 0.05.
Models whose null mean is 0 (no possible hits) get p = 1 and stay in the family (conservative).
Output: OUT/s06_bh.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

R = f"{ORIG}/outputs/atlas/comparative"
out = {"alpha": 0.05, "families": {}}


def fam(name, labels, ps, note=""):
    q = bh(ps)
    rows = [{"test": l, "p": p, "q_bh": qq, "survives_q05": qq < 0.05, "bonferroni": round(min(1.0, p * len(ps)), 5)}
            for l, p, qq in zip(labels, ps, q)]
    out["families"][name] = {"n_tests": len(ps), "n_p_lt_05": sum(p < 0.05 for p in ps),
                             "n_survive_bh": sum(r["survives_q05"] for r in rows), "note": note, "rows": rows}
    print(f"{name}: {len(ps)} tests, {sum(p < 0.05 for p in ps)} with p<0.05, {sum(r['survives_q05'] for r in rows)} survive BH")
    for r in rows:
        print(f"   {r['test']:<55} p={r['p']:<8} q={r['q_bh']:<8} {'*' if r['survives_q05'] else ''}")


t2 = json.load(open(f"{R}/hypothesis_trrust2.json"))["per_model"]
L1 = [f"per-model {m}" for m in ORDER]; P1 = [t2[m]["p_emp"] for m in ORDER]
fam("F1_ten_per_model_published", L1, P1, "hypothesis_trrust2.json:per_model.<m>.p_emp (500 rewirings, floor 1/501)")

rb = json.load(open(f"{R}/hypothesis_robust.json"))["variants"]
L2, P2 = [], []
for v, d in rb.items():
    for k in ("1", "2"):
        L2.append(f"robust {v} k>={k}"); P2.append(d[k]["p_emp"])
fam("F2_robust_variants_published", L2, P2, "hypothesis_robust.json:variants.<v>.<k>.p_emp (200 rewirings, floor 1/201)")
fam("F1_plus_F2_published", L1 + L2, P1 + P2, "union of F1 and F2")

sm = json.load(open(f"{R}/hypothesis_sizematched.json"))["models"]
L3, P3 = [], []
for m, d in sm.items():
    if isinstance(d, dict) and "topN" in d and isinstance(d["topN"], dict) and "p_emp" in d["topN"]:
        L3.append(f"equal-budget {m}"); P3.append(d["topN"]["p_emp"])
fam("F3_equal_budget_published", L3, P3, "hypothesis_sizematched.json:models.<m>.topN.p_emp (100 rewirings, floor 1/101); untestable models omitted")

if os.path.exists(f"{OUT}/s05_pooling.json"):
    s5 = json.load(open(f"{OUT}/s05_pooling.json"))
    L4, P4 = [], []
    for m, d in s5["per_model"].items():
        for s in ("all", "noL0", "even8", "even8noL0"):
            for W in s5["W"]:
                r = d[s][f"W{W}"]
                L4.append(f"{m} {s} W{W}"); P4.append(r.get("p_emp", 1.0) if r.get("n_pairs") else 1.0)
    fam("F4_pooling_variants_this_work", L4, P4, "s05_pooling.json:per_model.<m>.<layerset>.W<W>.p_emp (200 rewirings)")
    # sub-families: one per variant, across the ten models
    for s in ("all", "noL0", "even8", "even8noL0"):
        for W in s5["W"]:
            Ls = [f"{m} {s} W{W}" for m in ORDER]
            Ps = [(s5["per_model"][m][s][f"W{W}"].get("p_emp", 1.0) if s5["per_model"][m][s][f"W{W}"].get("n_pairs") else 1.0) for m in ORDER]
            fam(f"F4_{s}_W{W}_ten_models", Ls, Ps, "one pooling variant, BH across the ten models")
if os.path.exists(f"{OUT}/s04_null_choice.json"):
    s4 = json.load(open(f"{OUT}/s04_null_choice.json"))["per_model"]
    fam("F5_maslov_sneppen_seven_models", [f"MS {m}" for m in s4], [s4[m]["maslov_sneppen"]["p_emp"] for m in s4],
        "s04_null_choice.json:per_model.<m>.maslov_sneppen.p_emp (200 draws, floor 1/201)")
if os.path.exists(f"{OUT}/s03_hub_direction.json"):
    s3 = json.load(open(f"{OUT}/s03_hub_direction.json"))["models"]
    fam("F6_tf_to_nontf_hits_seven_models", [f"TF-nonTF {m}" for m in s3],
        [s3[m]["direction"]["hits_tf_nontf_vs_null"]["p_emp"] for m in s3],
        "s03_hub_direction.json:models.<m>.direction.hits_tf_nontf_vs_null.p_emp (200 draws)")
    fam("F7_leave_one_tf_out_worst_case_seven_models", [f"LOTO worst {m}" for m in s3],
        [s3[m]["leave_one_tf_out"]["max_p"] for m in s3],
        "s03_hub_direction.json:models.<m>.leave_one_tf_out.max_p (worst TF per model; 200 draws)")
    fam("F7b_drop_top3_tfs_seven_models", [f"drop top3 {m}" for m in s3],
        [s3[m]["leave_one_tf_out"]["drop_top3_together"]["p_emp"] for m in s3],
        "s03_hub_direction.json:models.<m>.leave_one_tf_out.drop_top3_together.p_emp")
if os.path.exists(f"{OUT}/s03b_family_masked.json"):
    s3b = json.load(open(f"{OUT}/s03b_family_masked.json"))["models"]
    L8, P8 = [], []
    for m, d in s3b.items():
        for v in ("M1_prefix2", "M2_struct_family", "M1_plus_M2"):
            L8.append(f"{m} {v}"); P8.append(d[v]["p_emp"])
    fam("F8_family_masked_seven_models_x3", L8, P8, "s03b_family_masked.json:models.<m>.<mask>.p_emp (200 draws)")
json.dump(out, open(f"{OUT}/s06_bh.json", "w"), indent=1)
