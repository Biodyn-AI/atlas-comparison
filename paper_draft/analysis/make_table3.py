"""Rebuild main-text Table 3 (tab:prediction) from the corrected run and the re-run analyses."""
import argparse
import json
import os
from pathlib import Path
AN = Path(__file__).resolve().parent
ROOT = AN.parents[1]
OUT = Path(os.environ.get("ATLAS_BASE", ROOT)) / "outputs/atlas/comparative"
NAME = {"AIDO": "AIDO.Cell", "Geneformer": "Geneformer-V2", "C2S": "C2S-Scale", "Tahoe": "Tahoe-x1"}
ORDER = ["AIDO", "scGPT", "scFoundation", "GeneCompass", "tGPT", "MaxToki", "Geneformer", "UCE", "C2S", "Tahoe"]
pm = json.load(open(f"{OUT}/hypothesis_trrust2.json"))["per_model"]
r1 = json.load(open(f"{AN}/review_12_trrust/r1_stratified_direction.json"))["models"]
ct = json.load(open(f"{AN}/coexpression_baseline/coexpr_trrust.json")); P = ct["kinds"]["pearson"]
dg = json.load(open(f"{AN}/coexpression_baseline/coexpr_degmatched.json"))["models"]
rc = json.load(open(f"{AN}/review_10_coexpr/review_cheap.json"))
PASS = [m for m in sorted(pm, key=lambda m: -pm[m]["fold"]) if pm[m]["p_emp"] < 0.05]
def pf(p):
    if p is None: return ""
    t = f"{p:.2g}"
    if "e" not in t:
        return t
    coefficient, exponent = t.split("e")
    return rf"${coefficient}\times 10^{{{int(exponent)}}}$"
rows = []
for m in ORDER:
    r = pm[m]; nm = NAME.get(m, m)
    fold = "--" if r["null_mean"] < 0.05 else f"{r['fold']:.2f}"
    strict = "--"
    if m in r1:
        s = r1[m]["stratified_null"]["both_blocks_total"]; strict = f"{s['fold']:.2f} ({pf(s['p_emp'])})"
    same = P["global"].get(str(r["n_pairs"]), {}).get("hits", "--")
    nmh = P["node_matched"].get(m, {}).get("N_cofire_in_U", {}).get("hits")
    pn = rc["paired_nodematched"]["pearson"].get(m, {}).get("binom_p_two_sided")
    sg = "--" if nmh is None else (f"{nmh} ({pf(pn)})" if pn is not None else f"{nmh}")
    if m in dg:
        pd_ = rc["paired_degmatched"].get(m, {}).get("binom_p_two_sided")
        deg = f"{dg[m]['coexpr_degmatched']['hits']} ({pf(pd_)})" if pd_ is not None else f"{dg[m]['coexpr_degmatched']['hits']}"
    else:
        deg = "--"
    rows.append(f"{nm:<13} & {r['n_pairs']:,} & {r['hits']} & {r['null_mean']:.2f} & {fold} ({pf(r['p_emp'])}) & {strict} & {same} & {sg} & {deg} \\\\\n"
                f"% src: OUT/hypothesis_trrust2.json:per_model.{m}; RV12/r1_stratified_direction.json:models.{m}; AN/co/coexpr_trrust.json:kinds.pearson.global.<N>, node_matched.{m}; RV10/review_cheap.json:paired_nodematched.pearson.{m}, paired_degmatched.{m}; AN/co/coexpr_degmatched.json:models.{m}")
g = rc["global_topN_own_size"]["pooled_5_passing"]; pd = rc["paired_degmatched_pooled_5_passing"]
known = sum(pm[m]["hits"] for m in PASS); degsum = sum(dg[m]["coexpr_degmatched"]["hits"] for m in PASS)
rows.append(f"\\midrule\nFour passing  & -- & {known} & -- & -- & -- & {g['pearson']} & -- & {degsum} ({pf(pd['binom_p_two_sided'])}) \\\\\n"
            "% src: RV10/review_cheap.json:global_topN_own_size.pooled_5_passing (now the four passing models), paired_degmatched_pooled_5_passing")
HEAD = ("\\begin{tabular}{@{}lrrrllrll@{}}\n\\toprule\n"
        " & \\multicolumn{5}{c}{Gene pairs that share features} & \\multicolumn{3}{c}{Plain co-expression: known pairs found} \\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(l){7-9}\n"
        " & Predicted & Known & Expected & Times & Times chance, & Same & Same & Same genes \\\\\n"
        "Model & pairs & pairs & by chance & chance ($p$) & stricter ($p$) & size & genes ($p$) & and partners ($p$) \\\\\n\\midrule\n")
parser = argparse.ArgumentParser(description=__doc__)
args = parser.parse_args()
print(HEAD + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
