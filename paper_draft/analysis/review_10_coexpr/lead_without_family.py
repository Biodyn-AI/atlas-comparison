#!/usr/bin/env python
"""Does a model's lead over degree-matched co-expression survive removing family and look-alike pairs?

Reads the discordant TRRUST hits of the degree-matched comparison on all 6,000 cells
(review_heavy.json:P1_recount_6000.degmatched.<m>.{cofire_only_hit_pairs, coexpr_only_hit_pairs}) and
drops the pairs that masks M1 + M2 of trrust_robustness/s03b_family_masked.py remove: the two symbols
share at least 2 leading characters (M1), or both genes are in one class of the hand-made
DNA-binding-domain map (s02b_family.json:family_map; M2). Exact two-sided binomial test on the rest.
usage: python lead_without_family.py  -> lead_without_family.json next to this script
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from family_mask import compare_without_family, gene_families, paired_p

heavy = json.loads((HERE / "review_heavy.json").read_text())
models = heavy["P1_recount_6000"]["degmatched"]
families = gene_families()
out = {
    "rule": "M1 (>= 2 shared leading characters) or M2 (same class of the hand-made DNA-binding-domain map)",
    "models": {},
}
for model, result in models.items():
    cofire_pairs = result["cofire_only_hit_pairs"]
    coexpr_pairs = result["coexpr_only_hit_pairs"]
    masked = compare_without_family(cofire_pairs, coexpr_pairs, families)
    cofire, coexpr = len(cofire_pairs), len(coexpr_pairs)
    out["models"][model] = {
        "cofire_only": cofire,
        "coexpr_only": coexpr,
        "binom_p": round(paired_p(cofire, coexpr), 5),
        "masked_cofire_only": cofire - masked["cofire_only"],
        "masked_coexpr_only": coexpr - masked["coexpr_only"],
        "cofire_only_unmasked": masked["cofire_only"],
        "coexpr_only_unmasked": masked["coexpr_only"],
        "binom_p_unmasked": round(masked["binom_p_two_sided"], 5),
    }
    print(model, out["models"][model])
(HERE / "lead_without_family.json").write_text(json.dumps(out, indent=1))
