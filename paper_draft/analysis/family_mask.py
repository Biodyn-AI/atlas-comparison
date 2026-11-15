"""Name and structural-family masks used in the atlas TRRUST comparisons."""

import json
from pathlib import Path

from scipy.stats import binomtest


def gene_families():
    path = Path(__file__).with_name("trrust_robustness") / "s02b_family.json"
    families = json.loads(path.read_text())["family_map"]
    return {gene: family for family, genes in families.items() for gene in genes.split()}


def excluded(pair, families):
    a, b = pair
    return a[:2] == b[:2] or (a in families and families[a] == families.get(b))


def paired_p(cofire_only, coexpr_only):
    total = cofire_only + coexpr_only
    return float(binomtest(cofire_only, total).pvalue) if total else 1.0


def compare_without_family(cofire_pairs, coexpr_pairs, families):
    cofire = sum(not excluded(pair, families) for pair in cofire_pairs)
    coexpr = sum(not excluded(pair, families) for pair in coexpr_pairs)
    return {
        "cofire_only": cofire,
        "coexpr_only": coexpr,
        "binom_p_two_sided": paired_p(cofire, coexpr),
    }
