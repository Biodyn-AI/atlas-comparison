"""Recount the name and structural-family masks in the trained and untrained controls."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from family_mask import excluded, gene_families


def load(path):
    return json.loads(path.read_text())


def known_hits(root, model, truth):
    catalogues = sorted((root / model).glob("feature_catalog_L*.json"))
    if not catalogues:
        raise FileNotFoundError(f"No layer catalogues for {model} under {root}")
    counts = Counter()
    for catalogue in catalogues:
        for feature in load(catalogue)["features"].values():
            genes = sorted({g.upper() for g in feature["top_genes"][:10] if not g.upper().startswith("ENSG")})
            counts.update(pair for pair in combinations(genes, 2) if pair in truth)
    return [pair for pair, n in counts.items() if n >= 5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trained", type=Path)
    parser.add_argument("untrained_model", type=Path)
    parser.add_argument("random_sae", type=Path)
    args = parser.parse_args()
    base = Path(os.environ.get("ATLAS_BASE", HERE.parents[2]))
    raw = load(base / "outputs/atlas/genesets/trrust_edges.json")
    truth = {tuple(sorted((a.upper(), b.upper()))) for a, targets in raw.items()
             for b in targets if a.upper() != b.upper()}
    families = gene_families()
    summary, uce = {}, {}
    for condition, root in vars(args).items():
        for model in ("UCE", "Geneformer"):
            hits = known_hits(root, model, truth)
            masked = sum(excluded(pair, families) for pair in hits)
            summary[f"{condition}:{model}"] = {"hits": len(hits), "family_or_lookalike": masked}
            if model == "UCE":
                uce[condition] = {
                    "hits": len(hits),
                    "M1": sum(a[:2] == b[:2] for a, b in hits),
                    "M2": sum(a in families and families[a] == families.get(b) for a, b in hits),
                    "M1_or_M2": masked,
                    "both_genes_in_map": sum(a in families and b in families for a, b in hits),
                }
    for name, result in (("hits_family_split", summary), ("uce_family_masks", uce)):
        (HERE / f"{name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
