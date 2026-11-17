"""Compare the trained and random-SAE cores using the strict middle-layer annotation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backbone_composition"))
from bb_common import Annotator, ORDER


def core(root, annotator):
    counts = Counter()
    for model in ORDER:
        catalogues = sorted((root / model).glob("feature_catalog_L*.json"))
        if not catalogues:
            raise FileNotFoundError(f"No layer catalogues for {model} under {root}")
        features = json.loads(catalogues[len(catalogues) // 2].read_text())["features"]
        terms = annotator.annotate_fast([f["top_genes"] for f in features.values()])
        counts.update(terms)
    return {term for term, n in counts.items() if n >= 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trained", type=Path)
    parser.add_argument("random_sae", type=Path)
    args = parser.parse_args()
    annotator = Annotator(200)
    trained = core(args.trained, annotator)
    random = core(args.random_sae, annotator)
    result = {"real_core": sorted(trained), "random_sae_core": sorted(random), "overlap": len(trained & random)}
    (HERE / "core_overlap.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
