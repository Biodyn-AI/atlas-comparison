#!/usr/bin/env python
"""Coarse module-theme matrix (9 themes x N models) for the 'axes meaning' heatmap. Classifies each
co-activation module's top label (mod_labels in modules_alllayers.json) into one of 9 themes by keyword,
summed across all layers. Rebuilds module_themes.json for all models present (adds Tahoe).
    python scripts/module_themes.py
"""
from __future__ import annotations
import json, re

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THEMES = ["translation", "immune", "mito/OXPHOS", "cell-cycle/DNA", "RNA processing",
          "metabolism", "membrane/traffic", "signaling", "tissue-specific"]
# priority-ordered keyword rules (first match wins)
RULES = [
    ("translation", r"translat|ribosom|peptide chain elong|aminoacyl|rrna process|eukaryotic translation|elongation"),
    ("cell-cycle/DNA", r"cell cycle|mitotic|dna replicat|dna repair|chromosom|spindle|kinetochore|s phase|g2/m|cell division|dna metabolic|telomere|nucleosome"),
    ("RNA processing", r"splic|spliceosome|mrna|rna processing|snrna|capping|polyadenyl|transcription|rna polymerase|nonsense-mediated"),
    ("mito/OXPHOS", r"respiratory electron|atp synth|oxidative phosph|mitochond|tca|citric acid|electron transport|\bmt-|oxphos|complex i\b"),
    ("immune", r"immune|neutrophil|interferon|cytokine|\bmhc\b|antigen|complement|inflamm|lymphocyte|t cell|b cell|interleukin|degranulation|leukocyte|nf-kappa"),
    ("membrane/traffic", r"transport|endocytos|golgi|vesicle|slc-mediated|transmembrane|secretion|traffick|exocytos|endosom|lysosom|clathrin"),
    ("signaling", r"signal|mapk|kinase|receptor|\bwnt\b|notch|gpcr|phosphoryl|pathway|erk|akt|calcium|rho gtpase"),
    ("metabolism", r"metabol|biosynth|catabol|glycol|lipid|fatty acid|cholesterol|amino acid|nucleotide|steroid|glucose|heme|oxidored"),
    ("immune", r"hla-|\bil\d|\bccl\d|\bcxcl\d|ifit|isg"),  # gene-symbol immune fallback
]


def theme_of(lab: str) -> str:
    s = (lab or "").lower()
    for th, pat in RULES:
        if re.search(pat, s):
            return th
    return "tissue-specific"  # bare gene symbols / tissue markers / unclassified


def main():
    g = json.load(open(f"{C}/modules_alllayers.json"))["graphs"]
    models = [m for m in ORDER if m in g]
    Z = [[0] * len(models) for _ in THEMES]
    ti = {t: i for i, t in enumerate(THEMES)}
    for j, m in enumerate(models):
        layers = sorted(g[m], key=lambda x: int(x))
        if len(layers) > 6:  # depth-match to 5 layers (0/25/50/75/100%) for parity across models
            layers = [layers[round(x * (len(layers) - 1) / 4)] for x in range(5)]
            layers = sorted(set(layers), key=lambda x: int(x))
        for L in layers:
            for lab in (g[m][L].get("mod_labels") or {}).values():
                Z[ti[theme_of(lab)]][j] += 1
    out = {"themes": THEMES, "models": models, "Z": Z}
    json.dump(out, open(f"{C}/module_themes.json", "w"))
    print(f"==> module_themes.json ({len(models)} models)")
    for i, t in enumerate(THEMES):
        print(f"  {t:18s} {Z[i]}")


if __name__ == "__main__":
    main()
