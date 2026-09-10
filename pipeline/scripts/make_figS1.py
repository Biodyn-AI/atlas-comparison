#!/usr/bin/env python
"""Supplementary Fig S1 — capacity-matched SAE-vs-SVD, projection across k (answers reviewer R2-Major-2).
(A) mean projected energy of SAE decoder directions onto the top-k PCA subspace vs k (log2 x), per model, with each
    model's isotropic random baseline k/d as a faint dotted line; (B) enrichment = SAE / (k/d), i.e. how many times
    more concentrated than random, vs k. Colorblind-safe (Okabe-Ito), PNG 300dpi + PDF."""
from __future__ import annotations
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
FIG = f"{C}/figures"
d = json.load(open(f"{C}/svd_projection_k.json"))

# Okabe-Ito per model (distinct, colorblind-safe)
COL = {"AIDO": "#E69F00", "C2S": "#009E73", "Geneformer": "#56B4E9", "MaxToki": "#CC79A7",
       "UCE": "#0072B2", "scGPT": "#D55E00", "tGPT": "#117733", "scFoundation": "#882255",
       "GeneCompass": "#AA4499", "Tahoe": "#333333"}
LABEL = {"C2S": "C2S-Scale", "Tahoe": "Tahoe-x1"}
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "svg.fonttype": "none"})

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0)); fig.subplots_adjust(wspace=0.28)
a, b = ax

for r in d:
    m = r["model"]; col = COL.get(m, "#777"); lab = LABEL.get(m, m)
    ks = sorted((int(k) for k in r["curve"]))
    sae = [r["curve"][str(k)]["sae_proj"] for k in ks]
    rnd = [r["curve"][str(k)]["random_baseline"] for k in ks]
    ratio = [s / max(x, 1e-9) for s, x in zip(sae, rnd)]
    a.plot(ks, sae, "-o", color=col, lw=1.5, ms=3.2, label=lab)
    a.plot(ks, rnd, ":", color=col, lw=0.9, alpha=0.55)
    b.plot(ks, ratio, "-o", color=col, lw=1.5, ms=3.2, label=lab)

for x in (a, b):
    x.set_xscale("log", base=2); x.set_xlabel("k  (top-k PCA subspace)")
    x.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256]); x.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256], fontsize=7.5)
a.set_ylabel("mean SAE-direction energy in top-k subspace"); a.set_ylim(0, 1.02)
a.axvline(32, color="#bbb", lw=0.8, ls="--"); a.text(33, 0.96, "k=32", color="#999", fontsize=7)
a.set_title("A  Projection onto top-k PCA subspace", loc="left", fontweight="bold", fontsize=9.5)
a.text(0.03, 0.60, "dotted = isotropic\nrandom baseline k/d", transform=a.transAxes, fontsize=7, color="#777")
a.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)

b.axhline(1.0, color="#bbb", lw=0.9, ls="--"); b.text(1.1, 1.05, "random (=1×)", color="#999", fontsize=7)
b.set_ylabel("enrichment over random  (SAE ÷ k/d)")
b.axvline(32, color="#bbb", lw=0.8, ls="--")
b.set_title("B  Concentration vs isotropic directions", loc="left", fontweight="bold", fontsize=9.5)
b.text(0.40, 0.90, "SAE directions align with high-variance axes\n(>1× at every k), yet only a minority of their\nenergy sits in the matched-capacity (k=32) subspace",
       transform=b.transAxes, fontsize=6.8, color="#555", va="top")

fig.suptitle("Fig S1   Capacity-matched SAE vs SVD: SAE directions are reachable by SVD given enough axes, "
             "not invisible to it", x=0.02, ha="left", fontweight="bold", fontsize=10)
fig.savefig(f"{FIG}/FigS1_svd_projection.png", bbox_inches="tight", dpi=300)
fig.savefig(f"{FIG}/FigS1_svd_projection.pdf", bbox_inches="tight")
plt.close(fig)
print("==> FigS1_svd_projection .png/.pdf in", FIG)
