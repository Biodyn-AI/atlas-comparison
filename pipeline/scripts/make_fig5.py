#!/usr/bin/env python
"""Fig 5: SAE features as a hypothesis generator, tested on held-out regulatory edges.
   A  per-model enrichment of TRRUST recovery over a degree-preserving null
   B  precision rises when independent models corroborate the same pair
   python scripts/make_fig5.py -> outputs/atlas/comparative/figures/Fig5_hypothesis.png
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
OUT = f"{C}/figures/Fig5_hypothesis.png"
DISP = {"AIDO": "AIDO.Cell", "C2S": "C2S-Scale", "Geneformer": "Geneformer-V2",
        "Tahoe": "Tahoe-x1", "scFoundation": "scFoundation", "GeneCompass": "GeneCompass",
        "MaxToki": "MaxToki", "UCE": "UCE", "scGPT": "scGPT", "tGPT": "tGPT"}
A = json.load(open(f"{C}/hypothesis_trrust2.json"))
F = json.load(open(f"{C}/hypothesis_final.json"))
per = A["per_model"]; curve = F["cross_model_curve"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- A: per-model enrichment ----
ms = sorted(per, key=lambda m: -(per[m]["fold"] or 0))
fold = [per[m]["fold"] or 0 for m in ms]
sig = [per[m]["p_emp"] <= 0.05 for m in ms]
cols = ["#2f7ed8" if s else "#c9ccd1" for s in sig]
y = np.arange(len(ms))[::-1]
ax1.barh(y, fold, color=cols, height=.68)
ax1.axvline(1, color="#444", lw=.9, ls="--", zorder=0)
ax1.set_yticks(y); ax1.set_yticklabels([DISP.get(m, m) for m in ms], fontsize=8)
ax1.set_xlabel("held-out TRRUST recovery, fold over degree-matched null")
for yi, m in zip(y, ms):
    d = per[m]
    ax1.text(max(d["fold"] or 0, 0) + .25, yi, f"{d['hits']} hits", va="center", fontsize=7,
             color="#2f7ed8" if d["p_emp"] <= 0.05 else "#8a8f96")
ax1.set_ylim(-0.7, len(ms) - 0.3)
ax1.set_xlim(0, max(fold) * 1.28)
ax1.set_title("A   Which models generate testable predictions", loc="left", fontsize=9, fontweight="bold")
ax1.text(.98, .04, "blue: p ≤ 0.05 (200 permutations)", transform=ax1.transAxes,
         ha="right", fontsize=6.8, color="#5a6169")

# ---- B: precision vs cross-model corroboration ----
ks = sorted(int(k) for k in curve)
ks = [k for k in ks if curve[str(k)]["hits"] > 0]
prec = [curve[str(k)]["precision"] for k in ks]
null = [curve[str(k)]["null_precision_mean"] for k in ks]
npairs = [curve[str(k)]["n_pairs"] for k in ks]
ax2.plot(ks, prec, "-o", color="#2f7ed8", lw=1.7, ms=5, label="observed", zorder=3)
nz = [(k, n) for k, n in zip(ks, null) if n > 0]                 # a log axis cannot show null = 0
ax2.plot([k for k, _ in nz], [n for _, n in nz], "-s", color="#b0b5bb", lw=1.4, ms=4,
         label="degree-matched null")
ax2.set_yscale("log")
ax2.set_xticks(ks)
ax2.set_xlabel("models independently predicting the pair")
ax2.set_ylabel("precision on held-out TRRUST")
ax2.legend(frameon=False, fontsize=7, loc="lower right")
for k, p, n in zip(ks, prec, npairs):
    f = curve[str(k)]["fold"]
    ax2.annotate(f"{f:g}×" if f else "null = 0", (k, p), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=7.5, color="#1b4f8a", fontweight="bold")
    ax2.annotate(f"n={n:,}", (k, p), textcoords="offset points", xytext=(0, -13),
                 ha="center", fontsize=6.5, color="#5a6169")
ax2.set_title("B   Agreement between models raises precision", loc="left", fontsize=9, fontweight="bold")
ax2.margins(y=.35)

fig.tight_layout(pad=.7)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
