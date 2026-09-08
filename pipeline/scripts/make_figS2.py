#!/usr/bin/env python
"""Supplementary Fig S2 — backbone robustness to layer choice. (A) ≥8/10 calibrated backbone size, real vs random-gene
null, across depth fractions, with fold-over-null annotated; (B) Jaccard of ≥8 concept membership vs the mid-depth set,
showing the phenomenon is depth-invariant while exact membership drifts. Colorblind-safe, PNG 300dpi + PDF."""
from __future__ import annotations
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
FIG = f"{C}/figures"
d = json.load(open(f"{C}/depth_backbone.json"))
BLUE, GREY_D, GREEN, VERM, INK = "#0072B2", "#7A7A7A", "#009E73", "#D55E00", "#222222"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "svg.fonttype": "none"})

rows = sorted(d["depths"], key=lambda r: r["frac"])
f = [r["frac"] for r in rows]
real = [r["tiers"][">=8"]["real"] for r in rows]
null = [r["tiers"][">=8"]["null"] for r in rows]
fold = [r["tiers"][">=8"]["fold"] for r in rows]
jac = [r["jaccard_vs_mid"] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0)); fig.subplots_adjust(wspace=0.30)
a, b = ax

x = np.arange(len(f)); w = 0.38
a.bar(x - w/2, real, w, color=BLUE, label="real backbone (≥8/10)")
a.bar(x + w/2, null, w, color=GREY_D, label="random-gene null")
for i in range(len(f)):
    a.text(i - w/2, real[i] + 2, f"{real[i]}", ha="center", fontsize=8, color=INK)
    a.text(i - w/2, real[i] + 9, f"{fold[i]:.0f}×", ha="center", fontsize=8, color=VERM, fontweight="bold")
a.set_xticks(x); a.set_xticklabels([f"{v:g}" for v in f])
a.set_xlabel("read-out depth (fraction)"); a.set_ylabel("≥8/10 shared concepts")
a.set_ylim(0, max(real) * 1.22)
a.set_title("A  Backbone is significant at every depth", loc="left", fontweight="bold", fontsize=9.5)
a.legend(frameon=False, fontsize=7.5, loc="upper right")
a.text(0.30, 0.92, "×N = real ÷ random-gene null (N=20 perms)", transform=a.transAxes, fontsize=7, color="#666")

b.plot(f, jac, "-o", color=GREEN, lw=1.6, ms=5)
b.axhline(1.0, color="#bbb", lw=0.8, ls="--")
b.set_ylim(0, 1.05); b.set_xlabel("read-out depth (fraction)")
b.set_ylabel("Jaccard of ≥8 membership vs mid")
b.set_title("B  Exact membership drifts with depth", loc="left", fontweight="bold", fontsize=9.5)
for xi, yi in zip(f, jac):
    b.text(xi, yi + 0.03, f"{yi:.2f}", ha="center", fontsize=7.5, color=INK)
b.text(0.02, 0.06, "existence & significance depth-invariant;\nconcept identity evolves across layers",
       transform=b.transAxes, fontsize=7, color="#555")

fig.suptitle("Fig S2   The shared backbone is robust to layer choice (a depth-invariant phenomenon, mid-depth-specific membership)",
             x=0.02, ha="left", fontweight="bold", fontsize=10)
fig.savefig(f"{FIG}/FigS2_depth_backbone.png", bbox_inches="tight", dpi=300)
fig.savefig(f"{FIG}/FigS2_depth_backbone.pdf", bbox_inches="tight")
plt.close(fig)
print("==> FigS2_depth_backbone .png/.pdf in", FIG)
