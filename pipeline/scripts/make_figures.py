#!/usr/bin/env python
"""Publication figures 2-3 for the manuscript (static, colorblind-safe Okabe-Ito, PNG 300dpi + PDF).
Fig 2: the artefact and its fix (permissive core < random null; calibrated backbone vs uniform + degree-matched
       nulls; held-out replication). Fig 3: the backbone (universality spectrum real vs null; concept categories;
       example programmes)."""
from __future__ import annotations
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
FIG = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative/figures"
import os; os.makedirs(FIG, exist_ok=True)
d = json.load(open(f"{C}/controls.json"))

# Okabe-Ito colorblind-safe
BLUE, ORANGE, GREEN, VERM, GREY_L, GREY_D, INK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CFCFCF", "#7A7A7A", "#222222"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#888",
                     "axes.linewidth": 0.8, "axes.grid": False, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# ============ FIG 2 ============
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6)); fig.subplots_adjust(wspace=0.42)

# 2A: permissive artefact — real core BELOW random-gene null
rn = d["random_null"]
a = ax[0]
bars = a.bar([0, 1], [rn["real_core"], rn["rand_core_mean"]], color=[BLUE, GREY_D], width=0.62,
             yerr=[0, rn["rand_core_sd"]], error_kw=dict(ecolor="#444", lw=1, capsize=3))
a.set_xticks([0, 1]); a.set_xticklabels(["real\nfeatures", "random-gene\nnull"])
a.set_ylabel("shared concepts (all 10 models)")
a.set_title("A  Naive annotation: an artefact", loc="left", fontweight="bold", fontsize=10)
for x, v in zip([0, 1], [rn["real_core"], rn["rand_core_mean"]]):
    a.text(x, v + 20, f"{v:.0f}", ha="center", fontsize=9, color=INK)
a.set_ylim(0, 720)
a.annotate("real BELOW null\n(z = %.1f)" % rn["core_z"], xy=(0, rn["real_core"]), xytext=(0.35, 500),
           fontsize=8.5, color=VERM, ha="center",
           arrowprops=dict(arrowstyle="->", color=VERM, lw=1))

# 2B: calibrated backbone vs uniform + degree-matched null (log scale)
b = ax[1]
KS = [">=7", ">=8", ">=9", ">=10"]; xlab = ["≥7", "≥8", "≥9", "≥10"]
real = [d["stats_final"]["backbone_null"][k]["real"] for k in KS]
uni = [max(d["stats_final"]["backbone_null"][k]["null_mean"], 0.05) for k in KS]
deg = [max(d["degree_null"]["tiers"][k]["degree_null_mean"], 0.05) for k in KS]
x = np.arange(4); w = 0.26
b.bar(x - w, real, w, color=BLUE, label="real")
b.bar(x, uni, w, color=GREY_L, label="uniform null")
b.bar(x + w, deg, w, color=GREY_D, label="degree-matched null")
b.set_yscale("log"); b.set_ylim(0.04, 400)
b.set_xticks(x); b.set_xticklabels(xlab); b.set_xlabel("shared by ≥ k of 10 models")
b.set_ylabel("concepts (log)")
b.set_title("B  Backbone vs nulls (log)", loc="left", fontweight="bold", fontsize=10)
for i, k in enumerate(KS):
    fold = d["degree_null"]["tiers"][k]["degree_fold"]
    b.text(i - w, real[i] * 1.25, f"{real[i]}", ha="center", fontsize=8, color=INK)
    b.text(i - w, real[i] * 1.9, f"{fold:.0f}×", ha="center", fontsize=8, color=VERM, fontweight="bold")
b.legend(frameon=False, fontsize=7.5, loc="upper right")
b.text(0.02, 0.02, "empirical p < 0.004 (250 perms); ×N = real ÷ degree-matched null", transform=b.transAxes, fontsize=7, color="#666")

# 2C: held-out replication
c = ax[2]
ho = d["stats_final"]["heldout_null"]
c.bar([0, 1], [ho["observed_jaccard"], ho["null_mean"]], color=[GREEN, GREY_D], width=0.62,
      yerr=[0, ho["null_sd"]], error_kw=dict(ecolor="#444", lw=1, capsize=3))
c.axhline(ho["null_max"], ls=":", color="#999", lw=1)
c.text(1.35, ho["null_max"], "null max", fontsize=7, color="#777", va="center")
c.set_xticks([0, 1]); c.set_xticklabels(["observed\n(held-out)", "random\ntwo-draw null"])
c.set_ylabel("backbone Jaccard (orig ↔ held-out)")
c.set_title("C  Held-out replication", loc="left", fontweight="bold", fontsize=10)
for xx, v in zip([0, 1], [ho["observed_jaccard"], ho["null_mean"]]):
    c.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
c.set_ylim(0, 0.62)
c.text(0.02, 0.94, f"p < {1/(d['stats_final']['n_perm_heldout']+1):.3f}, {ho['fold']}× over null",
       transform=c.transAxes, fontsize=7.5, color="#666")
save(fig, "Fig2_artefact_and_fix")

# ============ FIG 3 ============
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6)); fig.subplots_adjust(wspace=0.42)

# 3A: universality spectrum (concepts shared by exactly k) real vs null, calibrated
rf = d["recalibration"]["final"]
sp = rf["real_spectrum"]; ns = rf["null_spectrum_mean"]
ks = list(range(1, 11))
realv = [sp[str(k)] for k in ks]; nullv = [max(ns[str(k)], 0.05) for k in ks]
a = ax[0]
a.plot(ks, [max(v, 0.05) for v in realv], "-o", color=BLUE, ms=4, lw=1.8, label="real")
a.plot(ks, nullv, "--s", color=GREY_D, ms=3.5, lw=1.4, label="random-gene null")
a.set_yscale("log"); a.set_ylim(0.04, 2000)
a.set_xticks(ks); a.set_xlabel("shared by exactly k models"); a.set_ylabel("concepts (log)")
a.set_title("A  Sharing spectrum (calibrated)", loc="left", fontweight="bold", fontsize=10)
a.axvspan(7.5, 10.5, color=GREEN, alpha=0.08)
a.text(9, 900, "backbone\n(≥8)", fontsize=8, color=GREEN, ha="center")
a.legend(frameon=False, fontsize=7.5, loc="upper left")

# 3B: concept categories (specific vs housekeeping)
triv = d["recalibration"]["robust"]["triviality"]
hk = triv["pct_housekeeping"]; spc = 100 - hk; n = triv["n_ge8"]
b = ax[1]
wedges, _ = b.pie([spc, hk], colors=[GREEN, ORANGE], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
b.text(0, 0, f"{n}\nconcepts", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
b.set_title("B  Backbone is mostly specific biology", loc="left", fontweight="bold", fontsize=10)
b.legend([f"specific programmes  {spc:.0f}%", f"housekeeping  {hk:.0f}%"],
         frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.18))

# 3C: example backbone programmes
c = ax[2]; c.axis("off")
exs = ["Cytoplasmic translation", "SRP / ER protein targeting", "Nonsense-mediated mRNA decay",
       "Antigen processing (MHC-II)", "Defense response to bacterium", "Cytokine production / response",
       "Muscle contraction", "Adrenergic cardiac signalling"]
c.set_title("C  Example shared programmes (≥8/10)", loc="left", fontweight="bold", fontsize=10)
for i, e in enumerate(exs):
    col = ORANGE if i < 3 else GREEN
    c.text(0.02, 0.9 - i * 0.11, "▪", color=col, fontsize=11, transform=c.transAxes, va="center")
    c.text(0.09, 0.9 - i * 0.11, e, fontsize=9, transform=c.transAxes, va="center", color=INK)
yleg = 0.9 - len(exs) * 0.11
c.text(0.02, yleg, "▪", color=ORANGE, fontsize=10, transform=c.transAxes, va="center")
c.text(0.07, yleg, "housekeeping", fontsize=7.5, color="#777", transform=c.transAxes, va="center")
c.text(0.52, yleg, "▪", color=GREEN, fontsize=10, transform=c.transAxes, va="center")
c.text(0.57, yleg, "specific", fontsize=7.5, color="#777", transform=c.transAxes, va="center")
save(fig, "Fig3_backbone")
print("==> figures in", FIG)
