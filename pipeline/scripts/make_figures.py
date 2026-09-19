#!/usr/bin/env python
"""Publication figures 2-3 for the manuscript (static, colorblind-safe Okabe-Ito, PNG 300dpi + PDF).
Fig 2: the artefact and its fix (permissive core < random null; calibrated backbone vs uniform + degree-matched
       nulls; held-out replication). Fig 3: the backbone (universality spectrum real vs null; concept categories;
       example programmes)."""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

C = f"{_B}/outputs/atlas/comparative"
FIG = f"{_B}/outputs/atlas/comparative/figures"
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
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9)); fig.subplots_adjust(wspace=0.42, bottom=0.30)

# 2A: permissive artefact — real core BELOW random-gene null
rn = d["random_null"]
a = ax[0]
bars = a.bar([0, 1], [rn["real_core"], rn["rand_core_mean"]], color=[BLUE, GREY_D], width=0.62,
             yerr=[0, rn["rand_core_sd"]], error_kw=dict(ecolor="#444", lw=1, capsize=3))
a.set_xticks([0, 1]); a.set_xticklabels(["real\nfeatures", "random-gene\nnull"])
a.set_ylabel("shared concepts (all 10 models)")
AL = json.load(open(f"{C}/alllayer_null.json"))
a.set_title("A  Naive annotation: an artefact", loc="left", fontweight="bold", fontsize=10)
# clear the error-bar cap and the annotation arrow, which both used to strike through these labels
for x, v, sd in zip([0, 1], [rn["real_core"], rn["rand_core_mean"]], [0, rn["rand_core_sd"]]):
    a.text(x, v + sd + 26, f"{v:.0f}", ha="center", fontsize=9, color=INK)
a.set_ylim(0, 760)
a.text(0.02, 0.965, f"error bar: ±1 SD over {rn['n_perm']} permutations", transform=a.transAxes,
       fontsize=7, color="#666", va="top")
a.text(0.0, -0.22, "permissive annotator, mid layer\n"
                   f"all-layer union, same test: {AL['real_core']:,} vs {AL['null_core_mean']:,.0f} ± "
                   f"{AL['null_core_sd']:.0f} (z = {AL['core_z']})", transform=a.transAxes,
       fontsize=7, color="#666", va="top")
a.annotate("real BELOW null\n(z = %.1f)" % rn["core_z"], xy=(0.22, rn["real_core"] + 20), xytext=(0.40, 440),
           fontsize=8.5, color=VERM, ha="center",
           arrowprops=dict(arrowstyle="->", color=VERM, lw=1))

# 2B: calibrated backbone vs uniform + degree-matched null (log scale)
b = ax[1]
KS = [">=7", ">=8", ">=9", ">=10"]; xlab = ["≥7", "≥8", "≥9", "≥10"]
real = [d["stats_final"]["backbone_null"][k]["real"] for k in KS]
# plot the null means as measured. They were floored at 0.05 for display, which drew the ≥10
# uniform (0.03) and degree-matched (0.01) nulls as the same bar though they differ three-fold.
uni = [d["stats_final"]["backbone_null"][k]["null_mean"] for k in KS]
deg = [d["degree_null"]["tiers"][k]["degree_null_mean"] for k in KS]
FLOOR = 0.1   # degree_null.py divides by max(null, FLOOR), so a fold below it is a lower bound
# Points, not bars: on a log axis a bar's length is not proportional to its value and its baseline
# is arbitrary, so the same data as bars overstated the small nulls. Whiskers are ±1 SD across the
# 250 permutations, clipped at the axis floor where the SD of a near-zero null runs below it.
usd = [d["stats_final"]["backbone_null"][k]["null_sd"] for k in KS]
dsd = [d["degree_null"]["tiers"][k]["degree_null_sd"] for k in KS]
YLO = 0.004
x = np.arange(4); off = 0.17
def _err(vals, sds):
    lo = [v - max(v - s, YLO) for v, s in zip(vals, sds)]   # clipped lower arm
    return [lo, sds]
b.errorbar(x, uni, yerr=_err(uni, usd), fmt="s", ms=5, color=GREY_D, mfc=GREY_L, mew=1.1,
           capsize=3, lw=1, label="uniform null")
b.errorbar(x + off, deg, yerr=_err(deg, dsd), fmt="^", ms=5.5, color="#3f3f3f", mfc=GREY_D, mew=1.1,
           capsize=3, lw=1, label="degree-matched null")
b.plot(x - off, real, "o", ms=7, color=BLUE, label="real", zorder=3)
for i in range(4):   # the gap each tier's result spans, drawn once per tier
    b.plot([x[i] - off, x[i] - off], [max(uni[i], YLO), real[i]], "-", color=BLUE, lw=1, alpha=0.35, zorder=2)
b.set_yscale("log"); b.set_ylim(YLO, 900)
b.set_xlim(-0.5, 3.6)
b.set_xticks(x); b.set_xticklabels(xlab); b.set_xlabel("shared by ≥ k of 10 models")
b.set_ylabel("concepts (log)")
b.set_title("B  Backbone vs nulls (log)", loc="left", fontweight="bold", fontsize=10, pad=22)
for i, k in enumerate(KS):
    ufold = d["stats_final"]["backbone_null"][k]["fold"]
    dfold = d["degree_null"]["tiers"][k]["degree_fold"]
    ge = "≥" if deg[i] < FLOOR else ""      # the stored fold divided by the floor, so it under-states
    b.text(i - off, real[i] * 1.5, f"{real[i]}", ha="center", fontsize=8, color=INK)
    b.text(i - off, real[i] * 2.6, f"{ufold:.1f}× / {ge}{dfold:.0f}×", ha="center", fontsize=7.6,
           color=VERM, fontweight="bold")
b.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=3,
         bbox_to_anchor=(0.0, 1.02), borderaxespad=0, columnspacing=1.4, handlelength=1.2)
b.text(0.0, -0.30, "250 permutations, empirical p < 0.004 at every tier;  whiskers ±1 SD, clipped at the axis floor;\n"
                  "fold = real ÷ uniform null / real ÷ degree-matched null (a null mean below 0.1 is divided as 0.1, so ≥ is a lower bound)",
       transform=b.transAxes, fontsize=7, color="#666")   # below the axes: the bars ran through it inside

# 2C: held-out replication
c = ax[2]
ho = d["stats_final"]["heldout_null"]
c.bar([0, 1], [ho["observed_jaccard"], ho["null_mean"]], color=[GREEN, GREY_D], width=0.62,
      yerr=[0, ho["null_sd"]], error_kw=dict(ecolor="#444", lw=1, capsize=3))
c.axhline(ho["null_max"], ls=":", color="#999", lw=1)
c.text(1.46, ho["null_max"] + 0.012, "null max", fontsize=7, color="#777", va="bottom", ha="right")
c.set_xticks([0, 1]); c.set_xticklabels(["observed\n(held-out)", "random\ntwo-draw null"])
c.set_ylabel("backbone Jaccard (orig ↔ held-out)")
c.set_title("C  Held-out replication", loc="left", fontweight="bold", fontsize=10)
for xx, v in zip([0, 1], [ho["observed_jaccard"], ho["null_mean"]]):
    c.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
c.set_ylim(0, 0.62)
_nho = len(d.get("recalibration", {}).get("heldout_calibrated", {}).get("per_model", {})) or 9
c.text(0.02, 0.965, f"{_nho} of 10 models · p < {1/(d['stats_final']['n_perm_heldout']+1):.3f}, "
                    f"{ho['fold']}× over null\nerror bar: ±1 SD over {d['stats_final']['n_perm_heldout']} permutations",
       transform=c.transAxes, fontsize=7, color="#666", va="top")
save(fig, "Fig2_artefact_and_fix")

# ============ FIG 3 ============
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6)); fig.subplots_adjust(wspace=0.42)

# 3A: universality spectrum (concepts shared by exactly k) real vs null, calibrated
rf = d["recalibration"]["final"]
sp = rf["real_spectrum"]; ns = rf["null_spectrum_mean"]
ks = list(range(1, 11))
realv = [sp[str(k)] for k in ks]; nullv = [ns[str(k)] for k in ks]
a = ax[0]
# as measured: the old floor of 0.05 drew a null that never occurred in 250 permutations as if it had
SPEC_LO = 0.3
a.plot(ks, realv, "-o", color=BLUE, ms=4, lw=1.8, label="real")
_kv = [(k, v) for k, v in zip(ks, nullv) if v > 0]
a.plot([k for k, _ in _kv], [v for _, v in _kv], "--s", color=GREY_D, ms=3.5, lw=1.4, label="random-gene null")
for k, v in zip(ks, nullv):      # a null of exactly zero has no place on a log axis; say so instead
    if v == 0:
        a.plot([k], [SPEC_LO], "s", ms=3.5, mfc="white", mec=GREY_D, mew=1.1, clip_on=False)
        a.text(k, SPEC_LO * 1.35, "null = 0", fontsize=6.5, color=GREY_D, ha="center")
a.set_yscale("log"); a.set_ylim(SPEC_LO, 3000)
a.set_xticks(ks); a.set_xlabel("shared by exactly k models"); a.set_ylabel("concepts (log)")
a.set_title("A  Sharing spectrum (calibrated)", loc="left", fontweight="bold", fontsize=10)
a.axvspan(7.5, 10.5, color=GREEN, alpha=0.08)
a.text(9, 1300, "backbone\n(≥8)", fontsize=8, color=GREEN, ha="center")
a.text(0.0, -0.30, "null = mean over 250 permutations", transform=a.transAxes, fontsize=7, color="#666")
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
# read from the backbone itself, most widely shared first, under the paper's own housekeeping rule —
# these used to be eight paraphrases typed into this script, which no check could catch drifting
import re as _re
_BT = json.load(open(f"{C}/backbone_terms.json"))["tiers"][">=8"]
_HKW = _re.search(r"HK\s*=\s*\[(.*?)\]", open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "recalibrate_robust.py")).read(), _re.S).group(1)
_HK = [x.strip(" \"'") for x in _re.findall(r'"[^"]+"|\'[^\']+\'', _HKW)]
def _is_hk(t): return any(h in t.lower() for h in _HK)
def _clean(t):  # drop the accession, keep the name the source prints
    t = _re.sub(r"\s*\((GO|KEGG|R-HSA)[:\d]+\)\s*$", "", t)
    t = _re.sub(r"\s+R-HSA-\d+$", "", t).strip()
    return t if len(t) <= 30 else t[:28].rstrip(" ,") + "…"
_seen, _uniq = set(), []
for t in sorted(_BT, key=lambda t: (-t["n_models"], t["term"].lower())):
    key = _clean(t["term"]).lower()          # GO and Reactome both carry "muscle contraction"
    if key in _seen:
        continue
    _seen.add(key); _uniq.append(t)
# five specific and three housekeeping, the split the backbone itself has (63 % / 37 %)
_sp = [t for t in _uniq if not _is_hk(t["term"])][:5]
_hk = [t for t in _uniq if _is_hk(t["term"])][:3]
exs = [(_clean(t["term"]), t["n_models"], _is_hk(t["term"]))
       for t in sorted(_sp + _hk, key=lambda t: (-t["n_models"], t["term"].lower()))]
c.set_title("C  Most widely shared programmes", loc="left", fontweight="bold", fontsize=10)
for i, (name, nmod, is_hk) in enumerate(exs):
    col = ORANGE if is_hk else GREEN          # the rule decides the colour, not the row's position
    y = 0.9 - i * 0.11
    c.text(0.02, y, "▪", color=col, fontsize=11, transform=c.transAxes, va="center")
    c.text(0.09, y, f"{nmod}/10", fontsize=7.5, transform=c.transAxes, va="center", color="#777")
    c.text(0.26, y, name, fontsize=8.5, transform=c.transAxes, va="center", color=INK)
yleg = 0.9 - len(exs) * 0.11
c.text(0.02, yleg, "▪", color=ORANGE, fontsize=10, transform=c.transAxes, va="center")
c.text(0.07, yleg, "housekeeping", fontsize=7.5, color="#777", transform=c.transAxes, va="center")
c.text(0.52, yleg, "▪", color=GREEN, fontsize=10, transform=c.transAxes, va="center")
c.text(0.57, yleg, "specific", fontsize=7.5, color="#777", transform=c.transAxes, va="center")
save(fig, "Fig3_backbone")
print("==> figures in", FIG)
