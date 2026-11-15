"""Figure 5: plain co-expression from the same cells passes the same TRRUST test.
A times chance against network size; B known pairs at equal size and genes; C paired comparison of the
networks matched for genes and partner counts; D shared-feature pairs are co-expressed."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from style import WIDTH, setup, letter, fletter, out, an, save, ORDER, NAME, COL, COEX

setup()
mpl.rcParams.update({"font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "legend.fontsize": 6.6})
pm = out("hypothesis_trrust2.json")["per_model"]
ct = an("coexpression_baseline/coexpr_trrust.json"); K = ct["kinds"]
dg = an("coexpression_baseline/coexpr_degmatched.json")["models"]
rc = an("review_10_coexpr/review_cheap.json")["paired_degmatched"]
hv = an("review_10_coexpr/review_heavy.json")["P1_recount_6000"]["degmatched"]
import os
from style import AN
lwf = json.load(open(f"{AN}/review_10_coexpr/lead_without_family.json"))["models"] if os.path.exists(f"{AN}/review_10_coexpr/lead_without_family.json") else {}
PASS = [m for m in sorted(pm, key=lambda m: -pm[m]["fold"]) if pm[m]["p_emp"] < 0.05]
MK = {"AIDO": "v", "scGPT": "o", "scFoundation": "P", "GeneCompass": "X", "tGPT": "<", "MaxToki": ">",
      "Geneformer": "D", "UCE": "s", "C2S": "^", "Tahoe": "h"}

fig = plt.figure(figsize=(WIDTH, 520 / 72))
gA = fig.add_gridspec(1, 1, left=0.105, right=0.56, bottom=0.585, top=0.95)
gB = fig.add_gridspec(1, 1, left=0.70, right=0.99, bottom=0.62, top=0.95)
gC = fig.add_gridspec(1, 1, left=0.215, right=0.53, bottom=0.08, top=0.45)
gD = fig.add_gridspec(1, 1, left=0.755, right=0.99, bottom=0.08, top=0.45)

# ---------------- A ----------------
ax = fig.add_subplot(gA[0, 0])
for kind, lw, c, lab in [("pearson", 2.0, COEX[1], "top co-expressed pairs, Pearson (main)"),
                         ("spearman", 0.9, COEX[0], "Spearman"), ("withincomp", 0.9, COEX[3], "within each compartment")]:
    gl = K[kind]["global"]; Ns = sorted((int(n) for n in gl if int(n) >= 1000))
    xs = np.array(Ns); fo = np.array([gl[str(n)]["fold"] or np.nan for n in Ns])
    stable = np.array([gl[str(n)]["null_mean"] >= 0.1 for n in Ns])
    ax.plot(xs[stable], fo[stable], "-", color=c, lw=lw, label=lab, zorder=2)
    ax.plot(xs[stable], fo[stable], "o", color=c, ms=2.6 if lw > 1 else 2, zorder=2)
    if (~stable).any():
        i = np.where(stable)[0][0]
        ax.plot(xs[:i + 1], fo[:i + 1], ":", color=c, lw=0.9, zorder=2)
        ax.plot(xs[~stable], fo[~stable], "o", mfc="white", mec=c, ms=3, zorder=3)
OFF = {"scGPT": (-9, 0, "right"), "tGPT": (9, 8, "left"), "MaxToki": (8, 0, "left"), "Geneformer": (10, -10, "left"),
       "UCE": (8, 4, "left"), "C2S": (8, 6, "left"), "Tahoe": (-9, 3, "right"), "scFoundation": (-9, 7, "right")}
for m in ORDER:
    r = pm[m]
    if r["n_pairs"] < 1000 or r["hits"] == 0:
        continue
    sig = r["p_emp"] <= 0.05
    ax.plot(r["n_pairs"], r["fold"], MK[m], color=COL[m], mfc=COL[m] if sig else "white", mec=COL[m], ms=6, mew=1.1, zorder=5)
    dx, dy, ha = OFF.get(m, (7, 0, "left"))
    ax.annotate(NAME[m], (r["n_pairs"], r["fold"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.0, ha=ha,
                va="center")
ax.axhline(1, ls="--", lw=0.6, color="#555555"); ax.text(2.4e5, 1.08, "chance", fontsize=6.5, color="#555555", ha="right",
                                                         va="bottom")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3000, 2.6e5); ax.set_ylim(0.75, 160)
ax.set_xticks([1e4, 1e5]); ax.set_xticklabels(["10,000", "100,000"]); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
ax.set_xlabel("number of gene pairs (log scale)"); ax.set_ylabel("known TF–target pairs found,\ntimes chance (log scale)")
h, l = ax.get_legend_handles_labels()
UNSTABLE = any(K[k]["global"][str(n)]["null_mean"] < 0.1 for k in K for n in K[k]["global"] if int(n) >= 1000)
if UNSTABLE:
    h.append(Line2D([], [], ls=":", marker="o", mfc="white", mec=COEX[1], color=COEX[1], ms=3)); l.append("chance expects < 0.1: ratio unstable")
h.append(Line2D([], [], ls="", marker="o", color="#333333", ms=4.5)); l.append("one model's pairs (open: p > 0.05)")
ax.legend(h, l, loc="upper right", fontsize=6.3, handlelength=2.2)
fletter(ax, "A", "Times chance vs number of gene pairs", y=1.03)

# ---------------- B ----------------
ax = fig.add_subplot(gB[0, 0]); BM = [m for m in ORDER if pm[m]["n_pairs"] >= 5000 and pm[m]["hits"] > 0]
gl = K["pearson"]["global"]; nm = K["pearson"]["node_matched"]; w = 0.19
for i, m in enumerate(BM):
    own = ct["cofire"][m]["in_U"]["hits"]
    vals = [gl.get(str(pm[m]["n_pairs"]), {}).get("hits"), nm.get(m, {}).get("N_cofire_in_U", {}).get("hits"),
            dg[m]["coexpr_degmatched"]["hits"] if m in dg else None]
    ax.bar(i - 1.5 * w, own, w, color=COL[m])
    ax.text(i - 1.5 * w, own + 1.5, f"{own}", ha="center", fontsize=6.7)
    for j, v in enumerate(vals):
        x = i + (j - 0.5) * w
        if v is None:
            ax.plot(x, 1.2, "x", color="#555555", ms=3.2, mew=0.8)
        else:
            ax.bar(x, v, w, color=COEX[j])
ax.set_xticks(range(len(BM))); ax.set_xticklabels([NAME[m] for m in BM], rotation=35, ha="right", fontsize=7.0)
ax.set_ylabel("known TF–target pairs found"); ax.set_xlim(-0.6, len(BM) - 0.4)
top = max(ct["cofire"][m]["in_U"]["hits"] for m in BM); ax.set_ylim(0, top * 1.75)
ax.legend([plt.Rectangle((0, 0), 1, 1, color="#444444")] + [plt.Rectangle((0, 0), 1, 1, color=c) for c in COEX[:3]] +
          [Line2D([], [], marker="x", ls="", color="#555555", ms=3.2)],
          ["pairs sharing features (model colour)", "co-expression, same number of pairs", "… and same genes",
           "… and same partner counts", "none built"], loc="upper left", fontsize=6.3, handlelength=1.2)
fletter(ax, "B", "Known pairs found at equal size and genes", xfig=0.60, y=1.03)

# ---------------- C ----------------
ax = fig.add_subplot(gC[0, 0])
CM = [m for m in PASS if m in rc]
rows, gaps = [], []
for m in CM:
    r = rc[m]; rows.append((m, f"{NAME[m]}", r["coexpr_only_hits"], r["cofire_only_hits"], r["binom_p_two_sided"]))
    if m in lwf:
        u = lwf[m]
        rows.append((m, "without M1/M2 pairs", u["coexpr_only_unmasked"], u["cofire_only_unmasked"], u["binom_p_unmasked"]))
    else:
        lh = hv.get(m, {}).get("leave_top_hub_out")
        if lh and r["binom_p_two_sided"] < 0.05:
            rows.append((m, f"without {lh['hub']}", lh["coexpr_only"], lh["cofire_only"], lh["binom_p"]))
    gaps.append(len(rows))
yy, y = [], 0
for i in range(len(rows)):
    if i in gaps[:-1]:
        y += 0.6
    yy.append(-y); y += 1
ncorr = len(CM)
for yv, (m, lab, a, b, p) in zip(yy, rows):
    ax.barh(yv, -a, color=COEX[1], height=0.68); ax.barh(yv, b, color=COL[m], height=0.68)
    ax.text(-a - 1.5, yv, f"{a}", ha="right", va="center", fontsize=6.7); ax.text(b + 1.5, yv, f"{b}", ha="left", va="center", fontsize=6.7)
    strong = p * ncorr < 0.05
    ax.text(1.08, yv, f"p = {p:.2g}" if p >= 0.0001 else f"p < 0.0001", transform=ax.get_yaxis_transform(), va="center",
            fontsize=7.0, fontweight="bold" if strong else "normal", color="#222222" if p < 0.05 else "#666666")
ax.axvline(0, color="#444444", lw=0.6)
ax.set_yticks(yy); ax.set_yticklabels([r[1] for r in rows], fontsize=7.0); ax.tick_params(axis="y", length=0)
mx = max(max(r[2], r[3]) for r in rows); lim = np.ceil(mx * 1.25 / 10) * 10
ax.set_xlim(-lim, lim); t = [v for v in (-60, -30, 0, 30, 60) if abs(v) <= lim]
ax.set_xticks(t); ax.set_xticklabels([str(abs(v)) for v in t]); ax.set_xlabel("known pairs found by only one list")
ax.spines["left"].set_visible(False)
ax.text(-0.02 * lim, 1.0, "only\nco-expression", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=7.0)
ax.text(0.02 * lim, 1.0, "only shared\nfeatures", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7.0)
ax.text(1.08, 1.0, "paired\ntest", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0)
ax.set_ylim(min(yy) - 0.6, 0.6)
fletter(ax, "C", "Known pairs found by only one of the two lists (same genes and partner counts)", y=1.13)

# ---------------- D ----------------
ax = fig.add_subplot(gD[0, 0]); ov = K["pearson"]["overlap"]; y = np.arange(len(PASS))[::-1]
for yv, m in zip(y, PASS):
    o = ov[m]; inn, nn = o["cofire_in_global_top5"].get("hits", 0), o["cofire_not_global_top5"].get("hits", 0)
    ax.barh(yv + 0.17, 100 * o["share_in_global_top5pct"], height=0.34, color=COL[m], alpha=0.45)
    ax.barh(yv - 0.17, 100 * inn / max(inn + nn, 1), height=0.34, color=COL[m])
tr = 100 * K["pearson"]["trrust_share_in_global_top5pct"]
ytop = len(PASS) - 0.45   # lines stop below the legend
ax.plot([tr, tr], [-0.6, ytop], ls=":", color="#222222", lw=0.8); ax.plot([5, 5], [-0.6, ytop], ls="--", color="#777777", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in PASS], fontsize=7.0); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 100); ax.set_xlabel("% in the top 5 % of\nco-expressed gene pairs"); ax.set_ylim(-0.6, len(PASS) + 1.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.55), plt.Rectangle((0, 0), 1, 1, color="#444444"),
           Line2D([], [], ls=":", color="#222222"), Line2D([], [], ls="--", color="#777777")],
          ["pairs sharing features", "their known TF–target pairs", f"all known TF–target pairs ({tr:.1f} %)", "chance (5 %)"],
          loc="upper left", fontsize=6.3, handlelength=1.6, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
fletter(ax, "D", "Shared-feature pairs are co-expressed", xfig=0.60, y=1.13)
save(fig, "Figure_5")
print("Figure_5 written; B models:", BM, "C rows:", [(r[1], r[2], r[3], round(r[4], 4)) for r in rows])
