#!/usr/bin/env python
"""Fig 1 (overview schematic) + Fig 4 (annotation-free structure: cross-model CKA heatmap + cell-shuffle null;
SAE vs PCA variance at matched sparsity; linear decodability vs depth). Static, colorblind-safe, PNG 300dpi + PDF."""
from __future__ import annotations
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
FIG = f"{C}/figures"
d = json.load(open(f"{C}/controls.json"))
atlas = json.load(open(f"{C}/atlas_full_notf.json"))
nl = json.load(open(f"{C}/nonlinearity_alllayers.json"))
BLUE, ORANGE, GREEN, VERM, GREY_L, GREY_D, INK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CFCFCF", "#7A7A7A", "#222222"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "svg.fonttype": "none"})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ================= FIG 4 =================
fig, ax = plt.subplots(1, 3, figsize=(13, 4.1)); fig.subplots_adjust(wspace=0.50)

# 4A: cross-model CKA heatmap at 50% depth
CK = d["survivor_nulls"]["cka_null"]          # all 45 pairs, each with its own cell-shuffle null
M = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
_ix = {m: i for i, m in enumerate(M)}
Mat = np.eye(len(M))
for r in CK:
    p, q = r["pair"].split("-")
    if p in _ix and q in _ix:
        Mat[_ix[p], _ix[q]] = Mat[_ix[q], _ix[p]] = r["real_cka"]
_real = [r["real_cka"] for r in CK]; _shuf = [r["shuffled_cka"] for r in CK]
a = ax[0]
im = a.imshow(Mat, cmap="Blues", vmin=0, vmax=1)
a.set_xticks(range(len(M))); a.set_xticklabels(M, rotation=90, fontsize=6.5)
a.set_yticks(range(len(M))); a.set_yticklabels(M, fontsize=6.5)
a.set_title("A  Cross-model geometry — CKA, 50% depth", loc="left", fontweight="bold", fontsize=9.5)
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04); cb.set_label("linear CKA", fontsize=8); cb.ax.tick_params(labelsize=7)
a.text(0.0, -0.52, f"all 45 pairs null-tested: real {min(_real):.2f}–{max(_real):.2f} vs cell-shuffle "
                   f"{min(_shuf):.3f}–{max(_shuf):.3f}\n(every pair >10× its own shuffle floor)",
       transform=a.transAxes, fontsize=7, color="#666")

# 4B: SAE vs SVD variance at matched sparsity
b = ax[1]
sv = d["survivor_nulls"]["svd_null"]; mods = [x["model"] for x in sv]
x = np.arange(len(mods)); w = 0.36
b.bar(x - w/2, [x_["sae_var_explained"] for x_ in sv], w, color=BLUE, label="SAE (trained)")
b.bar(x + w/2, [x_["svd_var_at_k"] for x_ in sv], w, color=GREY_D, label="top-k PCA (SVD)")
b.set_xticks(x); b.set_xticklabels(mods, rotation=90, fontsize=7)
b.set_ylabel("variance explained at k=32")
b.set_ylim(0, 1.12)
b.set_title("B  SAE > PCA in all ten (margin varies)", loc="left", fontweight="bold", fontsize=9.5)
for i, x_ in enumerate(sv):   # gap annotation instead of two crowded numbers
    gap = x_["sae_var_explained"] - x_["svd_var_at_k"]
    b.text(i, max(x_["sae_var_explained"], x_["svd_var_at_k"]) + 0.03, f"+{gap:.2f}",
           ha="center", fontsize=6.5, color=GREEN if gap > 0.1 else VERM)
b.legend(frameon=False, fontsize=7.5, loc="upper left")


# 4C: linear decodability of cell identity vs depth (all models)
c = ax[2]
tm = nl["concepts"]["tissue"]["models"]; ch = nl["concepts"]["tissue"]["chance"]
for m in atlas["models"]:
    if m not in tm:
        continue
    layers = sorted(tm[m].keys(), key=int); n = len(layers)
    xs = [i / max(n - 1, 1) for i in range(n)]
    ys = [tm[m][L]["lin_res"] for L in layers]
    c.plot(xs, ys, "-", color=BLUE, lw=0.9, alpha=0.55)
c.axhline(ch, ls="--", color=VERM, lw=1.2); c.text(0.02, ch + 0.02, f"chance {ch:.2f}", color=VERM, fontsize=7.5)
c.set_ylim(0, 1.0); c.set_xlabel("relative depth"); c.set_ylabel("linear decodability (tissue)")
c.set_title("C  Cell identity is linearly readable", loc="left", fontweight="bold", fontsize=9.5)
c.text(0.02, 0.9, "each line = one model;\nMLP−linear gap ≈ 0 at every depth", transform=c.transAxes, fontsize=7.5, color="#555")
save(fig, "Fig4_annotation_free")

# ================= FIG 1 (schematic) =================
fig, ax = plt.subplots(figsize=(12, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc="#F4F6F9", ec="#8aa0c0", tc=INK, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.5",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, col="#7A7A7A"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.4))

# 1) models
box(1, 40, 15, 30, "10 single-cell\nfoundation models\n(10M – 3B)\nrank · magnitude ·\nprotein · sentence ·\nknowledge", fc="#EEF3FA", fs=7.8)
# 2) corpus
box(19, 46, 15, 18, "shared corpus\n6,000 Tabula\nSapiens cells\n(immune/kidney/lung)", fs=7.8)
arrow(16, 55, 19, 55)
# 3) SAE
box(37, 46, 14, 18, "TopK SAE\nper layer\n(k=32, 4×)", fs=8)
arrow(34, 55, 37, 55)
# 4) features
box(54, 46, 13, 18, "features →\ntop genes", fs=8)
arrow(51, 55, 54, 55)
# 5) two annotators
arrow(67, 58, 71, 74)   # up to permissive
arrow(67, 52, 71, 30)   # down to calibrated
box(71, 68, 27, 15, "Permissive annotation\n(top-5, 5 large DBs)", fc="#FBECEC", ec="#e0a0a0", fs=8)
box(71, 22, 27, 18, "Calibrated annotation\n(top-10, ≥3 in curated\n≤200 set, no PPI)\n+ random-gene null", fc="#EAF6F0", ec="#8fccb0", fs=7.8)
ax.text(72.5, 62.5, "✗  apparent 'universal core' ~2,000 — fails random-gene null (real < null)",
        fontsize=7.6, color=VERM, va="center")
ax.text(72.5, 15.5, "✓  reproducible backbone ~78 concepts (≥8/10), 22–59× over null, replicates held-out",
        fontsize=7.6, color=GREEN, va="center")
# annotation-free track
box(19, 6, 48, 9, "Annotation-FREE (orthogonal, pass their own nulls):  cross-model CKA · SAE > PCA · linear readout · tissue×depth",
    fc="#F3F0FA", ec="#b0a0d0", fs=7.6)
arrow(43, 46, 43, 15)
ax.set_title("Fig 1   A calibrated comparative SAE atlas of ten single-cell foundation models",
             loc="left", fontweight="bold", fontsize=10.5)
save(fig, "Fig1_overview")
print("==> Fig1, Fig4 in", FIG)
