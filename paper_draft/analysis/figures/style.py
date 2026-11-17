"""Shared look of the paper's figures: page width, Arial, panel letters, the model palette.
Colours are the ones of the original figures (read from their PDF drawing streams)."""
from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

WIDTH = 538.56 / 72          # inches: the original figures' width
# FIG_DATA=old draws the figure from the published run and the original analyses (to check a script
# against the original figure); the default draws it from the corrected run and the re-run analyses.
OLD = os.environ.get("FIG_DATA") == "old"
BASE = os.environ.get("ATLAS_BASE")
if BASE is None:
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = f"{BASE}/outputs/atlas/comparative"
AN = os.environ.get("ATLAS_ANALYSIS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
CACHE = os.environ.get("ATLAS_CACHE", os.path.join(AN, ".cache"))
FIGDIR = os.environ.get("ATLAS_FIGURES", os.path.join(CACHE, "figures_old" if OLD else "figures"))

ORDER = ["AIDO", "scGPT", "scFoundation", "GeneCompass", "tGPT", "MaxToki", "Geneformer", "UCE", "C2S", "Tahoe"]
NAME = {"AIDO": "AIDO.Cell", "scGPT": "scGPT", "scFoundation": "scFoundation", "GeneCompass": "GeneCompass",
        "tGPT": "tGPT", "MaxToki": "MaxToki", "Geneformer": "Geneformer-V2", "UCE": "UCE", "C2S": "C2S-Scale",
        "Tahoe": "Tahoe-x1"}
COL = {"AIDO": "#9b532a", "scGPT": "#4877e1", "scFoundation": "#e87ad0", "GeneCompass": "#355292",
       "tGPT": "#e8605d", "MaxToki": "#ad53b5", "Geneformer": "#991451", "UCE": "#43b2f6", "C2S": "#69bc8a",
       "Tahoe": "#738800"}
COEX = ["#e8a93a", "#d08c00", "#a36a00", "#6e4700"]      # co-expression, light to dark
CLASS = {"protein_synthesis": "#1a5561", "other_housekeeping": "#3f8f99", "specific": "#7fbcc0"}
GREY = "#555555"; LGREY = "#bbbbbb"


def setup():
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False, "savefig.dpi": 300})


def letter(ax, s, title="", x=-0.02, y=1.02, yfig=None):
    """Panel letter; with yfig, y is a figure fraction so that letters of panels in one row line up."""
    if yfig is not None:
        from matplotlib.transforms import blended_transform_factory
        tr = blended_transform_factory(ax.transAxes, ax.figure.transFigure)
        ax.annotate(f"{s}", (x, yfig), xycoords=tr, fontsize=7.5, fontweight="bold", va="bottom",
                    ha="left", annotation_clip=False)
        if title:
            ax.annotate(title, (x, yfig), xycoords=tr, xytext=(9, 0), textcoords="offset points", fontsize=7,
                        fontweight="bold", va="bottom", ha="left", annotation_clip=False)
        return
    ax.annotate(f"{s}", (x, y), xycoords="axes fraction", fontsize=7.5, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)
    if title:
        ax.annotate(title, (x, y), xycoords="axes fraction", xytext=(9, 0), textcoords="offset points",
                    fontsize=7, fontweight="bold", va="bottom", ha="left", annotation_clip=False)


def pletter(fig, s, title, x, y):
    """Panel letter and title at figure coordinates (x, y): letters of one column share x, of one row y."""
    from matplotlib.transforms import offset_copy
    fig.text(x, y, s, fontsize=7.5, fontweight="bold", va="bottom", ha="left")
    if title:
        fig.text(x, y, title, fontsize=7, fontweight="bold", va="bottom", ha="left",
                 transform=offset_copy(fig.transFigure, fig=fig, x=9, y=0, units="points"))


def fletter(ax, s, title="", xfig=0.004, y=1.03):
    """Panel letter at a fixed position across the page (figure fraction), so that the letters of
    panels stacked in one column line up whatever the width of their axis labels."""
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.figure.transFigure, ax.transAxes)
    ax.annotate(f"{s}", (xfig, y), xycoords=tr, fontsize=7.5, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)
    if title:
        ax.annotate(title, (xfig, y), xycoords=tr, xytext=(9, 0), textcoords="offset points", fontsize=7,
                    fontweight="bold", va="bottom", ha="left", annotation_clip=False)


def load(path):
    return json.load(open(path))


def an(rel):
    """a result file of the paper analyses, e.g. an('trrust_robustness/s04_null_choice.json')"""
    return json.load(open(f"{AN}/{rel}"))


def out(name):
    return json.load(open(f"{OUT}/{name}"))


def save(fig, name):
    """Save at exactly the page width (as the original figures), trimming only empty space above and
    below; report anything drawn outside the page width, which would be cut off."""
    from matplotlib.transforms import Bbox
    os.makedirs(FIGDIR, exist_ok=True)
    fig.canvas.draw()
    if os.environ.get("LETTERS"):
        r = fig.canvas.get_renderer(); H = fig.get_size_inches()[1] * fig.dpi
        for ax in fig.axes:
            for t in ax.texts:
                if len(t.get_text()) == 1 and t.get_text().isupper() and t.get_fontweight() in ("bold", 700):
                    e = t.get_window_extent(r)
                    print(f"LETTER {name} {t.get_text()} y0={e.y0 / H:.4f} x0={e.x0 / (fig.get_size_inches()[0] * fig.dpi):.3f}")
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    over = {"left": -bb.x0, "right": bb.x1 - w}
    over = {k: round(v * 72, 1) for k, v in over.items() if v > 0.01}
    if over:
        print(f"OVERFLOW {name} (pt):", over)
    box = Bbox([[0, bb.y0 - 0.04], [w, bb.y1 + 0.04]])
    fig.savefig(f"{FIGDIR}/{name}.pdf", bbox_inches=box)
    fig.savefig(f"{FIGDIR}/{name}.png", bbox_inches=box, dpi=200)
    plt.close(fig)
