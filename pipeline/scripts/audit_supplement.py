#!/usr/bin/env python
"""Check the hand-written supplementary tables against the data and code they describe.

The manuscript audit reads MANUSCRIPT.md only, so it saw none of this: Table S3 contradicting itself
on dead-feature fractions, Table S5 keeping a fold range the text had corrected, Table S2 printing
unfiltered totals under a "5-500 genes" heading. Those four tables are now generated from the
results (make_supp_tables.py) and cannot drift. The rest are prose written by hand:

    S1  per-model specification -- layers, dictionary size, variance explained
    S4  annotator parameters    -- must match the constants the scripts actually use
    S6  software environment    -- must match the interpreter this ran on
    S7  SAE vs PCA at k = 32    -- must match controls.json

This checks those, so every number in the supplement is either generated or verified.
    python scripts/audit_supplement.py
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, re, sys

BASE = _B
C = f"{BASE}/outputs/atlas/comparative"
SUP = open(f"{C}/SUPPLEMENTARY.md", encoding="utf-8").read()
SCRIPTS = _os.path.dirname(_os.path.abspath(__file__))   # the scripts whose constants are checked live next to this file
rows = []
DISP = {"AIDO.Cell": "AIDO", "C2S-Scale": "C2S", "Geneformer-V2": "Geneformer", "Tahoe-x1": "Tahoe"}


def load(n): return json.load(open(f"{C}/{n}"))


def rec(claim, expected, got, ok=None):
    ok = (str(expected) == str(got)) if ok is None else ok
    rows.append((claim, f"{expected} vs {got}", ok))


HEALTH = load("sae_health.json")["models"]
SVD = {x["model"]: x for x in load("controls.json")["survivor_nulls"]["svd_null"]}

# ---- S1: per-model specification -------------------------------------------
sec = SUP.split("## Table S1")[1].split("\n## ")[0]
seen = 0
for line in sec.split("\n"):
    m = re.match(r"\|\s*([A-Za-z0-9.\-]+)\s*\|[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", line)
    if not m:
        continue
    name, d_model, d_sae, layers, ve = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))
    key = DISP.get(name, name)
    if key not in HEALTH:
        rec(f"S1 {name}", "a known model", "not in the data", False); continue
    seen += 1
    rec(f"S1 {name} layers", HEALTH[key]["n_layers"], layers)
    rec(f"S1 {name} d_sae = 4 x d_model", 4 * d_model, d_sae)
    rec(f"S1 {name} VarExpl", round(SVD[key]["sae_var_explained"], 2), ve)
rec("S1 covers every model", len(HEALTH), seen)

# ---- S4: annotator parameters vs the constants the scripts use --------------
deg = open(f"{SCRIPTS}/degree_null.py", encoding="utf-8").read()
m = re.search(r"ALPHA,\s*TOP,\s*AMIN,\s*MX\s*=\s*([\d.]+),\s*(\d+),\s*(\d+),\s*(\d+)", deg)
if m:
    alpha, top, amin, mx = float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    s4 = SUP.split("## Table S4")[1].split("\n## ")[0]

    def cell(label):
        r = re.search(rf"\|\s*{re.escape(label)}\s*\|[^|]*\|\s*([^|]+)\|", s4)
        return r.group(1).strip() if r else None

    rec("S4 calibrated top genes", top, cell("Top genes tested"))
    rec("S4 calibrated min overlap", f"≥ {amin}", cell("Min overlap (a)"))
    rec("S4 calibrated max set size", mx, cell("Max set size"))
    rec("S4 calibrated alpha", "0.05", str(alpha))
    rec("S4 calibrated excludes PPI", True, "no PPI" in (cell("Databases") or ""),
        ok="no PPI" in (cell("Databases") or ""))
else:
    rec("S4 annotator constants", "readable from degree_null.py", "pattern not found", False)

# ---- S6: environment --------------------------------------------------------
import numpy, scipy, platform
s6 = SUP.split("## Table S6")[1].split("\n## ")[0]


def ver(label):
    """S6 has two value columns, GPU host then local; this interpreter is the local one"""
    r = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([^|]*)\|\s*([^|]*)\|", s6)
    return r.group(2).strip().split()[0] if r and r.group(2).strip() else None


py = ".".join(platform.python_version_tuple()[:2])
rec("S6 Python (local)", py, ver("Python"), ok=(ver("Python") or "").startswith(py))
rec("S6 NumPy (local)", numpy.__version__, ver("NumPy"), ok=(ver("NumPy") or "") == numpy.__version__)
rec("S6 SciPy (local)", scipy.__version__, ver("SciPy"), ok=(ver("SciPy") or "") == scipy.__version__)
import h5py
rec("S6 h5py (local)", h5py.__version__, ver("h5py"), ok=(ver("h5py") or "") == h5py.__version__)

# ---- S7: SAE vs PCA ---------------------------------------------------------
s7 = SUP.split("## Table S7")[1].split("\n## ")[0]
n7 = 0
for line in s7.split("\n"):
    m = re.match(r"\|\s*([A-Za-z0-9.\-]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\+([\d.]+)\s*\|\s*(-?[\d.]+)\s*\|", line)
    if not m:
        continue
    name, pca, sae, gap = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
    key = DISP.get(name, name)
    if key not in SVD:
        continue
    n7 += 1
    rec(f"S7 {name} PCA@k32", SVD[key]["svd_var_at_k"], pca, ok=abs(SVD[key]["svd_var_at_k"] - pca) < 5e-4)
    rec(f"S7 {name} SAE@k32", SVD[key]["sae_var_explained"], sae, ok=abs(SVD[key]["sae_var_explained"] - sae) < 5e-4)
    d = SVD[key]["sae_var_explained"] - SVD[key]["svd_var_at_k"]
    rec(f"S7 {name} gap", round(d, 3), gap, ok=abs(d - gap) < 5e-4)
rec("S7 covers every model", len(SVD), n7)

# ---- supplementary figure captions, which also carry data-derived numbers ----
DPB = load("depth_backbone.json")
folds = [r["tiers"][">=8"]["fold"] for r in DPB["depths"]]
caps = SUP.split("## Supplementary figures")[1].split("\n## ")[0]
m = re.search(r"significant at every depth \(([\d.]+)–([\d.]+)× over null, (\d+) permutations\)", caps)
if m:
    rec("FigS2 caption fold range", f"{min(folds)}-{max(folds)}", f"{m.group(1)}-{m.group(2)}")
    rec("FigS2 caption permutations", DPB["n_perm"], int(m.group(3)))
else:
    rec("FigS2 caption", "a fold range and permutation count", "pattern not found", False)
order = re.findall(r"- \*\*Fig S(\d+)\*\*", caps)
rec("supplementary figures in order", "1,2", ",".join(order))

# ---- report ------------------------------------------------------------------
bad = [r for r in rows if not r[2]]
w = max(len(r[0]) for r in rows)
for claim, val, ok in rows:
    if not ok:
        print(f"{claim:<{w}}  {val:>28}  *** MISMATCH ***")
print(f"\n{len(rows)-len(bad)}/{len(rows)} supplementary values verified "
      f"(S1, S4, S6, S7; S2/S3/S5/S8/S9/S10 are generated)")
if bad:
    print("\nneeds a look:")
    for claim, val, _ in bad:
        print(f"   {claim}: {val}")
sys.exit(1 if bad else 0)
