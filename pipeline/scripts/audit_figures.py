#!/usr/bin/env python
"""Catch figures rendered from data that has since changed.

Fig S2 shipped for four days showing folds of 24x/28x/20x/20x/17x and a legend reading "N=20 perms"
after the depth sweep had been recomputed at 250 permutations. Nothing tied the rendered PNG to the
JSON it came from, so the manuscript audit -- which checks text against data -- could not see it.

This closes that gap by content, not by timestamp: a figure re-rendered with no change to its data
is not stale, and a figure whose data changed is, whatever the file dates say. Each plotting script
is scanned for the JSON files it loads; their SHA-256s are recorded when the figure is stamped and
re-checked here.

    python scripts/audit_figures.py            # report stale figures, exit 1 if any
    python scripts/audit_figures.py --update   # re-stamp after regenerating them

Add a new plotting script and it is picked up automatically, as long as it loads its data with
json.load(open(...)) and its outputs are listed in FIGURES below.
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob, hashlib, json, os, re, sys

BASE = _B
C = f"{BASE}/outputs/atlas/comparative"
FIG = f"{C}/figures"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))   # the plotting scripts live next to this file
MANIFEST = f"{FIG}/.sources.json"

# which rendered figures each script is responsible for
FIGURES = {
    "make_figures.py": ["Fig2_artefact_and_fix", "Fig3_backbone"],
    "make_fig14.py": ["Fig1_overview", "Fig4_annotation_free"],
    "make_fig5.py": ["Fig5_hypothesis"],
    "make_figS1.py": ["FigS1_svd_projection"],
    "make_figS2.py": ["FigS2_depth_backbone"],
}


def sources_of(script: str) -> list[str]:
    """the .json files a plotting script reads"""
    src = open(f"{SCRIPTS}/{script}", encoding="utf-8").read()
    names = set()
    for call in re.findall(r"json\.load\(open\(([^)]*)\)", src):
        names.update(re.findall(r"([A-Za-z_0-9.]+\.json)", call))
    return sorted(names)


def sha(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def current() -> dict:
    out = {}
    for script in sorted(FIGURES):
        if not os.path.exists(f"{SCRIPTS}/{script}"):
            continue
        out[script] = {"figures": FIGURES[script],
                       "sources": {n: sha(f"{C}/{n}") for n in sources_of(script)}}
    return out


now = current()
old = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}

if "--update" in sys.argv:
    # Stamping says "these figures were rendered from these sources". It has to be earned: a figure
    # older than a source it is drawn from was not re-rendered, whatever the person stamping believes.
    # make_fig14.py once died on a typo *after* writing Fig4 and before Fig1, and a --update run
    # immediately after certified the un-rendered Fig1 as current.
    stale = []
    for script, v in now.items():
        newest_src = max((os.path.getmtime(f"{C}/{n}") for n in v["sources"]
                          if os.path.exists(f"{C}/{n}")), default=0)
        for f in v["figures"]:
            p = f"{FIG}/{f}.png"
            if not os.path.exists(p):
                stale.append((f, script, "не отрисована"))
            elif os.path.getmtime(p) < newest_src:
                stale.append((f, script, "старше своих данных — скрипт не перерисовал её"))
    if stale:
        print("НЕ ШТАМПУЮ — эти фигуры не были перерисованы:")
        for f, script, why in stale:
            print(f"   {f:<26} {script:<20} {why}")
        print("\nЗапусти генератор, убедись, что он завершился без ошибки, и повтори --update.")
        sys.exit(1)
    json.dump(now, open(MANIFEST, "w"), indent=1)
    n = sum(len(v["sources"]) for v in now.values())
    print(f"stamped {len(now)} scripts / {sum(len(v['figures']) for v in now.values())} figures "
          f"against {n} source files")
    sys.exit(0)

problems = []
print(f"{'figure':<26}{'script':<20}{'sources':>9}  status")
print("-" * 72)
for script, info in now.items():
    was = old.get(script, {}).get("sources", {})
    changed = [n for n, h in info["sources"].items() if was.get(n) != h]
    missing = [f for f in info["figures"] if not os.path.exists(f"{FIG}/{f}.png")]
    for f in info["figures"]:
        if f in missing:
            status, why = "*** MISSING ***", "not rendered"
        elif not was:
            status, why = "*** UNSTAMPED ***", "never stamped — run --update after rendering"
        elif changed:
            status, why = "*** STALE ***", "data changed since render: " + ", ".join(changed)
        else:
            status, why = "ok", ""
        print(f"{f:<26}{script:<20}{len(info['sources']):>9}  {status}")
        if why and status != "ok":
            print(f"{'':<55}{why}")
            problems.append((f, script, why))

if problems:
    print(f"\n{len(problems)} figure(s) need attention:")
    for f, script, why in problems:
        print(f"   {f}: {why}")
    print("\n   rerun the script, then: python scripts/audit_figures.py --update")
else:
    print(f"\nall {sum(len(v['figures']) for v in now.values())} figures match the data they were drawn from")
sys.exit(1 if problems else 0)
