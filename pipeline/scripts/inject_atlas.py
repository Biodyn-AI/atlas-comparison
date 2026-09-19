#!/usr/bin/env python
"""Inject the 8-model data blocks into the atlas HTML template and emit light (published) + full (local)
builds. Also patches the few hardcoded readouts to be N-model dynamic (svd bar/readout, theme readout,
gene-search denominator).
    python scripts/inject_atlas.py
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, re, os

C = f"{_B}/outputs/atlas/comparative"
# The site root: the repository this script sits in (pipeline/scripts/inject_atlas.py), or the
# working-tree checkout next to the data root when run from there.
_here = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.environ.get("ATLAS_REPO") or (
    _os.path.dirname(_os.path.dirname(_here)) if _os.path.exists(_os.path.join(_here, "..", "atlas_template.html"))
    else f"{_B}/comparative-sae-atlas")
TEMPLATE = f"{REPO}/pipeline/atlas_template.html"  # canonical editable markup (data blocks are __DATA__ placeholders)
OUT = _os.environ.get("ATLAS_OUT", f"{_B}/outputs/atlas/comparative")  # the two local builds land here

DATA = {
    "modules-data": f"{C}/modules_alllayers.json",
    "genes-data": f"{C}/genes_ts3.json",
    "atlas-data": f"{C}/atlas_full_notf.json",
    # dead-feature fractions per model-layer; the chooser reads it so a model whose lead rests on
    # the roster's weakest dictionaries (tGPT) carries that caveat next to its recommendation
    "health-data": f"{C}/sae_health.json",
}
N_MODELS = len(json.load(open(DATA["atlas-data"]))["models"])
NEW_THEME_READ = (
    " const Z=T.Z,tm=T.models,th=T.themes,N=tm.length;"
    "const inAll=th.filter((t,i)=>Z[i].every(v=>v>0));"
    "const perM=tm.map((m,j)=>Z.reduce((a,r)=>a+(r[j]>0?1:0),0));"
    "const rich=tm[perM.indexOf(Math.max(...perM))],poor=tm[perM.indexOf(Math.min(...perM))];"
    "const allTxt=inAll.length?inAll.join(', ')+' appear in all '+N+' models':'programs vary across all '+N+' models';"
    "document.getElementById('theme-read').innerHTML=`The co-activation modules map to "
    "canonical biological programs. <b style=\"color:var(--teal)\">${allTxt}</b>; the rest of the vocabulary is only "
    "partly shared. Coverage varies: <b>${rich}</b> resolves ${Math.max(...perM)} of "
    "the nine programs, <b>${poor}</b> only ${Math.min(...perM)}, so architecture and scale carve different numbers of "
    "distinct modules. The UMAP axes are non-linear, but the clusters <b>are</b> these programs — that is what the "
    "module structure means.`;})();"
)

SUBS = [
    # svd panel: iterate only models that actually have an SVD entry (Tahoe may be pending)
    ("box.innerHTML=M.map(m=>{const v=S[m].novel;",
     "box.innerHTML=M.filter(m=>S[m]).map(m=>{const v=S[m].novel;"),
    ("const mn=Math.min(...M.map(m=>S[m].novel));",
     "const mn=Math.min(...M.filter(m=>S[m]).map(m=>S[m].novel));"),
    ("+M.map(m=>{const sv=S[m].svd_var*100,se=S[m].sae_var*100;",
     "+M.filter(m=>S[m]).map(m=>{const sv=S[m].svd_var*100,se=S[m].sae_var*100;"),
    ("Across all seven models ", "Across all ${Object.keys(S).length} models "),
    # gene search denominator
    ("${rows.length}/7</b> models", "${rows.length}/${gModels.length}</b> models"),
    # caveat now resolved
    ("7 models (Tahoe pending)", "8 models"),
]


def build(explorer_path, out_path, label, lazy=None):
    """lazy: a path relative to the page (e.g. data/explorer.json). The explorer JSON is then written
    there and the page carries only a stub, so the published HTML is a fifth of the size and the maps
    are fetched the first time someone opens the explorer."""
    lines = open(TEMPLATE, encoding="utf-8").read().split("\n")
    blocks = dict(DATA); blocks["explorer-data"] = explorer_path
    if lazy:
        ex = json.load(open(explorer_path))
        dst = os.path.join(os.path.dirname(out_path), lazy)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(ex, open(dst, "w"), separators=(",", ":"))
        stub = os.path.join(os.path.dirname(out_path), "data", "explorer_stub.json")
        json.dump({"lazy": lazy, "models": ex["models"], "mb": round(os.path.getsize(dst) / 1048576)}, open(stub, "w"))
        blocks["explorer-data"] = stub
        print(f"explorer maps -> {dst} ({os.path.getsize(dst)/1048576:.1f} MB); the page keeps a stub")
    for i, ln in enumerate(lines):
        m = re.match(r'(\s*)<script id="([a-z-]+-data)" type="application/json">', ln)
        if m and m.group(2) in blocks:
            indent, sid = m.group(1), m.group(2)
            content = open(blocks[sid], encoding="utf-8").read().strip()
            lines[i] = f'{indent}<script id="{sid}" type="application/json">{content}</script>'
            continue
        # theme-read: replace the whole physical line
        if "getElementById('theme-read').innerHTML=" in ln:
            lines[i] = NEW_THEME_READ
    # model-count words in PROSE -> current N (only on short lines; never touch the multi-MB data blocks,
    # where a concept like "seven-transmembrane" must not be mangled)
    NW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
          8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}.get(N_MODELS, str(N_MODELS))
    # (the old seven/eight -> ten word substitution is gone: the template says "ten" itself, and the
    # rule rewrote any "eight" in prose, turning "at least eight of the ten" into "ten of the ten")
    html = "\n".join(lines)
    for a, b in SUBS:
        html = html.replace(a, b)
    import datetime
    html = html.replace("__BUILT__", datetime.date.today().strftime("%-d %B %Y"))
    open(out_path, "w", encoding="utf-8").write(html)
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"{label}: {out_path}  ({mb:.1f} MB)")
    return mb


if __name__ == "__main__":
    build(f"{C}/explorer_slim_light.json", f"{OUT}/atlas_ts3_light.html", "LIGHT")
    build(f"{C}/explorer_slim_full.json", f"{OUT}/atlas_ts3_full.html", "FULL")
    # the published page: same markup, explorer maps fetched on demand from data/explorer.json
    build(f"{C}/explorer_slim_full.json", f"{REPO}/index.html", "SITE", lazy="data/explorer.json")
    print("deployed to comparative-sae-atlas/index.html (+ data/explorer.json)")
