#!/usr/bin/env python
"""Inject the 8-model data blocks into the atlas HTML template and emit light (published) + full (local)
builds. Also patches the few 7-hardcoded readouts to be N-model dynamic (svd bar/readout, theme readout,
gene-search denominator, prose 'seven'->'eight', drops the 'Tahoe pending' caveat).
    python scripts/inject_atlas.py
"""
from __future__ import annotations
import json, re, os

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
SCRATCH = "/private/tmp/claude-501/-Users-annaantipova-Desktop-biomech/cff4254a-ea2f-48db-929f-897776ca3413/scratchpad"
TEMPLATE = f"{SCRATCH}/atlas_ts3.html"

DATA = {
    "modules-data": f"{C}/modules_alllayers.json",
    "genes-data": f"{C}/genes_ts3.json",
    "atlas-data": f"{C}/atlas_full_notf.json",
}
N_MODELS = len(json.load(open(DATA["atlas-data"]))["models"])
NEW_THEME_READ = (
    " const Z=T.Z,tm=T.models,th=T.themes,N=tm.length;"
    "const inAll=th.filter((t,i)=>Z[i].every(v=>v>0));"
    "const perM=tm.map((m,j)=>Z.reduce((a,r)=>a+(r[j]>0?1:0),0));"
    "const rich=tm[perM.indexOf(Math.max(...perM))],poor=tm[perM.indexOf(Math.min(...perM))];"
    "const allTxt=inAll.length?inAll.join(', ')+' appear in all '+N+' models':'programs vary across all '+N+' models';"
    "document.getElementById('theme-read').innerHTML=`The co-activation modules aren't arbitrary — they map to "
    "canonical biological programs. <b style=\"color:var(--teal)\">${allTxt}</b> — the SAE organises features into a "
    "shared module vocabulary across architectures. Coverage varies: <b>${rich}</b> resolves ${Math.max(...perM)} of "
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


def build(explorer_path, out_path, label):
    lines = open(TEMPLATE, encoding="utf-8").read().split("\n")
    blocks = dict(DATA); blocks["explorer-data"] = explorer_path
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
    CW = [(r"\bseven\b", NW), (r"\bSeven\b", NW.capitalize()),
          (r"\beight\b", NW), (r"\bEight\b", NW.capitalize())]
    for i, l in enumerate(lines):
        if len(l) < 4000:
            for pat, rep in CW:
                l = re.sub(pat, rep, l)
            lines[i] = l
    html = "\n".join(lines)
    for a, b in SUBS:
        html = html.replace(a, b)
    open(out_path, "w", encoding="utf-8").write(html)
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"{label}: {out_path}  ({mb:.1f} MB)")
    return mb


if __name__ == "__main__":
    build(f"{C}/explorer_slim_light.json", f"{SCRATCH}/atlas_ts3_light.html", "LIGHT")
    build(f"{C}/explorer_slim_full.json", f"{SCRATCH}/atlas_ts3.html", "FULL")
