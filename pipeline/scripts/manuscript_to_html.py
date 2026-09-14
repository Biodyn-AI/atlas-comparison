#!/usr/bin/env python
"""Render MANUSCRIPT.md as a single self-contained HTML with the figures embedded
(base64), so it can be opened in a browser and copy-pasted straight into Google Docs.
    python scripts/manuscript_to_html.py  ->  outputs/atlas/comparative/MANUSCRIPT.html
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import base64, html, re, os

C = f"{_B}/outputs/atlas/comparative"
FIG = f"{C}/figures"
OUT = f"{C}/MANUSCRIPT.html"

FIGFILE = {
    "Fig 1": "Fig1_overview.png",
    "Fig 2": "Fig2_artefact_and_fix.png",
    "Fig 3": "Fig3_backbone.png",
    "Fig 4": "Fig4_annotation_free.png",
    "Fig 5": "Fig5_hypothesis.png",
    "Fig S1": "FigS1_svd_projection.png",
    "Fig S2": "FigS2_depth_backbone.png",
}


def data_uri(fn):
    p = f"{FIG}/{fn}"
    if not os.path.exists(p):
        return None
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"(?<!href=\")(?<!\">)(https?://[^\s<),]+)", r'<a href="\1">\1</a>', s)
    return s


def figure_block(tag: str, caption_html: str) -> str:
    uri = data_uri(FIGFILE.get(tag, ""))
    img = f'<img src="{uri}" alt="{tag}">' if uri else f"<p><em>[{tag} image missing]</em></p>"
    return f'<figure>{img}<figcaption>{caption_html}</figcaption></figure>'


def convert(md: str) -> str:
    lines = md.split("\n")
    out, para, i = [], [], 0
    in_refs = False

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            flush(); i += 1; continue
        if s.startswith("---") and set(s) <= {"-"}:
            flush(); i += 1; continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            lvl = len(m.group(1))
            txt = re.sub(r"\s*\*\(.*?\)\*\s*$", "", m.group(2))  # drop editorial asides in headings
            out.append(f"<h{lvl}>{inline(txt)}</h{lvl}>")
            in_refs = txt.lower().startswith("references")
            i += 1; continue

        # figure bullets: "- **Fig 2.** caption..."  (may wrap onto indented lines)
        fm = re.match(r"^-\s+\*\*(Fig S?\d+)\.\*\*\s*(.*)$", s)
        if fm:
            flush()
            tag, cap = fm.group(1), fm.group(2)
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip() and not lines[j].strip().startswith("- "):
                cap += " " + lines[j].strip(); j += 1
            out.append(figure_block(tag, f"<strong>{tag}.</strong> " + inline(cap)))
            i = j; continue

        if in_refs and not ln.startswith("  "):
            flush()
            ref = s
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip():
                ref += " " + lines[j].strip(); j += 1
            out.append('<p class="ref">' + inline(ref) + "</p>")
            i = j; continue

        if s.startswith("- "):
            flush()
            out.append("<ul><li>" + inline(s[2:]) + "</li></ul>")
            i += 1; continue

        para.append(ln); i += 1
    flush()
    return "\n".join(out)


md = open(f"{C}/MANUSCRIPT.md", encoding="utf-8").read()
body = convert(md)

# supplementary figures, appended after the main Figures list
sup = open(f"{C}/SUPPLEMENTARY.md", encoding="utf-8").read()
sup_caps = {}
for m in re.finditer(r"^- \*\*(Fig S\d+)\*\*\s*[—-]\s*(.*?)(?=\n- \*\*Fig S|\n\n|\Z)", sup, re.S | re.M):
    sup_caps[m.group(1)] = " ".join(m.group(2).split())
extra = ""
for tag in ("Fig S1", "Fig S2"):
    if tag in sup_caps:
        extra += figure_block(tag, f"<strong>{tag}.</strong> " + inline(sup_caps[tag]))
if extra:
    body += "\n<h2>Supplementary figures</h2>\n" + extra

htmlpage = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Comparative SAE atlas — manuscript</title>
<style>
 body{{font-family:Georgia,'Times New Roman',serif;font-size:12pt;line-height:1.55;color:#111;
   background:#fff;max-width:780px;margin:36px auto;padding:0 24px}}
 h1{{font-size:20pt;line-height:1.25;margin:0 0 4px}}
 h2{{font-size:14.5pt;margin:26px 0 6px;border-bottom:1px solid #ddd;padding-bottom:3px}}
 h3{{font-size:12.5pt;margin:18px 0 4px}}
 p{{margin:0 0 10px;text-align:justify}}
 p.ref{{margin:0 0 6px;padding-left:22px;text-indent:-22px;font-size:11pt;text-align:left}}
 code{{font-family:Menlo,Consolas,monospace;font-size:10.5pt;background:#f4f4f4;padding:1px 3px}}
 a{{color:#0b57d0}}
 figure{{margin:18px 0 22px;page-break-inside:avoid}}
 figure img{{width:100%;height:auto;border:1px solid #e3e3e3}}
 figcaption{{font-size:10.5pt;line-height:1.45;color:#333;margin-top:6px;text-align:left}}
 ul{{margin:0 0 10px 20px;padding:0}}
</style></head><body>
{body}
</body></html>"""

open(OUT, "w", encoding="utf-8").write(htmlpage)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024/1024:.1f} MB)")
