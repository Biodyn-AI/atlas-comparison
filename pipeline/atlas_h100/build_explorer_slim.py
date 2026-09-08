#!/usr/bin/env python
"""Build the Layer-Explorer slim (pts format the atlas renders) for ALL layers, in two sizes:
  explorer_slim_full.json  — every alive feature (for the LOCAL, no-size-limit atlas)
  explorer_slim_light.json — top ~CAP features/layer by firing rate (for the PUBLISHED <=16MB artifact)

Reads: out_alllayers/explorer/<M>_explorer.json (id,x,y,tx,ty,freq,svd + edges, all layers),
       out_alllayers/<M>/feature_catalog_L*.json (top gene per feature),
       out_alllayers/modules_alllayers.json (module + tf per top-degree feature, n_mod, mod_labels),
       out_alllayers/annotations/<M>_L*.json (annotation rate).
pts = [x*100, y*100, module, svd, geneIdx, freqKey, id, tf, tx*100, ty*100]  (x,y=decoder-UMAP; tx,ty=t-SNE)

    python build_explorer_slim.py --out out_alllayers --cap 500
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def load_ann(p):
    if not os.path.exists(p):
        return {}
    d = json.load(open(p)); return d.get("annotations", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--cap", type=int, default=500, help="features/layer in the light build")
    args = ap.parse_args()
    mods = json.load(open(f"{args.out}/modules_alllayers.json"))["graphs"]
    present = [m for m in MODELS if os.path.exists(f"{args.out}/explorer/{m}_explorer.json")]
    full = {"models": present}; light = {"models": present}
    for m in present:
        exp = json.load(open(f"{args.out}/explorer/{m}_explorer.json"))
        genes, gidx = [], {}
        def gi(g):
            g = str(g).upper()
            if g not in gidx:
                gidx[g] = len(genes); genes.append(g)
            return gidx[g]
        Lfull, Llight = {}, {}
        for L, ld in exp["layers"].items():
            LL = int(L)
            cat = json.load(open(f"{args.out}/{m}/feature_catalog_L{LL:02d}.json"))["features"]
            ann = load_ann(f"{args.out}/annotations/{m}_L{LL:02d}_annotations.json")
            n_annot = sum(1 for v in ann.values() if v)
            g = mods.get(m, {}).get(str(LL), {})
            modmap = {int(n["id"]): (int(n["m"]), int(bool(n.get("tf")))) for n in g.get("nodes", [])}
            n_mod = int(g.get("n_mod", 0)); mod_labels = g.get("mod_labels", {})
            pts = []
            for f in ld["features"]:
                fid = int(f["id"])
                tg = cat.get(str(fid), {}).get("top_genes", [])
                gid = gi(tg[0]) if tg else gi("?")
                mm, tf = modmap.get(fid, (n_mod, 0))     # not in the top-degree graph -> uncoloured (grey)
                fk = int(round(float(f.get("freq", 0)) * 1000))
                pts.append([round(f["x"] * 100, 1), round(f["y"] * 100, 1), mm, int(f.get("svd", 0)),
                            gid, fk, fid, tf, round(f.get("tx", f["x"]) * 100, 1), round(f.get("ty", f["y"]) * 100, 1)])
            meta = {"alive": int(ld.get("n_alive", len(pts))), "dead": int(ld.get("dead", 0)),
                    "svd_aligned": int(ld.get("n_svd_aligned", 0)), "n_mod": n_mod,
                    "mod_labels": mod_labels, "rate": round(100 * n_annot / max(len(pts), 1), 1)}
            Lfull[str(LL)] = {**meta, "pts": pts, "n_shown": len(pts)}
            top = sorted(pts, key=lambda p: -p[5])[:args.cap]
            Llight[str(LL)] = {**meta, "pts": top, "n_shown": len(top)}
        full[m] = {"genes": genes, "layers": Lfull}
        light[m] = {"genes": genes, "layers": Llight}
        print(f"{m}: {len(exp['layers'])} layers, {len(genes)} genes, "
              f"full {sum(len(v['pts']) for v in Lfull.values())} pts / light {sum(len(v['pts']) for v in Llight.values())}", flush=True)
    json.dump(full, open(f"{args.out}/explorer_slim_full.json", "w"))
    json.dump(light, open(f"{args.out}/explorer_slim_light.json", "w"))
    fs = os.path.getsize(f"{args.out}/explorer_slim_full.json") // 1024 // 1024
    lsz = os.path.getsize(f"{args.out}/explorer_slim_light.json") // 1024 // 1024
    print(f"==> explorer_slim_full.json ({fs} MB) + explorer_slim_light.json ({lsz} MB)", flush=True)


if __name__ == "__main__":
    main()
