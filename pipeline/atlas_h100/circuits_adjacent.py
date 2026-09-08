#!/usr/bin/env python
"""Adjacent-layer feature circuits — the methodologically-correct version (coupling between CONSECUTIVE
layers, not the depth-matched jumps that skip 4-5 transformer blocks and understate coupling, which is why
the earlier circuits panel was shelved). Weight-based, no forward:

  coupling C = rownorm(W_enc[L+1]) @ colnorm(W_dec[L])   ->  [d_sae(L+1), d_sae(L)]
  C[g,f] ~ how much feature g at layer L+1 READS what feature f at layer L WROTE into the residual
  (valid to first order because adjacent layers mostly preserve the residual + a block's delta).

Per model, at a mid anchor L (and the two neighbouring transitions), summarise coupling strength +
connectivity, with a shuffled-column baseline. CPU/GPU, SAEs only.
    python circuits_adjacent.py --out out_alllayers   -> out_alllayers/circuits_adjacent.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def load_sae(path, device):
    import torch
    from common.sae import TopKSAE
    ck = torch.load(path, map_location=device); cfg = ck["cfg"]
    s = TopKSAE(int(cfg["d_model"]), int(cfg["expansion"]) * int(cfg["d_model"]), int(cfg["k"])).to(device)
    s.load_state_dict(ck["state_dict"]); s.eval()
    return s


def coupling_stats(Wdec_L, Wenc_L1, device, thr=(0.3, 0.5)):
    """Wdec_L: [d_model, d_sae_L]; Wenc_L1: [d_sae_L1, d_model]. Returns coupling summary for f@L -> g@L+1."""
    import torch
    Wd = Wdec_L / (Wdec_L.norm(dim=0, keepdim=True) + 1e-9)          # unit write directions (cols)
    We = Wenc_L1 / (Wenc_L1.norm(dim=1, keepdim=True) + 1e-9)        # unit read directions (rows)
    # C[g,f] = We[g] . Wd[:,f]  -> [d_sae_L1, d_sae_L]; per f@L take strongest reader g
    out = {}
    with torch.no_grad():
        C = (We @ Wd).abs()                                          # [d_sae_L1, d_sae_L]
        maxcoup = C.max(0).values                                    # per f@L: strongest downstream reader
        out["mean_top1"] = round(float(maxcoup.mean()), 4)
        out["med_top1"] = round(float(maxcoup.median()), 4)
        for t in thr:
            out[f"conn_{int(t*100)}"] = round(float((C > t).float().mean()), 5)  # frac of all pairs coupled
        # shuffled baseline: permute f columns of Wd -> destroys any structure
        perm = torch.randperm(Wd.shape[1], device=device)
        Cs = (We @ Wd[:, perm]).abs()
        out["mean_top1_shuffled"] = round(float(Cs.max(0).values.mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--device", default="cuda"); args = ap.parse_args()
    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    res = {}
    for m in MODELS:
        cats = sorted(glob.glob(f"{args.out}/{m}/sae_L*.pt"),
                      key=lambda p: int(re.search(r"sae_L(\d+)", p).group(1)))
        if len(cats) < 2:
            print(f"  {m}: <2 SAEs, skip", flush=True); continue
        Ls = [int(re.search(r"sae_L(\d+)", p).group(1)) for p in cats]
        mid = len(cats) // 2
        trans = []
        for i in (mid - 1, mid):                                     # two adjacent transitions around the middle
            if i < 0 or i + 1 >= len(cats):
                continue
            sa, sb = load_sae(cats[i], dev), load_sae(cats[i + 1], dev)
            st = coupling_stats(sa.W_dec.weight.detach(), sb.W_enc.weight.detach(), dev)
            st["from"] = Ls[i]; st["to"] = Ls[i + 1]; trans.append(st)
        res[m] = {"layers": Ls, "transitions": trans}
        pk = trans[0] if trans else {}
        print(f"  {m}: L{pk.get('from')}->L{pk.get('to')} mean_top1={pk.get('mean_top1')} "
              f"(shuffled {pk.get('mean_top1_shuffled')}) conn30={pk.get('conn_30')}", flush=True)
    res = {m: res[m] for m in MODELS if m in res}
    json.dump(res, open(f"{args.out}/circuits_adjacent.json", "w"))
    print(f"==> {args.out}/circuits_adjacent.json ({len(res)} models)", flush=True)


if __name__ == "__main__":
    main()
