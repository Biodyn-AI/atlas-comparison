#!/usr/bin/env python
"""AIDO.Cell MLP audit — 3rd model in the cross-model key-value check. AIDO uses a GATED
(SwiGLU) MLP: neuron activation = silu(gate_proj(x))*up_proj(x) = the INPUT to down_proj
(640/layer x8 = 5120 neurons); value = down_proj columns. Capture/ablate via a pre-hook
on down_proj. Stage 2 (regulon + cell-type detectors, null) + stage 5 (ablate -> Δ
predicted expression-bin distribution).

    conda activate scprint
    python scripts/aido_mlp.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B
MARKERS = {"T": ["CD3D", "CD3E", "TRAC", "IL7R"], "B": ["MS4A1", "CD79A", "CD79B"],
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "NK": ["NKG7", "GNLY", "KLRD1"]}


def main():
    import numpy as np, scanpy as sc, torch, torch.nn as nn, pandas as pd
    from scipy.stats import rankdata
    from gb_cell.models import CellFoundationConfig
    from gb_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
    from gb_cell.utils import align_adata, preprocess_counts

    N2, N5 = 32, 16
    genes = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    sym2pos = {g: i for i, g in enumerate(genes)}; ng = len(genes)
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    keep = sorted(sym2pos[g] for g in tr_g if g in sym2pos); keep_sym = np.array([genes[i] for i in keep])
    uni = sorted(set(keep_sym)); uni_i = {g: i for i, g in enumerate(uni)}
    regs = {tf: np.array([uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i])
            for tf, grp in tr.groupby(tr.tf.str.upper())}
    regs = {k: v for k, v in regs.items() if len(v) >= 15}

    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationForMaskedLM.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    nL, dff = cfg.num_hidden_layers, 640; nN = nL * dff; layer_of = np.repeat(np.arange(nL), dff)
    blocks = next(mod for _, mod in m.named_modules() if isinstance(mod, nn.ModuleList) and len(mod) == nL)
    print(f"==> AIDO gated-MLP: {nN} neurons ({nL}x{dff}); {len(uni)} TRRUST genes, {len(regs)} regulons")

    adata = sc.datasets.pbmc3k()[:N2].copy()
    raw = adata.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    Scr = {}
    for ct, ms in MARKERS.items():
        pres = [g for g in ms if g in raw.var_names]
        X = raw[:, pres].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        Scr[ct] = np.mean([z(X[:, j]) for j in range(X.shape[1])], 0)
    S = np.stack([Scr[c] for c in MARKERS], 1); cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")
    print("==> cell types:", {c: int((cell_type == c).sum()) for c in np.unique(cell_type)})

    ad, attn = align_adata(adata); attn_t = torch.from_numpy(attn).unsqueeze(0)
    keep_t = torch.as_tensor(keep)

    def forward_batch(xb, ablate=None, want_logits=False):
        inp = preprocess_counts(xb, device="cpu")
        am = torch.cat([attn_t.repeat(inp.shape[0], 1), torch.ones((inp.shape[0], 2))], 1)
        cap = {}
        hs = []
        for li, blk in enumerate(blocks):
            def pre(mod, a, li=li):
                x = a[0]
                if ablate is not None and ablate[0] == li:
                    x = x.clone(); x[..., ablate[1]] = 0.0; a = (x,) + a[1:]
                cap[li] = x[:, keep_t, :].detach() if not want_logits else None
                return a
            hs.append(blk.mlp.down_proj.register_forward_pre_hook(pre))
        with torch.no_grad():
            out = m(input_ids=inp, attention_mask=am)
        for h in hs:
            h.remove()
        logits = (out.logits if hasattr(out, "logits") else out[0]).float()[:, keep_t, :].numpy() if want_logits else None
        return cap, logits

    # ---- stage 2: capture activations -----------------------------------
    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni), np.float32)
    P = np.zeros((N2, nN), np.float32)
    Xall = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
    for s in range(0, N2, 4):
        cap, _ = forward_batch(Xall[s:s+4])
        Akeep = np.concatenate([cap[li].numpy() for li in range(nL)], -1)   # [b, |keep|, nN]
        for bi in range(Akeep.shape[0]):
            np.add.at(G, [uni_i[g] for g in keep_sym], Akeep[bi]); np.add.at(cnt, [uni_i[g] for g in keep_sym], 1)
            P[s+bi] = Akeep[bi].mean(0)
        print(f"  s2 {min(s+4,N2)}/{N2}", end="\r")
    G /= np.maximum(cnt[:, None], 1)
    print(f"\n==> captured")

    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    def best(mat, groups, nrows):
        Rk = rankdata(mat, axis=0); b = np.full(nN, 0.5); nm = np.array(["-"]*nN, dtype=object)
        for gname, idx in groups.items():
            au = auroc(Rk, idx, nrows); up = np.abs(au-0.5) > np.abs(b-0.5); b[up]=au[up]; nm[up]=gname
        return b, nm
    breg, bregn = best(G, {k: v for k, v in regs.items()}, len(uni))
    rng = np.random.default_rng(0); breg0, _ = best(G[rng.permutation(len(uni))], regs, len(uni))
    thr = np.percentile(2*np.abs(breg0-0.5), 99)
    print(f"== (a) REGULON detectors: REAL {int((2*np.abs(breg-0.5)>thr).sum())}/{nN} "
          f"({100*(2*np.abs(breg-0.5)>thr).mean():.1f}%) vs NULL {100*(2*np.abs(breg0-0.5)>thr).mean():.1f}%")
    types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 6 and c != "other"]
    bct, bctn = best(P, {c: np.where(cell_type == c)[0] for c in types}, N2)
    perm = rng.permutation(N2); bct0, _ = best(P, {c: np.where(cell_type[perm] == c)[0] for c in types}, N2)
    thrc = np.percentile(2*np.abs(bct0-0.5), 99)
    print(f"== (b) CELL-TYPE detectors: REAL {int((2*np.abs(bct-0.5)>thrc).sum())}/{nN} "
          f"({100*(2*np.abs(bct-0.5)>thrc).mean():.1f}%) vs NULL {100*(2*np.abs(bct0-0.5)>thrc).mean():.1f}%")

    # ---- stage 5: causal on top detectors ------------------------------
    targets = []
    for c in ("B", "Mono", "T"):
        cand = np.where(bctn == c)[0]
        if len(cand):
            j = cand[np.argmax(np.abs(bct[cand]-0.5))]; targets.append(("celltype", c, int(j), bct[j]))
    for j in np.argsort(np.abs(breg-0.5))[::-1][:2]:
        targets.append(("regulon", bregn[j], int(j), breg[j]))
    for j in rng.integers(0, nN, 2):
        targets.append(("random", "-", int(j), np.nan))

    Xd5 = Xall[:N5]; ct5 = cell_type[:N5]
    base5 = np.concatenate([forward_batch(Xd5[s:s+4], want_logits=True)[1] for s in range(0, N5, 4)], 0)
    def pred_delta(ablate):
        a = np.concatenate([forward_batch(Xd5[s:s+4], ablate=ablate, want_logits=True)[1]
                            for s in range(0, N5, 4)], 0)
        return np.abs(a - base5).mean(-1)                                  # [N5, |keep|] mean|Δlogit| over bins

    print("\n==> CAUSAL ablation (Δ predicted expr-bin distribution):")
    print(f"    {'kind':<9} {'concept':<8} {'neuron':>5} {'layer':>4} {'detAUROC':>8} {'causAUROC':>9} {'ratio':>6}")
    for kind, concept, j, det in targets:
        li, loc = j // dff, j % dff
        D = pred_delta((li, loc))                                          # [N5, |keep|]
        if kind in ("regulon", "random"):
            dv = D.mean(0)                                                 # per gene
            tset = set(regs.get(concept, next(iter(regs.values()))).tolist())
            mem = np.array([uni_i.get(g, -1) in tset for g in keep_sym])
            cau = auroc(rankdata(dv)[:, None], np.where(mem)[0], len(dv))[0] if 3 <= mem.sum() < len(mem) else np.nan
            ratio = dv[mem].mean()/max(dv[~mem].mean(), 1e-9)
        else:
            cd = D.mean(1); y = ct5 == concept
            cau = auroc(rankdata(cd)[:, None], np.where(y)[0], N5)[0] if 3 <= y.sum() < len(y) else np.nan
            ratio = cd[y].mean()/max(cd[~y].mean(), 1e-9)
        ds = f"{det:.3f}" if not np.isnan(det) else "  -  "
        print(f"    {kind:<9} {concept:<8} {j:>5} {li:>4} {ds:>8} {cau:>9.3f} {ratio:>6.2f}")
    print("\n==> AIDO MLP audit done")


if __name__ == "__main__":
    main()
