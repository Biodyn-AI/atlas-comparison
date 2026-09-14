#!/usr/bin/env python
"""scPRINT MLP audit — 3rd model (light: gene-panel forward, fast on CPU). Block MLP =
mlp.fc1 (1024/layer x8 = 8192 neurons) -> GELU -> mlp.fc2. Hooks on mlp.fc1 fire directly
(custom Block, no fused path). Neuron activation = gelu(fc1(x)); ablate by zeroing an fc1
output column (GELU(0)=0). Stage 2 (regulon + cell-type detectors, null) + stage 5 (ablate
-> Δ NB-decoder predicted expression).

    conda activate scprint
    python scripts/scprint_mlp.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B
MARKERS = {"T": ["CD3D", "CD3E", "TRAC", "IL7R"], "B": ["MS4A1", "CD79A", "CD79B"],
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "NK": ["NKG7", "GNLY", "KLRD1"]}


def main():
    import numpy as np, scanpy as sc, torch, torch.nn.functional as Fn, pandas as pd
    from scipy.stats import rankdata
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint

    N2, N5 = 48, 32
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    sym2ens = {}
    for e, s in ens2sym.items():
        sym2ens.setdefault(s, e)
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())

    m = scPrint.load_from_checkpoint(f"{BASE}/ckpt_scprint/medium-v1.5.ckpt",
                                     precpt_gene_emb=None, transformer="normal").eval()
    nL = len(m.transformer.blocks); dff = m.transformer.blocks[0].mlp.fc1.weight.shape[0]
    nN = nL * dff; layer_of = np.repeat(np.arange(nL), dff)
    mgenes = set(m.genes)
    fixed = [sym2ens[s] for s in tr_g if s in sym2ens and sym2ens[s] in mgenes]
    fixed = list(dict.fromkeys(fixed))[:500]
    print(f"==> scPRINT MLP: {nN} neurons ({nL}x{dff}); panel {len(fixed)} TRRUST genes")

    adata = sc.datasets.pbmc3k(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)
    # cell types from markers on full preprocessed expression (var['symbol'])
    vs = adata.var["symbol"].astype(str).str.upper().values
    Xf = adata.layers["norm"] if "norm" in adata.layers else adata.X
    Xf = Xf.toarray() if hasattr(Xf, "toarray") else np.asarray(Xf)
    Xf = np.log1p(Xf)
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    Scr = {}
    for ct, ms in MARKERS.items():
        cols = [np.where(vs == g)[0][0] for g in ms if (vs == g).any()]
        Scr[ct] = np.mean([z(Xf[:, c]) for c in cols], 0) if cols else np.zeros(adata.n_obs)
    S = np.stack([Scr[c] for c in MARKERS], 1); cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")

    def make_batch(n):
        ds = SimpleAnnDataset(adata[:n], obs_to_output=["organism_ontology_term_id"])
        col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=fixed, max_len=0)
        return next(iter(DataLoader(ds, collate_fn=col, batch_size=n, shuffle=False)))

    def run(batch, ablate=None, capture=False):
        gp, expr = batch["genes"], batch["x"]; ng = gp.shape[1]
        cap = {}
        hs = []
        for li, blk in enumerate(m.transformer.blocks):
            def hook(mod, inp, out, li=li):
                if ablate is not None and ablate[0] == li:
                    out = out.clone(); out[..., ablate[1]] = 0.0
                if capture:
                    cap[li] = Fn.gelu(out)[:, out.shape[1]-ng:, :].detach()
                return out
            hs.append(blk.mlp.fc1.register_forward_hook(hook))
        with torch.no_grad():
            o = m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
        for h in hs:
            h.remove()
        A = np.concatenate([cap[li].numpy() for li in range(nL)], -1) if capture else None
        return o["mean"].cpu().numpy(), A, gp

    # ---- stage 2 --------------------------------------------------------
    b2 = make_batch(N2)
    _, A, gp = run(b2, capture=True)
    gene_syms = np.array([ens2sym.get(str(g), "") for g in np.array(m.genes)[gp[0].cpu().numpy()]])
    uni = sorted(set(s for s in gene_syms if s in tr_g)); uni_i = {g: i for i, g in enumerate(uni)}
    regs = {tf: np.array([uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i])
            for tf, grp in tr.groupby(tr.tf.str.upper())}
    regs = {k: v for k, v in regs.items() if len(v) >= 8}
    ct2 = cell_type[:N2]
    print(f"==> {len(uni)} regulon-panel genes, {len(regs)} regulons; cells "
          f"{ {c:int((ct2==c).sum()) for c in np.unique(ct2)} }")
    gi = np.array([uni_i.get(s, -1) for s in gene_syms]); valid = gi >= 0
    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni), np.float32)
    for bi in range(N2):
        np.add.at(G, gi[valid], A[bi, valid]); np.add.at(cnt, gi[valid], 1)
    G /= np.maximum(cnt[:, None], 1)
    P = A.mean(1)                                                          # [N2, nN] per-cell mean

    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    def best(mat, groups, nrows):
        Rk = rankdata(mat, axis=0); b = np.full(nN, 0.5); nm = np.array(["-"]*nN, dtype=object)
        for gn, idx in groups.items():
            au = auroc(Rk, idx, nrows); up = np.abs(au-0.5) > np.abs(b-0.5); b[up]=au[up]; nm[up]=gn
        return b, nm
    breg, bregn = best(G, regs, len(uni)); rng = np.random.default_rng(0)
    breg0, _ = best(G[rng.permutation(len(uni))], regs, len(uni))
    thr = np.percentile(2*np.abs(breg0-0.5), 99)
    print(f"== (a) REGULON detectors: REAL {int((2*np.abs(breg-0.5)>thr).sum())}/{nN} "
          f"({100*(2*np.abs(breg-0.5)>thr).mean():.1f}%) vs NULL {100*(2*np.abs(breg0-0.5)>thr).mean():.1f}%")
    types = [c for c in np.unique(ct2) if (ct2 == c).sum() >= 6 and c != "other"]
    bct, bctn = best(P, {c: np.where(ct2 == c)[0] for c in types}, N2)
    perm = rng.permutation(N2); bct0, _ = best(P, {c: np.where(ct2[perm] == c)[0] for c in types}, N2)
    thrc = np.percentile(2*np.abs(bct0-0.5), 99)
    print(f"== (b) CELL-TYPE detectors: REAL {int((2*np.abs(bct-0.5)>thrc).sum())}/{nN} "
          f"({100*(2*np.abs(bct-0.5)>thrc).mean():.1f}%) vs NULL {100*(2*np.abs(bct0-0.5)>thrc).mean():.1f}%")

    # ---- stage 5 --------------------------------------------------------
    # target selection = strongest POSITIVE detectors (au>0.5), not anti-correlated
    targets = []
    Rp = rankdata(P, axis=0)
    for c in ("B", "Mono", "T"):
        if (ct2 == c).sum() >= 6:
            au = auroc(Rp, np.where(ct2 == c)[0], N2); j = int(au.argmax())
            targets.append(("celltype", c, j, au[j]))
    Rg = rankdata(G, axis=0); posb = np.full(nN, 0.5); posn = np.array(["-"]*nN, dtype=object)
    for tf, t in regs.items():
        au = auroc(Rg, t, len(uni)); up = au > posb; posb[up] = au[up]; posn[up] = tf
    for j in np.argsort(posb)[::-1][:2]:
        targets.append(("regulon", posn[j], int(j), posb[j]))
    for j in rng.integers(0, nN, 2):
        targets.append(("random", "-", int(j), np.nan))

    b5 = make_batch(N5); ct5 = cell_type[:N5]
    base5, _, _ = run(b5)
    print("\n==> CAUSAL ablation (Δ NB predicted expr):")
    print(f"    {'kind':<9} {'concept':<8} {'neuron':>5} {'layer':>4} {'detAUROC':>8} {'causAUROC':>9} {'ratio':>6}")
    for kind, concept, j, det in targets:
        li, loc = j // dff, j % dff
        abl5, _, _ = run(b5, ablate=(li, loc))
        D = np.abs(abl5 - base5)                                           # [N5, ng]
        if kind in ("regulon", "random"):
            dv = D.mean(0); tset = set(regs.get(concept, next(iter(regs.values()))).tolist())
            mem = np.array([uni_i.get(s, -1) in tset for s in gene_syms])
            cau = auroc(rankdata(dv)[:, None], np.where(mem)[0], len(dv))[0] if 3 <= mem.sum() < len(mem) else np.nan
            ratio = dv[mem].mean()/max(dv[~mem].mean(), 1e-9)
        else:
            cd = D.mean(1); y = ct5 == concept
            cau = auroc(rankdata(cd)[:, None], np.where(y)[0], N5)[0] if 3 <= y.sum() < len(y) else np.nan
            ratio = cd[y].mean()/max(cd[~y].mean(), 1e-9)
        ds = f"{det:.3f}" if not np.isnan(det) else "  -  "
        print(f"    {kind:<9} {concept:<8} {j:>5} {li:>4} {ds:>8} {cau:>9.3f} {ratio:>6.2f}")
    print("\n==> scPRINT MLP audit done")


if __name__ == "__main__":
    main()
