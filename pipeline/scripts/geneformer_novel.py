#!/usr/bin/env python
"""Phase 2 — NOVEL fact discovery (H100-ready, three controls). On SAE features of the
MLP activations, a feature is a candidate novel module only if, AFTER removing the
expression-RANK/positional confound, it is still (i) REPRODUCIBLE across two disjoint
halves of the cells, (ii) COHERENT (a few genes over-activate it), and (iii) UNANNOTATED
by the broad vocabulary. The rank residualization is the key fix for Geneformer (it
encodes expression as token position, so features can lock onto the low-expression tail
— recurring GPCRs/ZZZ3 — which is reproducible+coherent but NOT a module).

    conda activate scprint
    python scripts/geneformer_novel.py --layer 5 --min-obs 30 [--no-residualize]
Real validation still needs external GT (ChIP/perturb-seq) + a full vocab (GO/GPCRs).
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import argparse, os, pickle, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geneformer_mlp_sae import broad_vocab, tokenize_cell
BASE = _B; GF = f"{BASE}/ckpt_geneformer"

KNOWN = {
    "ribosomal": lambda g: g.startswith(("RPL", "RPS", "MRPL", "MRPS")),
    "mito": lambda g: g.startswith("MT-") or g.startswith(("NDUF", "COX", "ATP5")),
    "HLA/immune": lambda g: g.startswith("HLA-") or g in {"B2M", "CD74"},
    "IFN/ISG": lambda g: g.startswith(("IFI", "ISG", "OAS", "MX", "IRF")),
    "histone": lambda g: g.startswith(("HIST", "H2AC", "H2BC", "H3C", "H4C")),
    "heat-shock": lambda g: g.startswith(("HSP", "DNAJ")),
    "GPCR": lambda g: g.startswith("GPR") or g.startswith(("ADGR", "GPN")),
    "AP-1/IEG": lambda g: g in {"FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "DUSP1", "NR4A1", "ZFP36"},
}


def known_flag(top_genes):
    for fam, fn in KNOWN.items():
        if sum(fn(g) for g in top_genes) >= 3:
            return fam
    return "MYSTERIOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5); ap.add_argument("--min-obs", type=int, default=30)
    ap.add_argument("--no-residualize", action="store_true"); A = ap.parse_args()
    import numpy as np, scanpy as sc, torch, pandas as pd
    from scipy.stats import rankdata
    from transformers import BertForMaskedLM
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    model = BertForMaskedLM.from_pretrained(GF).eval()
    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))
    med = pickle.load(open(f"{GF}/gene_median_dictionary_gc30M.pkl", "rb"))
    pad_id = tok.get("<pad>", 0)
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}; ensg2sym = {}
    for e, s in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(s).upper(), e); ensg2sym[e] = str(s).upper()

    N = 128
    adata = sc.datasets.pbmc3k()[:N].copy()
    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    ensg = np.array([sym2ensg.get(s.upper(), "NA") for s in adata.var_names])
    seqs = [tokenize_cell(Xd[i], ensg, tok, med) for i in range(N)]

    acts, gsym_rows, cell_rows, pos_rows = [], [], [], []
    for s in range(0, N, 8):
        sub = list(range(s, min(s+8, N))); ids, esl = zip(*[seqs[i] for i in sub])
        L = max(len(x) for x in ids)
        inp = torch.full((len(sub), L), pad_id, dtype=torch.long); at = torch.zeros((len(sub), L), dtype=torch.long)
        for k, x in enumerate(ids):
            inp[k, :len(x)] = torch.tensor(x); at[k, :len(x)] = 1
        cap = {}
        h = model.bert.encoder.layer[A.layer].intermediate.register_forward_hook(
            lambda m, i, o: cap.__setitem__("a", o.detach()))
        with torch.no_grad():
            model(input_ids=inp, attention_mask=at)
        h.remove()
        Aar = cap["a"].numpy()
        for bi, ci in enumerate(sub):
            n = len(esl[bi]); acts.append(Aar[bi, :n])
            gsym_rows.extend([ensg2sym.get(e, "") for e in esl[bi]]); cell_rows.extend([ci]*n)
            pos_rows.extend(range(n))                                   # token position = expression rank
    X = np.concatenate(acts, 0); gsym = np.array(gsym_rows); cellid = np.array(cell_rows); posn = np.array(pos_rows)

    subs = np.random.default_rng(0).choice(X.shape[0], min(40000, X.shape[0]), replace=False)
    sae, stats, log = train_sae(X[subs], SAEConfig(d_sae=4096, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    print(f"==> SAE FVU {log.history[-1]['fvu']:.3f}; {X.shape[0]} tokens")

    uni = sorted(set(g for g in gsym if g)); uni_i = {g: i for i, g in enumerate(uni)}
    gi = np.array([uni_i.get(g, -1) for g in gsym]); ok = gi >= 0
    U = F.shape[1]

    def agg(mask):
        G = np.zeros((len(uni), U), np.float32); c = np.zeros(len(uni)); pos = np.zeros(len(uni))
        m = ok & mask
        np.add.at(G, gi[m], F[m]); np.add.at(c, gi[m], 1); np.add.at(pos, gi[m], posn[m])
        return G, c, pos
    hf = (cellid % 2 == 0)
    G, cnt, possum = agg(np.ones_like(ok)); GA, cA, _ = agg(hf); GB, cB, _ = agg(~hf)
    # ubiquity / background control: genes present in ~all cells are the always-on housekeeping
    # background (mito/ribosomal/B2M/TMSB4X). Keep only stable BUT cell-variable genes, so a
    # feature's top genes reflect a specific program, not the shared background.
    present = np.zeros((len(uni), N), bool); present[gi[ok], cellid[ok]] = True
    cellfrac = present.sum(1) / N
    high = (cnt >= A.min_obs) & (cellfrac < 0.6)
    hi = np.where(high)[0]; nH = len(hi)
    print(f"==> {int((cnt>=A.min_obs).sum())} stable genes; {nH} after dropping ubiquitous (cellfrac>=0.6)")
    posrank = (possum[hi] / np.maximum(cnt[hi], 1)).astype(np.float64)   # per-gene mean expression rank
    print(f"==> genes with >= {A.min_obs} obs: {nH}/{len(uni)}")

    def resid(Gsub):
        """residualize each feature's per-gene activation against gene mean rank (+ rank^2)."""
        Z = Gsub[hi]                                                     # [nH, U]
        if A.no_residualize:
            return Z
        Xd_ = np.column_stack([np.ones(nH), posrank, posrank**2])        # design
        beta, *_ = np.linalg.lstsq(Xd_, Z, rcond=None)
        return Z - Xd_ @ beta
    Rg = resid(G); RA = resid(GA); RB = resid(GB)
    # per-gene centering: subtract each gene's mean activation ACROSS features, so genes that are
    # globally high for every feature (the recurring MT/RPL/housekeeping background) don't dominate
    # every feature's top-gene list. Leaves each feature's gene-SPECIFIC over-activation.
    Rg = Rg - Rg.mean(1, keepdims=True); RA = RA - RA.mean(1, keepdims=True); RB = RB - RB.mean(1, keepdims=True)
    # diagnostic: how position-confounded were features before residualization
    Zc = G[hi]; corr = np.array([np.corrcoef(Zc[:, f], posrank)[0, 1] if Zc[:, f].std() > 0 else 0 for f in range(U)])
    print(f"==> features strongly position-confounded (|corr(activation, rank)|>0.3): "
          f"{int((np.abs(corr)>0.3).sum())}/{U}  <- what the rank control removes")

    order = np.argsort(Rg, 0)[::-1]                                     # top genes by residual (over-activation)
    tA = np.argsort(RA, 0)[::-1][:10]; tB = np.argsort(RB, 0)[::-1][:10]
    repro = np.array([len(set(tA[:, f]) & set(tB[:, f]))/10 for f in range(U)])
    pos_res = np.clip(Rg, 0, None)
    conc = np.array([pos_res[order[:10, f], f].sum()/(pos_res[:, f].sum()+1e-9) for f in range(U)])

    vocab, fam = broad_vocab(set(uni))
    remap = {g: k for k, g in enumerate(hi)}
    vidx = {k: np.array([remap[uni_i[g]] for g in s if g in uni_i and uni_i[g] in remap]) for k, s in vocab.items()}
    vidx = {k: v for k, v in vidx.items() if len(v) >= 8}
    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    Rk = rankdata(Rg, axis=0); best = np.full(U, 0.5)
    for _, idx in vidx.items():
        best = np.maximum(best, auroc(Rk, idx, nH))
    rng = np.random.default_rng(1); Rk0 = rankdata(Rg[rng.permutation(nH)], axis=0); best0 = np.full(U, 0.5)
    for _, idx in vidx.items():
        best0 = np.maximum(best0, auroc(Rk0, idx, nH))
    thr = np.percentile(best0, 99)

    live = F.std(0) > 1e-4
    reproducible = repro >= 0.4
    coherent = conc > np.percentile(conc[live], 75)
    annotated = best > thr
    novel = live & reproducible & coherent & (~annotated)
    print(f"\n== filter cascade (rank-residualized; {int(live.sum())} live SAE features) ==")
    print(f"   reproducible across halves      : {int((live&reproducible).sum())}")
    print(f"   + coherent (post-residual)      : {int((live&reproducible&coherent).sum())}")
    print(f"   + unannotated                   : {int(novel.sum())}  <- NOVEL candidates")

    cand = np.where(novel)[0]; cand = cand[np.argsort(repro[cand]*conc[cand])[::-1]]
    print(f"\n== rank-controlled novel-candidate modules ({len(cand)}) ==")
    print(f"   {'feat':>5} {'repro':>5} {'flag':<11} top genes (over-activated beyond expression rank)")
    myst = 0
    for f in cand[:15]:
        tg = [uni[hi[i]] for i in order[:8, f]]; fl = known_flag(tg); myst += fl == "MYSTERIOUS"
        print(f"   {f:>5} {repro[f]:>5.1f} {fl:<11} {', '.join(tg)}")
    print(f"\n   {myst}/{min(15,len(cand))} MYSTERIOUS after rank control. Real validation = external GT + full vocab.")
    # save candidate signatures (top-20 genes each) for multi-label annotation + external validation
    import os as _os
    sigs = [",".join(uni[hi[i]] for i in order[:20, f]) for f in cand]
    np.savez_compressed(f"{BASE}/outputs/singlecell/novel_candidates.npz",
                        feats=np.array(cand), layer=np.full(len(cand), A.layer), sigs=np.array(sigs, dtype=object))
    print(f"==> saved {len(cand)} candidate signatures -> outputs/singlecell/novel_candidates.npz "
          f"(feed to scripts/mlp_validate.py)")


if __name__ == "__main__":
    main()
