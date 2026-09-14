#!/usr/bin/env python
"""How many MLP neurons can we actually describe? Honest coverage measurement on
Geneformer: per neuron, is it (A) NAMED — matches a cell-type or TRRUST regulon above
the permuted null; (B) ACTIVE but UNNAMED — fires coherently but no DB match (candidate
novel / polysemantic); or (C) DEAD/flat — nothing to describe. Answers "can we describe
ALL MLPs, or only some?" with a number.

    conda activate scprint
    python scripts/geneformer_coverage.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B; GF = f"{BASE}/ckpt_geneformer"
MARKERS = {"T": ["CD3D", "CD3E", "TRAC", "IL7R"], "B": ["MS4A1", "CD79A", "CD79B"],
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "NK": ["NKG7", "GNLY", "KLRD1"]}


def tokenize_cell(counts, ensg, tok, med, max_len=2048):
    import numpy as np
    total = counts.sum()
    if total == 0:
        return [], []
    norm = counts / total; val, ids, es = [], [], []
    for j, g in enumerate(ensg):
        if counts[j] > 0 and g in tok and g in med and med[g] > 0:
            val.append(norm[j] / med[g]); ids.append(tok[g]); es.append(g)
    if not val:
        return [], []
    order = np.argsort(val)[::-1][:max_len]
    return [ids[i] for i in order], [es[i] for i in order]


def main():
    import numpy as np, scanpy as sc, torch, pandas as pd
    from scipy.stats import rankdata
    from transformers import BertForMaskedLM

    model = BertForMaskedLM.from_pretrained(GF).eval()
    nL, dff = model.config.num_hidden_layers, model.config.intermediate_size
    nN = nL * dff; layer_of = np.repeat(np.arange(nL), dff)
    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))
    med = pickle.load(open(f"{GF}/gene_median_dictionary_gc30M.pkl", "rb"))
    pad_id = tok.get("<pad>", 0)
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}; ensg2sym = {}
    for e, s in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(s).upper(), e); ensg2sym[e] = str(s).upper()

    N = 128
    adata = sc.datasets.pbmc3k()[:N].copy()
    raw = adata.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    Sc = {}
    for ct, ms in MARKERS.items():
        pres = [g for g in ms if g in raw.var_names]
        X = raw[:, pres].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        Sc[ct] = np.mean([z(X[:, j]) for j in range(X.shape[1])], 0)
    S = np.stack([Sc[c] for c in MARKERS], 1); cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")

    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    ensg = np.array([sym2ensg.get(s.upper(), "NA") for s in adata.var_names])
    seqs = [tokenize_cell(Xd[i], ensg, tok, med) for i in range(N)]
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    uni = sorted(g for g in tr_g if g.upper() in sym2ensg and sym2ensg[g.upper()] in tok)
    uni_i = {g: i for i, g in enumerate(uni)}
    regs = {tf: np.array([uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i])
            for tf, grp in tr.groupby(tr.tf.str.upper())}
    regs = {k: v for k, v in regs.items() if len(v) >= 15}

    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni), np.float32)
    P = np.zeros((N, nN), np.float32); actsum = np.zeros(nN); actsq = np.zeros(nN); nfire = np.zeros(nN); ntok = 0
    for s in range(0, N, 8):
        sub = list(range(s, min(s+8, N))); ids, esl = zip(*[seqs[i] for i in sub])
        L = max(len(x) for x in ids)
        inp = torch.full((len(sub), L), pad_id, dtype=torch.long); at = torch.zeros((len(sub), L), dtype=torch.long)
        for k, x in enumerate(ids):
            inp[k, :len(x)] = torch.tensor(x); at[k, :len(x)] = 1
        acts = {}; hs = [model.bert.encoder.layer[li].intermediate.register_forward_hook(
            lambda m, i, o, li=li: acts.__setitem__(li, o.detach())) for li in range(nL)]
        with torch.no_grad():
            model(input_ids=inp, attention_mask=at)
        for h in hs:
            h.remove()
        A = np.concatenate([acts[li].numpy() for li in range(nL)], -1)     # [b,L,nN]
        for bi, ci in enumerate(sub):
            n = len(esl[bi]); a = A[bi, :n]
            syms = [ensg2sym.get(e, "") for e in esl[bi]]
            valid = [(p, uni_i[sy]) for p, sy in enumerate(syms) if sy in uni_i]
            if valid:
                np.add.at(G, [k for _, k in valid], a[[p for p, _ in valid]]); np.add.at(cnt, [k for _, k in valid], 1)
            P[ci] = a.mean(0)
            actsum += a.sum(0); actsq += (a**2).sum(0); nfire += (a > 0.05).sum(0); ntok += n
    G /= np.maximum(cnt[:, None], 1)
    amean = actsum/ntok; astd = np.sqrt(np.maximum(actsq/ntok - amean**2, 0)); firefrac = nfire/ntok

    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    def bestpos(mat, groups, nrows):
        Rk = rankdata(mat, axis=0); b = np.full(nN, 0.5)
        for gn, idx in groups.items():
            au = auroc(Rk, idx, nrows); b = np.maximum(b, au)          # positive detectors only
        return b
    rng = np.random.default_rng(0)
    reg_s = bestpos(G, regs, len(uni)); reg_null = bestpos(G[rng.permutation(len(uni))], regs, len(uni))
    types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 8 and c != "other"]
    ct_s = bestpos(P, {c: np.where(cell_type == c)[0] for c in types}, N)
    perm = rng.permutation(N); ct_null = bestpos(P, {c: np.where(cell_type[perm] == c)[0] for c in types}, N)
    thr_r = np.percentile(reg_null, 99); thr_c = np.percentile(ct_null, 99)

    dead = (astd < 1e-3) | (firefrac < 0.002)
    named = (~dead) & ((reg_s > thr_r) | (ct_s > thr_c))
    unnamed = (~dead) & (~named)
    print(f"==> Geneformer MLP coverage over {nN} neurons ({nL}x{dff}), {N} cells, "
          f"vocab = {len(regs)} regulons + {len(types)} cell types")
    print(f"   (A) NAMED (cell-type or regulon > null)     : {named.sum():>4} ({100*named.mean():.1f}%)")
    print(f"       - regulon-named                          : {((~dead)&(reg_s>thr_r)).sum():>4} ({100*((~dead)&(reg_s>thr_r)).mean():.1f}%)")
    print(f"       - cell-type-named                        : {((~dead)&(ct_s>thr_c)).sum():>4} ({100*((~dead)&(ct_s>thr_c)).mean():.1f}%)")
    print(f"   (B) ACTIVE but UNNAMED (no DB match)         : {unnamed.sum():>4} ({100*unnamed.mean():.1f}%)")
    print(f"   (C) DEAD / flat                              : {dead.sum():>4} ({100*dead.mean():.1f}%)")
    print(f"\n   -> we can NAME {100*named.mean():.0f}% with just regulons+cell-types; "
          f"{100*unnamed.mean():.0f}% fire coherently but need a broader vocabulary / SAE; "
          f"{100*dead.mean():.0f}% dead.")
    np.savez_compressed(f"{BASE}/outputs/singlecell/geneformer_coverage.npz",
                        named=named, unnamed=unnamed, dead=dead, layer_of=layer_of,
                        reg_s=reg_s, ct_s=ct_s, firefrac=firefrac)
    # coverage by layer
    print("\n   named fraction by layer:", {int(li): round(float(named[layer_of == li].mean()), 2) for li in range(nL)})
    print("==> coverage done")


if __name__ == "__main__":
    main()
