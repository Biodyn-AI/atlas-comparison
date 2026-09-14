#!/usr/bin/env python
"""Geneformer MLP — scale the stage-2 detection to catalog which 'facts' the MLP stores
BEYOND the dominant RFX5 immune module. Runs detection on more cells and reports the
DIVERSITY of regulons & cell types across all detector neurons (best-concept per neuron),
with examples. Answers: what non-RFX regulatory facts live in the MLP?

    conda activate scprint
    python scripts/geneformer_catalog.py --n-cells 400
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import argparse, os, pickle, sys
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
    ap = argparse.ArgumentParser(); ap.add_argument("--n-cells", type=int, default=400); Aa = ap.parse_args()
    import numpy as np, scanpy as sc, torch, pandas as pd
    from scipy.stats import rankdata
    from collections import Counter
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

    N = Aa.n_cells
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
    print("==> cell types:", {c: int((cell_type == c).sum()) for c in np.unique(cell_type)})

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
    P = np.zeros((N, nN), np.float32)
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
        A = np.concatenate([acts[li].numpy() for li in range(nL)], -1)
        for bi, ci in enumerate(sub):
            syms = [ensg2sym.get(e, "") for e in esl[bi]]
            valid = [(p, uni_i[sy]) for p, sy in enumerate(syms) if sy in uni_i]
            if valid:
                np.add.at(G, [k for _, k in valid], A[bi, [p for p, _ in valid]])
                np.add.at(cnt, [k for _, k in valid], 1)
            P[ci] = A[bi, :len(esl[bi])].mean(0)
        print(f"  {min(s+8,N)}/{N}", end="\r")
    G /= np.maximum(cnt[:, None], 1)
    print(f"\n==> captured {N} cells")

    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    Rg = rankdata(G, axis=0); best = np.full(nN, 0.5); name = np.array(["-"]*nN, dtype=object)
    for tf, t in regs.items():
        au = auroc(Rg, t, len(uni)); up = np.abs(au-0.5) > np.abs(best-0.5); best[up]=au[up]; name[up]=tf
    rng = np.random.default_rng(0); best0 = np.full(nN, 0.5)
    perm = rng.permutation(len(uni)); Rg0 = rankdata(G[perm], axis=0)
    for tf, t in regs.items():
        au = auroc(Rg0, t, len(uni)); best0 = np.where(np.abs(au-0.5) > np.abs(best0-0.5), au, best0)
    thr = np.percentile(2*np.abs(best0-0.5), 99)
    strong = 2*np.abs(best-0.5) > thr
    print(f"\n== REGULON fact catalog (detectors: {int(strong.sum())}/{nN}, thr {thr:.3f}) ==")
    cnt_reg = Counter(name[strong])
    print("  regulon        #neurons   best example (neuron/layer/AUROC)")
    for reg, c in sorted(cnt_reg.items(), key=lambda x: -x[1])[:14]:
        cand = np.where(strong & (name == reg))[0]; j = cand[np.argmax(np.abs(best[cand]-0.5))]
        print(f"  {reg:<12} {c:>7}     n{j} L{layer_of[j]} {best[j]:.3f}")

    Rp = rankdata(P, axis=0); bct = np.full(nN, 0.5); bctn = np.array(["-"]*nN, dtype=object)
    types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 10 and c != "other"]
    for c in types:
        au = auroc(Rp, np.where(cell_type == c)[0], N); up = np.abs(au-0.5) > np.abs(bct-0.5); bct[up]=au[up]; bctn[up]=c
    perm = rng.permutation(N); bct0 = np.full(nN, 0.5)
    for c in types:
        au = auroc(Rp, np.where(cell_type[perm] == c)[0], N); bct0 = np.where(np.abs(au-0.5) > np.abs(bct0-0.5), au, bct0)
    thrc = np.percentile(2*np.abs(bct0-0.5), 99); strongc = 2*np.abs(bct-0.5) > thrc
    print(f"\n== CELL-TYPE fact catalog (detectors: {int(strongc.sum())}/{nN}) ==")
    for ct, c in sorted(Counter(bctn[strongc]).items(), key=lambda x: -x[1]):
        print(f"  {ct:<6} {c:>4} neurons")
    np.savez_compressed(f"{BASE}/outputs/singlecell/geneformer_catalog.npz",
                        reg_counts=np.array(list(cnt_reg.items()), dtype=object),
                        name=name.astype(str), strong=strong, best=best, layer_of=layer_of)
    print("\n==> catalog done")


if __name__ == "__main__":
    main()
