#!/usr/bin/env python
"""Tissue-agnostic MLP fact catalog. Point it at ANY single-cell h5ad (Tabula Sapiens
style: Ensembl gene ids in var, real cell types in obs, raw counts) and it runs the
Geneformer MLP stage-2 audit on that tissue — regulon detectors (gene-level) + cell-type
detectors (REAL labels, no marker guessing) vs a permuted null — and prints the tissue's
fact profile. This is the engine for the multi-tissue catalog (blood only shows immune
facts; other tissues surface their own regulons). Loop it over tissues on the H100.

    conda activate scprint
    python scripts/tissue_mlp.py --h5ad <file.h5ad> --n-cells 256 [--layer 5]
"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "/Users/annaantipova/Desktop/biomech"; GF = f"{BASE}/ckpt_geneformer"
DEFAULT = f"{BASE}/external/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"


def load_tissue(h5ad, n_cells, seed=0):
    import scanpy as sc, numpy as np
    a = sc.read_h5ad(h5ad)
    # cell types (real labels): prefer 'cell_type'
    ctcol = next((c for c in ("cell_type", "free_annotation", "cell_ontology_class") if c in a.obs), None)
    cell_type = a.obs[ctcol].astype(str).values
    # raw counts: a.raw.X if integer-like, else a layer, else X
    def pick_counts():
        for src, name in [(a.raw.X if a.raw is not None else None, "raw"),
                          (a.layers.get("decontXcounts") if "decontXcounts" in a.layers else None, "decontX"),
                          (a.layers.get("counts") if "counts" in a.layers else None, "counts"), (a.X, "X")]:
            if src is None:
                continue
            s = src[:20].toarray() if hasattr(src, "toarray") else np.asarray(src[:20])
            if np.allclose(s, np.round(s)) and s.max() > 1:
                return src, name
        return a.X, "X(fallback)"
    counts, cname = pick_counts()
    ensg = np.array([str(g).split(".")[0] for g in a.var.index])       # strip Ensembl version
    # subsample cells
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.n_obs, min(n_cells, a.n_obs), replace=False)
    C = counts[idx]; C = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
    print(f"==> {os.path.basename(h5ad)}: {a.n_obs} cells, counts='{cname}'; using {len(idx)} cells")
    print(f"    cell types: {dict(list(__import__('collections').Counter(cell_type[idx]).most_common(8)))}")
    return C, ensg, cell_type[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=DEFAULT); ap.add_argument("--n-cells", type=int, default=256)
    ap.add_argument("--layer", type=int, default=5); A = ap.parse_args()
    import numpy as np, torch, pandas as pd
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
    ensg2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}

    C, ensg, cell_type = load_tissue(A.h5ad, A.n_cells)
    N = C.shape[0]

    def tokenize(counts_row):
        total = counts_row.sum()
        if total == 0:
            return [], []
        norm = counts_row / total; val, ids, es = [], [], []
        for j, g in enumerate(ensg):
            if counts_row[j] > 0 and g in tok and g in med and med[g] > 0:
                val.append(norm[j] / med[g]); ids.append(tok[g]); es.append(g)
        order = np.argsort(val)[::-1][:2048]
        return [ids[i] for i in order], [es[i] for i in order]
    seqs = [tokenize(C[i]) for i in range(N)]

    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    sym2ens = {}
    for e, s in ensg2sym.items():
        sym2ens.setdefault(s, e)
    uni = sorted(g for g in tr_g if g.upper() in sym2ens and sym2ens[g.upper()] in tok)
    uni_i = {g: i for i, g in enumerate(uni)}
    regs = {tf: np.array([uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i])
            for tf, grp in tr.groupby(tr.tf.str.upper())}
    regs = {k: v for k, v in regs.items() if len(v) >= 15}

    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni)); P = np.zeros((N, nN), np.float32)
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
        Aar = np.concatenate([acts[li].numpy() for li in range(nL)], -1)
        for bi, ci in enumerate(sub):
            syms = [ensg2sym.get(e, "") for e in esl[bi]]
            valid = [(p, uni_i[sy]) for p, sy in enumerate(syms) if sy in uni_i]
            if valid:
                np.add.at(G, [k for _, k in valid], Aar[bi, [p for p, _ in valid]]); np.add.at(cnt, [k for _, k in valid], 1)
            P[ci] = Aar[bi, :len(esl[bi])].mean(0)
        print(f"  {min(s+8,N)}/{N}", end="\r")
    G /= np.maximum(cnt[:, None], 1)

    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    def best(mat, groups, nrows, name):
        Rk = rankdata(mat, axis=0); b = np.full(nN, 0.5); nm = np.array(["-"]*nN, dtype=object)
        for gn, idx in groups.items():
            au = auroc(Rk, idx, nrows); up = au > b; b[up] = au[up]; nm[up] = gn
        return b, nm
    breg, bregn = best(G, regs, len(uni), "reg")
    rng = np.random.default_rng(0); breg0, _ = best(G[rng.permutation(len(uni))], regs, len(uni), "reg")
    thr = np.percentile(breg0, 99)
    types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 12]
    bct, bctn = best(P, {c: np.where(cell_type == c)[0] for c in types}, N, "ct")
    perm = rng.permutation(N); bct0, _ = best(P, {c: np.where(cell_type[perm] == c)[0] for c in types}, N, "ct")
    thrc = np.percentile(bct0, 99)

    print(f"\n\n===== TISSUE FACT PROFILE: {os.path.basename(A.h5ad)} =====")
    print(f"REGULON detectors: {int((breg>thr).sum())}/{nN} vs null 1% | CELL-TYPE detectors: {int((bct>thrc).sum())}/{nN}")
    print("\ntop REGULON facts (by # detector neurons):")
    for reg, c in Counter(bregn[breg > thr]).most_common(12):
        cand = np.where((breg > thr) & (bregn == reg))[0]; j = cand[breg[cand].argmax()]
        print(f"   {reg:<10} {c:>4} neurons   (best n{j} L{layer_of[j]} AUROC {breg[j]:.2f})")
    print("\ntop CELL-TYPE facts (real labels):")
    for ct, c in Counter(bctn[bct > thrc]).most_common(10):
        print(f"   {str(ct)[:40]:<42} {c:>4} neurons")
    print("\n==> tissue done. (Blood gives RFX/CEBPB; a non-immune tissue should surface its own regulons.)")


if __name__ == "__main__":
    main()
