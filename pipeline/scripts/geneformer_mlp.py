#!/usr/bin/env python
"""Geneformer MLP audit — cross-model confirmation of scGPT's key-value memory. Stage 2
(keys: regulon + cell-type detectors, null-controlled) + stage 5 (causal: ablate a
detector -> Δ MLM logit of the true gene token -> does it isolate the neuron's concept?).

    conda activate scprint
    python scripts/geneformer_mlp.py
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
    print(f"==> {len(uni)} TRRUST genes, {len(regs)} regulons, {nN} neurons")

    def batches(idxs, bs=8):
        for s in range(0, len(idxs), bs):
            sub = idxs[s:s+bs]; ids, esl = zip(*[seqs[i] for i in sub])
            L = max(len(x) for x in ids)
            inp = torch.full((len(sub), L), pad_id, dtype=torch.long)
            at = torch.zeros((len(sub), L), dtype=torch.long)
            for k, x in enumerate(ids):
                inp[k, :len(x)] = torch.tensor(x); at[k, :len(x)] = 1
            yield sub, inp, at, esl

    def hook_capture():
        acts = {}
        hs = [model.bert.encoder.layer[li].intermediate.register_forward_hook(
            lambda m, i, o, li=li: acts.__setitem__(li, o.detach())) for li in range(nL)]
        return acts, hs

    # ---- stage 2: capture activations over corpus -----------------------
    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni), np.float32)
    P = np.zeros((N, nN), np.float32)
    for sub, inp, at, esl in batches(list(range(N))):
        acts, hs = hook_capture()
        with torch.no_grad():
            model(input_ids=inp, attention_mask=at)
        for h in hs:
            h.remove()
        A = np.concatenate([acts[li].numpy() for li in range(nL)], -1)   # [b, L, nN]
        for bi, ci in enumerate(sub):
            es = esl[bi]; syms = [ensg2sym.get(e, "") for e in es]
            valid = [(p, uni_i[s]) for p, s in enumerate(syms) if s in uni_i]
            if valid:
                gt = [p for p, _ in valid]; gi = [k for _, k in valid]
                np.add.at(G, gi, A[bi, gt]); np.add.at(cnt, gi, 1)
            P[ci] = A[bi, :len(es)].mean(0)
    G /= np.maximum(cnt[:, None], 1)
    print(f"==> captured; genes covered {int((cnt>0).sum())}/{len(uni)}")

    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))

    def best_reg(Gmat):
        Rk = rankdata(Gmat, axis=0); best = np.full(nN, 0.5); name = np.array(["-"]*nN, dtype=object)
        for tf, t in regs.items():
            au = auroc(Rk, t, len(uni)); up = np.abs(au-0.5) > np.abs(best-0.5); best[up]=au[up]; name[up]=tf
        return best, name
    breg, bregn = best_reg(G)
    rng = np.random.default_rng(0)
    breg0, _ = best_reg(G[rng.permutation(len(uni))])
    thr_r = np.percentile(2*np.abs(breg0-0.5), 99)
    print(f"\n== (a) REGULON detectors: REAL {int((2*np.abs(breg-0.5)>thr_r).sum())}/{nN} "
          f"({100*(2*np.abs(breg-0.5)>thr_r).mean():.1f}%) vs NULL "
          f"{int((2*np.abs(breg0-0.5)>thr_r).sum())}/{nN} ({100*(2*np.abs(breg0-0.5)>thr_r).mean():.1f}%)")

    Rp = rankdata(P, axis=0); bct = np.full(nN, 0.5); bctn = np.array(["-"]*nN, dtype=object)
    types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 8 and c != "other"]
    for c in types:
        au = auroc(Rp, np.where(cell_type == c)[0], N); up = np.abs(au-0.5) > np.abs(bct-0.5); bct[up]=au[up]; bctn[up]=c
    perm = rng.permutation(N); bct0 = np.full(nN, 0.5)
    for c in types:
        au = auroc(Rp, np.where(cell_type[perm] == c)[0], N); bct0 = np.where(np.abs(au-0.5)>np.abs(bct0-0.5), au, bct0)
    thr_c = np.percentile(2*np.abs(bct0-0.5), 99)
    print(f"== (b) CELL-TYPE detectors: REAL {int((2*np.abs(bct-0.5)>thr_c).sum())}/{nN} "
          f"({100*(2*np.abs(bct-0.5)>thr_c).mean():.1f}%) vs NULL {int((2*np.abs(bct0-0.5)>thr_c).sum())}/{nN}")

    # ---- stage 5: causal ablation on top detectors ---------------------
    targets = []
    for c in ("B", "Mono", "T"):
        cand = np.where(bctn == c)[0]
        if len(cand):
            j = cand[np.argmax(np.abs(bct[cand]-0.5))]; targets.append(("celltype", c, int(j), bct[j]))
    for j in np.argsort(np.abs(breg-0.5))[::-1][:3]:
        targets.append(("regulon", bregn[j], int(j), breg[j]))
    for j in rng.integers(0, nN, 3):
        targets.append(("random", "-", int(j), np.nan))

    idxs5 = list(range(min(64, N)))

    def true_logit(ablate=None):
        """return {cell: {sym: logit_of_true_token}} on idxs5."""
        res = {}
        for sub, inp, at, esl in batches(idxs5):
            hs = []
            if ablate is not None:
                li, loc = ablate
                def abl_hook(m, i, o, loc=loc):
                    o = o.clone(); o[..., loc] = 0.0; return o
                hs.append(model.bert.encoder.layer[li].intermediate.register_forward_hook(abl_hook))
            with torch.no_grad():
                lo = model(input_ids=inp, attention_mask=at).logits
                tl = torch.gather(lo, 2, inp.unsqueeze(-1)).squeeze(-1)   # [b,L] logit of true token
            for h in hs:
                h.remove()
            for bi, ci in enumerate(sub):
                es = esl[bi]
                res[ci] = {ensg2sym.get(es[p], ""): float(tl[bi, p]) for p in range(len(es))}
        return res

    base = true_logit(None)
    print("\n==> CAUSAL ablation (Δ MLM logit of true gene):")
    print(f"    {'kind':<9} {'concept':<8} {'neuron':>5} {'layer':>5} {'detAUROC':>9} {'causAUROC':>9} {'ratio':>6}")
    for kind, concept, j, det in targets:
        li, loc = j // dff, j % dff
        abl = true_logit((li, loc))
        if kind in ("regulon", "random"):
            per = {}
            for c in base:
                for g, v in base[c].items():
                    if g in abl[c] and g in uni_i:
                        per.setdefault(g, []).append(abs(abl[c][g]-v))
            gg = [g for g in per]; dv = np.array([np.mean(per[g]) for g in gg])
            tset = set(regs.get(concept, next(iter(regs.values()))).tolist())
            mem = np.array([uni_i[g] in tset for g in gg])
            cau = auroc(rankdata(dv)[:, None], np.where(mem)[0], len(dv))[0] if 3 <= mem.sum() < len(mem) else np.nan
            ratio = dv[mem].mean()/max(dv[~mem].mean(), 1e-9) if mem.sum() else np.nan
        else:
            cd = np.array([np.mean([abs(abl[c][g]-base[c][g]) for g in base[c] if g in abl[c]]) for c in idxs5])
            y = cell_type[idxs5] == concept
            cau = auroc(rankdata(cd)[:, None], np.where(y)[0], len(idxs5))[0] if 3 <= y.sum() < len(y) else np.nan
            ratio = cd[y].mean()/max(cd[~y].mean(), 1e-9)
        ds = f"{det:.3f}" if not np.isnan(det) else "  -  "
        print(f"    {kind:<9} {concept:<8} {j:>5} {li:>5} {ds:>9} {cau:>9.3f} {ratio:>6.2f}")
    print("\n==> Geneformer cross-model done")


if __name__ == "__main__":
    main()
