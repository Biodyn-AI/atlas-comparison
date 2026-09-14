#!/usr/bin/env python
"""MLP fact EDITING module (Phase 3) — turn a stored fact up/down and measure the
dose-response. A detector neuron's activation is scaled by alpha (0 = ablate/remove the
fact, 1 = baseline, >1 = amplify), and we read how the model's predictions for THAT
neuron's concept move vs everything else. If target predictions rise monotonically with
alpha while non-targets stay flat, the neuron is a causal, editable 'knob' on the fact.
Demonstrated on Geneformer (CPU); device-agnostic -> ready to scale on H100 (bigger
models, all genes, + rank-1 ROME insertion).

    conda activate scprint
    python scripts/mlp_edit.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B; GF = f"{BASE}/ckpt_geneformer"
MARKERS = {"Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "B": ["MS4A1", "CD79A", "CD79B"],
           "T": ["CD3D", "CD3E", "TRAC", "IL7R"]}


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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForMaskedLM.from_pretrained(GF).eval().to(dev)
    print(f"==> device: {dev}")
    nL, dff = model.config.num_hidden_layers, model.config.intermediate_size
    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))
    med = pickle.load(open(f"{GF}/gene_median_dictionary_gc30M.pkl", "rb"))
    pad_id = tok.get("<pad>", 0)
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}; ensg2sym = {}
    for e, s in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(s).upper(), e); ensg2sym[e] = str(s).upper()

    N = 64
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
    nN = nL * dff

    def batches(idxs, bs=8):
        for s in range(0, len(idxs), bs):
            sub = idxs[s:s+bs]; ids, esl = zip(*[seqs[i] for i in sub])
            L = max(len(x) for x in ids)
            inp = torch.full((len(sub), L), pad_id, dtype=torch.long)
            at = torch.zeros((len(sub), L), dtype=torch.long)
            for k, x in enumerate(ids):
                inp[k, :len(x)] = torch.tensor(x); at[k, :len(x)] = 1
            yield sub, inp.to(dev), at.to(dev), esl

    # ---- detect a strong RFX5 (regulon) neuron + a Mono (cell-type) neuron ----
    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni), np.float32); P = np.zeros((N, nN), np.float32)
    for sub, inp, at, esl in batches(list(range(N))):
        acts = {}; hs = [model.bert.encoder.layer[li].intermediate.register_forward_hook(
            lambda m, i, o, li=li: acts.__setitem__(li, o.detach())) for li in range(nL)]
        with torch.no_grad():
            model(input_ids=inp, attention_mask=at)
        for h in hs:
            h.remove()
        A = np.concatenate([acts[li].cpu().numpy() for li in range(nL)], -1)
        for bi, ci in enumerate(sub):
            syms = [ensg2sym.get(e, "") for e in esl[bi]]
            valid = [(p, uni_i[s]) for p, s in enumerate(syms) if s in uni_i]
            if valid:
                np.add.at(G, [k for _, k in valid], A[bi, [p for p, _ in valid]]); np.add.at(cnt, [k for _, k in valid], 1)
            P[ci] = A[bi, :len(esl[bi])].mean(0)
    G /= np.maximum(cnt[:, None], 1)
    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
    Rg = rankdata(G, axis=0); au_rfx = auroc(Rg, regs["RFX5"], len(uni)); rfx_j = int(au_rfx.argmax())
    Rp = rankdata(P, axis=0); au_mono = auroc(Rp, np.where(cell_type == "Mono")[0], N); mono_j = int(au_mono.argmax())
    print(f"==> editing targets: RFX5 neuron {rfx_j} (L{rfx_j//dff}, det {au_rfx[rfx_j]:.3f}); "
          f"Mono neuron {mono_j} (L{mono_j//dff}, det {au_mono[mono_j]:.3f})")

    # ---- EDIT: scale the neuron's activation by alpha, read the concept -------
    def edited_logits(neuron, alpha):
        li, loc = neuron // dff, neuron % dff
        def hook(m, i, o):
            o = o.clone(); o[..., loc] = o[..., loc] * alpha; return o
        h = model.bert.encoder.layer[li].intermediate.register_forward_hook(hook)
        out = {}
        for sub, inp, at, esl in batches(list(range(N))):
            with torch.no_grad():
                lo = model(input_ids=inp, attention_mask=at).logits
                tl = torch.gather(lo, 2, inp.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for bi, ci in enumerate(sub):
                out[ci] = {ensg2sym.get(esl[bi][p], ""): float(tl[bi, p]) for p in range(len(esl[bi]))}
        h.remove()
        return out

    alphas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    rfx_set = set(uni[k] for k in regs["RFX5"])
    print("\n== EDIT dose-response: RFX5 neuron -> RFX5-target prediction ==")
    print(f"   {'alpha':>6} {'target logit':>13} {'non-target':>11} {'target-nontarget':>17}")
    rfx_curve = []
    for a in alphas:
        L = edited_logits(rfx_j, a)
        tg = [L[c][g] for c in L for g in L[c] if g in rfx_set]
        nt = [L[c][g] for c in L for g in L[c] if g in uni_i and g not in rfx_set]
        rfx_curve.append((a, np.mean(tg), np.mean(nt)))
        print(f"   {a:>6.1f} {np.mean(tg):>13.3f} {np.mean(nt):>11.3f} {np.mean(tg)-np.mean(nt):>17.3f}")

    print("\n== EDIT dose-response: Mono neuron -> prediction shift in Mono vs other cells ==")
    print(f"   {'alpha':>6} {'Mono cells':>11} {'other cells':>12}")
    base = edited_logits(mono_j, 1.0)
    mono_curve = []
    for a in alphas:
        L = edited_logits(mono_j, a)
        dcell = np.array([np.mean([abs(L[c][g]-base[c][g]) for g in base[c] if g in L[c]]) for c in range(N)])
        y = cell_type == "Mono"
        mono_curve.append((a, dcell[y].mean(), dcell[~y].mean()))
        print(f"   {a:>6.1f} {dcell[y].mean():>11.3f} {dcell[~y].mean():>12.3f}")

    np.savez_compressed(f"{BASE}/outputs/singlecell/mlp_edit.npz",
                        rfx_curve=np.array(rfx_curve), mono_curve=np.array(mono_curve),
                        rfx_j=rfx_j, mono_j=mono_j)
    print("\n==> editing demo done (0=remove fact, >1=amplify). Saved mlp_edit.npz")


if __name__ == "__main__":
    main()
