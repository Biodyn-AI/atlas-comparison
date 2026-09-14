#!/usr/bin/env python
"""ROME rank-1 fact editing + RIGOROUS validation (Phase 3). Following Meng et al. 2022,
insert a regulatory fact into the MLP weights least-disturbingly:
    ΔW = (v* - W k*)(C^-1 k*)^T / (k*^T C^-1 k*),   C = E[k k^T], k = MLP post-activation.
The edit is COMPUTED on TRAIN cells and EVALUATED on held-out cells, with a control
regulon, giving the three ROME metrics:
  - EFFICACY      : target regulon's predictions rise on the edited cells.
  - GENERALIZATION: they also rise on HELD-OUT cells (not used for k*/C).
  - SPECIFICITY   : non-target genes AND a control regulon stay flat.
Device-agnostic (uses cuda if available -> H100-ready).

    conda activate scprint
    python scripts/mlp_rome.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B; GF = f"{BASE}/ckpt_geneformer"


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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LAYER, GAMMA = 4, 1.5
    TARGET_TFS = ("RFX5", "RFXANK", "RFXAP")     # fact to insert
    CONTROL_TF = "E2F1"                          # unrelated regulon (specificity check)
    model = BertForMaskedLM_load(dev)
    print(f"==> device {dev}; editing layer {LAYER}, gamma {GAMMA}; target=RFX, control={CONTROL_TF}")
    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))
    med = pickle.load(open(f"{GF}/gene_median_dictionary_gc30M.pkl", "rb"))
    pad_id = tok.get("<pad>", 0)
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}
    for e, s in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(s).upper(), e)

    N, NTRAIN = 80, 48
    TRAIN, HELD = list(range(NTRAIN)), list(range(NTRAIN, N))
    adata = sc.datasets.pbmc3k()[:N].copy()
    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    ensg = np.array([sym2ensg.get(s.upper(), "NA") for s in adata.var_names])
    seqs = [tokenize_cell(Xd[i], ensg, tok, med) for i in range(N)]

    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    def tf_tokens(tfs):
        gs = {t.upper() for tf in tfs for t in tr[tr.tf.str.upper() == tf]["tg"]}
        return {tok[sym2ensg[g]] for g in gs if g in sym2ensg and sym2ensg[g] in tok}
    rfx_tokens = tf_tokens(TARGET_TFS)
    ctrl_tokens = tf_tokens((CONTROL_TF,)) - rfx_tokens
    uni_tokens = {tok[sym2ensg[g]] for g in tr_g if g in sym2ensg and sym2ensg[g] in tok}

    def batches(idxs, bs=8):
        for s in range(0, len(idxs), bs):
            sub = idxs[s:s+bs]; ids, esl = zip(*[seqs[i] for i in sub])
            L = max(len(x) for x in ids)
            inp = torch.full((len(sub), L), pad_id, dtype=torch.long); at = torch.zeros((len(sub), L), dtype=torch.long)
            for k, x in enumerate(ids):
                inp[k, :len(x)] = torch.tensor(x); at[k, :len(x)] = 1
            yield sub, inp.to(dev), at.to(dev), [list(e) for e in esl]

    dff = model.config.intermediate_size
    W_out = model.bert.encoder.layer[LAYER].output.dense
    W0 = W_out.weight.data.clone()
    WU = model.cls.predictions.decoder.weight.detach()

    # ---- collect C and k* on TRAIN cells only ---------------------------
    C = torch.zeros(dff, dff, device=dev); nk = 0
    ksum = torch.zeros(dff, device=dev); nstar = 0
    cap = {}
    h = W_out.register_forward_pre_hook(lambda m, a: cap.__setitem__("k", a[0].detach()))
    with torch.no_grad():
        for sub, inp, at, esl in batches(TRAIN):
            model(input_ids=inp, attention_mask=at)
            k = cap["k"]; mask = at.bool(); kk = k[mask]
            C += kk.T @ kk; nk += kk.shape[0]
            for bi in range(len(sub)):
                for p, e in enumerate(esl[bi]):
                    if tok.get(e) in rfx_tokens:
                        ksum += k[bi, p]; nstar += 1
    h.remove()
    C = C / nk + 1e-2 * torch.eye(dff, device=dev) * (C.diag().mean() / nk)
    kstar = ksum / max(nstar, 1)
    Cinv_kstar = torch.linalg.solve(C, kstar); denom = (kstar @ Cinv_kstar).clamp_min(1e-6)
    print(f"==> C from {nk} TRAIN keys; k* from {nstar} RFX-target tokens")

    rfx_ids = [i for i in rfx_tokens if i < WU.shape[0]]
    u = WU[rfx_ids].mean(0) - WU.mean(0); u = u / u.norm()
    scale = (W0 @ kstar).norm()

    def measure(idxs):
        tg, nt, cl = [], [], []
        with torch.no_grad():
            for sub, inp, at, esl in batches(idxs):
                lo = model(input_ids=inp, attention_mask=at).logits
                tl = torch.gather(lo, 2, inp.unsqueeze(-1)).squeeze(-1)
                for bi in range(len(sub)):
                    for p, e in enumerate(esl[bi]):
                        ti = tok.get(e)
                        if ti in rfx_tokens: tg.append(float(tl[bi, p]))
                        elif ti in ctrl_tokens: cl.append(float(tl[bi, p]))
                        elif ti in uni_tokens: nt.append(float(tl[bi, p]))
        return np.mean(tg), np.mean(nt), np.mean(cl)

    b_tr = measure(TRAIN); b_he = measure(HELD)
    vstar = W0 @ kstar + GAMMA * scale * u
    W_out.weight.data = W0 + torch.outer(vstar - W0 @ kstar, Cinv_kstar) / denom
    e_tr = measure(TRAIN); e_he = measure(HELD)
    W_out.weight.data = W0.clone()

    print(f"\n== ROME rank-1: insert 'RFX context -> boost RFX targets', edit W_out[L{LAYER}], gamma={GAMMA} ==")
    print(f"   {'metric':<34} {'Δ logit':>9}")
    print(f"   EFFICACY   RFX targets, TRAIN cells   {e_tr[0]-b_tr[0]:>+9.3f}")
    print(f"   GENERALIZE RFX targets, HELD-OUT cells {e_he[0]-b_he[0]:>+9.3f}   <- key: edit transfers to unseen cells")
    print(f"   SPECIFIC   non-target genes (held-out)  {e_he[1]-b_he[1]:>+9.3f}")
    print(f"   SPECIFIC   {CONTROL_TF} control regulon (held-out){e_he[2]-b_he[2]:>+9.3f}")
    eff, gen = e_tr[0]-b_tr[0], e_he[0]-b_he[0]
    spec = max(abs(e_he[1]-b_he[1]), abs(e_he[2]-b_he[2]))
    print(f"\n   generalization/efficacy = {gen/max(eff,1e-6):.2f} (≈1 = edit transfers); "
          f"efficacy/specificity = {abs(gen)/max(spec,1e-6):.1f}x")
    print("==> ROME held-out validation done")


def BertForMaskedLM_load(dev):
    from transformers import BertForMaskedLM
    return BertForMaskedLM.from_pretrained(GF).eval().to(dev)


if __name__ == "__main__":
    main()
