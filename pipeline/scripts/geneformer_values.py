#!/usr/bin/env python
"""Geneformer MLP — Geva stage-3 (value->vocab), the CLEAN static logit-lens that scGPT
lacked. Geneformer is a BertForMaskedLM whose MLM decoder is TIED to the input gene
embeddings, so projecting each MLP value vector (output.dense column) through that
unembedding gives a genuine per-neuron gene distribution. Annotate vs TRRUST regulons
WITH a random-vector null. TEST: does real >> null here (unlike scGPT, where the
expression-regression head made the static lens an artifact)?

    conda activate scprint
    python scripts/geneformer_values.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B; GF = f"{BASE}/ckpt_geneformer"


def main():
    import numpy as np, pandas as pd, torch
    from scipy.stats import rankdata
    from transformers import BertForMaskedLM

    model = BertForMaskedLM.from_pretrained(GF).eval()
    nL, dff, H = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    nN = nL * dff; layer_of = np.repeat(np.arange(nL), dff)
    WU = model.cls.predictions.decoder.weight.detach().numpy()          # [vocab, H] tied unembedding
    # value vectors = columns of output.dense.weight [H, d_ff] per layer
    V = np.concatenate([model.bert.encoder.layer[li].output.dense.weight.detach().numpy().T
                        for li in range(nL)], 0)                         # [nN, H]
    print(f"==> {nN} MLP neurons, hidden {H}, tied unembedding {WU.shape}")

    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}
    for e, s in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(s).upper(), e)
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    uni, ids = [], []
    for g in sorted(tr_g):
        e = sym2ensg.get(g.upper())
        if e in tok:
            uni.append(g); ids.append(tok[e])
    uni_i = {g: i for i, g in enumerate(uni)}
    regs = {tf: np.array([uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i])
            for tf, grp in tr.groupby(tr.tf.str.upper())}
    regs = {k: v for k, v in regs.items() if len(v) >= 15}
    print(f"==> {len(uni)} TRRUST genes in vocab, {len(regs)} regulons")

    def unit(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    Ug = unit(WU[ids])                                                  # [n_uni, H] readout dirs
    reg_items = list(regs.items()); nGuni = len(uni)

    def annotate(Vmat):
        G = Ug @ unit(Vmat).T                                          # [genes, neurons] logit-lens
        Rk = rankdata(G, axis=0); best = np.full(Vmat.shape[0], 0.5); name = np.array(["-"]*Vmat.shape[0], dtype=object)
        for tf, t in reg_items:
            npos = len(t); au = (Rk[t].sum(0)-npos*(npos+1)/2)/(npos*(nGuni-npos))
            up = np.abs(au-0.5) > np.abs(best-0.5); best[up]=au[up]; name[up]=tf
        return best, name

    best, name = annotate(V)
    rng = np.random.default_rng(0)
    best0, _ = annotate(rng.standard_normal(V.shape))
    strg, strg0 = 2*np.abs(best-0.5), 2*np.abs(best0-0.5)
    thr = np.percentile(strg0, 99)
    real, null = int((strg > thr).sum()), int((strg0 > thr).sum())
    print(f"\n== value->vocab (tied unembedding) ==")
    print(f"  threshold (99th pct null) = {thr:.3f}")
    print(f"  REAL neurons above threshold: {real}/{nN} ({100*real/nN:.1f}%)  vs NULL {null}/{nN} ({100*null/nN:.1f}%)")
    print(f"  -> {'CLEAN LENS WORKS (real >> null)' if real > 3*null else 'still artifact (real ~ null)'}")
    order = np.argsort(strg)[::-1]
    print("  strongest value-writing neurons (neuron/layer/regulon/AUROC):")
    for j in order[:12]:
        print(f"    n{j:>4} L{layer_of[j]} {name[j]:<8} {best[j]:.3f}")
    # regulon diversity among strong value neurons
    strong = strg > thr
    from collections import Counter
    div = Counter(name[strong])
    print(f"  regulon diversity among {int(strong.sum())} strong value-neurons: "
          f"{dict(sorted(div.items(), key=lambda x:-x[1])[:10])}")
    np.savez_compressed(f"{BASE}/outputs/singlecell/geneformer_values.npz",
                        strg=strg, best=best, name=name.astype(str), layer_of=layer_of)
    print("\n==> stage-3 value->vocab done")


if __name__ == "__main__":
    main()
