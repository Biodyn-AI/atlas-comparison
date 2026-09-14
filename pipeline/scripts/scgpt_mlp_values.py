#!/usr/bin/env python
"""scGPT MLP audit — stage 3 (values), static / weights-only. Each MLP neuron writes a
value vector (column of linear2) to the residual. Logit-lens: project each of the 6144
value vectors onto the 60697 gene-token embeddings (the readout directions) and annotate
the per-neuron gene signature against TRRUST regulons — do MLP neurons store coherent
TF-regulon 'programs' (the key-value memory hypothesis)? No forward pass, no scgpt install.

    conda activate bae
    python scripts/scgpt_mlp_values.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

BASE = _B
CKPT = f"{BASE}/external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human"
TRRUST = f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv"
TFS = f"{BASE}/external/scprint_data/TFs.txt"


def main():
    from mechaudit.annotate.genesets import load_trrust_genesets, annotate_axes, top_annotations_per_axis

    state = torch.load(f"{CKPT}/best_model.pt", map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state) if isinstance(state, dict) and "model_state_dict" in state else state
    vocab = json.load(open(f"{CKPT}/vocab.json"))               # symbol -> token id
    upper2id = {}
    for g, i in vocab.items():
        upper2id.setdefault(g.upper(), i)
    E = state["encoder.embedding.weight"].float().numpy()      # [60697, 512]
    # optional: readout via MVC gene2query (how gene identity is read at decoding)
    Wq = state["mvc_decoder.gene2query.weight"].float().numpy()
    bq = state["mvc_decoder.gene2query.bias"].float().numpy()
    Eq = E @ Wq.T + bq                                          # [60697, 512] readout directions

    n_layers = sum(1 for k in state if k.endswith("linear1.weight") and "transformer_encoder.layers." in k)
    V = np.concatenate([state[f"transformer_encoder.layers.{L}.linear2.weight"].float().numpy().T
                        for L in range(n_layers)], 0)           # [n_layers*512, 512] neuron value vectors
    layer_of = np.repeat(np.arange(n_layers), 512)
    print(f"==> {V.shape[0]} MLP neurons ({n_layers} layers x 512), d_model {V.shape[1]}")

    # gene universe = TRRUST genes present in vocab; regulons = TFs with >=10 targets
    import pandas as pd
    from scipy.stats import rankdata
    tr = pd.read_csv(TRRUST, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    uni = sorted(g for g in tr_genes if g in upper2id)
    uni_i = {g: i for i, g in enumerate(uni)}
    ids = [upper2id[g] for g in uni]
    regulons = {}
    for tf, grp in tr.groupby(tr.tf.str.upper()):
        tgs = [uni_i[t.upper()] for t in grp.tg if t.upper() in uni_i]
        if len(tgs) >= 10:
            regulons[tf] = np.array(tgs)
    print(f"==> annotating {V.shape[0]} neurons over {len(uni)} TRRUST genes, {len(regulons)} regulons")

    def unit(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    nG = len(uni)
    reg_items = list(regulons.items())

    def annotate(Vmat, readout, center=False):
        Rg = readout[ids].copy()
        if center:                                             # remove the anisotropic mean direction
            Rg = Rg - Rg.mean(0, keepdims=True)
        Rg = unit(Rg); Vn = unit(Vmat)
        G = Rg @ Vn.T                                          # [genes, neurons]
        Rk = rankdata(G, axis=0)
        best_au = np.full(Vmat.shape[0], 0.5)
        best_reg = np.array(["-"] * Vmat.shape[0], dtype=object)
        for reg, tgs in reg_items:
            npos = len(tgs)
            au = (Rk[tgs].sum(0) - npos*(npos+1)/2) / (npos*(nG-npos))
            upd = np.abs(au-0.5) > np.abs(best_au-0.5)
            best_au[upd] = au[upd]; best_reg[upd] = reg
        return best_au, best_reg, 2*np.abs(best_au-0.5)

    rng = np.random.default_rng(0)
    Vrand = rng.standard_normal(V.shape)                       # null: random value vectors
    for tag, readout, center in (("raw-embedding", E, False),
                                 ("raw-embedding CENTERED", E, True),
                                 ("mvc-gene2query", Eq, False)):
        au, reg, strength = annotate(V, readout, center)
        au0, _, str0 = annotate(Vrand, readout, center)
        strong = strength > 0.6; order = np.argsort(strength)[::-1]
        print(f"\n===== readout: {tag} =====")
        print(f"  real  neurons strength>0.6: {int(strong.sum()):>4}/{V.shape[0]}  ({100*strong.mean():.0f}%)")
        print(f"  NULL  random-vec strength>0.6: {int((str0>0.6).sum()):>4}/{V.shape[0]}  ({100*(str0>0.6).mean():.0f}%)  <- if ~equal, it's an artifact")
        pos = au[order] > 0.5
        print(f"  of real 'strong' hits, {100*(au[strong]>0.5).mean():.0f}% are +correlation (rest are anti-correlation)")
        print(f"    top: " + ", ".join(f"L{layer_of[j]}:{reg[j]}({au[j]:.2f})" for j in order[:8]))
    np.savez_compressed(f"{BASE}/outputs/singlecell/scgpt_mlp_values.npz", layer_of=layer_of)
    print("\n==> stage-3 static pass done (read the NULL before believing anything)")


if __name__ == "__main__":
    main()
