#!/usr/bin/env python
"""Geneformer MLP audit — stand up CPU forward (cross-model check vs scGPT). Standard HF
BertForMaskedLM: MLP = encoder.layer[i].intermediate.dense (W_in) -> gelu -> output.dense
(W_out); hooks on submodules fire directly (no fused fast path). Rank-value tokenization
reimplemented from token/median dicts (no geneformer package). Step 1 = validate.

    conda activate scprint
    python scripts/geneformer_forward.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B
GF = f"{BASE}/ckpt_geneformer"


def load_dicts():
    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))      # ENSG -> token id
    med = pickle.load(open(f"{GF}/gene_median_dictionary_gc30M.pkl", "rb"))  # ENSG -> median
    return tok, med


def tokenize_cell(counts, ensg, tok, med, max_len=2048):
    """Geneformer rank encoding: normalize per cell, divide by gene median, rank desc."""
    import numpy as np
    total = counts.sum()
    if total == 0:
        return []
    norm = counts / total
    val, ids = [], []
    for j, g in enumerate(ensg):
        if counts[j] > 0 and g in tok and g in med and med[g] > 0:
            val.append(norm[j] / med[g]); ids.append(tok[g])
    if not val:
        return []
    order = np.argsort(val)[::-1][:max_len]
    return [ids[i] for i in order]


def main():
    import numpy as np, scanpy as sc, torch, pandas as pd
    from transformers import BertForMaskedLM

    model = BertForMaskedLM.from_pretrained(GF).eval()
    nL = model.config.num_hidden_layers; dff = model.config.intermediate_size
    print(f"==> Geneformer loaded: {nL} layers, hidden {model.config.hidden_size}, "
          f"intermediate {dff} -> {nL*dff} MLP neurons, vocab {model.config.vocab_size}")

    tok, med = load_dicts()
    pad_id = tok.get("<pad>", 0)
    print(f"    token dict {len(tok)} genes; special: "
          f"{[k for k in tok if k.startswith('<')][:5]}")

    # symbol -> Ensembl via biomart
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}
    for ensg, sym in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(sym).upper(), ensg)

    adata = sc.datasets.pbmc3k()[:8].copy()
    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    ensg = np.array([sym2ensg.get(s.upper(), "NA") for s in adata.var_names])
    seqs = [tokenize_cell(Xd[i], ensg, tok, med) for i in range(adata.n_obs)]
    L = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s); attn[i, :len(s)] = 1
    print(f"==> batch {tuple(input_ids.shape)}; seq lens {[len(s) for s in seqs]}")

    acts = {}
    hooks = [model.bert.encoder.layer[li].intermediate.register_forward_hook(
        lambda m, i, o, li=li: acts.__setitem__(li, o.detach())) for li in range(nL)]
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
    for h in hooks:
        h.remove()

    print(f"==> MLM logits {tuple(out.logits.shape)} (B, L, vocab)")
    print(f"==> layer-0 MLP activations {tuple(acts[0].shape)} (B, L, d_ff), post-gelu")
    m = attn.bool()
    for li in (0, nL // 2, nL - 1):
        A = acts[li][m]
        print(f"    layer {li}: mean {A.mean():.3f}, frac>0 {(A>0).float().mean():.3f}, max {A.max():.2f}")
    print("==> FORWARD WORKS — ready for stage-2/5 (hooks fire directly, clean MLM head)")


if __name__ == "__main__":
    main()
