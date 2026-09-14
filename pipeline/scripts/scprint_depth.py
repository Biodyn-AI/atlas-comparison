#!/usr/bin/env python
"""Depth profile of a regulon across scPRINT layers — companion to aido_depth.py.
Captures each TRRUST gene's residual at every layer (pre-block-0 = input embedding,
then each block output), linear-probes RFX / E2F1 target membership, AUROC vs depth.
Claim: scPRINT (ESM-augmented) front-loads the regulon at the INPUT, unlike AIDO.

    conda activate scprint
    python scripts/scprint_depth.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import numpy as np, pandas as pd, scanpy as sc, torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint

    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    rfx = {t.upper() for tf in ("RFXANK", "RFXAP", "RFX5") for t in tr[tr.tf.str.upper() == tf]["tg"]}
    e2f1 = {t.upper() for t in tr[tr.tf.str.upper() == "E2F1"]["tg"]}
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    sym2ens = {}
    for e, s in ens2sym.items():
        sym2ens.setdefault(s, e)

    m = scPrint.load_from_checkpoint(f"{BASE}/ckpt_scprint/medium-v1.5.ckpt",
                                     precpt_gene_emb=None, transformer="normal"); m.eval()
    mgenes = set(m.genes)
    # prioritise the concept target genes (RFX + E2F1) so the probe has enough positives,
    # then fill the rest of the panel with other TRRUST genes
    concept = rfx | e2f1 | {"RFXANK", "RFXAP", "RFX5", "E2F1"}
    prio = [sym2ens[s] for s in sorted(concept) if s in sym2ens and sym2ens[s] in mgenes]
    prio = list(dict.fromkeys(prio))
    other = [sym2ens[s] for s in tr_genes if s in sym2ens and sym2ens[s] in mgenes and sym2ens[s] not in set(prio)]
    fixed = list(dict.fromkeys(prio + other))[:1200]
    print(f"==> forcing {len(fixed)} TRRUST genes ({len(prio)} concept targets prioritised) into scPRINT panel")

    adata = sc.datasets.pbmc3k(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)
    ds = SimpleAnnDataset(adata[:48], obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=fixed, max_len=0)
    batch = next(iter(DataLoader(ds, collate_fn=col, batch_size=48, shuffle=False)))
    gp, expr = batch["genes"], batch["x"]; ng = gp.shape[1]
    gene_syms = np.array([ens2sym.get(str(g), "") for g in np.array(m.genes)[gp[0].cpu().numpy()]])
    y_rfx = np.array([s in rfx for s in gene_syms]); y_e2f = np.array([s in e2f1 for s in gene_syms])
    print(f"==> panel {ng} genes; RFX targets {y_rfx.sum()}, E2F1 targets {y_e2f.sum()}")

    caps = {}
    hooks = []
    hooks.append(m.transformer.blocks[0].register_forward_pre_hook(
        lambda mod, inp: caps.__setitem__(0, (inp[0] if isinstance(inp, tuple) else inp)[:, -ng:, :].detach())))
    for i, blk in enumerate(m.transformer.blocks):
        hooks.append(blk.register_forward_hook(
            lambda mod, inp, out, i=i: caps.__setitem__(
                i+1, (out[0] if isinstance(out, tuple) else out)[:, -ng:, :].detach())))
    with torch.no_grad():
        m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
    for h in hooks:
        h.remove()

    def probe(G, y):
        G = (G - G.mean(0)) / (G.std(0) + 1e-9)
        pr = cross_val_predict(LogisticRegression(max_iter=1000, C=1.0), G, y, cv=5,
                               method="predict_proba")[:, 1]
        return roc_auc_score(y, pr)

    rfx_au, e2f_au = [], []
    for L in sorted(caps):
        G = caps[L].mean(0).numpy()
        rfx_au.append(probe(G, y_rfx)); e2f_au.append(probe(G, y_e2f))
        print(f"   layer {L:>2}: RFX {rfx_au[-1]:.3f}  E2F1 {e2f_au[-1]:.3f}")
    np.savez_compressed(f"{BASE}/outputs/singlecell/depth_scPRINT.npz",
                        rfx=np.array(rfx_au), e2f1=np.array(e2f_au))
    print("==> saved depth_scPRINT.npz")


if __name__ == "__main__":
    main()
