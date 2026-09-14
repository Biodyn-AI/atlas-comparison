#!/usr/bin/env python
"""Generate AIDO feature_catalog (top-20 genes/feature) for the comparative atlas.
Captures layer-4 residual over TRRUST genes, trains TopK SAE, dumps catalog JSON.

    conda activate scprint
    python scripts/aido_catalog.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import numpy as np, pandas as pd, scanpy as sc, torch
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    genes = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    sym2pos = {g: i for i, g in enumerate(genes)}
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    keep = sorted(sym2pos[g] for g in tr_genes if g in sym2pos)
    keep_sym = np.array([genes[i] for i in keep])

    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationModel.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    adata = sc.datasets.pbmc3k()[:48].copy()
    ad, attn = align_adata(adata); attn_t = torch.from_numpy(attn).unsqueeze(0)
    acts, gid = [], []
    with torch.no_grad():
        for s in range(0, ad.n_obs, 8):
            xb = ad.X[s:s+8]; xb = xb.toarray() if hasattr(xb, "toarray") else xb
            inp = preprocess_counts(xb, device="cpu")
            am = torch.cat([attn_t.repeat(inp.shape[0], 1), torch.ones((inp.shape[0], 2))], 1)
            out = m(input_ids=inp, attention_mask=am, output_hidden_states=True)
            hid = out.hidden_states[4][:, keep, :]
            for b in range(hid.shape[0]):
                acts.append(hid[b].numpy()); gid.extend(keep_sym)
    X = np.concatenate(acts, 0); gid = np.array(gid)
    sae, stats, log = train_sae(X, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    uniq = np.array(sorted(set(gid))); gi = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        G[gi[g]] += row; c[gi[g]] += 1
    G /= np.maximum(c[:, None], 1)
    freq = (F > 0).mean(0)
    feats = {}
    for f in np.where(freq > 0)[0]:
        order = np.argsort(G[:, f])[::-1][:20]
        feats[str(int(f))] = {"top_genes": [str(uniq[i]) for i in order],
                              "max_activation": float(G[:, f].max()),
                              "activation_frequency": float(freq[f])}
    os.makedirs(f"{BASE}/outputs/atlas/AIDO", exist_ok=True)
    cat = {"model": "AIDO", "layer": 4, "d_model": 256, "d_sae": 2048, "k": 32,
           "n_alive": int((freq > 0).sum()), "n_genes": int(len(uniq)),
           "fvu": float(log.history[-1]["fvu"]), "features": feats}
    json.dump(cat, open(f"{BASE}/outputs/atlas/AIDO/feature_catalog_L4.json", "w"))
    print(f"==> AIDO L4: {cat['n_alive']} alive, FVU {cat['fvu']:.3f}, {len(uniq)} genes")


if __name__ == "__main__":
    main()
