#!/usr/bin/env python
"""HEADLINE test — does scPRINT (GRN-designed) encode directed TF->target CAUSAL logic,
or only co-expression (the 6.2% null)? Systematic activation-patching across TRRUST
TFs: for each TF, find its regulon-detecting SAE feature (correlational), ablate it, and
measure whether the TF's TARGET genes shift more than non-targets on scPRINT's NB decoder
(causal). Report the fraction of TFs with a regulon-SPECIFIC causal effect — our in-model
analog of the prior perturbation-specificity rate (Geneformer/scGPT: 6.2%, 3/48 TFs).

    conda activate scprint
    python scripts/scprint_reglogic.py --layer 4 --n-cells 48 --n-genes 1500
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=48)
    ap.add_argument("--n-genes", type=int, default=1500)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--max-tfs", type=int, default=0, help="cap number of TFs (0 = all); picks TFs with most targets")
    ap.add_argument("--causal-auroc", type=float, default=0.60, help="threshold for 'regulon-specific causal'")
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np, pandas as pd, scanpy as sc, torch
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    bm = pd.read_parquet(args.biomart)
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    sym2ens = {}
    for e, s in ens2sym.items():
        sym2ens.setdefault(s, e)
    tr = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tf2tg = {}
    for tf, tg in zip(tr.tf.str.upper(), tr.tg.str.upper()):
        tf2tg.setdefault(tf, set()).add(tg)

    # model + broad panel (force TRRUST genes so many TFs have targets present)
    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal"); m.eval()
    mgenes = set(m.genes)
    adata = sc.datasets.pbmc3k(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)
    tr_syms = set(tf2tg) | set().union(*tf2tg.values())
    forced = [sym2ens[s] for s in tr_syms if s in sym2ens and sym2ens[s] in mgenes]
    forced = list(dict.fromkeys(forced))
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, flavor="seurat_v3")
    hv = [g for g in adata.var.index[adata.var.highly_variable] if g in mgenes]
    panel = list(dict.fromkeys(forced + hv))[:args.n_genes]
    ds = SimpleAnnDataset(adata[: args.n_cells], obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=panel, max_len=0)
    batch = next(iter(DataLoader(ds, collate_fn=col, batch_size=args.n_cells, shuffle=False)))
    gp, expr = batch["genes"], batch["x"]; ng = gp.shape[1]
    gene_syms = np.array([ens2sym.get(str(g), "") for g in np.array(m.genes)[gp[0].cpu().numpy()]])
    print(f"==> panel {ng} genes; {len(set(gene_syms) & set(tf2tg))} TRRUST TFs have >=1 target in panel")

    # ---- SAE on layer-L residual (capture via a hook on one baseline forward) ----
    cap = {}
    hcap = m.transformer.blocks[args.layer].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        base_out = m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
    hcap.remove()
    base = base_out["mean"].cpu().numpy()                       # [B, ng]
    H = cap["h"][:, cap["h"].shape[1] - ng:, :]                 # gene-token residual [B, ng, d]
    X = H.reshape(-1, H.shape[-1]).numpy()
    gid = np.tile(gene_syms, H.shape[0])
    sae, stats, _ = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    uniq = np.array(sorted(set(gid))); gi = {g: i for i, g in enumerate(uniq)}
    Gp = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        Gp[gi[g]] += row; c[gi[g]] += 1
    Gp /= np.maximum(c[:, None], 1)
    R = np.apply_along_axis(rankdata, 0, Gp)

    # ---- patch hook (ablate one feature, preserve recon error) ----
    mean_t = torch.as_tensor(stats.mean); scale = float(stats.scale)
    state = {"feat": None}

    def patch(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        g0 = h.shape[1] - ng; hg = h[:, g0:, :]
        xn = (hg - mean_t) / scale
        f, _ = sae.encode(xn); recon = sae.decode(f); err = xn - recon
        if state["feat"] is not None:
            f = f.clone(); f[..., state["feat"]] = 0.0
        h = h.clone(); h[:, g0:, :] = (sae.decode(f) + err) * scale + mean_t
        return (h,) + out[1:] if isinstance(out, tuple) else h

    hp = m.transformer.blocks[args.layer].register_forward_hook(patch)

    def ablated(feat):
        state["feat"] = feat
        with torch.no_grad():
            o = m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
        return o["mean"].cpu().numpy()

    # ---- loop over TFs ----
    tfs = [tf for tf in tf2tg if tf in gi and len(tf2tg[tf] & set(uniq)) >= args.min_targets]
    tfs = sorted(tfs, key=lambda t: -len(tf2tg[t] & set(uniq)))
    if args.max_tfs:
        tfs = tfs[: args.max_tfs]
    print(f"==> testing {len(tfs)} TFs (>= {args.min_targets} targets in panel)", flush=True)
    rows = []
    for ti, tf in enumerate(tfs):
        tgt_syms = tf2tg[tf] & set(uniq)
        y = np.array([g in tgt_syms for g in uniq])
        npos = y.sum()
        au_feat = (R[y].sum(0) - npos*(npos+1)/2) / (npos*(len(uniq)-npos))    # per-feature detect AUROC
        feat = int(np.abs(au_feat - 0.5).argmax())
        detect = float(au_feat[feat])                                          # correlational (L3)
        abl = ablated(feat)
        delta = np.abs(abl - base).mean(0)                                     # [ng]
        ytg = np.array([s in tgt_syms for s in gene_syms])
        if 3 <= ytg.sum() < ng:
            causal = float(roc_auc_score(ytg, delta))                          # causal (L4)
            ratio = float(delta[ytg].mean() / max(delta[~ytg].mean(), 1e-12))
        else:
            causal, ratio = np.nan, np.nan
        rows.append((tf, int(npos), detect, causal, ratio))
        print(f"   [{ti+1}/{len(tfs)}] {tf:<8} targets {int(npos):>3}  detect {detect:.2f}  causal {causal:.2f}", flush=True)

    hp.remove()
    df = pd.DataFrame(rows, columns=["TF", "n_targets", "detect_auroc", "causal_auroc", "ratio"]).dropna()
    df = df.sort_values("causal_auroc", ascending=False)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "reglogic.csv"), index=False)

    thr = args.causal_auroc
    frac_detect = (df.detect_auroc > 0.6).mean()
    frac_causal = (df.causal_auroc > thr).mean()
    print(df.head(12).to_string(index=False))
    print(f"\n==> scPRINT regulatory-logic test ({len(df)} TFs):")
    print(f"    correlational: {100*frac_detect:.0f}% of TFs have a feature that DETECTS their regulon (AUROC>0.6)")
    print(f"    CAUSAL       : {100*frac_causal:.0f}% of TFs are regulon-SPECIFIC causal (AUROC>{thr})  "
          f"[the prior Geneformer/scGPT atlases: ~6.2% (3/48)]")
    print(f"    median detect {df.detect_auroc.median():.3f} vs median causal {df.causal_auroc.median():.3f} "
          f"-> the correlational->causal GAP is the regulatory-logic deficit")
    print("==> saved outputs/scprint/reglogic.csv")


if __name__ == "__main__":
    main()
