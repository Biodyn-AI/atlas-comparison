#!/usr/bin/env python
"""VALID regulatory-logic test (Igor's Phase 8, non-circular). External perturbation:
run CRISPRi-knockdown cells + controls through scPRINT, find SAE features that
DIFFERENTIALLY RESPOND to each TF knockdown (Wilcoxon KD vs control, BH<0.05), then
SEPARATELY test whether those responders detect that TF's TRRUST targets (Fisher).
No feature is selected on the target set -> no circularity. Reports detection rate and
TF-specific rate (Igor's Geneformer/scGPT: 92% detect, 6.2% TF-specific).

    conda activate scprint
    python scripts/scprint_pert_response.py --inspect          # check h5ad structure first
    python scripts/scprint_pert_response.py --n-tfs 8 --n-kd 40 --n-ctrl 200
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="external/perturb/replogle_k562_essential.h5ad")
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--n-tfs", type=int, default=8)
    ap.add_argument("--n-kd", type=int, default=40)
    ap.add_argument("--n-ctrl", type=int, default=200)
    ap.add_argument("--n-genes", type=int, default=1200)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np, pandas as pd, scanpy as sc, torch
    from scipy.stats import ranksums, fisher_exact
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    adata = sc.read_h5ad(args.h5ad)
    if args.inspect:
        print("obs cols:", list(adata.obs.columns))
        for c in adata.obs.columns:
            nu = adata.obs[c].nunique()
            if 2 <= nu <= 5000 and adata.obs[c].dtype.name in ("category", "object"):
                vc = adata.obs[c].value_counts().head(5)
                print(f"  {c}: {nu} uniq | top {dict(vc)}")
        print("var index sample:", list(adata.var_names[:5]), "| n_var", adata.n_vars)
        return

    tr = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tf2tg = {}
    for tf, tg in zip(tr.tf.str.upper(), tr.tg.str.upper()):
        tf2tg.setdefault(tf, set()).add(tg)

    pert = adata.obs[args.pert_col].astype(str).str.upper()
    ctrl_labels = {"CONTROL", "NON-TARGETING", "NEGATIVE", "NTC", "NT"}
    is_ctrl = pert.isin(ctrl_labels)
    perturbed_tfs = [t for t in tf2tg if (pert == t).sum() >= args.n_kd and len(tf2tg[t]) >= args.min_targets]
    perturbed_tfs = sorted(perturbed_tfs, key=lambda t: -(pert == t).sum())[: args.n_tfs]
    print(f"==> {is_ctrl.sum()} control cells; testing {len(perturbed_tfs)} TFs: {perturbed_tfs}", flush=True)

    rng = np.random.default_rng(0)
    ctrl_idx = np.where(is_ctrl.values)[0]; rng.shuffle(ctrl_idx); ctrl_idx = ctrl_idx[: args.n_ctrl]
    tf_idx = {t: np.where((pert == t).values)[0][: args.n_kd] for t in perturbed_tfs}
    all_idx = np.concatenate([ctrl_idx] + [tf_idx[t] for t in perturbed_tfs])
    sub = adata[all_idx].copy()
    group = np.array(["control"] * len(ctrl_idx) + sum(([t] * len(tf_idx[t]) for t in perturbed_tfs), []))

    # ---- scPRINT forward -> per-cell SAE feature profile ----------------
    sub.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    sub.var_names = [str(s).upper() for s in sub.var_names]
    sub = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=500,
                       min_nnz_genes=50, filter_gene_by_counts=False)(sub)
    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal"); m.eval()
    tr_syms = set(tf2tg) | set().union(*tf2tg.values())
    sc.pp.highly_variable_genes(sub, n_top_genes=args.n_genes, flavor="seurat_v3")
    panel = [g for g in sub.var.index[sub.var.highly_variable] if g in set(m.genes)]
    forced = [g for g in tr_syms if g in set(m.genes)]
    panel = list(dict.fromkeys(panel + forced))[: args.n_genes]
    ds = SimpleAnnDataset(sub, obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=panel, max_len=0)
    dl = DataLoader(ds, collate_fn=col, batch_size=32, shuffle=False)

    cap = {}
    hh = m.transformer.blocks[args.layer].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    resid, cell_syms = [], None
    with torch.no_grad():
        for batch in dl:
            gp, expr = batch["genes"], batch["x"]; ng = gp.shape[1]
            cap.clear()
            m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
            hg = cap["h"][:, cap["h"].shape[1]-ng:, :]
            resid.append(hg.numpy())
            if cell_syms is None:
                ens2sym = {e: str(s).upper() for e, s in pd.read_parquet(
                    "external/scprint_data/biomart_pos.parquet")["hgnc_symbol"].items()}
                cell_syms = np.array([ens2sym.get(str(g), str(g)) for g in np.array(m.genes)[gp[0].cpu().numpy()]])
    hh.remove()
    Hcells = np.concatenate(resid, 0)                         # [n_cells, ng, d]

    # SAE on CONTROL residual only (reference dictionary)
    isc = group == "control"
    Xc = Hcells[isc].reshape(-1, Hcells.shape[-1])
    sae, stats, _ = train_sae(Xc, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    # per-cell feature profile = mean feature activation over that cell's gene tokens
    def cell_profiles(H):
        n = H.shape[0]
        P = np.zeros((n, 2048), np.float32)
        for i in range(n):
            P[i] = feature_activations(sae, H[i], stats, device="cpu").mean(0)
        return P
    Pall = cell_profiles(Hcells)
    # per-gene feature activation (for regulon-detection of a feature) from controls
    Fc = feature_activations(sae, Xc, stats, device="cpu")
    gidc = np.tile(cell_syms, isc.sum())
    uniq = np.array(sorted(set(gidc))); gi = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), 2048), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(Fc, gidc):
        G[gi[g]] += row; c[gi[g]] += 1
    G /= np.maximum(c[:, None], 1)
    top_genes = {f: set(uniq[np.argsort(G[:, f])[::-1][:20]]) for f in range(2048)}
    bg = len(uniq)

    # ---- per-TF: responding features (Wilcoxon) -> regulon Fisher -------
    Pc = Pall[isc]
    rows = []
    for t in perturbed_tfs:
        Pk = Pall[group == t]
        eff, pv = [], []
        for f in range(2048):
            a, b = Pk[:, f], Pc[:, f]
            if a.std() + b.std() == 0:
                eff.append(0); pv.append(1); continue
            s, p = ranksums(a, b)
            eff.append((a.mean()-b.mean())/(np.concatenate([a, b]).std()+1e-9)); pv.append(p)
        eff = np.array(eff); pv = np.array(pv)
        order = np.argsort(pv); ranks = np.arange(1, len(pv)+1)
        q = np.minimum.accumulate((pv[order]*len(pv)/ranks)[::-1])[::-1]; qv = np.empty(len(pv)); qv[order] = q
        responders = np.where((qv < 0.05) & (np.abs(eff) > 0.5))[0]
        # regulon-specificity: any responder whose top-genes enrich for this TF's targets?
        tgt = tf2tg[t] & set(uniq); specific = False; best_p = 1.0
        for f in responders:
            ov = top_genes[f] & tgt
            if len(ov) < 2:
                continue
            a2 = len(ov); b2 = 20-a2; c2 = len(tgt)-a2; d2 = bg-a2-b2-c2
            _, pf = fisher_exact([[a2, b2], [c2, d2]], alternative="greater")
            best_p = min(best_p, pf); specific = specific or pf < 0.05
        rows.append((t, int((group == t).sum()), len(responders), int(specific), best_p))
        print(f"   {t:<8} KD {int((group==t).sum()):>3} | responders {len(responders):>4} | "
              f"regulon-specific {'YES' if specific else 'no':<3} (min Fisher p {best_p:.1e})", flush=True)

    df = pd.DataFrame(rows, columns=["TF", "n_kd", "n_responders", "specific", "min_fisher_p"])
    df.to_csv(os.path.join(args.out, "pert_response.csv"), index=False)
    detect = (df.n_responders > 0).mean(); spec = df.specific.mean()
    print(f"\n==> scPRINT perturbation-response ({len(df)} TFs, non-circular):")
    print(f"    detection rate : {100*detect:.0f}% of TFs have >=1 responding feature   [Igor: 92%]")
    print(f"    TF-SPECIFIC    : {100*spec:.0f}% of TFs have a responder enriched for their regulon   [Igor: 6.2%]")
    print("==> saved outputs/scprint/pert_response.csv")


if __name__ == "__main__":
    main()
