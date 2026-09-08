"""VALID regulatory-logic test (Igor's Phase 8, NON-circular), model-agnostic.
External CRISPRi perturbation: run knockdown + control cells through a model, find SAE
features that DIFFERENTIALLY RESPOND to each TF knockdown (Wilcoxon KD vs control,
BH<0.05), then SEPARATELY test whether responders detect that TF's TRRUST targets
(Fisher). No feature is selected on the target set -> no circularity.

Ported + validated from scripts/scprint_pert_response.py (first run: scPRINT 6 TFs,
detect 67%, TF-specific 17% = 1/6 MAX). Igor's Geneformer/scGPT: 92% detect, 6.2% specific.
"""
from __future__ import annotations
import json, os
import numpy as np
from .sae import SAECfg, train_sae, feature_activations


def _bh(pv):
    order = np.argsort(pv); ranks = np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate((pv[order] * len(pv) / ranks)[::-1])[::-1]
    out = np.empty(len(pv)); out[order] = np.clip(q, 0, 1); return out


def run_perturbation(adapter, adata, trrust_path, layer, out_dir,
                     pert_col="perturbation", ctrl_labels=("control", "non-targeting"),
                     n_tfs=48, n_kd=40, n_ctrl=300, min_targets=5, device="cuda", log=print):
    import pandas as pd
    from scipy.stats import ranksums, fisher_exact

    tr = pd.read_csv(trrust_path, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tf2tg = {}
    for tf, tg in zip(tr.tf.str.upper(), tr.tg.str.upper()):
        tf2tg.setdefault(tf, set()).add(tg)

    pert = adata.obs[pert_col].astype(str).str.upper()
    ctrl = {c.upper() for c in ctrl_labels}
    is_ctrl = pert.isin(ctrl)
    tfs = [t for t in tf2tg if (pert == t).sum() >= n_kd and len(tf2tg[t]) >= min_targets]
    tfs = sorted(tfs, key=lambda t: -(pert == t).sum())[:n_tfs]
    log(f"  {is_ctrl.sum()} controls; {len(tfs)} TFs perturbed with >= {n_kd} cells")

    rng = np.random.default_rng(0)
    ci = np.where(is_ctrl.values)[0]; rng.shuffle(ci); ci = ci[:n_ctrl]
    ki = {t: np.where((pert == t).values)[0][:n_kd] for t in tfs}
    idx = np.concatenate([ci] + [ki[t] for t in tfs])
    sub = adata[idx].copy()
    sub.obs["pert_group"] = ["control"] * len(ci) + sum(([t] * len(ki[t]) for t in tfs), [])

    # ---- adapter forward -> per-(cell,gene) residual at `layer` ----
    acts_all, syms_all, cid_all = [], [], []
    for acts, syms, cell_ids in adapter.iter_activations(sub, batch_size=32):
        acts_all.append(acts[layer]); syms_all.append(syms); cid_all.append(cell_ids)
    A = np.concatenate(acts_all); syms = np.concatenate(syms_all); cids = np.concatenate(cid_all)
    obs = adapter.processed_obs
    group = obs["pert_group"].values
    ncell = len(group)
    log(f"  processed {ncell} cells x {A.shape[0]//max(ncell,1)} gene-tokens")

    isc_cell = group == "control"
    isc_pos = isc_cell[cids]
    d = A.shape[1]

    # SAE on control residual (reference dictionary)
    sae, stats = train_sae(A[isc_pos], SAECfg(d_model=d, expansion=8, k=32, epochs=40), device=device, log=lambda *a: None)
    Fpos = feature_activations(sae, A, device=device)                 # [pos, d_sae]

    # per-cell feature profile = mean over that cell's gene positions
    dsae = Fpos.shape[1]
    P = np.zeros((ncell, dsae), np.float32); cnt = np.zeros(ncell)
    np.add.at(P, cids, Fpos); np.add.at(cnt, cids, 1)
    P /= np.maximum(cnt[:, None], 1)

    # per-gene feature activation from controls -> each feature's top-20 genes
    uniq = np.array(sorted(set(syms[isc_pos]))); gi = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), dsae), np.float32); gc = np.zeros(len(uniq))
    sidx = np.array([gi[g] for g in syms[isc_pos]])
    np.add.at(G, sidx, Fpos[isc_pos]); np.add.at(gc, sidx, 1)
    G /= np.maximum(gc[:, None], 1)
    top_genes = {f: set(uniq[np.argsort(G[:, f])[::-1][:20]]) for f in range(dsae)}
    bg = len(uniq)

    Pc = P[isc_cell]
    rows = []
    for t in tfs:
        Pk = P[group == t]
        eff = np.zeros(dsae); pv = np.ones(dsae)
        for f in range(dsae):
            a, b = Pk[:, f], Pc[:, f]
            if a.std() + b.std() == 0:
                continue
            _, pv[f] = ranksums(a, b)
            eff[f] = (a.mean() - b.mean()) / (np.concatenate([a, b]).std() + 1e-9)
        qv = _bh(pv)
        resp = np.where((qv < 0.05) & (np.abs(eff) > 0.5))[0]
        tgt = tf2tg[t] & set(uniq); specific = False; best = 1.0
        for f in resp:
            ov = top_genes[f] & tgt
            if len(ov) < 2:
                continue
            a2 = len(ov); b2 = 20 - a2; c2 = len(tgt) - a2; d2 = bg - a2 - b2 - c2
            _, pf = fisher_exact([[a2, b2], [c2, d2]], alternative="greater")
            best = min(best, pf); specific = specific or pf < 0.05
        rows.append((t, int((group == t).sum()), len(resp), int(specific), float(best)))
        log(f"    {t:<8} KD {int((group==t).sum()):>3} | responders {len(resp):>4} | "
            f"specific {'YES' if specific else 'no'} (p {best:.1e})")

    df = pd.DataFrame(rows, columns=["TF", "n_kd", "n_responders", "specific", "min_fisher_p"])
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{adapter.name}_pert_response.csv"), index=False)
    detect = float((df.n_responders > 0).mean()); spec = float(df.specific.mean())
    summary = {"model": adapter.name, "n_tfs": len(df), "detection_rate": detect,
               "tf_specific_rate": spec, "igor_detect": 0.92, "igor_specific": 0.062}
    json.dump(summary, open(os.path.join(out_dir, f"{adapter.name}_pert_summary.json"), "w"))
    log(f"\n  ==> {adapter.name}: detection {100*detect:.0f}% [Igor 92%] | "
        f"TF-specific {100*spec:.0f}% [Igor 6.2%]  ({len(df)} TFs)")
    return df, summary
