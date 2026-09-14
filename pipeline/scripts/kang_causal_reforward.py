#!/usr/bin/env python
"""Causal->prediction, STRONGER version: full mid-network re-forward ablation. Patch the
IFN feature in scGPT's layer-6 residual and let it propagate through layers 7-11, then
read the final-layer cell representation the predictor uses. Unlike the representational
ablation, this measures the true downstream causal effect through the rest of the network.

    conda activate scprint
    python scripts/kang_causal_reforward.py
"""
from __future__ import annotations
import os, sys
import numpy as np, scanpy as sc
sys.path.insert(0, "atlas_h100"); sys.path.insert(0, "external/single_cell_mechinterp/external/scGPT")
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

CKPT = os.path.abspath("external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human")
ISG = set("ISG15 IFI6 IFI27 IFI44 IFI44L IFIT1 IFIT2 IFIT3 IFITM1 IFITM3 MX1 MX2 OAS1 OAS2 OAS3 OASL RSAD2 "
          "IRF7 STAT1 STAT2 USP18 LY6E IFI35 XAF1 BST2 ISG20 HERC5 DDX58 DDX60 EIF2AK2 GBP1 CMPK2 CXCL10 "
          "SIGLEC1 IFI16 SAMD9 SAMD9L PARP9 PARP14".split())
SRC_L, TGT_L = 6, 11


def auroc_col(M, y):
    y = np.asarray(y, bool); R = np.apply_along_axis(rankdata, 0, M)
    npos = y.sum(); n = len(y); return (R[y].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))


def main():
    from adapters.scgpt import ScGPTAdapter
    ad = sc.read_h5ad("data/kang/kang_batch2.h5ad")
    ad.var_names = ad.var["sym"].astype(str).values; ad.var_names_make_unique()
    rng = np.random.default_rng(0); idx = []
    for (s, d), g in ad.obs.groupby(["stim", "ind"], observed=True):
        idx += list(rng.choice(g.index.values, min(len(g), 12), replace=False))
    ad = ad[idx].copy()
    var_syms = np.array([str(x) for x in ad.var_names])
    stim = (ad.obs["stim"].values == "stim").astype(int); don = ad.obs["ind"].values.astype(str)
    print(f"==> {ad.n_obs} cells ({stim.sum()} stim)")

    adp = ScGPTAdapter(ckpt_dir=CKPT, layers=[SRC_L, TGT_L], max_seq_len=1200); adp.load(device="cpu")
    adp._var = var_syms
    mean_t = torch.as_tensor(np.asarray(None) if False else 0.0)   # placeholder, set after stats

    # ---- one baseline forward: collect layer-6 (SAE) + layer-11 (rep) tokens ----
    def run(ablate_feat=None, sae=None, stats=None):
        """Return per-cell pooled layer-11 rep; if collecting, also layer-6 tokens+meta."""
        # HARD-disable the encoder's NestedTensor fast path — its final to_padded_tensor is
        # not implemented on the CPU backend (only MPS/CUDA), which crashes any forward here.
        import torch.nn as nn
        for mod in adp.model.modules():
            if isinstance(mod, nn.TransformerEncoder):
                mod.enable_nested_tensor = False
                mod.use_nested_tensor = False
        enc = adp.model.transformer_encoder.layers
        cap = {}
        def denest(h): return h.to_padded_tensor(0.0) if getattr(h, "is_nested", False) else h
        hooks = [enc[TGT_L].register_forward_hook(
            lambda m, i, o: cap.__setitem__("t", denest(o[0] if isinstance(o, tuple) else o).detach()))]
        if ablate_feat is None:
            hooks.append(enc[SRC_L].register_forward_hook(
                lambda m, i, o: cap.__setitem__("s", denest(o[0] if isinstance(o, tuple) else o).detach())))
        else:
            mt = torch.as_tensor(stats.mean, dtype=torch.float32); sc_ = float(stats.scale)
            def patch(m, i, o):
                h = denest(o[0] if isinstance(o, tuple) else o); shp = h.shape
                xn = (h.reshape(-1, shp[-1]) - mt) / sc_
                f, _ = sae.encode(xn); err = xn - sae.decode(f)
                f = f.clone(); f[:, ablate_feat] = 0.0
                hn = (sae.decode(f) + err) * sc_ + mt
                hn = hn.reshape(shp)
                return (hn,) + o[1:] if isinstance(o, tuple) else hn
            hooks.append(enc[SRC_L].register_forward_hook(patch))
        rep = np.zeros((ad.n_obs, adp.d_model), np.float32)
        S6, sym6, cid6 = [], [], []
        cid = 0
        for s in range(0, ad.n_obs, 16):
            rows = [adp._tokenise(ad.X[j]) for j in range(s, min(s+16, ad.n_obs))]
            b = len(rows); L = max(1, max(len(r[0]) for r in rows))
            gid = np.full((b, L), adp.pad_id, np.int64); val = np.zeros((b, L), np.float32); gidx = np.full((b, L), -1, np.int64)
            for i, (g, vv, gi) in enumerate(rows):
                gid[i, :len(g)] = g; val[i, :len(vv)] = vv; gidx[i, :len(gi)] = gi
            src = torch.as_tensor(gid); mask = src == adp.pad_id
            cap.clear()
            with torch.no_grad():
                adp.model(src=src, values=torch.as_tensor(val), src_key_padding_mask=mask)
            keep = ~mask.numpy()
            h11 = cap["t"].float().numpy()
            for i in range(b):
                pos = np.where(keep[i])[0]; rep[cid+i] = h11[i, pos, :].mean(0)
            if ablate_feat is None:
                h6 = cap["s"].float().numpy()
                for i in range(b):
                    pos = np.where(keep[i])[0]
                    S6.append(h6[i, pos, :]); sym6.append(np.array([str(adp._var[k]).upper() for k in gidx[i, pos]])); cid6.append(np.full(len(pos), cid+i))
            cid += b
        for hk in hooks: hk.remove()
        if ablate_feat is None:
            return rep, np.concatenate(S6, 0).astype(np.float32), np.concatenate(sym6), np.concatenate(cid6)
        return rep

    rep0, X6, syms, cids = run()
    print(f"==> layer-6 residual {X6.shape}; layer-11 rep {rep0.shape}")
    sae, stats, _ = train_sae(X6, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X6, stats, device="cpu")

    # IFN feature (conjunctive: stim-cell + ISG-gene)
    Pf = np.zeros((ad.n_obs, F.shape[1])); c = np.zeros(ad.n_obs)
    for row, ci in zip(F, cids): Pf[ci] += row; c[ci] += 1
    Pf /= np.maximum(c[:, None], 1)
    fj = int(np.abs(auroc_col(Pf, stim)-0.5).argmax())
    print(f"==> IFN feature {fj}")

    # predictor on baseline layer-11 rep
    oof = np.zeros(ad.n_obs)
    for d in np.unique(don):
        te = don == d; cl = LogisticRegression(max_iter=2000, class_weight="balanced").fit(rep0[~te], stim[~te]); oof[te] = cl.predict_proba(rep0[te])[:, 1]
    print(f"==> predictor on FINAL-layer rep, LODO AUROC {roc_auc_score(stim, oof):.3f}")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(rep0, stim)
    base = clf.predict_proba(rep0)[:, 1]; sm = stim == 1

    # ablated re-forwards
    rep_ifn = run(ablate_feat=fj, sae=sae, stats=stats)
    ab = clf.predict_proba(rep_ifn)[:, 1]
    dprob = (base[sm]-ab[sm]).mean(); flip = ((base[sm] >= .5) & (ab[sm] < .5)).mean()
    active = np.where(F.max(0) > 0)[0]; rndf = rng.choice(active[active != fj], 5, replace=False)
    rnd_d = []
    for j in rndf:
        rep_r = run(ablate_feat=int(j), sae=sae, stats=stats)
        rnd_d.append((base[sm]-clf.predict_proba(rep_r)[:, 1][sm]).mean())
    print(f"\n==> FULL RE-FORWARD causal->prediction (patch L{SRC_L} -> propagate -> L{TGT_L}):")
    print(f"   IFN feature: stim-prob drop {dprob:+.3f}, {100*flip:.1f}% of stim cells flip to ctrl")
    print(f"   5 random features: mean drop {np.mean(rnd_d):+.3f} ± {np.std(rnd_d):.3f}")
    print(f"   -> {'SPECIFIC causal driver through the network' if dprob > 5*np.std(rnd_d)+abs(np.mean(rnd_d)) else 'distributed'}")
    np.savez_compressed("outputs/singlecell/kang_causal_reforward.npz", base=base, ab=ab, stim=stim,
                        dprob=dprob, flip=flip, rnd_d=np.array(rnd_d), fj=fj)
    print("==> saved kang_causal_reforward.npz")


if __name__ == "__main__":
    main()
