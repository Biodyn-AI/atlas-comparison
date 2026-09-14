#!/usr/bin/env python
"""Bidirectional causal steering (Kang / scGPT, full re-forward). The IFN feature both
ways: ABLATE it in stim cells -> do they flip to ctrl? CLAMP it ON in ctrl cells ->
do they flip to 'response'? Plus a random-feature clamp control. Patch at layer 6,
propagate through the network, read the predictor. Steering the prediction in BOTH
directions via one interpretable feature = symmetric causal control.

    conda activate scprint
    python scripts/kang_bidirectional_steer.py
"""
from __future__ import annotations
import os, sys
import numpy as np, scanpy as sc
sys.path.insert(0, "atlas_h100"); sys.path.insert(0, "external/single_cell_mechinterp/external/scGPT")
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))
import torch, torch.nn as nn
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

CKPT = os.path.abspath("external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human")
ISG = set("ISG15 IFI6 IFI27 IFI44 IFI44L IFIT1 IFIT2 IFIT3 IFITM1 IFITM3 MX1 MX2 OAS1 OAS2 OAS3 OASL RSAD2 "
          "IRF7 STAT1 STAT2 USP18 LY6E IFI35 XAF1 BST2 ISG20 HERC5 DDX58 DDX60 EIF2AK2 GBP1 CMPK2 CXCL10".split())
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
    ad = ad[idx].copy(); var_syms = np.array([str(x) for x in ad.var_names])
    stim = (ad.obs["stim"].values == "stim").astype(int)
    print(f"==> {ad.n_obs} cells ({stim.sum()} stim / {(1-stim).sum()} ctrl)")

    adp = ScGPTAdapter(ckpt_dir=CKPT, layers=[SRC_L, TGT_L], max_seq_len=1200); adp.load(device="cpu")
    adp._var = var_syms
    holder = {}                                                     # patch config

    def run(collect=False):
        for mod in adp.model.modules():
            if isinstance(mod, nn.TransformerEncoder):
                mod.enable_nested_tensor = False; mod.use_nested_tensor = False
        enc = adp.model.transformer_encoder.layers; cap = {}
        def denest(h): return h.to_padded_tensor(0.0) if getattr(h, "is_nested", False) else h
        hooks = [enc[TGT_L].register_forward_hook(lambda m, i, o: cap.__setitem__("t", denest(o[0] if isinstance(o, tuple) else o).detach()))]
        if collect:
            hooks.append(enc[SRC_L].register_forward_hook(lambda m, i, o: cap.__setitem__("s", denest(o[0] if isinstance(o, tuple) else o).detach())))
        elif holder.get("feat") is not None:
            mt = torch.as_tensor(holder["mean"], dtype=torch.float32); scl = float(holder["scale"]); sae = holder["sae"]
            feat, val = holder["feat"], holder["val"]
            def patch(m, i, o):
                h = denest(o[0] if isinstance(o, tuple) else o); shp = h.shape
                xn = (h.reshape(-1, shp[-1]) - mt) / scl
                f, _ = sae.encode(xn); err = xn - sae.decode(f)
                f = f.clone(); f[:, feat] = val
                hn = ((sae.decode(f) + err) * scl + mt).reshape(shp)
                return (hn,) + o[1:] if isinstance(o, tuple) else hn
            hooks.append(enc[SRC_L].register_forward_hook(patch))
        rep = np.zeros((ad.n_obs, adp.d_model), np.float32); S6, sym6, cid6 = [], [], []; cid = 0
        for s in range(0, ad.n_obs, 16):
            rows = [adp._tokenise(ad.X[j]) for j in range(s, min(s+16, ad.n_obs))]
            b = len(rows); L = max(1, max(len(r[0]) for r in rows))
            gid = np.full((b, L), adp.pad_id, np.int64); val_ = np.zeros((b, L), np.float32); gidx = np.full((b, L), -1, np.int64)
            for i, (g, vv, gi) in enumerate(rows):
                gid[i, :len(g)] = g; val_[i, :len(vv)] = vv; gidx[i, :len(gi)] = gi
            src = torch.as_tensor(gid); mask = src == adp.pad_id; cap.clear()
            with torch.no_grad():
                adp.model(src=src, values=torch.as_tensor(val_), src_key_padding_mask=mask)
            keep = ~mask.numpy(); h11 = cap["t"].float().numpy()
            for i in range(b):
                pos = np.where(keep[i])[0]; rep[cid+i] = h11[i, pos, :].mean(0)
            if collect:
                h6 = cap["s"].float().numpy()
                for i in range(b):
                    pos = np.where(keep[i])[0]
                    S6.append(h6[i, pos, :]); sym6.append(np.array([str(adp._var[k]).upper() for k in gidx[i, pos]])); cid6.append(np.full(len(pos), cid+i))
            cid += b
        for hk in hooks: hk.remove()
        return (rep, np.concatenate(S6, 0).astype(np.float32), np.concatenate(sym6), np.concatenate(cid6)) if collect else rep

    rep0, X6, syms, cids = run(collect=True)
    sae, stats, _ = train_sae(X6, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X6, stats, device="cpu")
    Pf = np.zeros((ad.n_obs, F.shape[1])); c = np.zeros(ad.n_obs)
    for row, ci in zip(F, cids): Pf[ci] += row; c[ci] += 1
    Pf /= np.maximum(c[:, None], 1)
    fj = int(np.abs(auroc_col(Pf, stim)-0.5).argmax())
    clampv = float(np.median(F[(F[:, fj] > 0) & np.isin(cids, np.where(stim == 1)[0]), fj]))  # stim-level firing
    print(f"==> IFN feature {fj}; stim-level clamp value {clampv:.3f}")
    holder.update(sae=sae, mean=stats.mean, scale=stats.scale)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(rep0, stim)
    base = clf.predict_proba(rep0)[:, 1]
    print(f"==> predictor LODO baseline established (rep AUROC {roc_auc_score(stim, base):.3f} in-sample)")
    sm, cm = stim == 1, stim == 0

    def steer(feat, val): holder.update(feat=feat, val=val); return clf.predict_proba(run())[:, 1]

    down = steer(fj, 0.0)                                          # ablate in all -> effect on stim
    up = steer(fj, clampv)                                        # clamp ON in all -> effect on ctrl
    active = np.where(F.max(0) > 0)[0]; rj = int(rng.choice(active[active != fj]))
    rup = steer(rj, clampv)                                       # random feature clamp -> ctrl control

    fdown = ((base[sm] >= .5) & (down[sm] < .5)).mean()
    fup = ((base[cm] < .5) & (up[cm] >= .5)).mean()
    fup_rnd = ((base[cm] < .5) & (rup[cm] >= .5)).mean()
    print(f"\n==> BIDIRECTIONAL STEERING (patch L6 -> full re-forward):")
    print(f"   DOWN  ablate IFN feat in stim -> {100*fdown:.1f}% of stim cells flip to ctrl "
          f"(mean prob {base[sm].mean():.2f} -> {down[sm].mean():.2f})")
    print(f"   UP    clamp IFN feat in ctrl  -> {100*fup:.1f}% of ctrl cells flip to response "
          f"(mean prob {base[cm].mean():.2f} -> {up[cm].mean():.2f})")
    print(f"   CTRL  clamp RANDOM feat       -> {100*fup_rnd:.1f}% of ctrl cells flip "
          f"(mean prob {base[cm].mean():.2f} -> {rup[cm].mean():.2f})")
    np.savez_compressed("outputs/singlecell/kang_steer.npz", base=base, down=down, up=up, rup=rup,
                        stim=stim, fdown=fdown, fup=fup, fup_rnd=fup_rnd, fj=fj)
    print("==> saved kang_steer.npz")


if __name__ == "__main__":
    main()
