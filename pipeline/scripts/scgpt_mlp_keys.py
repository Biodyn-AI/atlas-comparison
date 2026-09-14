#!/usr/bin/env python
"""scGPT MLP audit — stage 2 (KEYS), on real activations. Runs the CPU forward over a
PBMC corpus, captures relu(linear1) neuron activations per (cell,gene) token, and asks
per neuron: (a) does it fire on a specific TRRUST regulon's genes? (b) is it specific to
a cell type? BOTH with a permuted null (the static weights-only pass failed its null, so
every claim here is measured against chance).

    conda activate scprint
    python scripts/scgpt_mlp_keys.py --n-cells 160
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCGPT = f"{_B}/external/single_cell_mechinterp/external/scGPT"
sys.path.insert(0, SCGPT)
CKPT = f"{_B}/external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human"
BASE = _B
MARKERS = {"T": ["CD3D", "CD3E", "TRAC", "IL7R"], "B": ["MS4A1", "CD79A", "CD79B"],
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "NK": ["NKG7", "GNLY", "KLRD1"],
           "DC": ["FCER1A", "CST3", "CLEC10A"]}


def load_model():
    import importlib.util, types
    for pkg, path in (("scgpt", f"{SCGPT}/scgpt"), ("scgpt.model", f"{SCGPT}/scgpt/model")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg); m.__path__ = [path]; sys.modules[pkg] = m
    for sub in ("dsbn", "grad_reverse"):
        name = f"scgpt.model.{sub}"
        spec = importlib.util.spec_from_file_location(name, f"{SCGPT}/scgpt/model/{sub}.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("scgpt.model.model", f"{SCGPT}/scgpt/model/model.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["scgpt.model.model"] = mod; spec.loader.exec_module(mod)
    import torch
    vocab = json.load(open(f"{CKPT}/vocab.json")); args = json.load(open(f"{CKPT}/args.json"))
    model = mod.TransformerModel(ntoken=len(vocab), d_model=args["embsize"], nhead=args["nheads"],
        d_hid=args["d_hid"], nlayers=args["nlayers"], vocab=vocab, dropout=0.0,
        pad_token=args["pad_token"], pad_value=args["pad_value"], do_mvc=True,
        input_emb_style="continuous", cell_emb_style="cls", use_fast_transformer=False,
        pre_norm=False, mvc_decoder_style="inner product").eval()
    sd = torch.load(f"{CKPT}/best_model.pt", map_location="cpu", weights_only=False)
    remap = {}
    for k, v in sd.items():
        k2 = k.replace("Wqkv.weight", "in_proj_weight").replace("Wqkv.bias", "in_proj_bias")
        remap[k2] = v
    model.load_state_dict(remap, strict=False)
    return model, vocab, args


def bin_cell(vals, n_bins=51):
    import numpy as np
    if vals.size == 0:
        return vals
    return np.digitize(vals, np.quantile(vals, np.linspace(0, 1, n_bins - 1))).astype(np.float32)


def auroc_vec(Rk, tgt_idx, nG):
    npos = len(tgt_idx)
    return (Rk[tgt_idx].sum(0) - npos * (npos + 1) / 2) / (npos * (nG - npos))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-cells", type=int, default=160)
    ap.add_argument("--batch", type=int, default=4); args_ = ap.parse_args()
    import numpy as np, scanpy as sc, torch
    from scipy.stats import rankdata
    import pandas as pd

    model, vocab, cfg = load_model()
    nL = cfg["nlayers"]; dff = cfg["d_hid"]; nN = nL * dff
    layer_of = np.repeat(np.arange(nL), dff)
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]

    adata = sc.datasets.pbmc3k()[: args_.n_cells].copy()
    raw = adata.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    Sc = {}
    for ct, ms in MARKERS.items():
        pres = [g for g in ms if g in raw.var_names]
        X = raw[:, pres].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        Sc[ct] = np.mean([z(X[:, j]) for j in range(X.shape[1])], 0)
    S = np.stack([Sc[c] for c in MARKERS], 1); cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")
    print("==> cell types:", {c: int((cell_type == c).sum()) for c in np.unique(cell_type)})

    syms = np.array([s.upper() for s in adata.var_names])
    in_vocab = np.array([s in vocab for s in syms])
    Xd = raw.X.toarray() if hasattr(raw.X, "toarray") else raw.X

    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    uni = sorted(g for g in tr_g if g.upper() in vocab); uni_i = {g: i for i, g in enumerate(uni)}
    regs = {}
    for tf, grp in tr.groupby(tr.tf.str.upper()):
        t = [uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i]
        if len(t) >= 15:
            regs[tf] = np.array(t)
    print(f"==> {len(uni)} TRRUST genes, {len(regs)} regulons (>=15 targets); {nN} neurons")

    G = np.zeros((len(uni), nN), np.float32); cnt = np.zeros(len(uni), np.float32)
    P = np.zeros((args_.n_cells, nN), np.float32)
    for s in range(0, adata.n_obs, args_.batch):
        e = min(s + args_.batch, adata.n_obs); rows, vals, gsym = [], [], []
        for i in range(s, e):
            nz = np.where((Xd[i] > 0) & in_vocab)[0]
            order = nz[np.argsort(Xd[i][nz])[::-1]][:1199]
            rows.append([cls_id] + [vocab[syms[j]] for j in order])
            vals.append([0.0] + list(bin_cell(Xd[i][order])))
            gsym.append(["<cls>"] + [syms[j] for j in order])
        L = max(len(r) for r in rows)
        src = torch.full((len(rows), L), pad_id, dtype=torch.long)
        value = torch.full((len(rows), L), float(cfg["pad_value"]), dtype=torch.float)
        for i, (g, v) in enumerate(zip(rows, vals)):
            src[i, :len(g)] = torch.tensor(g); value[i, :len(v)] = torch.tensor(v)
        kpm = src.eq(pad_id)
        cap = {}
        ph = model.transformer_encoder.register_forward_pre_hook(
            lambda m, a, k: cap.__setitem__("x", a[0].detach()), with_kwargs=True)
        with torch.no_grad():
            model._encode(src, value, src_key_padding_mask=kpm)
        ph.remove()
        with torch.no_grad():
            x = cap["x"]; A = []
            for layer in model.transformer_encoder.layers:
                attn, _ = layer.self_attn(x, x, x, key_padding_mask=kpm, need_weights=False)
                x = layer.norm1(x + attn)
                act = layer.activation(layer.linear1(x))
                x = layer.norm2(x + layer.linear2(act))
                A.append(act.numpy())
        A = np.concatenate(A, -1)                                   # [b, L, nN]
        for bi in range(e - s):
            gl = gsym[bi]; valid = [(p, uni_i[g]) for p, g in enumerate(gl) if g in uni_i]
            genetok = [p for p, _ in valid]; gi = [k for _, k in valid]
            if gi:
                np.add.at(G, gi, A[bi, genetok]); np.add.at(cnt, gi, 1)
            P[s + bi] = A[bi, 1:len(gl)].mean(0)                    # per-cell mean over gene tokens
        print(f"  cells {s}-{e}", end="\r")
    G /= np.maximum(cnt[:, None], 1)
    print(f"\n==> captured. genes covered {int((cnt>0).sum())}/{len(uni)}")

    # ---- (a) regulon detectors: real vs gene-permuted null -------------
    nGuni = len(uni)
    def best_strength(Gmat):
        Rk = rankdata(Gmat, axis=0)
        best = np.full(nN, 0.5)
        for tf, t in regs.items():
            au = auroc_vec(Rk, t, nGuni); upd = np.abs(au - 0.5) > np.abs(best - 0.5); best[upd] = au[upd]
        return 2 * np.abs(best - 0.5), best
    strg, best_au = best_strength(G)
    rng = np.random.default_rng(0)
    strg_null, _ = best_strength(G[rng.permutation(nGuni)])          # permute gene labels
    thr = np.percentile(strg_null, 99)
    real_hits = int((strg > thr).sum()); null_hits = int((strg_null > thr).sum())
    print(f"\n== (a) REGULON detectors (gene-level activation) ==")
    print(f"  strength threshold (99th pct of null) = {thr:.3f}")
    print(f"  REAL neurons above threshold: {real_hits}/{nN} ({100*real_hits/nN:.1f}%)  "
          f"vs NULL {null_hits}/{nN} ({100*null_hits/nN:.1f}%)  -> excess {real_hits-null_hits}")

    # ---- (b) cell-type detectors: real vs shuffled-label null ----------
    print(f"\n== (b) CELL-TYPE detectors (per-cell activation) ==")
    types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 8 and c != "other"]
    Rp = rankdata(P, axis=0)
    best_ct = np.full(nN, 0.5); best_ct_name = np.array(["-"] * nN, dtype=object)
    for c in types:
        y = np.where(cell_type == c)[0]
        au = auroc_vec(Rp, y, args_.n_cells); upd = np.abs(au - 0.5) > np.abs(best_ct - 0.5)
        best_ct[upd] = au[upd]; best_ct_name[upd] = c
    ct_str = 2 * np.abs(best_ct - 0.5)
    perm = rng.permutation(args_.n_cells)
    best_ct0 = np.full(nN, 0.5)
    for c in types:
        y = np.where(cell_type[perm] == c)[0]
        au = auroc_vec(Rp, y, args_.n_cells); best_ct0 = np.where(np.abs(au-0.5) > np.abs(best_ct0-0.5), au, best_ct0)
    thr_ct = np.percentile(2*np.abs(best_ct0-0.5), 99)
    rct = int((ct_str > thr_ct).sum()); nct = int((2*np.abs(best_ct0-0.5) > thr_ct).sum())
    print(f"  threshold (99th pct null) = {thr_ct:.3f}; REAL {rct}/{nN} ({100*rct/nN:.1f}%) vs NULL {nct}/{nN} -> excess {rct-nct}")
    order = np.argsort(ct_str)[::-1]
    print("  top cell-type-specific neurons:")
    for j in order[:12]:
        print(f"    neuron {j:>4} L{layer_of[j]:>2} -> {best_ct_name[j]:<5} AUROC {best_ct[j]:.3f}")
    strong_ct = ct_str > thr_ct
    print(f"  layer distribution of cell-type detectors: {dict(pd.Series(layer_of[strong_ct]).value_counts().sort_index())}")
    np.savez_compressed(f"{BASE}/outputs/singlecell/scgpt_mlp_keys.npz",
                        strg=strg, strg_null=strg_null, ct_str=ct_str, best_ct=best_ct,
                        best_ct_name=best_ct_name, layer_of=layer_of)
    print("\n==> stage-2 done")


if __name__ == "__main__":
    main()
