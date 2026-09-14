#!/usr/bin/env python
"""scGPT MLP audit — stage 5 (CAUSAL, the flagship). For top detector neurons from
stage 2, ablate the neuron (zero its activation at its layer), run scGPT's expression
decoder, and test whether the prediction shift concentrates on THAT neuron's concept
(its regulon's target genes, or its cell type's cells) vs random-neuron controls. If it
does, the neuron stores that regulatory 'fact' (key-value memory, Dai knowledge-neuron).

    conda activate scprint
    python scripts/scgpt_mlp_causal.py --n-cells 64
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
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "NK": ["NKG7", "GNLY", "KLRD1"]}


def load_model():
    import importlib.util, types, torch
    for pkg, path in (("scgpt", f"{SCGPT}/scgpt"), ("scgpt.model", f"{SCGPT}/scgpt/model")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg); m.__path__ = [path]; sys.modules[pkg] = m
    for sub in ("dsbn", "grad_reverse"):
        name = f"scgpt.model.{sub}"
        spec = importlib.util.spec_from_file_location(name, f"{SCGPT}/scgpt/model/{sub}.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("scgpt.model.model", f"{SCGPT}/scgpt/model/model.py")
    mm = importlib.util.module_from_spec(spec); sys.modules["scgpt.model.model"] = mm; spec.loader.exec_module(mm)
    vocab = json.load(open(f"{CKPT}/vocab.json")); args = json.load(open(f"{CKPT}/args.json"))
    model = mm.TransformerModel(ntoken=len(vocab), d_model=args["embsize"], nhead=args["nheads"],
        d_hid=args["d_hid"], nlayers=args["nlayers"], vocab=vocab, dropout=0.0,
        pad_token=args["pad_token"], pad_value=args["pad_value"], do_mvc=True,
        input_emb_style="continuous", cell_emb_style="cls", use_fast_transformer=False,
        pre_norm=False, mvc_decoder_style="inner product").eval()
    sd = torch.load(f"{CKPT}/best_model.pt", map_location="cpu", weights_only=False)
    remap = {k.replace("Wqkv.weight", "in_proj_weight").replace("Wqkv.bias", "in_proj_bias"): v
             for k, v in sd.items()}
    model.load_state_dict(remap, strict=False)
    return model, vocab, args


def bin_cell(v, n=51):
    import numpy as np
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, n - 1))).astype(np.float32) if v.size else v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-cells", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8); A = ap.parse_args()
    import numpy as np, scanpy as sc, torch, pandas as pd
    from scipy.stats import rankdata

    model, vocab, cfg = load_model()
    nL, dff = cfg["nlayers"], cfg["d_hid"]; nN = nL * dff
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]

    adata = sc.datasets.pbmc3k()[: A.n_cells].copy()
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

    syms = np.array([s.upper() for s in adata.var_names]); in_vocab = np.array([s in vocab for s in syms])
    Xd = raw.X.toarray() if hasattr(raw.X, "toarray") else raw.X
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_g = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    uni = sorted(g for g in tr_g if g in vocab); uni_i = {g: i for i, g in enumerate(uni)}
    regs = {tf: np.array([uni_i[x.upper()] for x in grp.tg if x.upper() in uni_i])
            for tf, grp in tr.groupby(tr.tf.str.upper())}
    regs = {k: v for k, v in regs.items() if len(v) >= 15}

    # pre-tokenize all batches
    batches = []
    for s in range(0, adata.n_obs, A.batch):
        e = min(s + A.batch, adata.n_obs); rows, vals, gsym = [], [], []
        for i in range(s, e):
            nz = np.where((Xd[i] > 0) & in_vocab)[0]
            order = nz[np.argsort(Xd[i][nz])[::-1]][:1199]
            rows.append([cls_id] + [vocab[syms[j]] for j in order])
            vals.append([0.0] + list(bin_cell(Xd[i][order]))); gsym.append(["<cls>"] + list(syms[order]))
        L = max(len(r) for r in rows)
        src = torch.full((len(rows), L), pad_id, dtype=torch.long)
        val = torch.full((len(rows), L), float(cfg["pad_value"]), dtype=torch.float)
        for i, (g, v) in enumerate(zip(rows, vals)):
            src[i, :len(g)] = torch.tensor(g); val[i, :len(v)] = torch.tensor(v)
        batches.append((s, e, src, val, src.eq(pad_id), gsym))

    def run(ablate=None):
        """forward all batches; return per-cell decoder pred dict {cell: {gene: pred}} and
        accumulators. ablate=(layer, local_idx) zeroes that neuron."""
        preds = {}; G = np.zeros((len(uni), nN), np.float32) if ablate is None else None
        cnt = np.zeros(len(uni), np.float32) if ablate is None else None
        P = np.zeros((A.n_cells, nN), np.float32) if ablate is None else None
        for (s, e, src, val, kpm, gsym) in batches:
            cap = {}
            ph = model.transformer_encoder.register_forward_pre_hook(
                lambda m, a, k: cap.__setitem__("x", a[0].detach()), with_kwargs=True)
            with torch.no_grad():
                model._encode(src, val, src_key_padding_mask=kpm)
            ph.remove()
            with torch.no_grad():
                x = cap["x"]; acts_all = []
                for li, layer in enumerate(model.transformer_encoder.layers):
                    at, _ = layer.self_attn(x, x, x, key_padding_mask=kpm, need_weights=False)
                    x = layer.norm1(x + at); act = layer.activation(layer.linear1(x))
                    if ablate is not None and ablate[0] == li:
                        act = act.clone(); act[..., ablate[1]] = 0.0
                    x = layer.norm2(x + layer.linear2(act))
                    if ablate is None:
                        acts_all.append(act.numpy())
                pred = model.decoder(x)
                pred = pred["pred"] if isinstance(pred, dict) else pred
                pred = pred.squeeze(-1).numpy()                       # [b, L]
            for bi in range(e - s):
                gl = gsym[bi]
                preds[s + bi] = {g: pred[bi, p] for p, g in enumerate(gl) if g != "<cls>"}
                if ablate is None:
                    Aacc = np.concatenate([a[bi] for a in acts_all], -1)  # [L, nN]
                    valid = [(p, uni_i[g]) for p, g in enumerate(gl) if g in uni_i]
                    if valid:
                        gi = [k for _, k in valid]; gt = [p for p, _ in valid]
                        np.add.at(G, gi, Aacc[gt]); np.add.at(cnt, gi, 1)
                    P[s + bi] = Aacc[1:len(gl)].mean(0)
        return preds, (G, cnt, P)

    base, (G, cnt, P) = run(None); G /= np.maximum(cnt[:, None], 1)
    print(f"==> baseline done; genes covered {int((cnt>0).sum())}")

    # pick target neurons from THIS batch's detection
    def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0) - npos*(npos+1)/2)/(npos*(n-npos))
    Rp = rankdata(P, axis=0)
    targets = []
    for ct in ("B", "Mono", "T"):
        y = np.where(cell_type == ct)[0]
        if len(y) >= 6:
            au = auroc(Rp, y, A.n_cells); j = int(np.abs(au-0.5).argmax())
            targets.append(("celltype", ct, j, au[j]))
    Rg = rankdata(G, axis=0)
    best = np.full(nN, 0.5); bestr = np.array(["-"]*nN, dtype=object)
    for tf, t in regs.items():
        au = auroc(Rg, t, len(uni)); upd = np.abs(au-0.5) > np.abs(best-0.5); best[upd]=au[upd]; bestr[upd]=tf
    for j in np.argsort(np.abs(best-0.5))[::-1][:3]:
        targets.append(("regulon", bestr[j], int(j), best[j]))
    rng = np.random.default_rng(1)
    for j in rng.integers(0, nN, 3):
        targets.append(("random", "-", int(j), np.nan))

    print("\n==> CAUSAL ablation test (Δ decoder prediction):")
    print(f"    {'kind':<9} {'concept':<8} {'neuron':>5} {'layer':>5} {'detectAUROC':>11} {'causalAUROC':>11} {'ratio':>6}")
    for kind, concept, j, det in targets:
        li, loc = j // dff, j % dff
        abl, _ = run((li, loc))
        # per-gene Δ (mean over cells) for regulon; per-cell Δ for celltype
        if kind in ("regulon", "random"):
            per = {}
            for c in base:
                for g, pv in base[c].items():
                    if g in abl[c]:
                        per.setdefault(g, []).append(abs(abl[c][g] - pv))
            gd = {g: np.mean(v) for g, v in per.items() if g in uni_i}
            gg = [g for g in gd]; dv = np.array([gd[g] for g in gg])
            tgt = regs.get(concept, np.array([]))
            ismem = np.array([uni_i[g] in set(tgt.tolist()) for g in gg]) if kind == "regulon" else \
                    np.array([uni_i[g] in set(next(iter(regs.values())).tolist()) for g in gg])
            if 3 <= ismem.sum() < len(ismem):
                cau = auroc(rankdata(dv)[:, None], np.where(ismem)[0], len(dv))[0]
                ratio = dv[ismem].mean()/max(dv[~ismem].mean(), 1e-9)
            else:
                cau, ratio = np.nan, np.nan
        else:  # celltype: per-cell Δ
            cd = np.array([np.mean([abs(abl[c][g]-base[c][g]) for g in base[c] if g in abl[c]])
                           for c in range(A.n_cells)])
            y = (cell_type == concept)
            cau = auroc(rankdata(cd)[:, None], np.where(y)[0], A.n_cells)[0] if 3 <= y.sum() < len(y) else np.nan
            ratio = cd[y].mean()/max(cd[~y].mean(), 1e-9)
        dets = f"{det:.3f}" if not np.isnan(det) else "  -  "
        print(f"    {kind:<9} {concept:<8} {j:>5} {li:>5} {dets:>11} {cau:>11.3f} {ratio:>6.2f}")
    print("\n==> stage-5 done (target neurons should beat random on causal AUROC & ratio)")


if __name__ == "__main__":
    main()
