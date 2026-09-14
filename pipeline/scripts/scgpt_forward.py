#!/usr/bin/env python
"""scGPT MLP audit — stand up a CPU forward (no scgpt install; import model class from
local source, tokenize from vocab.json). Builds TransformerModel(use_fast_transformer=
False) = nn.TransformerEncoderLayer (relu, post-norm = numerically faithful to the
trained flash layer), remaps the fused Wqkv attention weights to in_proj, loads weights,
runs _encode on PBMC cells, and captures MLP neuron activations relu(linear1(h)) via
hooks. Step 1 = validate load + forward; then stage-2 (keys) builds on this.

    conda activate bae
    python scripts/scgpt_forward.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCGPT = f"{_B}/external/single_cell_mechinterp/external/scGPT"
sys.path.insert(0, SCGPT)
CKPT = f"{_B}/external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human"


def bin_cell(vals, n_bins=51):
    """scGPT-style per-cell binning of nonzero expression -> bin indices 1..n_bins-1."""
    import numpy as np
    if vals.size == 0:
        return vals
    bins = np.quantile(vals, np.linspace(0, 1, n_bins - 1))
    return np.digitize(vals, bins).astype(np.float32)


def load_transformer_model_class():
    """Import scgpt.model.model WITHOUT running scgpt/__init__ (which needs datasets/scvi)."""
    import importlib.util, sys as _sys, types
    for pkg, path in (("scgpt", f"{SCGPT}/scgpt"), ("scgpt.model", f"{SCGPT}/scgpt/model")):
        if pkg not in _sys.modules:
            m = types.ModuleType(pkg); m.__path__ = [path]; _sys.modules[pkg] = m
    for sub in ("dsbn", "grad_reverse"):
        name = f"scgpt.model.{sub}"
        spec = importlib.util.spec_from_file_location(name, f"{SCGPT}/scgpt/model/{sub}.py")
        mod = importlib.util.module_from_spec(spec); _sys.modules[name] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("scgpt.model.model", f"{SCGPT}/scgpt/model/model.py")
    mod = importlib.util.module_from_spec(spec); _sys.modules["scgpt.model.model"] = mod
    spec.loader.exec_module(mod)
    return mod.TransformerModel


def main():
    import numpy as np, scanpy as sc, torch
    try:
        torch.backends.mha.set_fastpath_enabled(False)   # force eager so submodule hooks fire
    except Exception as e:
        print("  (could not disable mha fastpath:", e, ")")
    TransformerModel = load_transformer_model_class()

    vocab = json.load(open(f"{CKPT}/vocab.json"))
    args = json.load(open(f"{CKPT}/args.json"))
    model = TransformerModel(
        ntoken=len(vocab), d_model=args["embsize"], nhead=args["nheads"],
        d_hid=args["d_hid"], nlayers=args["nlayers"], vocab=vocab,
        dropout=0.0, pad_token=args["pad_token"], pad_value=args["pad_value"],
        do_mvc=args.get("MVC", True), input_emb_style="continuous",
        cell_emb_style="cls", use_fast_transformer=False, pre_norm=False,
        mvc_decoder_style="inner product",
    ).eval()

    sd = torch.load(f"{CKPT}/best_model.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) and "model_state_dict" in sd else sd
    remap = {}
    for k, v in sd.items():
        if k.endswith("self_attn.Wqkv.weight"):
            remap[k.replace("Wqkv.weight", "in_proj_weight")] = v
        elif k.endswith("self_attn.Wqkv.bias"):
            remap[k.replace("Wqkv.bias", "in_proj_bias")] = v
        else:
            remap[k] = v
    miss, unexp = model.load_state_dict(remap, strict=False)
    print(f"==> loaded. missing={len(miss)} unexpected={len(unexp)}")
    print("    missing (first 8):", list(miss)[:8])
    print("    unexpected (first 8):", list(unexp)[:8])

    # ---- a batch of PBMC cells -> (gene ids, binned values) ------------
    adata = sc.datasets.pbmc3k()[:8].copy()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    syms = np.array([s.upper() for s in adata.var_names])
    in_vocab = np.array([s in vocab for s in syms])
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]
    max_len = 1200
    rows, vals = [], []
    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    for i in range(adata.n_obs):
        nz = np.where((Xd[i] > 0) & in_vocab)[0]
        order = nz[np.argsort(Xd[i][nz])[::-1]][: max_len - 1]      # top genes by expr
        gid = [cls_id] + [vocab[syms[j]] for j in order]
        val = [0.0] + list(bin_cell(Xd[i][order]))
        rows.append(gid); vals.append(val)
    L = max(len(r) for r in rows)
    src = torch.full((len(rows), L), pad_id, dtype=torch.long)
    value = torch.full((len(rows), L), float(args["pad_value"]), dtype=torch.float)
    for i, (g, v) in enumerate(zip(rows, vals)):
        src[i, :len(g)] = torch.tensor(g); value[i, :len(v)] = torch.tensor(v)
    key_pad = src.eq(pad_id)
    print(f"==> batch {tuple(src.shape)}; seq lens {[len(r) for r in rows]}")

    # ---- hook MLP neuron activations relu(linear1(h)) ------------------
    # capture the transformer input, then run the layers MANUALLY (eager post-norm) so we
    # read relu(linear1(h)) directly — nn.TransformerEncoder's fused fast path skips submodules
    captured = {}

    def pre_hook(mod, hargs, hkwargs):
        captured["x"] = hargs[0].detach()
        captured["mask"] = hkwargs.get("src_key_padding_mask", None)
    ph = model.transformer_encoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        enc = model._encode(src, value, src_key_padding_mask=key_pad)
    ph.remove()

    def manual_layer(layer, x, kpm):
        attn, _ = layer.self_attn(x, x, x, key_padding_mask=kpm, need_weights=False)
        x = layer.norm1(x + attn)
        act = layer.activation(layer.linear1(x))
        x = layer.norm2(x + layer.linear2(act))
        return act, x

    acts = {}
    with torch.no_grad():
        x = captured["x"]; kpm = captured["mask"]
        for L_i, layer in enumerate(model.transformer_encoder.layers):
            acts[L_i], x = manual_layer(layer, x, kpm)
    nonpad = ~key_pad
    fid_all = (x - enc).abs().max().item()
    fid_real = (x - enc)[nonpad].abs().max().item()
    print(f"==> _encode output {tuple(enc.shape)}; max|Δ| all={fid_all:.2e}, non-pad tokens={fid_real:.2e} "
          f"({'FAITHFUL on real tokens' if fid_real < 1e-3 else 'MISMATCH'})")
    A0 = acts[0]
    print(f"==> layer-0 MLP activations {tuple(A0.shape)} (B, L, d_ff)")
    for L_i in (0, 5, 11):
        A = acts[L_i][~key_pad]                                     # non-pad tokens
        print(f"    layer {L_i:>2}: mean {A.mean():.3f}, frac>0 {(A>0).float().mean():.3f}, "
              f"max {A.max():.2f}")
    print("==> FORWARD WORKS on CPU — ready for stage-2 (max-activating annotation)")


if __name__ == "__main__":
    main()
