#!/usr/bin/env python
"""scGPT MLP audit — step 0: light-path checkpoint inventory (no scgpt install, no
forward). Confirms the key-value MLP structure (linear1 = keys, linear2 = values) per
layer and the output-head weights, so stages 2-3 can start. torch.load a state_dict of
plain tensors needs no model class.

    conda activate bae
    python scripts/scgpt_mlp_inventory.py
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os, re
import torch

BASE = _B
CKPT = f"{BASE}/external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human"

args = json.load(open(f"{CKPT}/args.json"))
print("==> args.json (model config):")
for k in ("embsize", "d_model", "nheads", "nhead", "d_hid", "nlayers", "n_layers",
          "dropout", "n_bins", "input_emb_style", "cell_emb_style", "MVC", "use_fast_transformer"):
    if k in args:
        print(f"    {k} = {args[k]}")
vocab = json.load(open(f"{CKPT}/vocab.json"))
print(f"==> vocab: {len(vocab)} gene tokens (e.g. {list(vocab)[:6]})")

sd = torch.load(f"{CKPT}/best_model.pt", map_location="cpu", weights_only=False)
if not isinstance(sd, dict) or all(hasattr(v, "shape") for v in sd.values()):
    state = sd
else:
    state = sd.get("model_state_dict", sd.get("state_dict", sd))
print(f"\n==> checkpoint: {len(state)} tensors")

# MLP submodules per encoder layer
lin1 = sorted(k for k in state if re.search(r"linear1\.weight$", k))
lin2 = sorted(k for k in state if re.search(r"linear2\.weight$", k))
print(f"\n==> MLP found in {len(lin1)} layers")
if lin1:
    w1 = state[lin1[0]]; w2 = state[lin2[0]]
    print(f"    linear1 (W_in / keys)   {tuple(w1.shape)}  = [d_ff, d_model]")
    print(f"    linear2 (W_out / values){tuple(w2.shape)}  = [d_model, d_ff]")
    print(f"    -> {w1.shape[0]} neurons/layer x {len(lin1)} layers = {w1.shape[0]*len(lin1)} MLP neurons")
    print(f"    sample layer keys: {lin1[0]} | {lin2[0]}")

# heads / embeddings for value->gene projection (stage 3)
print("\n==> heads & embeddings (for value->gene projection):")
for k in sorted(state):
    if re.search(r"decoder|embedding|encoder\.embedding|gene_encoder|expr|cls|mvc|value", k, re.I) \
       and "layers." not in k:
        print(f"    {k:<55} {tuple(state[k].shape)}")


if __name__ == "__main__":
    pass
