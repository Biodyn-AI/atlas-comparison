"""Untrained-model control: the same adapter, corpus, sampling (all layers, at most 168 positions per
cell, seed 0) and SAE recipe as the atlas (run_model.py --all-layers --per-cell 168), but the model's
transformer weights are replaced by a fresh random initialisation (torch seed 0).

  Geneformer: a new model built from the same config (AutoModelForMaskedLM.from_config), so the
              token embeddings are random too.
  UCE:        the TransformerModel is constructed but its trained weights are not loaded; the ESM2
              protein embeddings at the input are kept, because UCE does not learn them.

usage: ATLAS_GPU_BUNDLE=<bundle> python run_untrained.py MODEL
"""
import os, sys, importlib
BUNDLE = os.path.abspath(os.environ["ATLAS_GPU_BUNDLE"])
CACHE = os.path.abspath(os.environ.get("ATLAS_CACHE", os.path.join(os.path.dirname(__file__), "..", ".cache")))
OUT = os.path.abspath(os.environ.get("ATLAS_UNTRAINED_OUTPUT", os.path.join(CACHE, "out_untrained")))
CORPUS = os.path.abspath(os.environ.get("ATLAS_GPU_CORPUS", os.path.join(BUNDLE, "external/perturb/tabula_3tissue_6k.h5ad")))
sys.path.insert(0, BUNDLE)
os.chdir(BUNDLE)                                   # adapters resolve checkpoints relative to the bundle
import torch
import run_model
from common import pipeline

m = sys.argv[1]
mod, cls = run_model.ADAPTERS[m]
adapter = getattr(importlib.import_module(mod), cls)()
if m == "UCE":                                     # build the model without loading the trained state dict
    import torch.nn as nn
    _load = torch.load
    def _no_state(path, *a, **k):
        obj = _load(path, *a, **k)
        if str(path) == str(getattr(adapter, "model_loc", "")):
            torch.manual_seed(0)
            return adapter.model.state_dict()      # the freshly initialised weights
        return obj
    torch.load = _no_state
    torch.manual_seed(0)
    adapter.load(device="cuda")
    torch.load = _load
else:
    adapter.load(device="cuda")
    from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
    cfg = adapter.model.config
    torch.manual_seed(0)
    loader = AutoModelForCausalLM if getattr(adapter, "hf_auto", "") == "causal" else AutoModelForMaskedLM
    adapter.model = loader.from_config(cfg).eval().to("cuda")
adapter.layers = tuple(range(adapter.n_layers if getattr(adapter, "n_layers", None) else max(adapter.layers) + 1))
print(f"==> untrained {m}: layers {list(adapter.layers)}", flush=True)
import scanpy as sc
adata = sc.read_h5ad(CORPUS)   # the corpus of run_full_fixed.sh
out_dir = os.path.join(OUT, m)
pipeline.extract(adapter, adata, out_dir, batch_size=8, max_positions=4_000_000, per_cell=168, seed=0)
pipeline.build_catalog(adapter, out_dir, device="cuda", seed=0)
print("UNTRAINED_DONE", m, flush=True)
