"""Tahoe-x1 adapter (Tahoe Therapeutics 2025). Transformer-encoder MLM over ranked gene
tokens, custom MosaicML Composer model (NOT HF AutoModel). Apache-2.0, no gate. For the atlas
we take Tx1-70M (smallest; context 1024) — a perturbation-trained gene-token FM axis.

Loading (per the model card):
    from tahoe_x1.model import ComposerTX
    model, vocab, model_cfg, collator_cfg = ComposerTX.from_hf(repo_id="tahoebio/Tahoe-x1", model_size="3b")
Vocabulary is Ensembl-keyed (gene_id_key="ensembl_id"); tokenisation ranks a cell's genes.

SCAFFOLD — reconstructed from the repo; VERIFY on H100 (expect 1-3 fixes, like the uce adapter):
  * the transformer block path for per-layer hooks — try model.model.transformer.blocks (MPT-style),
    else model.model.encoder.layers / grep the ComposerTX module for the nn.ModuleList of blocks.
  * exact tokenisation: whether Tx1 bins expression or uses pure rank; here we replicate a
    Geneformer-style rank (by raw count) over in-vocab Ensembl genes, capped to context. If their
    collator value-bins, mirror collator_cfg.
  * d_model / n_layers / context are read from model_cfg (not hardcoded).
Corpus var must expose Ensembl ids (var['ensembl_id'] or ENSG var_names) + symbols (feature_name/var_names).
"""
from __future__ import annotations
import os, sys
import numpy as np
from .base import Adapter


class TahoeAdapter(Adapter):
    name = "Tahoe"

    def __init__(self, repo_id="tahoebio/Tahoe-x1", model_size="3b", tahoe_repo=None,
                 layers=None, max_len=2048):
        self.repo_id = repo_id; self.model_size = model_size; self.tahoe_repo = tahoe_repo
        self._layers_override = layers; self.max_len = max_len

    def load(self, device="cuda"):
        import torch
        if self.tahoe_repo:
            sys.path.insert(0, self.tahoe_repo)
        from tahoe_x1.model import ComposerTX
        self.torch = torch; self.device = device
        model, vocab, model_cfg, collator_cfg = ComposerTX.from_hf(repo_id=self.repo_id, model_size=self.model_size)
        self.model = model.eval().to(device)
        self.collator_cfg = collator_cfg
        # vocab: ensembl (or gene) -> id. GeneVocab-like; support dict or torchtext-style
        self.vocab = vocab
        self.stoi = vocab.get_stoi() if hasattr(vocab, "get_stoi") else dict(vocab)
        self.pad_id = self.stoi.get("<pad>", self.stoi.get("[PAD]", 0))
        self.d_model = int(getattr(model_cfg, "d_model", getattr(model_cfg, "hidden_size", 0)) or model_cfg["d_model"])
        self.n_layers = int(getattr(model_cfg, "n_layers", getattr(model_cfg, "num_hidden_layers", 0)) or model_cfg["n_layers"])
        from common.layers import depth_matched
        self.layers = tuple(self._layers_override) if self._layers_override else depth_matched(self.n_layers)
        # locate the transformer blocks (ModuleList) to hook -- VERIFY path on H100
        self.blocks = self._find_blocks(self.model)

    @staticmethod
    def _find_blocks(model):
        import torch.nn as nn
        for path in ("model.transformer.blocks", "transformer.blocks", "model.encoder.layers",
                     "encoder.layers", "model.blocks", "blocks"):
            obj = model
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                if isinstance(obj, (list, nn.ModuleList)) and len(obj) > 1:
                    return obj
            except AttributeError:
                continue
        # last resort: the longest ModuleList of identical blocks
        best = None
        for m in model.modules():
            if isinstance(m, nn.ModuleList) and len(m) > 1:
                if best is None or len(m) > len(best):
                    best = m
        if best is None:
            raise RuntimeError("could not locate transformer blocks for hooking; inspect ComposerTX")
        return best

    def _var_maps(self, adata):
        vn = np.array([str(x) for x in adata.var_names])
        if "ensembl_id" in adata.var.columns:
            ens = np.array([str(x).split(".")[0] for x in adata.var["ensembl_id"]])
        elif vn[0].startswith("ENSG"):
            ens = np.array([x.split(".")[0] for x in vn])
        else:
            raise ValueError("Tahoe needs Ensembl ids in corpus var")
        sym = (np.array([str(x).upper() for x in adata.var["feature_name"]])
               if "feature_name" in adata.var.columns else np.array([x.upper() for x in vn]))
        tok = np.array([self.stoi.get(e, -1) for e in ens])
        return sym, tok

    def iter_activations(self, adata, batch_size=8):
        import torch
        self.processed_obs = adata.obs.reset_index(drop=True)
        sym, tok = self._var_maps(adata)
        in_vocab = tok >= 0
        caps = {}
        hooks = [self.blocks[L].register_forward_hook(
            (lambda L: (lambda m, i, o: caps.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())))(L))
            for L in self.layers]
        cid = 0
        for s in range(0, adata.n_obs, batch_size):
            xb = adata.X[s:min(s + batch_size, adata.n_obs)]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            ids_batch, sym_batch = [], []
            for row in xb:
                v = np.where((row > 0) & in_vocab)[0]
                order = v[np.argsort(-row[v])][: self.max_len]       # rank by expression (VERIFY vs their collator)
                ids_batch.append(tok[order]); sym_batch.append(sym[order])
            L = max(1, max(len(x) for x in ids_batch))
            inp = np.full((len(ids_batch), L), self.pad_id, dtype=np.int64)
            for i, ids in enumerate(ids_batch):
                inp[i, :len(ids)] = ids
            input_ids = torch.as_tensor(inp, device=self.device)
            mask = (input_ids != self.pad_id)
            caps.clear()
            with torch.no_grad():
                try:
                    self.model(input_ids=input_ids, attention_mask=mask.long())
                except TypeError:
                    self.model({"gene": input_ids, "key_padding_mask": ~mask})   # VERIFY collator batch keys
            b = len(ids_batch)
            acts = {}
            for Ly in self.layers:
                h = caps[Ly].float().cpu().numpy()
                acts[Ly] = np.concatenate([h[i, :len(sym_batch[i]), :] for i in range(b)], 0)
            syms = np.concatenate(sym_batch)
            cids = np.concatenate([np.full(len(sym_batch[i]), cid + i) for i in range(b)])
            cid += b
            yield acts, syms, cids
        for hk in hooks:
            hk.remove()
