"""AIDO.Cell-10M adapter (GenBio gb_cell). VERIFIED forward path (used in our audit).
Reads all 19,264 genes/cell; residual per gene token. Env needs `gb_cell`.
"""
from __future__ import annotations
import numpy as np
from .base import Adapter


class AIDOAdapter(Adapter):
    name = "AIDO"
    d_model = 256

    def __init__(self, ckpt: str = "ckpt_aido", genes_tsv: str = "external/scprint_data/aido_genes.tsv",
                 layers=None):
        self.ckpt = ckpt
        self.genes_tsv = genes_tsv
        self._layers_override = layers          # None -> depth_matched after load

    def load(self, device: str = "cuda"):
        import torch
        from gb_cell.models import CellFoundationModel, CellFoundationConfig
        self.torch = torch
        self.device = device
        cfg = CellFoundationConfig.from_pretrained(self.ckpt)
        self.n_layers = cfg.num_hidden_layers
        from common.layers import depth_matched
        self.layers = tuple(self._layers_override) if self._layers_override else depth_matched(self.n_layers)
        self.model = CellFoundationModel.from_pretrained(self.ckpt, config=cfg).eval().to(device)
        self.genes = np.array([l.split("\t")[0].upper()
                               for l in open(self.genes_tsv).read().splitlines()[1:]])
        self.ng = len(self.genes)

    def iter_activations(self, adata, batch_size: int = 8):
        import torch
        from gb_cell.utils import align_adata, preprocess_counts
        ad, attn = align_adata(adata)
        self.processed_obs = ad.obs.reset_index(drop=True)
        attn_t = torch.from_numpy(attn).unsqueeze(0)
        for s in range(0, ad.n_obs, batch_size):
            xb = ad.X[s:s + batch_size]
            xb = xb.toarray() if hasattr(xb, "toarray") else xb
            inp = preprocess_counts(xb, device=self.device)
            if inp.is_floating_point():          # GB.Cell preprocessing returns bf16 on GPU; weights are f32
                inp = inp.float()
            am = torch.cat([attn_t.repeat(inp.shape[0], 1).to(self.device),
                            torch.ones((inp.shape[0], 2), device=self.device)], 1)
            with torch.no_grad():
                out = self.model(input_ids=inp, attention_mask=am, output_hidden_states=True)
            hs = out.hidden_states  # tuple len n_layers+1
            b = inp.shape[0]
            acts = {L: hs[L][:, :self.ng, :].reshape(-1, self.d_model).float().cpu().numpy()
                    for L in self.layers}
            syms = np.tile(self.genes, b)
            cell_ids = np.repeat(np.arange(s, s + b), self.ng)
            yield acts, syms, cell_ids
