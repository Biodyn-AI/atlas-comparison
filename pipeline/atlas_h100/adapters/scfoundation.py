"""scFoundation / xTrimoGene adapter (Hao et al., Nature Methods 2024; BioMap biomap-research/scFoundation).
Large read-depth-aware expression model: asymmetric MAE, a value (token) embedding of continuous expression
+ gene position embedding, a SPARSE encoder over NON-ZERO genes, and two auxiliary read-depth tokens T,S
(log10 total / sampled counts) appended to the sequence. We take the ENCODER's per-gene hidden states
(dropping T/S + padding) as the residual — matching our per-gene-token contract. New axis: large expression
model with an explicit sequencing-depth token.

REQUIRES on the cluster (weights are PUBLIC — biomap-research/scFoundation, models/models.ckpt via their
release form / figshare):
  * scFoundation repo `model/` dir on sys.path -> sf_repo=<path> (provides load.py + pretrainmodels)
  * checkpoint -> ckpt=<path>/models.ckpt
  * their gene index tsv (19,264 genes) -> gene_index=<path>/OS_scRNA_gene_index.19264.tsv

VERIFY on H100 (written from their get_embedding.py; expect real iteration):
  * exact load util + key: `load_model_frommmf(ckpt, key='gene')` (encoder+gene head) vs 'cell'/'rde'.
  * the value-tokenisation + T/S append + gatherData sparse packing (their pretrain preprocessing).
  * model.token_emb / pos_emb / encoder attribute names + forward signature.
  * gene ordering: their gene index is HGNC symbols already; if Ensembl, map via biomart.
"""
from __future__ import annotations
import sys
import numpy as np
from .base import Adapter


class ScFoundationAdapter(Adapter):
    name = "scFoundation"; d_model = 768

    def __init__(self, sf_repo="external/scFoundation/model",
                 ckpt="external/scFoundation/model/models/models.ckpt",
                 gene_index="external/scFoundation/model/OS_scRNA_gene_index.19264.tsv", layers=None):
        self.sf_repo = sf_repo; self.ckpt = ckpt; self.gene_index = gene_index
        self._layers_override = tuple(layers) if layers else None
        self.layers = self._layers_override or (0, 3, 6, 9, 12)

    def load(self, device="cuda"):
        import torch, pandas as pd
        sys.path.insert(0, self.sf_repo)
        from load import load_model_frommmf, gatherData           # their load.py API
        self.torch = torch; self.device = device; self._gather = gatherData
        self.model, self.model_config = load_model_frommmf(self.ckpt, key="gene")   # VERIFY key ('gene'/'cell'/'rde')
        self.model = self.model.eval().to(device)
        self.pad_id = int(self.model_config.get("pad_token_id", 103))
        enc = getattr(self.model, "encoder", self.model)
        blocks = getattr(enc, "transformer_encoder", getattr(enc, "layers", None))
        self.n_layers = len(blocks) if blocks is not None else int(self.model_config.get("encoder_depth", 12))
        if not self._layers_override:
            from common.layers import depth_matched
            self.layers = depth_matched(self.n_layers)
        gi = pd.read_csv(self.gene_index, sep="\t")
        col = "gene_name" if "gene_name" in gi.columns else gi.columns[-1]
        self.genes = np.array([str(g).upper() for g in gi[col].values])   # 19264 panel, HGNC
        self.ng = len(self.genes)
        # hook the encoder blocks so we get every layer's post-residual output (index_add over depth-matched)
        self._hs = {}
        def mk(i):
            def hook(_m, _in, out):
                self._hs[i] = out[0] if isinstance(out, tuple) else out
            return hook
        if blocks is not None:
            for i, blk in enumerate(blocks):
                blk.register_forward_hook(mk(i))
        self._blocks = blocks

    def _align(self, adata):
        """Align adata counts to the 19,264-gene panel (their main_gene_selection)."""
        import numpy as np
        var = np.array([str(v).upper() for v in adata.var_names])
        pos = {g: i for i, g in enumerate(var)}
        take = np.array([pos.get(g, -1) for g in self.genes])
        return take  # index into adata columns for each panel gene (-1 = absent -> 0)

    def iter_activations(self, adata, batch_size=4):
        import torch
        take = self._align(adata)
        self.processed_obs = adata.obs.reset_index(drop=True)
        X = adata.X
        for s in range(0, adata.n_obs, batch_size):
            xb = X[s:s + batch_size]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            # to the 19264 panel
            panel = np.zeros((xb.shape[0], self.ng), dtype=np.float32)
            present = take >= 0
            panel[:, present] = xb[:, take[present]]
            # read-depth tokens: log10 total counts, appended twice (T = S at inference)
            tot = np.log10(panel.sum(1, keepdims=True) + 1.0)
            full = np.concatenate([panel, tot, tot], axis=1)                       # [b, ng+2]
            value = torch.tensor(full, dtype=torch.float32, device=self.device)
            data_mask = value > 0                                                  # non-zero + T,S kept
            # sparse-pack non-zero positions (their gatherData); position ids track gene index
            posids = torch.arange(full.shape[1], device=self.device).unsqueeze(0).repeat(full.shape[0], 1)
            x, x_pad = self._gather(value, data_mask, self.pad_id)                 # VERIFY signature/return
            pos_g, _ = self._gather(posids, data_mask, self.pad_id)
            self._hs.clear()
            with torch.no_grad():
                xe = self.model.token_emb(x.unsqueeze(2).float(), output_weight=0) # VERIFY token_emb API
                xe = xe + self.model.pos_emb(pos_g)                                # VERIFY pos_emb
                _ = self.model.encoder(xe, x_pad)                                  # fills hooks
            # per position: recover which are GENE tokens (pos_g < ng) and their symbol
            pg = pos_g.cpu().numpy(); keepmask = (pg < self.ng) & (~x_pad.cpu().numpy())
            acts = {}
            for L in self.layers:
                h = self._hs.get(L)
                if h is None:
                    continue
                acts[L] = h.float().cpu().numpy()[keepmask]
            syms = self.genes[pg[keepmask]]
            # cell id per kept position
            rows = np.repeat(np.arange(xb.shape[0]), pg.shape[1]).reshape(pg.shape)[keepmask] + s
            yield acts, syms.astype(str), rows.astype(int)
