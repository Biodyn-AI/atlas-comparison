"""scBERT adapter (Yao et al. 2022). Performer attention, binned expression, gene tokens
in a FIXED reference panel (~16,906 genes). Position i = gene i of the reference panel;
value = the cell's expression of that gene, binned 0..5 (capped). Open weights (manual DL).

REQUIRES on the cluster:
  * scBERT repo on sys.path (for `performer_pytorch`) -> scbert_repo=<path>
  * pretrain checkpoint (their Google Drive: `panglao_pretrain.pth`) -> ckpt=<path>
  * the reference gene panel = the ordered gene list the model was trained on
    (their preprocessing reference; a text/npy of `gene_num` HGNC symbols) -> genes_file=<path>

VERIFY on H100:
  * gene_num (default 16906) + bin_num (5) must match the checkpoint; SEQ_LEN = gene_num+1.
  * hidden capture: this hooks the Performer stack's OUTPUT (top-layer residual). For
    intermediate layers, hook model.performer.net sublayers (SequentialSequence).
  * load_state_dict may need strict=False if the ckpt has a finetune head; we keep the LM body.
"""
from __future__ import annotations
import sys
import numpy as np
from .base import Adapter


class ScBERTAdapter(Adapter):
    name = "scBERT"; d_model = 200

    def __init__(self, scbert_repo="external/scBERT", ckpt="external/scBERT/panglao_pretrain.pth",
                 genes_file="external/scBERT/reference_genes.txt", gene_num=16906, bin_num=5,
                 depth=6, layers=(6,)):
        self.scbert_repo = scbert_repo; self.ckpt = ckpt; self.genes_file = genes_file
        self.gene_num = gene_num; self.bin_num = bin_num; self.depth = depth; self.layers = tuple(layers)

    def load(self, device="cuda"):
        import torch
        sys.path.insert(0, self.scbert_repo)
        from performer_pytorch import PerformerLM
        self.torch = torch; self.device = device
        CLASS = self.bin_num + 2
        self.model = PerformerLM(num_tokens=CLASS, dim=self.d_model, depth=self.depth, heads=10,
                                 max_seq_len=self.gene_num + 1, g2v_position_emb=True)
        ckpt = torch.load(self.ckpt, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.eval().to(device)
        self.ref_genes = np.array([l.strip().upper() for l in open(self.genes_file) if l.strip()])
        assert len(self.ref_genes) == self.gene_num, f"{len(self.ref_genes)} != gene_num {self.gene_num}"

    def iter_activations(self, adata, batch_size=8):
        import torch
        # align adata expression to the reference gene panel (missing genes -> 0)
        var = {str(v).upper(): i for i, v in enumerate(adata.var_names)}
        cols = np.array([var.get(g, -1) for g in self.ref_genes])
        present = cols >= 0
        self.processed_obs = adata.obs.reset_index(drop=True)
        cap = {}
        h = self.model.performer.register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        cid = 0
        X = adata.X
        for s in range(0, adata.n_obs, batch_size):
            xb = X[s:s + batch_size]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            seqs = np.zeros((xb.shape[0], self.gene_num + 1), np.int64)
            aligned = np.zeros((xb.shape[0], self.gene_num), np.float32)
            aligned[:, present] = xb[:, cols[present]]
            aligned = np.minimum(np.rint(aligned), self.bin_num).astype(np.int64)
            seqs[:, :self.gene_num] = aligned                      # last col = 0 (CLS-like)
            inp = torch.from_numpy(seqs).to(self.device)
            cap.clear()
            with torch.no_grad():
                self.model(inp)                                    # fills the performer hook
            H = cap["h"][:, :self.gene_num, :].float().cpu().numpy()   # [b, gene_num, 200]
            b = H.shape[0]
            acts = {L: H.reshape(-1, self.d_model) for L in self.layers}
            syms = np.tile(self.ref_genes, b)
            cids = np.repeat(np.arange(cid, cid + b), self.gene_num); cid += b
            yield acts, syms, cids
        h.remove()
