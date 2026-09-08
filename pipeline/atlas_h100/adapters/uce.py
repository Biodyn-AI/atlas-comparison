"""UCE adapter (Rosen et al. 2023/2026) — 33-layer / 650M, d_model 1280, token_dim 5120.
Gene tokens ARE ESM2 protein embeddings -> the independent replication of our ESM-prior
finding. This WRAPS UCE's own model + tokenisation (do NOT reimplement blind): loads their
TransformerModel, reuses their MultiDatasetSentences to build per-cell token sentences, hooks
`model.transformer_encoder.layers[L]` for per-layer residual, and maps token index -> gene via
the inverted protein-embedding order (special/chrom/CLS tokens are skipped, as they aren't in
the gene map).

REQUIRES on the cluster (clone + download; open weights, not gated, but large):
  * git clone https://github.com/snap-stanford/UCE  -> pass uce_repo=<path>
  * UCE model_files (their download script): all_tokens.torch (token_file), protein_embeddings/,
    and the 33-layer checkpoint (model_loc). token_dim 5120, nlayers 33, d_model 1280.

VERIFY on H100 (this is a faithful reconstruction of evaluate.run_eval; expect 1-2 fixes):
  * exact import paths: `from model import TransformerModel`, `from evaluate import AnndataProcessor`,
    `from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator`, and the helper that
    returns species->{gene:row} (grep evaluate/utils for `get_species_to_pe`).
  * the args Namespace fields (defaults from eval_single_anndata.py argparse) — fill any I missed.
  * chrom/CLS/pad token indices default 1/2 (chrom L/R), 3 (CLS), 0 (pad); gene tokens map via idx2gene.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
from .base import Adapter


class UCEAdapter(Adapter):
    name = "UCE"; d_model = 1280

    def __init__(self, uce_repo="external/UCE",
                 model_loc="external/UCE/model_files/33l_8ep_1024t_1280.torch",   # 650M (33-layer); VERIFY exact name in lza1/uce_hf
                 token_file="external/UCE/model_files/all_tokens.torch",
                 protein_embeddings_dir="external/UCE/model_files/protein_embeddings/",
                 nlayers=33, layers=None, species="human", batch_size=16):
        self.uce_repo = uce_repo; self.model_loc = model_loc; self.token_file = token_file
        self.protein_embeddings_dir = protein_embeddings_dir; self.nlayers = nlayers
        from common.layers import depth_matched
        self.layers = tuple(layers) if layers else depth_matched(nlayers)   # 33 -> (0,8,16,24,32)
        self.species = species; self.batch_size = batch_size

    def _args(self, work_dir):
        mf = "external/UCE/model_files"
        return argparse.Namespace(
            dir=work_dir + "/", species=self.species, filter=True, skip=True,
            spec_chrom_csv_path=f"{mf}/species_chrom.csv", offset_pkl_path=f"{mf}/species_offsets.pkl",
            token_file=self.token_file, protein_embeddings_dir=self.protein_embeddings_dir,
            model_loc=self.model_loc, batch_size=self.batch_size, pad_length=1536,
            pad_token_idx=0, chrom_token_left_idx=1, chrom_token_right_idx=2, cls_token_idx=3, sample_size=1024,
            CHROM_TOKEN_OFFSET=143574, token_dim=5120, multi_gpu=False,
            emsize=1280, d_hid=5120, nlayers=self.nlayers, nhead=20, dropout=0.05, output_dim=1280)

    def load(self, device="cuda"):
        import torch, torch.nn as nn
        sys.path.insert(0, self.uce_repo)
        from model import TransformerModel
        self.torch = torch; self.device = device
        a = self._args(".")
        self.model = TransformerModel(token_dim=a.token_dim, d_model=a.emsize, nhead=a.nhead,
                                      d_hid=a.d_hid, nlayers=a.nlayers, output_dim=a.output_dim,
                                      dropout=a.dropout)
        all_pe = torch.load(self.token_file)                      # [n_tokens, 5120]
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.load_state_dict(torch.load(self.model_loc, map_location="cpu"), strict=True)
        self.model = self.model.eval().to(device)
        # idx2gene for the target species (invert protein-embedding order)
        try:
            from utils import get_species_to_pe
        except ImportError:
            from evaluate import get_species_to_pe
        sp2pe = get_species_to_pe(self.protein_embeddings_dir)   # {species: {gene: ESM2 emb tensor}}
        import pickle
        offsets = pickle.load(open(os.path.join(os.path.dirname(self.token_file), "species_offsets.pkl"), "rb"))
        genes = list(sp2pe[self.species].keys())                 # concatenation order within the species block
        off = int(offsets[self.species])
        self.idx2gene = {off + i: str(g).upper() for i, g in enumerate(genes)}

    def iter_activations(self, adata, batch_size=None):
        import torch, scanpy as sc
        from torch.utils.data import DataLoader
        from evaluate import AnndataProcessor
        from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
        bs = batch_size or self.batch_size
        work = "uce_work"; os.makedirs(work, exist_ok=True)
        name = "atlas"; adata.write_h5ad(f"{work}/{name}.h5ad")
        a = self._args(work); a.adata_path = f"{work}/{name}.h5ad"
        from accelerate import Accelerator
        proc = AnndataProcessor(a, accelerator=Accelerator())
        proc.preprocess_anndata(); proc.generate_idxs()
        self.processed_obs = adata.obs.reset_index(drop=True)
        import pickle
        shapes = pickle.load(open(proc.shapes_dict_path, "rb"))
        ds = MultiDatasetSentences(sorted_dataset_names=[name], shapes_dict=shapes, args=a, npzs_dir=a.dir,
                                   dataset_to_protein_embeddings_path=proc.pe_idx_path,
                                   datasets_to_chroms_path=proc.chroms_path,
                                   datasets_to_starts_path=proc.starts_path)
        dl = DataLoader(ds, batch_size=bs, shuffle=False,
                        collate_fn=MultiDatasetSentenceCollator(a))
        caps = {}
        hooks = [self.model.transformer_encoder.layers[L].register_forward_hook(
            (lambda L: (lambda m, i, o: caps.__setitem__(L, o.detach())))(L)) for L in self.layers]
        cid = 0
        for batch in dl:
            sent, mask = batch[0], batch[1]
            sent = sent.to(self.device); mask = mask.to(self.device)
            src = sent.permute(1, 0)                              # [seq, batch]
            emb = torch.nn.functional.normalize(self.model.pe_embedding(src.long()), dim=2)
            caps.clear()
            with torch.no_grad():
                self.model(emb, mask)                             # fills hooks
            tok = sent.cpu().numpy()                              # [batch, seq] token indices
            b = tok.shape[0]
            # gene positions = tokens present in idx2gene (special/chrom/cls skipped)
            keep = np.array([[t in self.idx2gene for t in row] for row in tok])  # [batch, seq]
            syms, cids = [], []
            for i in range(b):
                gp = np.where(keep[i])[0]
                syms.append(np.array([self.idx2gene[tok[i, p]] for p in gp]))
                cids.append(np.full(len(gp), cid + i))
            syms = np.concatenate(syms); cids = np.concatenate(cids)
            acts = {}
            for L in self.layers:
                h = caps[L].permute(1, 0, 2).float().cpu().numpy()   # [batch, seq, d]
                acts[L] = np.concatenate([h[i, np.where(keep[i])[0], :] for i in range(b)], 0)
            cid += b
            yield acts, syms, cids
        for hk in hooks:
            hk.remove()
