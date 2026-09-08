"""scPRINT-12M adapter. VERIFIED forward path (used in our audit). Env needs `scprint`
+ scdataloader. Broad gene panel: top-N HVG UNION forced TRRUST genes.
"""
from __future__ import annotations
import numpy as np
from .base import Adapter


class ScPrintAdapter(Adapter):
    name = "scPRINT"
    d_model = 256

    def __init__(self, ckpt="ckpt_scprint/medium-v1.5.ckpt", biomart="external/scprint_data/biomart_pos.parquet",
                 trrust="external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                 n_genes=2000, layers=(0, 2, 4, 6, 8)):
        self.ckpt, self.biomart, self.trrust, self.n_genes = ckpt, biomart, trrust, n_genes
        self.layers = tuple(layers)

    def load(self, device="cuda"):
        import torch, pandas as pd
        from scprint import scPrint
        self.torch = torch; self.device = device
        self.model = scPrint.load_from_checkpoint(self.ckpt, precpt_gene_emb=None,
                                                  transformer="normal").eval().to(device)
        bm = pd.read_parquet(self.biomart)
        self.ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
        self.sym2ens = {}
        for e, s in self.ens2sym.items():
            self.sym2ens.setdefault(s, e)

    def iter_activations(self, adata, batch_size=16):
        import torch, scanpy as sc, pandas as pd
        from scdataloader import Preprocessor, SimpleAnnDataset, Collator
        from torch.utils.data import DataLoader
        m = self.model
        adata = adata.copy(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
        adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                             min_nnz_genes=100, filter_gene_by_counts=False)(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_genes, flavor="seurat_v3")
        self.processed_obs = adata.obs.reset_index(drop=True)
        tr = pd.read_csv(self.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
        forced = {self.sym2ens[s] for s in (set(tr.tf.str.upper()) | set(tr.tg.str.upper()))
                  if s in self.sym2ens}
        hv = set(adata.var.index[adata.var.highly_variable])
        panel = [g for g in (hv | forced) if g in set(m.genes)]
        ds = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
        col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=panel, max_len=0)
        dl = DataLoader(ds, collate_fn=col, batch_size=batch_size, shuffle=False)
        cid = 0
        for batch in dl:
            gp, expr = batch["genes"].to(self.device), batch["x"].to(self.device)
            ng = gp.shape[1]
            caps = {}
            hooks = [m.transformer.blocks[0].register_forward_pre_hook(
                lambda mod, i: caps.__setitem__(0, (i[0] if isinstance(i, tuple) else i)[:, -ng:, :].detach()))]
            for L in self.layers:
                if L == 0:
                    continue
                hooks.append(m.transformer.blocks[L - 1].register_forward_hook(
                    lambda mod, i, o, L=L: caps.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, -ng:, :].detach())))
            with torch.no_grad():
                m(gene_pos=gp, expression=expr, req_depth=batch["depth"].to(self.device),
                  depth_mult=expr.sum(1))
            for h in hooks:
                h.remove()
            b = gp.shape[0]
            acts = {L: caps[L].reshape(-1, self.d_model).float().cpu().numpy() for L in self.layers}
            ens = np.array(m.genes)[gp[0].cpu().numpy()]
            syms = np.tile(np.array([self.ens2sym.get(str(e), str(e)) for e in ens]), b)
            cell_ids = np.repeat(np.arange(cid, cid + b), ng); cid += b
            yield acts, syms, cell_ids
