"""GeneCompass adapter (Yang et al. 2024, Cell Research). Knowledge-graph-informed, cross-species BERT
over rank-ordered gene tokens + expression VALUES; injects 4 prior-knowledge embeddings (promoter / co-exp /
gene-family / PECA-GRN) via KnowledgeBertEmbeddings passed to the model CONSTRUCTOR. The knowledge/GRN-prior
axis. We run it on the HUMAN TS corpus and read only human-gene features.

WIRING (verified against genecompass/{modeling_bert,utils}.py + pretraining/pretrain_genecompass_w_human_mouse_base.py):
  * knowledges = load_prior_embedding(token_dictionary_or_path=token_dict) -> tuple(promoter, co_exp,
    gene_family, peca_grn, homologous); model = BertForMaskedLM(config, knowledges=dict); load_state_dict.
    (prior defaults resolve to external/GeneCompass/prior_knowledge/* — all subfolders present.)
  * BertModel.forward accepts `values=` (continuous expression) — passed alongside input_ids.
  * token dict is Ensembl-keyed (<pad>=0,<mask>=1,ENSG…); corpus is HGNC symbols -> map sym->Ensembl->token.
REQUIRES on cluster: repo at gc_repo (genecompass pkg + prior_knowledge/), weights dir ckpt_dir
(config.json + pytorch_model.bin), token_dict = prior_knowledge/h&m_token1000W.pickle (load_prior default).
"""
from __future__ import annotations
import os, pickle, sys
import numpy as np
from .base import Adapter


class GeneCompassAdapter(Adapter):
    name = "GeneCompass"; d_model = 768

    def __init__(self, gc_repo="external/GeneCompass",
                 ckpt_dir="external/GeneCompass/pretrained_models/GeneCompass_Base",
                 token_dict="external/GeneCompass/prior_knowledge/human_mouse_tokens.pickle",  # 50558 = checkpoint vocab
                 biomart="external/scprint_data/biomart_pos.parquet",
                 max_len=2048, layers=None):
        self.gc_repo = gc_repo; self.ckpt_dir = ckpt_dir; self.token_dict = token_dict
        self.biomart = biomart; self.max_len = max_len
        self._layers_override = tuple(layers) if layers else None
        self.layers = self._layers_override or (0, 3, 6, 9, 12)

    def load(self, device="cuda"):
        import torch, pandas as pd
        from transformers import BertConfig
        sys.path.insert(0, self.gc_repo)
        from genecompass.modeling_bert import BertForMaskedLM   # submodule (skips __init__'s Trainer-heavy pretrainer)
        from genecompass.utils import load_prior_embedding
        self.torch = torch; self.device = device
        out = load_prior_embedding(token_dictionary_or_path=self.token_dict)   # tuple of 5
        knowledges = {"promoter": out[0], "co_exp": out[1], "gene_family": out[2],
                      "peca_grn": out[3], "homologous_gene_human2mouse": out[4]}
        config = BertConfig.from_pretrained(self.ckpt_dir)
        self.model = BertForMaskedLM(config, knowledges=knowledges)
        sd = torch.load(os.path.join(self.ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        miss, unexp = self.model.load_state_dict(sd, strict=False)
        print(f"[GeneCompass] weights loaded (missing={len(miss)} unexpected={len(unexp)}); "
              f"knowledge prior wired=True (via constructor)", flush=True)
        self.model = self.model.eval().to(device)
        self.n_layers = int(getattr(config, "num_hidden_layers", 12))
        if not self._layers_override:
            from common.layers import depth_matched
            self.layers = depth_matched(self.n_layers)
        # gene-id lookups: corpus is HGNC symbols, GeneCompass keys are Ensembl (human = ENSG…)
        self.td = pickle.load(open(self.token_dict, "rb"))
        bm = pd.read_parquet(self.biomart)
        ens2sym = {str(e).split(".")[0]: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
        self.ens2tok = {str(k).split(".")[0]: int(v) for k, v in self.td.items() if str(k).startswith("ENS")}
        self.sym2tok = {}
        for k, v in self.td.items():
            if str(k).startswith("ENSG"):                       # human only
                sym = ens2sym.get(str(k).split(".")[0])
                if sym and sym not in self.sym2tok:
                    self.sym2tok[sym] = int(v)
        self.id2sym = {int(v): ens2sym.get(str(k).split(".")[0], str(k)).upper() for k, v in self.td.items()}
        print(f"[GeneCompass] {self.n_layers} layers; {len(self.sym2tok)} human symbols mapped to tokens", flush=True)

    def iter_activations(self, adata, batch_size=8):
        import torch
        var = np.array([str(v) for v in adata.var_names])
        is_ens = np.mean([v.upper().startswith("ENSG") for v in var]) > 0.5
        col_tok = np.array([(self.ens2tok.get(g.split(".")[0], -1) if is_ens
                             else self.sym2tok.get(g.upper(), -1)) for g in var])
        print(f"[GeneCompass] gene id type={'Ensembl' if is_ens else 'symbol'}, "
              f"{int((col_tok >= 0).sum())}/{len(var)} genes mapped to tokens", flush=True)
        self.processed_obs = adata.obs.reset_index(drop=True)
        X = adata.X; cid = 0
        for s in range(0, adata.n_obs, batch_size):
            xb = X[s:s + batch_size]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            ids_b, val_b, lens = [], [], []
            for row in xb:
                order = np.argsort(row)[::-1]
                order = order[(col_tok[order] >= 0) & (row[order] > 0)][: self.max_len]
                ids_b.append(col_tok[order]); val_b.append(row[order]); lens.append(len(order))
            L = max(lens) if lens else 0
            if L == 0:
                cid += len(ids_b); continue
            inp = torch.zeros((len(ids_b), L), dtype=torch.long, device=self.device)
            val = torch.zeros((len(ids_b), L), dtype=torch.float, device=self.device)
            am = torch.zeros((len(ids_b), L), dtype=torch.long, device=self.device)
            for i, (ids, v, n) in enumerate(zip(ids_b, val_b, lens)):
                inp[i, :n] = torch.as_tensor(ids, device=self.device)
                val[i, :n] = torch.as_tensor(v, dtype=torch.float, device=self.device); am[i, :n] = 1
            with torch.no_grad():
                out = self.model(input_ids=inp, attention_mask=am, values=val, output_hidden_states=True)
            hs = out.hidden_states
            acts = {Ly: [] for Ly in self.layers}
            syms, cids = [], []
            for i, n in enumerate(lens):
                g = np.array([self.id2sym.get(int(t), str(int(t))) for t in ids_b[i]])
                syms.append(g); cids.append(np.full(n, cid + i))
                for Ly in self.layers:
                    acts[Ly].append(hs[Ly][i, :n, :].float().cpu().numpy())
            acts = {Ly: np.concatenate(v, 0) for Ly, v in acts.items()}
            cid += len(ids_b)
            yield acts, np.concatenate(syms), np.concatenate(cids)
