"""tGPT adapter (Shen et al. 2023). Autoregressive GPT-2 over rank-ordered genes —
the AR-objective axis vs the MLM models. HF: lixiangchun/transcriptome-gpt-1024-8-16-64.
Tokenisation (per the paper "generative pretraining on rankings of top-expressing genes"):
each cell -> its genes sorted by expression DESCENDING -> the top ~max_len as a token
sequence. Open weights (HF), no gating.

VERIFY on H100 (1-2 likely tweaks):
  * the tokenizer's gene-id convention: run `tok.get_vocab()` and check whether tokens are
    HGNC SYMBOLS or Ensembl IDs — set GENE_ID accordingly (default: symbol).
  * whether tGPT expects a space-joined string (tok(text)) or direct input_ids — this uses
    direct symbol->id lookup; if the vocab is BPE over a joined string, switch to
    tok(" ".join(ranked_genes), ...) and drop non-gene subword positions.
"""
from __future__ import annotations
import numpy as np
from .base import Adapter


class TGPTAdapter(Adapter):
    name = "tGPT"

    def __init__(self, hf_id="lixiangchun/transcriptome-gpt-1024-8-16-64", max_len=64,
                 layers=None, gene_id="symbol"):
        self.hf_id = hf_id; self.max_len = max_len; self._layers_override = layers; self.gene_id = gene_id

    def load(self, device="cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch; self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, output_hidden_states=True, use_safetensors=True).eval().to(device)  # torch<2.6 CVE: force safetensors
        self.d_model = self.model.config.n_embd
        self.n_layers = self.model.config.n_layer
        from common.layers import depth_matched
        self.layers = tuple(self._layers_override) if self._layers_override else depth_matched(self.n_layers)
        self.vocab = {k.upper(): v for k, v in self.tok.get_vocab().items()}   # gene-token -> id

    def iter_activations(self, adata, batch_size=16):
        import torch
        X = adata.X
        var = np.array([str(v).upper() for v in adata.var_names])
        # keep only genes present in the tokenizer vocab (positions we can attribute)
        in_vocab = np.array([g in self.vocab for g in var])
        self.processed_obs = adata.obs.reset_index(drop=True)
        cid = 0
        for s in range(0, adata.n_obs, batch_size):
            xb = X[s:s + batch_size]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            ids_batch, syms_batch, lens = [], [], []
            for row in xb:
                order = np.argsort(row)[::-1]                       # highest expression first
                order = order[in_vocab[order]]
                order = order[row[order] > 0][: self.max_len]
                genes = var[order]
                ids = [self.vocab[g] for g in genes]
                ids_batch.append(ids); syms_batch.append(genes); lens.append(len(ids))
            L = max(lens)
            inp = torch.full((len(ids_batch), L), self.tok.pad_token_id or 0, dtype=torch.long, device=self.device)
            for i, ids in enumerate(ids_batch):
                inp[i, :len(ids)] = torch.tensor(ids, device=self.device)
            with torch.no_grad():
                out = self.model(input_ids=inp)
            hs = out.hidden_states                                  # tuple len n_layer+1, [B, L, d]
            acts = {}
            syms_all, cids_all = [], []
            for Ly in self.layers:
                rows = []
                for i, n in enumerate(lens):
                    rows.append(hs[Ly][i, :n, :].float().cpu().numpy())
                acts[Ly] = np.concatenate(rows, 0)
            for i, (g, n) in enumerate(zip(syms_batch, lens)):
                syms_all.append(np.asarray(g)); cids_all.append(np.full(n, cid + i))
            cid += len(ids_batch)
            yield acts, np.concatenate(syms_all), np.concatenate(cids_all)
