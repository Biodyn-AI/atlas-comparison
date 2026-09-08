"""Geneformer adapter (Theodoris et al. 2023; V2-316M for the atlas). BertForMaskedLM over
rank-value-encoded cells: genes ranked by (count / corpus gene-median), mapped to Ensembl
token ids, top max_len kept. Per-gene residual = hidden state at each gene-token position,
hooked at bert.encoder.layer[L]. Also the anchor for the theodoris-lab family (MaxToki).

Rank-value note: Geneformer normalises count_i by the gene's corpus median and by total
counts; total-count and the 10k scale are per-cell constants that do NOT change the ranking,
so only count_i / gene_median_i matters for the token ORDER (what we need).

REQUIRES a checkpoint dir with: config.json, token_dictionary*.pkl (gene->id, Ensembl),
gene_median_dictionary*.pkl (Ensembl->median). V2-316M: theodoris-lab / ctheodoris HF.
Corpus var must expose Ensembl ids (var['ensembl_id'] or ENSG var_names) AND gene symbols
(var_names or var['feature_name']) so output positions carry symbols for annotation.

VERIFY on H100: V2 may prepend <cls>/<eos> — handled generically via special tokens in the
token dict (only ids present in idx2ens are emitted as gene positions).
"""
from __future__ import annotations
import glob, json, os, pickle
import numpy as np
from .base import Adapter


class GeneformerAdapter(Adapter):
    name = "Geneformer"
    hf_auto = "masked"        # "masked" -> AutoModelForMaskedLM (BERT); "causal" -> AutoModelForCausalLM (Llama, MaxToki)

    def __init__(self, ckpt="ckpt_geneformer", layers=None, max_len=2048,
                 token_dict_dir=None):
        self.ckpt = ckpt
        self._layers_override = layers            # None -> set from depth after load
        self.max_len = max_len
        self.token_dict_dir = token_dict_dir or ckpt   # where token/median pickles live (may differ from weights)

    def _find(self, pattern):
        hits = glob.glob(os.path.join(self.token_dict_dir, pattern))
        if not hits:
            raise FileNotFoundError(f"{pattern} not in {self.token_dict_dir}")
        return hits[0]

    def load(self, device="cuda"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
        self.torch = torch; self.device = device
        cfg = json.load(open(os.path.join(self.ckpt, "config.json")))
        self.d_model = int(cfg["hidden_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        self.token_dict = pickle.load(open(self._find("token_dictionary*.pkl"), "rb"))   # ens/special -> id
        self.median = pickle.load(open(self._find("gene_median_dictionary*.pkl"), "rb"))  # ens -> median
        self.idx2ens = {i: g for g, i in self.token_dict.items() if str(g).startswith("ENSG")}
        loader = AutoModelForCausalLM if self.hf_auto == "causal" else AutoModelForMaskedLM
        model = loader.from_pretrained(self.ckpt, output_hidden_states=True)
        self.model = model.eval().to(device)
        if self._layers_override is not None:
            self.layers = tuple(self._layers_override)
        else:
            from common.layers import depth_matched
            self.layers = depth_matched(self.n_layers)

    def _cell_var_maps(self, adata):
        """Return per-var-gene arrays: ensembl id, symbol, token id (or -1), median (or nan)."""
        vn = np.array([str(x) for x in adata.var_names])
        ens = None
        if "ensembl_id" in adata.var.columns:
            ens = np.array([str(x).split(".")[0] for x in adata.var["ensembl_id"]])
        elif vn[0].startswith("ENSG"):
            ens = np.array([x.split(".")[0] for x in vn])
        else:
            raise ValueError("corpus var needs Ensembl ids (var['ensembl_id'] or ENSG var_names)")
        if "feature_name" in adata.var.columns:
            sym = np.array([str(x).upper() for x in adata.var["feature_name"]])
        else:
            sym = np.array([x.upper() for x in vn])
        tok = np.array([self.token_dict.get(e, -1) for e in ens])
        med = np.array([self.median.get(e, np.nan) for e in ens], dtype=np.float64)
        return ens, sym, tok, med

    def iter_activations(self, adata, batch_size=8):
        import torch
        self.processed_obs = adata.obs.reset_index(drop=True)
        ens, sym, tok, med = self._cell_var_maps(adata)
        usable = (tok >= 0) & np.isfinite(med) & (med > 0)          # genes Geneformer knows
        pad_id = self.token_dict.get("<pad>", 0)
        cid = 0
        for s in range(0, adata.n_obs, batch_size):
            xb = adata.X[s:min(s + batch_size, adata.n_obs)]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            ids_batch, sym_batch = [], []
            for row in xb:
                v = np.where((row > 0) & usable)[0]
                if v.size == 0:
                    ids_batch.append(np.array([pad_id])); sym_batch.append(np.array([""])); continue
                rankval = row[v] / med[v]                            # count / gene-median -> rank key
                order = v[np.argsort(-rankval)][: self.max_len]
                ids_batch.append(tok[order]); sym_batch.append(sym[order])
            L = max(len(x) for x in ids_batch)
            inp = np.full((len(ids_batch), L), pad_id, dtype=np.int64)
            for i, ids in enumerate(ids_batch):
                inp[i, :len(ids)] = ids
            input_ids = torch.as_tensor(inp, device=self.device)
            attn = (input_ids != pad_id).long()
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn)
            hs = out.hidden_states                                  # tuple len n_layers+1, [B, L, d]
            b = len(ids_batch)
            acts = {}
            for Ly in self.layers:
                h = hs[Ly + 1].float().cpu().numpy()                # +1: hidden_states[0] is embeddings
                acts[Ly] = np.concatenate([h[i, :len(sym_batch[i]), :] for i in range(b)], 0)
            syms = np.concatenate(sym_batch)
            cids = np.concatenate([np.full(len(sym_batch[i]), cid + i) for i in range(b)])
            keep = syms != ""
            cid += b
            yield {L: acts[L][keep] for L in self.layers}, syms[keep], cids[keep]
