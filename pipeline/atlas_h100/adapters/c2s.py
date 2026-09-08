"""C2S-Scale adapter (Cell2Sentence-Scale-Gemma-2-2B; Levine/van Dijk lab). A Gemma-2 decoder-
only LM where each cell is a "cell sentence" = space-separated gene SYMBOLS ranked by expression
high->low (top ~512). Gene names are BPE-split by the Gemma tokenizer, so per-gene residual is
read at each gene's LAST subword position (in a causal LM that token has attended to the whole
gene) via offset-mapping. Same idea as Igor's c2s-mechinterp sub-word->gene attribution.

Weights: HF vandijklab/C2S-Scale-Gemma-2-2B (CC-BY-4.0, no gate), transformers AutoModelForCausalLM.
d_model 2304, 26 layers (d_sae 9216 = 4x2304). 2B params -> keep batch small on H100.

VERIFY on H100: (1) the sub-word representative (last vs mean over a gene's subwords) — c2s-mechinterp
has the attribution convention; last is the causal-LM default here. (2) whether C2S prepends a prompt/
instruction to the cell sentence (if so, offset those char spans out). (3) sentencepiece/protobuf installed.
"""
from __future__ import annotations
import numpy as np
from .base import Adapter


class C2SAdapter(Adapter):
    name = "C2S"

    def __init__(self, hf_id="ckpt_c2s", layers=None, max_genes=512):   # local dir (downloaded); or the HF repo id
        self.hf_id = hf_id; self._layers_override = layers; self.max_genes = max_genes

    def load(self, device="cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch; self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, output_hidden_states=True, torch_dtype=torch.bfloat16).eval().to(device)
        self.d_model = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        from common.layers import depth_matched
        self.layers = tuple(self._layers_override) if self._layers_override else depth_matched(self.n_layers)

    def _sentence(self, row, var):
        """Cell -> (sentence str, list of (gene_symbol, char_start, char_end))."""
        nz = np.where(row > 0)[0]
        order = nz[np.argsort(-row[nz])][: self.max_genes]
        genes = [str(var[i]).upper() for i in order]
        spans, pos, parts = [], 0, []
        for g in genes:
            spans.append((g, pos, pos + len(g)))
            parts.append(g); pos += len(g) + 1                      # +1 for the joining space
        return " ".join(parts), spans

    def iter_activations(self, adata, batch_size=4):
        import torch
        self.processed_obs = adata.obs.reset_index(drop=True)
        var = np.array([str(v).upper() for v in adata.var_names])
        cid = 0
        for s in range(0, adata.n_obs, batch_size):
            xb = adata.X[s:min(s + batch_size, adata.n_obs)]
            xb = xb.toarray() if hasattr(xb, "toarray") else np.asarray(xb)
            sents, gene_pos_batch, sym_batch = [], [], []
            for row in xb:
                sent, spans = self._sentence(row, var)
                enc = self.tok(sent, return_offsets_mapping=True, add_special_tokens=True)
                offs = enc["offset_mapping"]
                # for each gene, its last subword token index (token span inside the gene's char span)
                reps, syms = [], []
                ti = 0
                for g, gs, ge in spans:
                    last = -1
                    while ti < len(offs) and offs[ti][0] < ge:
                        a, b = offs[ti]
                        if a >= gs and b <= ge and b > a:
                            last = ti
                        ti += 1
                    if last >= 0:
                        reps.append(last); syms.append(g)
                    ti = max(0, ti - 1)                             # allow shared boundary token
                sents.append(enc["input_ids"]); gene_pos_batch.append(reps); sym_batch.append(syms)
            L = max(len(x) for x in sents)
            pad_id = self.tok.pad_token_id or 0
            inp = torch.full((len(sents), L), pad_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((len(sents), L), dtype=torch.long, device=self.device)
            for i, ids in enumerate(sents):
                inp[i, :len(ids)] = torch.tensor(ids, device=self.device); attn[i, :len(ids)] = 1
            with torch.no_grad():
                out = self.model(input_ids=inp, attention_mask=attn)
            hs = out.hidden_states                                 # tuple len n_layers+1
            b = len(sents)
            acts = {L_: [] for L_ in self.layers}
            syms_all, cids_all = [], []
            for i in range(b):
                pos = np.array(gene_pos_batch[i], dtype=int)
                if pos.size == 0:
                    continue
                for L_ in self.layers:
                    acts[L_].append(hs[L_ + 1][i, pos, :].float().cpu().numpy())
                syms_all.append(np.array(sym_batch[i])); cids_all.append(np.full(len(pos), cid + i))
            cid += b
            if not syms_all:
                continue
            yield ({L_: np.concatenate(acts[L_], 0) for L_ in self.layers},
                   np.concatenate(syms_all), np.concatenate(cids_all))
