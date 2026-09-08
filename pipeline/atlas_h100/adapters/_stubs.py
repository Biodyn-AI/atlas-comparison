"""Scaffolded adapters for models we cannot run locally (fill + VERIFY on H100).
Each carries the exact loading / forward-hook / tokenisation recipe. Implement
`iter_activations` to yield ({layer: [n_pos, d_model]}, hgnc_symbols[n_pos]), taking
only gene-token positions. Then remove the NotImplementedError.
"""
from __future__ import annotations
import numpy as np
from .base import Adapter

_TODO = "IMPLEMENT + VERIFY on H100 (see docstring recipe), then delete this raise."


class ScBERTAdapter(Adapter):
    """scBERT (Yao et al. 2022). Performer attention, binned expression, gene tokens.
    Load: repo github.com/TencentAILabHealthcare/scBERT — PerformerLM checkpoint;
      gene2vec vocab ~16,906 genes (symbols). Forward: hidden states per Performer layer
      (register hooks on each `PerformerLM.performer.net.layers[i]`). d_model ~200, 6 layers.
    Tokenise: bin expression to 5+ bins per the repo's preprocessing; sequence = genes.
    Symbols: vocab is already gene symbols.
    """
    name = "scBERT"; d_model = 200; layers = (0, 1, 2, 3, 4, 5, 6)
    def load(self, device="cuda"): raise NotImplementedError(_TODO)
    def iter_activations(self, adata, batch_size=8): raise NotImplementedError(_TODO)


class TGPTAdapter(Adapter):
    """tGPT (Shen et al. 2023). AUTOREGRESSIVE GPT-2 over rank-ordered gene sequences.
    Load: HF `lixiangchun/transcriptome-gpt-*` (GPT2LMHeadModel + tokenizer); genes ranked
      by expression descending -> token ids. Forward: output_hidden_states=True (13 layers,
      d_model 768). Symbols: map tokenizer gene tokens -> HGNC.
    Note: AR objective is the key axis vs MLM models — expect different feature depth grammar.
    """
    name = "tGPT"; d_model = 768; layers = (0, 3, 6, 9, 12)
    def load(self, device="cuda"): raise NotImplementedError(_TODO)
    def iter_activations(self, adata, batch_size=8): raise NotImplementedError(_TODO)


class CellPLMAdapter(Adapter):
    """CellPLM (Wen et al. 2024). CELL-TOKEN (not gene-token): cells are tokens, genes are
    features. Load: github.com/OmicsML/CellPLM pretrained ckpt. Residual is per-CELL, so
    'positions' = cells, and there is no per-gene symbol — annotate via each cell's top
    expressed/attributed genes instead (adapt catalog: gene attribution per feature).
    We already have a wrapper (mechaudit/models/cellplm_wrapper.py) to start from.
    CAVEAT: cell-token breaks the per-gene top-20 annotation — needs a gene-attribution step.
    """
    name = "CellPLM"; d_model = 512; layers = (0, 1, 2, 3, 4)
    def load(self, device="cuda"): raise NotImplementedError(_TODO)
    def iter_activations(self, adata, batch_size=8): raise NotImplementedError(_TODO)


class UCEAdapter(Adapter):
    """UCE — Universal Cell Embedding (Rosen et al. 2023/2026). Uses ESM2 protein
    embeddings for gene tokens -> REPLICATES our ESM-prior finding independently.
    Load: github.com/snap-stanford/UCE (33-layer, d_model 1280, 650M) + its ESM2 token
    embeddings. Forward: hidden states per transformer block (hooks). Sequence = genes
    (protein-embedding tokens) + a CLS cell token; take gene positions only.
    Symbols: UCE gene vocab -> HGNC (its ESM2 gene mapping table).
    HIGH VALUE: if UCE shows the same complexes/PPI/localization geometry as scPRINT,
    the ESM-prior effect reproduces across two independent architectures.
    """
    name = "UCE"; d_model = 1280; layers = (0, 8, 16, 24, 32)
    def load(self, device="cuda"): raise NotImplementedError(_TODO)
    def iter_activations(self, adata, batch_size=4): raise NotImplementedError(_TODO)


class GeneCompassAdapter(Adapter):
    """GeneCompass (Yang et al. 2024). Knowledge-graph-informed, cross-species (human+mouse).
    Load: github.com/xCompass-AI/GeneCompass checkpoint; injects promoter/GRN/co-expression/
    gene-family priors. Forward: hidden states per layer (hooks), gene-token model.
    HIGH VALUE for the headline test: does a KNOWLEDGE prior break the 6.2% TF-logic null?
    Symbols: GeneCompass vocab -> HGNC (human side).
    """
    name = "GeneCompass"; d_model = 768; layers = (0, 3, 6, 9, 12)
    def load(self, device="cuda"): raise NotImplementedError(_TODO)
    def iter_activations(self, adata, batch_size=4): raise NotImplementedError(_TODO)
