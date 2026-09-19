"""MaxToki adapter (theodoris-lab; MaxToki-217M / 1B). LlamaForCausalLM over rank-value-encoded
cells — same theodoris-lab rank-value Ensembl tokenisation as Geneformer (vocab 20,275 matches
Geneformer-V2), but an autoregressive Llama backbone (d=1232, 11 layers, RoPE). So it reuses the
Geneformer adapter's tokenisation and per-gene hook wholesale; only the HF auto-class differs.

Weights: HF theodoris-lab/MaxToki -> MaxToki-217M-HF/ (Apache-2.0, no gate). Pass ckpt to that dir.
VERIFY on H100: the token_dictionary + gene_median source. MaxToki-HF likely ships them or reuses
Geneformer-V2's (shared 20,275 vocab) — point token_dict_dir at whichever has the pickles. If MaxToki
ships an HF tokenizer.json instead of the pickle dicts, adapt _find/load accordingly.
"""
from __future__ import annotations
from .geneformer import GeneformerAdapter


class MaxTokiAdapter(GeneformerAdapter):
    name = "MaxToki"
    hf_auto = "causal"        # LlamaForCausalLM

    def __init__(self, ckpt="ckpt_maxtoki/MaxToki-217M-HF", layers=None, max_len=2048,
                 token_dict_dir=None):
        # token dicts default to the Geneformer-V2 checkpoint (shared vocab) unless overridden
        super().__init__(ckpt=ckpt, layers=layers, max_len=max_len,
                         token_dict_dir=token_dict_dir or "ckpt_geneformer_v2/geneformer")
