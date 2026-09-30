#!/usr/bin/env python
"""Extract the UCE input tokens (ESM2 protein embeddings, human) from the local HF cache
(minwoosun/uce-misc, protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt).
Writes: the gene order of the dict (= UCE's per-species token order) and float16 embeddings.
Output goes to $ATLAS_CACHE/input_encoding (big file); a small JSON summary goes next to this script.
Run with system python3 (needs torch)."""
import json, os, torch, numpy as np
SRC = os.path.expanduser("~/.cache/huggingface/hub/models--minwoosun--uce-misc/snapshots/"
                         "bffb91084e4476698984e7e01f6170ce291f4074/protein_embeddings/"
                         "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
SCR = os.path.join(os.environ.get("ATLAS_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")), "input_encoding")
os.makedirs(SCR, exist_ok=True)
d = torch.load(SRC, map_location="cpu")
keys = list(d.keys())
E = torch.stack([d[k].float() for k in keys]).numpy().astype(np.float16)
np.save(f"{SCR}/uce_esm2_emb_f16.npy", E)
json.dump(keys, open(f"{SCR}/uce_esm2_keys.json", "w"))
summ = {"source": SRC, "n_genes": len(keys), "dim": int(E.shape[1]), "first_keys": keys[:10],
        "keys_sorted_alphabetically": keys == sorted(keys)}
json.dump(summ, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "uce_esm2_extract.json"), "w"), indent=1)
print(summ)
