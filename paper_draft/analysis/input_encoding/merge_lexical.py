#!/usr/bin/env python
"""Merge the two lexical.py batch outputs (run as two ~10-minute jobs) into lexical.json."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(f"{D}/lexical_AIDO_C2S_Geneformer_MaxToki_UCE.json"))
B = json.load(open(f"{D}/lexical_scGPT_tGPT_scFoundation_GeneCompass_Tahoe.json"))
assert A["design"] == B["design"]
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
out = {"design": A["design"], "batches": ["lexical.py 50 200 50 AIDO,C2S,Geneformer,MaxToki,UCE",
                                          "lexical.py 50 200 50 scGPT,tGPT,scFoundation,GeneCompass,Tahoe"],
       "models": {m: (A["models"].get(m) or B["models"][m]) for m in ORDER}}
json.dump(out, open(f"{D}/lexical.json", "w"), indent=1)
print("merged", list(out["models"]))
