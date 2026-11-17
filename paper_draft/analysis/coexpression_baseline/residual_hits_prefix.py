"""Which TRRUST hits of the co-firing graphs are NOT highly co-expressed (Pearson r below the
top-5 % threshold of all U pairs), and how many of those share a 3-letter symbol prefix (the
same crude family rule as notes/08 prefix_check.py)? Reads coexpr_trrust.json only."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "coexpr_trrust.json")))["kinds"]["pearson"]
t5 = d["thresholds"]["top5pct_r"]
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))
TF = {str(k).upper() for k in json.load(open(f"{BASE}/outputs/atlas/genesets/trrust_edges.json"))}
out = {"top5pct_r": t5, "n_trrust_tfs": len(TF), "models": {}}
for m, o in d["overlap"].items():
    low = [h for h in o["hit_pairs_r"] if h[2] < t5]
    pre = [h for h in low if h[0][:3] == h[1][:3]]
    out["models"][m] = {"hits_total": len(o["hit_pairs_r"]), "hits_not_top5": len(low),
                        "hits_not_top5_same_prefix": len(pre),
                        "hits_not_top5_other": len(low) - len(pre),
                        "hits_not_top5_tf_tf": sum(a in TF and b in TF for a, b, _ in low),
                        "hits_top5_tf_tf": sum(a in TF and b in TF for a, b, r in o["hit_pairs_r"] if r >= t5),
                        "hits_top5": len(o["hit_pairs_r"]) - len(low),
                        "not_top5_hits": low}
    print(m, {k: v for k, v in out["models"][m].items() if k != "not_top5_hits"})
import pickle
PRED, _ = pickle.load(open(f"{BASE}/outputs/atlas/comparative/.cofire_cache.pkl", "rb"))
U = {g for mm in PRED for e in PRED[mm] for g in e}
TR = set()
for a, bs in json.load(open(f"{BASE}/outputs/atlas/genesets/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v and u in U and v in U: TR.add((u, v))
out["trrust_edges_in_cofire_union"] = len(TR)
out["trrust_share_tf_tf"] = round(sum(a in TF and b in TF for a, b in TR) / len(TR), 4)
print("TRRUST edges in union", len(TR), "share TF-TF", out["trrust_share_tf_tf"])
json.dump(out, open(os.path.join(H, "residual_hits_prefix.json"), "w"), indent=1)
