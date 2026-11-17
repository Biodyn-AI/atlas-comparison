"""Values that the original analyses typed in for the superseded run, read from the corrected
results instead: the models passing the per-model TRRUST test (p < 0.05, strongest first), the
equal-budget size, and each model's co-firing graph size."""
import json, os
_B = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
_C = os.path.join(_B, "outputs", "atlas", "comparative")
_T2 = json.load(open(os.path.join(_C, "hypothesis_trrust2.json")))["per_model"]
PASSED = [m for m, v in sorted(_T2.items(), key=lambda kv: -kv[1]["fold"]) if v["p_emp"] < 0.05]
BUDGET = json.load(open(os.path.join(_C, "hypothesis_sizematched.json")))["budget"]
GRAPH_SIZE = {m: v["n_pairs"] for m, v in _T2.items()}
# models with at least one known pair in the corrected run (was the hard-coded SEVEN of the published run)
WITH_HITS = [m for m, v in sorted(_T2.items(), key=lambda kv: -kv[1]["fold"]) if v["hits"] >= 1]
