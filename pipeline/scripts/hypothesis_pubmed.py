#!/usr/bin/env python
"""Literature co-mention for the predicted gene pairs, from NCBI gene2pubmed (bulk, versioned,
no API calls -- so the analysis is reproducible from a dated file rather than from live queries).

Two questions, one file:

(1) VALIDATION. Are the predicted pairs co-mentioned in the literature more often than chance?
    This is independent of TRRUST. Null: each gene keeps its own publication count exactly, and
    the papers are reassigned at random -- so a pair cannot score merely because both genes are
    heavily studied. Implemented as a degree-preserving shuffle of the gene->paper bipartite graph
    (stub matching on the paper side), which fixes both gene publication counts and paper sizes.

(2) NOVELTY. Among pairs the models predict confidently, which have NEVER been co-mentioned?
    Those are the candidates for genuinely unexplored biology. This is only meaningful because
    (1) and the TRRUST test establish that the predictor is calibrated: an uncorroborated pair
    from a validated predictor is a lead, whereas the same list from an unvalidated one is noise.

    python scripts/hypothesis_pubmed.py [NPERM]  -> outputs/atlas/comparative/hypothesis_pubmed.json
"""
from __future__ import annotations
import gzip, json, sys
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; P = f"{BASE}/outputs/atlas/pubmed"
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TAXID = "9606"
rng = np.random.default_rng(0)

pairs = json.load(open(f"{C}/hypothesis_pairs_full.json"))["pairs"]
WANT = {g for c in pairs for g in c["pair"]}
print(f"{len(pairs)} predicted pairs over {len(WANT)} genes", flush=True)

# ---- symbol -> GeneID (human, current symbols + synonyms) -------------------
sym2id, id2sym = {}, {}
with gzip.open(f"{P}/Homo_sapiens.gene_info.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[0] != TAXID: continue
        gid, sym, syns = f[1], f[2].upper(), f[4]
        id2sym[gid] = sym
        sym2id.setdefault(sym, gid)
        if sym not in WANT:
            for s in syns.split("|"):
                s = s.strip().upper()
                if s and s != "-" and s in WANT: sym2id.setdefault(s, gid)
mapped = {g: sym2id[g] for g in WANT if g in sym2id}
print(f"mapped to GeneID: {len(mapped)}/{len(WANT)} genes", flush=True)

# ---- gene -> papers ---------------------------------------------------------
IDS = set(mapped.values())
g2p = defaultdict(set)
with gzip.open(f"{P}/gene2pubmed.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        t, gid, pmid = line.rstrip("\n").split("\t")
        if t == TAXID and gid in IDS: g2p[gid].add(pmid)
npub = {g: len(g2p.get(mapped[g], ())) for g in mapped}
print(f"papers linked: {sum(len(v) for v in g2p.values())} gene-paper links; "
      f"median papers/gene {np.median(list(npub.values())):.0f}", flush=True)

# ---- observed co-mentions ---------------------------------------------------
def co(a, b):
    ga, gb = mapped.get(a), mapped.get(b)
    if not ga or not gb: return None
    return len(g2p[ga] & g2p[gb])

obs = []
for c in pairs:
    a, b = c["pair"]; n = co(a, b)
    c["co_mentions"] = n
    c["papers"] = [npub.get(a), npub.get(b)]
    if n is not None: obs.append(n)
obs = np.array(obs)
testable = [c for c in pairs if c["co_mentions"] is not None]
print(f"\nboth genes mappable for {len(testable)} pairs", flush=True)
print(f"observed: {int((obs > 0).sum())} pairs co-mentioned ({100*(obs>0).mean():.1f}%), "
      f"total {int(obs.sum())} co-mentions", flush=True)

# ---- degree-preserving null on the gene-paper bipartite graph ---------------
gid_list = sorted(IDS); gpos = {g: i for i, g in enumerate(gid_list)}
rows, cols = [], []
paper_ix = {}
for g, ps in g2p.items():
    for p_ in ps:
        j = paper_ix.setdefault(p_, len(paper_ix))
        rows.append(gpos[g]); cols.append(j)
rows = np.array(rows, np.int32); cols = np.array(cols, np.int32)
print(f"bipartite graph: {len(gid_list)} genes x {len(paper_ix)} papers, {len(rows)} links", flush=True)

tp = [(gpos[mapped[c['pair'][0]]], gpos[mapped[c['pair'][1]]]) for c in testable]
tp_a = np.array([x[0] for x in tp], np.int32); tp_b = np.array([x[1] for x in tp], np.int32)

nz, ntot = [], []
for it in range(NPERM):
    sc = rng.permutation(cols)                       # keep each gene's paper count exactly
    order = np.lexsort((sc, rows))
    r2, c2 = rows[order], sc[order]
    bypaper = defaultdict(list)
    for g, p_ in zip(r2, c2): bypaper[p_].append(g)
    cnt = defaultdict(int)
    for p_, gs in bypaper.items():
        if len(gs) < 2 or len(gs) > 60: continue     # skip mega-papers, as in the observed data
        gs = sorted(set(gs))
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)): cnt[(gs[i], gs[j])] += 1
    vals = np.array([cnt.get((min(a, b), max(a, b)), 0) for a, b in zip(tp_a, tp_b)])
    nz.append(int((vals > 0).sum())); ntot.append(int(vals.sum()))
    if (it + 1) % 50 == 0: print(f"  perm {it+1}/{NPERM}: {np.mean(nz):.1f} co-mentioned", flush=True)

# observed with the same mega-paper cap, for a like-for-like comparison
big = {j for j, n in zip(*np.unique(cols, return_counts=True)) if n > 60}
g2p_cap = {g: {p_ for p_ in ps if paper_ix[p_] not in big} for g, ps in g2p.items()}
obs_cap = np.array([len(g2p_cap[mapped[c['pair'][0]]] & g2p_cap[mapped[c['pair'][1]]]) for c in testable])

nz = np.array(nz, float); ntot = np.array(ntot, float)
o_nz = int((obs_cap > 0).sum()); o_tot = int(obs_cap.sum())
res = {
 "source": "NCBI gene2pubmed + Homo_sapiens.gene_info, downloaded 2026-09-11",
 "null": "degree-preserving shuffle of the gene-paper bipartite graph (each gene keeps its paper count)",
 "n_perm": NPERM, "n_pairs_testable": len(testable), "mega_paper_cap": 60,
 "observed_pairs_comentioned": o_nz, "observed_total_comentions": o_tot,
 "null_pairs_comentioned_mean": round(float(nz.mean()), 1), "null_sd": round(float(nz.std()), 1),
 "fold": round(o_nz / max(float(nz.mean()), 1e-9), 2),
 "z": round(float((o_nz - nz.mean()) / (nz.std() or 1)), 2),
 "p_emp": round((int((nz >= o_nz).sum()) + 1) / (NPERM + 1), 5),
}
print(f"\n=== (1) VALIDATION: literature co-mention ===")
print(f"observed {o_nz} of {len(testable)} pairs co-mentioned  vs null {nz.mean():.1f}±{nz.std():.1f}"
      f"  ->  {res['fold']}x, z={res['z']}, p={res['p_emp']}", flush=True)

never = [c for c in testable if c["co_mentions"] == 0]
never.sort(key=lambda c: (-c["n_models"], -c["weight"]))
studied = [c for c in never if min(c["papers"]) >= 20]      # both genes are studied, just never together
res["n_never_comentioned"] = len(never)
res["n_never_but_both_studied"] = len(studied)
res["never_comentioned"] = never[:150]
res["never_but_both_studied"] = studied[:60]
print(f"\n=== (2) NOVELTY ===")
print(f"never co-mentioned in any paper: {len(never)} of {len(testable)} pairs")
print(f"  ... of which both genes have >=20 papers each (studied, but never together): {len(studied)}")
for c in studied[:20]:
    print(f"   {c['pair'][0]:>12} — {c['pair'][1]:<12} {c['n_models']} models, "
          f"weight {c['weight']:>4}, papers {c['papers'][0]}/{c['papers'][1]}")

json.dump(res, open(f"{C}/hypothesis_pubmed.json", "w"), indent=1)
json.dump({"n": len(pairs), "pairs": pairs}, open(f"{C}/hypothesis_pairs_full.json", "w"))
print("\n==> hypothesis_pubmed.json", flush=True)
