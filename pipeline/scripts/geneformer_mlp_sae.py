#!/usr/bin/env python
"""SAE on MLP activations + BROAD vocabulary (Phase-1 upgrade). Decomposes a layer's
MLP activation space into an overcomplete sparse dictionary (monosemantic FEATURES, the
right unit vs polysemantic neurons) and annotates against a broad gene-set vocabulary
(TRRUST regulons + protein complexes + subcellular localization + co-regulated clusters
+ cell types). Measures whether SAE + broad vocab raise COVERAGE over raw neurons.

    conda activate scprint
    python scripts/geneformer_mlp_sae.py --layer 5
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import argparse, os, pickle, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B; GF = f"{BASE}/ckpt_geneformer"
MARKERS = {"T": ["CD3D", "CD3E", "TRAC", "IL7R"], "B": ["MS4A1", "CD79A", "CD79B"],
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8"], "NK": ["NKG7", "GNLY", "KLRD1"]}
ACC = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")


def broad_vocab(panel):
    """Assemble {name: set(symbols in panel)} from local sources (no downloads)."""
    import pandas as pd
    fam = {}; sets = {}
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    for tf, g in tr.groupby(tr.tf.str.upper()):
        s = {x.upper() for x in g.tg} & panel
        if len(s) >= 12:
            sets[f"reg:{tf}"] = s; fam[f"reg:{tf}"] = "regulon"
    # protein complexes (Complex Portal + HGNC uniprot->symbol)
    try:
        hg = pd.read_csv(f"{BASE}/external/corum/hgnc.txt", sep="\t", dtype=str, low_memory=False)
        u2s = {u.strip(): str(s).upper() for s, ups in zip(hg["symbol"], hg["uniprot_ids"])
               if isinstance(ups, str) for u in ups.split("|")}
        cp = pd.read_csv(f"{BASE}/external/corum/cp.tsv", sep="\t", dtype=str)
        for name, mol in zip(cp["Recommended name"], cp["Identifiers (and stoichiometry) of molecules in complex"]):
            if isinstance(mol, str):
                s = {u2s[t.split("(")[0]] for t in mol.split("|") if t.split("(")[0] in u2s} & panel
                if len(s) >= 12:
                    sets[f"cplx:{name}"[:40]] = s; fam[f"cplx:{name}"[:40]] = "complex"
    except Exception as e:
        print("  (complexes skipped:", e, ")")
    # subcellular localization (HPA)
    try:
        COARSE = {"Nucleus": ["Nucleoplasm", "Nucleoli", "Nuclear bodies", "Nuclear membrane", "Nuclear speckles"],
                  "Cytosol": ["Cytosol"], "Plasma membrane": ["Plasma membrane", "Cell Junctions"],
                  "Mitochondria": ["Mitochondria"], "ER": ["Endoplasmic reticulum"], "Golgi": ["Golgi apparatus"],
                  "Vesicles": ["Vesicles"], "Cytoskeleton": ["Actin filaments", "Microtubules", "Centrosome"]}
        f2c = {f: c for c, fs in COARSE.items() for f in fs}
        hpa = pd.read_csv(f"{BASE}/external/hpa/subcellular_location.tsv", sep="\t")
        loc = {c: set() for c in COARSE}
        for g, mn in zip(hpa["Gene name"], hpa["Main location"]):
            if isinstance(mn, str):
                for x in mn.split(";"):
                    if x in f2c:
                        loc[f2c[x]].add(str(g).upper())
        for c, s in loc.items():
            s = s & panel
            if len(s) >= 12:
                sets[f"loc:{c}"] = s; fam[f"loc:{c}"] = "localization"
    except Exception as e:
        print("  (localization skipped:", e, ")")
    # co-regulated clusters
    cl = {"HLA": lambda g: g.startswith("HLA-"), "Histones": lambda g: g.startswith(("HIST", "H2AC", "H2BC", "H3C", "H4C")),
          "HOX": lambda g: bool(re.match(r"^HOX[ABCD]\d+$", g)), "Keratins": lambda g: bool(re.match(r"^KRT\d+$", g)),
          "IFN-a": lambda g: bool(re.match(r"^IFNA\d+$", g))}
    for name, fn in cl.items():
        s = {g for g in panel if fn(g)}
        if len(s) >= 8:
            sets[f"clust:{name}"] = s; fam[f"clust:{name}"] = "cluster"
    # GO / Reactome / Hallmark gene sets (GMT files in external/genesets/) — the completeness that
    # makes 'unannotated' mean genuinely novel (Block C). Download once with gseapy (see run-guide).
    import glob
    gdir = f"{BASE}/external/genesets"
    fam_map = {"GO_Biological_Process": "GO:BP", "GO_Molecular_Function": "GO:MF",
               "GO_Cellular_Component": "GO:CC", "Reactome": "Reactome", "MSigDB_Hallmark": "Hallmark"}
    for gm in glob.glob(f"{gdir}/*.gmt"):
        b = os.path.basename(gm)
        fname = next((v for k, v in fam_map.items() if b.startswith(k)), "geneset")
        for line in open(gm):
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            s = {g.upper() for g in p[2:] if g} & panel
            if 12 <= len(s) <= 500:                                # skip tiny & over-broad terms
                key = f"{fname}:{p[0]}"[:60]
                sets[key] = s; fam[key] = fname
    return sets, fam


def tokenize_cell(counts, ensg, tok, med, max_len=2048):
    import numpy as np
    total = counts.sum()
    if total == 0:
        return [], []
    norm = counts / total; val, ids, es = [], [], []
    for j, g in enumerate(ensg):
        if counts[j] > 0 and g in tok and g in med and med[g] > 0:
            val.append(norm[j] / med[g]); ids.append(tok[g]); es.append(g)
    if not val:
        return [], []
    order = np.argsort(val)[::-1][:max_len]
    return [ids[i] for i in order], [es[i] for i in order]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, default=5); A = ap.parse_args()
    import numpy as np, scanpy as sc, torch, pandas as pd
    from scipy.stats import rankdata
    from transformers import BertForMaskedLM
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    model = BertForMaskedLM.from_pretrained(GF).eval()
    dff = model.config.intermediate_size
    tok = pickle.load(open(f"{GF}/token_dictionary_gc30M.pkl", "rb"))
    med = pickle.load(open(f"{GF}/gene_median_dictionary_gc30M.pkl", "rb"))
    pad_id = tok.get("<pad>", 0)
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    sym2ensg = {}; ensg2sym = {}
    for e, s in bm["hgnc_symbol"].items():
        sym2ensg.setdefault(str(s).upper(), e); ensg2sym[e] = str(s).upper()

    N = 128
    adata = sc.datasets.pbmc3k()[:N].copy()
    raw = adata.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    Sc = {}
    for ct, ms in MARKERS.items():
        pres = [g for g in ms if g in raw.var_names]
        X = raw[:, pres].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        Sc[ct] = np.mean([z(X[:, j]) for j in range(X.shape[1])], 0)
    S = np.stack([Sc[c] for c in MARKERS], 1); cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")

    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    ensg = np.array([sym2ensg.get(s.upper(), "NA") for s in adata.var_names])
    seqs = [tokenize_cell(Xd[i], ensg, tok, med) for i in range(N)]

    # capture layer-L MLP activations per (cell,gene) token
    acts_rows, gsym_rows, cell_rows = [], [], []
    for s in range(0, N, 8):
        sub = list(range(s, min(s+8, N))); ids, esl = zip(*[seqs[i] for i in sub])
        L = max(len(x) for x in ids)
        inp = torch.full((len(sub), L), pad_id, dtype=torch.long); at = torch.zeros((len(sub), L), dtype=torch.long)
        for k, x in enumerate(ids):
            inp[k, :len(x)] = torch.tensor(x); at[k, :len(x)] = 1
        cap = {}
        h = model.bert.encoder.layer[A.layer].intermediate.register_forward_hook(
            lambda m, i, o: cap.__setitem__("a", o.detach()))
        with torch.no_grad():
            model(input_ids=inp, attention_mask=at)
        h.remove()
        Aar = cap["a"].numpy()
        for bi, ci in enumerate(sub):
            n = len(esl[bi])
            acts_rows.append(Aar[bi, :n]); gsym_rows.extend([ensg2sym.get(e, "") for e in esl[bi]])
            cell_rows.extend([ci]*n)
    Xacts = np.concatenate(acts_rows, 0)                          # [tokens, dff]
    gsym = np.array(gsym_rows); cellid = np.array(cell_rows)
    print(f"==> layer {A.layer}: {Xacts.shape[0]} tokens x {dff} neurons")

    panel = set(g for g in gsym if g)
    vocab, fam = broad_vocab(panel)
    sym2uni = {g: i for i, g in enumerate(sorted(panel))}
    uni = sorted(panel); nGuni = len(uni)
    vidx = {name: np.array([sym2uni[g] for g in s if g in sym2uni]) for name, s in vocab.items()}
    vidx = {k: v for k, v in vidx.items() if len(v) >= 8}
    print(f"==> broad vocab: {len(vidx)} gene sets across "
          f"{ {f: sum(1 for k in vidx if fam.get(k)==f) for f in set(fam.values())} } + {len(MARKERS)} cell types")

    def coverage(F, tag):
        """F [tokens, U]; return fraction of live units NAMED by broad vocab (gene-sets) or cell type."""
        U = F.shape[1]
        # per-gene aggregate
        G = np.zeros((nGuni, U), np.float32); cnt = np.zeros(nGuni)
        gi = np.array([sym2uni.get(g, -1) for g in gsym]); ok = gi >= 0
        np.add.at(G, gi[ok], F[ok]); np.add.at(cnt, gi[ok], 1); G /= np.maximum(cnt[:, None], 1)
        # per-cell aggregate
        P = np.zeros((N, U), np.float32); cc = np.zeros(N)
        np.add.at(P, cellid, F); np.add.at(cc, cellid, 1); P /= np.maximum(cc[:, None], 1)
        live = F.std(0) > 1e-4
        def auroc(Rk, idx, n): npos = len(idx); return (Rk[idx].sum(0)-npos*(npos+1)/2)/(npos*(n-npos))
        def bestpos(mat, groups, nrows):
            Rk = rankdata(mat, axis=0); b = np.full(U, 0.5)
            for _, idx in groups.items():
                b = np.maximum(b, auroc(Rk, idx, nrows))
            return b
        rng = np.random.default_rng(0)
        gs_s = bestpos(G, vidx, nGuni); gs_n = bestpos(G[rng.permutation(nGuni)], vidx, nGuni)
        types = [c for c in np.unique(cell_type) if (cell_type == c).sum() >= 8 and c != "other"]
        ctg = {c: np.where(cell_type == c)[0] for c in types}
        ct_s = bestpos(P, ctg, N); perm = rng.permutation(N)
        ct_n = bestpos(P, {c: np.where(cell_type[perm] == c)[0] for c in types}, N)
        thr_g, thr_c = np.percentile(gs_n, 99), np.percentile(ct_n, 99)
        named = live & ((gs_s > thr_g) | (ct_s > thr_c))
        print(f"   {tag:<32} live {live.sum():>4}/{U}  NAMED {named.sum():>4} ({100*named[live].mean():.1f}% of live)")
        return named, live

    # (1) raw neurons
    coverage(Xacts, f"raw neurons (broad vocab)")
    # (2) SAE features
    sub = np.random.default_rng(0).choice(Xacts.shape[0], min(40000, Xacts.shape[0]), replace=False)
    sae, stats, log = train_sae(Xacts[sub], SAEConfig(d_sae=4096, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, Xacts, stats, device="cpu")
    print(f"   [SAE trained: FVU {log.history[-1]['fvu']:.3f}, dead {log.history[-1]['dead_frac']:.2f}]")
    coverage(F, f"SAE features (broad vocab)")
    print("\n==> coverage upgrade done — compare NAMED% of live: raw neurons vs SAE features")


if __name__ == "__main__":
    main()
