#!/usr/bin/env python
"""Does the ESM advantage persist into AIDO's DEEP residual, or is it input-only?
Re-runs the three protein probes (complexes / PPI / localization) on AIDO's residual
at layers 0(input)/4/8, vs the known input-embedding numbers and scPRINT's input.
Tests 'the prior sets the DEPTH of protein geometry, not its existence': if so, AIDO's
deep residual should recover some of the structure absent at its input.

    conda activate scprint
    python scripts/aido_deep_probes.py
"""
from __future__ import annotations
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")


def aido_residuals():
    """Return {layer: {symbol: vec}} for AIDO at layers 0/4/8 (mean over PBMC cells)."""
    import numpy as np, scanpy as sc, torch, torch.nn as nn
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts
    genes = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    ng = len(genes)
    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationModel.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    blocks = next(mod for _, mod in m.named_modules()
                  if isinstance(mod, nn.ModuleList) and len(mod) == cfg.num_hidden_layers)
    adata = sc.datasets.pbmc3k()[:32].copy()
    ad, attn = align_adata(adata)
    xb = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
    inp = preprocess_counts(xb, device="cpu")
    am = torch.cat([torch.from_numpy(attn).unsqueeze(0).repeat(inp.shape[0], 1),
                    torch.ones((inp.shape[0], 2))], 1)
    cap = {}
    hks = [blocks[0].register_forward_pre_hook(
        lambda mod, i: cap.__setitem__(0, (i[0] if isinstance(i, tuple) else i)[:, :ng, :].mean(0).detach()))]
    for L, blk in ((4, blocks[3]), (8, blocks[7])):
        hks.append(blk.register_forward_hook(
            lambda mod, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, :ng, :].mean(0).detach())))
    with torch.no_grad():
        m(input_ids=inp, attention_mask=am)
    for h in hks:
        h.remove()
    return {L: {g: cap[L][i].numpy() for i, g in enumerate(genes)} for L in cap}


def main():
    import numpy as np, pandas as pd
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    def auroc(s, y):
        y = np.asarray(y, bool); r = rankdata(s); npos = y.sum(); n = len(y)
        return (r[y].sum() - npos*(npos+1)/2) / (npos*(n-npos))

    res = aido_residuals()
    # input embedding (pure gene identity) for reference
    Ea = np.load(f"{BASE}/outputs/aido/gene_embedding.npz")["embedding"]
    ag = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    emb_variants = {"input-emb": {g: Ea[i] for i, g in enumerate(ag) if i < len(Ea)},
                    "L0": res[0], "L4": res[4], "L8": res[8]}
    panel = sorted(set(ag))
    rng = np.random.default_rng(0)

    # ---- ground truths -------------------------------------------------
    hg = pd.read_csv(f"{BASE}/external/corum/hgnc.txt", sep="\t", dtype=str, low_memory=False)
    uni2sym = {u.strip(): str(s).upper() for s, ups in zip(hg["symbol"], hg["uniprot_ids"])
               if isinstance(ups, str) for u in ups.split("|")}
    cp = pd.read_csv(f"{BASE}/external/corum/cp.tsv", sep="\t", dtype=str)
    complexes = []
    for mol in cp["Identifiers (and stoichiometry) of molecules in complex"]:
        if not isinstance(mol, str):
            continue
        s = {uni2sym[t.split("(")[0]] for t in mol.split("|")
             if ACC.match(t.split("(")[0]) and t.split("(")[0] in uni2sym and uni2sym[t.split("(")[0]] in set(panel)}
        if 3 <= len(s) <= 60:
            complexes.append(sorted(s))
    cpx_pos = list({(a, b) for s in complexes for i, a in enumerate(s) for b in s[i+1:]})

    hpa = pd.read_csv(f"{BASE}/external/hpa/subcellular_location.tsv", sep="\t")
    COARSE = {"Nucleus": ["Nucleoplasm", "Nuclear bodies", "Nuclear membrane", "Nuclear speckles", "Nucleoli",
              "Nucleoli fibrillar center", "Nucleoli rim", "Kinetochore", "Mitotic chromosome"],
              "Cytosol": ["Cytosol", "Cytoplasmic bodies", "Rods & Rings", "Aggresome"],
              "Plasma membrane": ["Plasma membrane", "Cell Junctions"], "Mitochondria": ["Mitochondria"],
              "ER": ["Endoplasmic reticulum"], "Golgi": ["Golgi apparatus"],
              "Vesicles": ["Vesicles", "Lysosomes", "Peroxisomes", "Endosomes", "Lipid droplets"],
              "Cytoskeleton": ["Actin filaments", "Centrosome", "Centriolar satellite", "Microtubules",
              "Intermediate filaments", "Microtubule ends", "Cytokinetic bridge", "Midbody", "Midbody ring",
              "Focal adhesion sites", "Cleavage furrow"]}
    f2c = {f: c for c, fs in COARSE.items() for f in fs}
    g2c = {}
    for g, mn in zip(hpa["Gene name"], hpa["Main location"]):
        if isinstance(mn, str):
            cc = {f2c[x] for x in mn.split(";") if x in f2c}
            if cc:
                g2c[str(g).upper()] = cc

    # STRING (high-conf edges within panel)
    import gzip
    ens2sym = {}
    with gzip.open(f"{BASE}/external/string/9606.protein.info.v12.0.txt.gz", "rt") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t"); ens2sym[p[0]] = p[1].upper()
    pset = set(panel); keepE = {e for e, s in ens2sym.items() if s in pset}
    string_hi = set()
    with gzip.open(f"{BASE}/external/string/9606.protein.links.v12.0.txt.gz", "rt") as f:
        next(f)
        for ln in f:
            a, b, sc = ln.split()
            if a in keepE and b in keepE and int(sc) >= 700:
                sa, sb = ens2sym[a], ens2sym[b]
                if sa != sb:
                    string_hi.add((min(sa, sb), max(sa, sb)))
    string_hi = list(string_hi)

    # shared random negatives
    negs = []
    while len(negs) < 16000:
        i, j = rng.integers(0, len(panel), 2)
        if i != j:
            negs.append((min(panel[i], panel[j]), max(panel[i], panel[j])))
    negs = list(set(negs) - set(cpx_pos) - set(string_hi))

    def cos_auroc(D, pos):
        keys = [g for g in panel if g in D]
        idx = {g: k for k, g in enumerate(keys)}
        Mn = np.stack([D[g] for g in keys]).astype(np.float64)
        Mn = Mn / (np.linalg.norm(Mn, axis=1, keepdims=True) + 1e-9)
        def c(pairs):
            pp = [(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx]
            return np.array([(Mn[i]*Mn[j]).sum() for i, j in pp])
        pc, nc = c(pos), c(negs)
        return auroc(np.concatenate([pc, nc]), [1]*len(pc)+[0]*len(nc))

    def loc_macro(D):
        keys = [g for g in panel if g in D and g in g2c]
        M = np.stack([D[g] for g in keys]).astype(np.float64)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        comps = list(COARSE); Y = np.array([[c in g2c[g] for c in comps] for g in keys])
        keep = Y.sum(0) >= 40; comps = [c for c, k in zip(comps, keep) if k]; Y = Y[:, keep]
        aus = []
        for i in range(len(comps)):
            pr = cross_val_predict(LogisticRegression(max_iter=1000), M, Y[:, i], cv=5,
                                   method="predict_proba")[:, 1]
            aus.append(roc_auc_score(Y[:, i], pr))
        return float(np.mean(aus))

    print(f"\n==> probes across AIDO depth  (complexes {len(cpx_pos)} pairs, STRING {len(string_hi)} hi-conf, HPA loc)")
    print(f"    scPRINT input reference: complexes 0.694 | PPI 0.599 | localization 0.681\n")
    print(f"    {'variant':<10} {'complexes':>10} {'PPI':>8} {'localiz.':>10}")
    for name, D in emb_variants.items():
        print(f"    {name:<10} {cos_auroc(D, cpx_pos):>10.3f} {cos_auroc(D, string_hi):>8.3f} {loc_macro(D):>10.3f}")
    print("\n[test] if these RISE from input->L8, the ESM prior sets DEPTH (not existence) of protein geometry")


if __name__ == "__main__":
    main()
