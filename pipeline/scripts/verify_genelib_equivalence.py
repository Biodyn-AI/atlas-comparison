"""Equivalence check: does routing the co-firing graph through genelib change Fig 5A?

genelib.top_genes() drops Ensembl ids BEFORE truncating to top-N, where hypothesis_trrust.py
truncated first and dropped after. That matters unevenly -- ENSG ids are 19.5% of Tahoe-x1's
top-gene slots and 0% of five other models' -- so "top-10 genes" meant ~8.3 usable symbols for
Tahoe against 10 for UCE.

Result (2026-09-11, 200 permutations): the headline is unchanged. The same five models pass, in
the same order, and the same five do not.

    model          published            recomputed
    Tahoe-x1       59 hits, 13.29x      62 hits, 13.23x
    scGPT          10 hits, 11.90x      10 hits,  9.62x
    UCE            43 hits, 10.54x      43 hits, 10.80x
    C2S-Scale      83 hits,  4.12x      84 hits,  3.96x
    Geneformer-V2  12 hits,  3.87x      12 hits,  3.98x
    MaxToki         2 hits,  3.45x       2 hits,  3.74x  (n.s. both ways)
    tGPT            3 hits,  1.31x       3 hits,  1.21x  (n.s. both ways)
    AIDO.Cell / scFoundation / GeneCompass: no edges recovered either way

Tahoe gains pairs (41,952 -> 50,228) because the fix returns its real symbols, and three more
recovered edges, at an unchanged fold. scGPT's fold moves most (11.9 -> 9.6) on an unchanged hit
count of 10: that is noise in the null estimate at small counts, and the reason the manuscript
ranks models rather than reading small differences between them.

    python pipeline/scripts/verify_genelib_equivalence.py
"""
import sys, json, itertools
import numpy as np
from collections import defaultdict
sys.path.insert(0,"/Users/annaantipova/Desktop/biomech/comparative-sae-atlas/pipeline/scripts")
from genelib import MODELS, catalogs, features, top_genes

BASE="/Users/annaantipova/Desktop/biomech"; C=f"{BASE}/outputs/atlas/comparative"; G=f"{BASE}/outputs/atlas/genesets"
TOPG,W0,NPERM = 10,5,200
rng=np.random.default_rng(0)
TRUTH=set()
for a,bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u,v=sorted((a.upper(),b.upper()))
        if u!=v: TRUTH.add((u,v))

PRED={}
for m in MODELS:
    cnt=defaultdict(int)
    for p in catalogs(m):
        for _,f in features(p):
            s=sorted(set(top_genes(f, TOPG)))          # <-- ENSG dropped first
            if len(s)<2: continue
            for a,b in itertools.combinations(s,2): cnt[(a,b)]+=1
    PRED[m]={k:v for k,v in cnt.items() if v>=W0}
    print(f"  {m}: {len(PRED[m])} pairs", flush=True)

GENES=sorted({g for m in MODELS for e in PRED[m] for g in e}); GID={g:i for i,g in enumerate(GENES)}
TK=np.array(sorted({(lambda i,j:i*(1<<21)+j)(*sorted((GID[a],GID[b]))) for a,b in TRUTH if a in GID and b in GID}),np.int64)
def shuf(E):
    a=np.fromiter((GID[e[0]] for e in E),np.int32,len(E)); b=np.fromiter((GID[e[1]] for e in E),np.int32,len(E))
    s=np.concatenate([a,b]); rng.shuffle(s); h,t=s[::2],s[1::2]; ok=h!=t; h,t=h[ok],t[ok]
    return np.unique(np.minimum(h,t).astype(np.int64)*(1<<21)+np.maximum(h,t).astype(np.int64))

pub=json.load(open(f"{C}/hypothesis_trrust2.json"))["per_model"]
print(f"\n{'model':<14}{'published':>22}{'recomputed':>24}")
out={}
for m in MODELS:
    E=list(PRED[m])
    if not E: continue
    k=np.empty(len(E),np.int64)
    for i,(a,b) in enumerate(E):
        x,y=sorted((GID[a],GID[b])); k[i]=x*(1<<21)+y
    obs=int(np.isin(k,TK).sum())
    nul=np.array([int(np.isin(shuf(E),TK).sum()) for _ in range(NPERM)],float)
    fold=obs/max(nul.mean(),1e-9); p=(int((nul>=obs).sum())+1)/(NPERM+1)
    o=pub.get(m,{})
    out[m]={"hits":obs,"fold":round(fold,2),"p":round(p,4)}
    lhs = "%s hits, %sx" % (o.get("hits", "-"), o.get("fold", "-"))
    rhs = "%d hits, %.2fx (p=%.4f)" % (obs, fold, p)
    print("%-14s%22s%24s" % (m, lhs, rhs), flush=True)
json.dump(out,open("/tmp/verify_ensg.json","w"),indent=1)
