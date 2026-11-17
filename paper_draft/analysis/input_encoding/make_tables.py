#!/usr/bin/env python
"""Print the markdown tables of notes/11_input_encoding.md from the JSON outputs of this folder.
Every cell comes from a JSON key; the key path is printed above each table."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
LX = json.load(open(f"{D}/lexical.json"))["models"]
TR = json.load(open(f"{D}/trrust_lexical.json")) if os.path.exists(f"{D}/trrust_lexical.json") else None
AN = json.load(open(f"{D}/annot_clean.json")) if os.path.exists(f"{D}/annot_clean.json") else None


def pct(x, nd=2): return "-" if x is None else f"{100 * x:.{nd}f}"
def f_(x, nd=2): return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def cell(mt, k, as_pct=True, nd=2):
    e = mt.get(k)
    if not isinstance(e, dict): return ["-"] * 5
    o, n = (pct(e["obs"], nd), pct(e["null_mean"], nd)) if as_pct else (f_(e["obs"], 3), f_(e["null_mean"], 3))
    return [o, n, f_(e["fold"]), f_(e["z"], 1), f_(e["p_emp"], 4)]


print("\n#### T-IE-1: tie tails per model (lexical.json:models.<m>.tie_tail, universe_size_reconstructed, n_features_all)\n")
print("| model | universe (genes) | features | tie tail reaches top-10 (share of features) | certain-tie slots caught by block rule | tie->top10 share by firing count 1-9 / 10-99 / 100-999 / >=1000 | tail genes' position in index | most frequent tail genes |")
print("|---|---|---|---|---|---|---|---|")
for m in ORDER:
    r = LX[m]; t = r["tie_tail"]
    bins = [t.get(f"tail_into_top10_share_firing_{a}", {}).get("share_final") for a in ("1_10", "10_100", "100_1000", "1000_inf")]
    pos = sorted({x[2] for x in t["top_tail_genes"][:5]})
    print(f"| {m} | {r['universe_size_reconstructed']} | {r['n_features_all']} | {pct(t['share_features_tail_into_top10'],1)} % | "
          f"{pct(t.get('certain_tie_slots_flagged_by_block_rule_share'),1) if 'certain_tie_slots_flagged_by_block_rule_share' in t else 'n/a'}"
          f"{' % of ' + str(t['certain_tie_slots']) if 'certain_tie_slots' in t else ''} | "
          f"{' / '.join(pct(b,1) if b is not None else '-' for b in bins)} | {', '.join(str(p) for p in pos) if pos else '-'} | "
          f"{', '.join(f'{g} ({c})' for g, c, _ in t['top_tail_genes'][:5])} |")

print("\n#### T-IE-2: most featured genes, share of their top-10 appearances that sit in a tie tail (lexical.json:models.<m>.hubs_raw / hubs_clean)\n")
print("| model | top-5 featured genes, raw (count, share in tie tail) | top-5 after removing tie tails (count) |")
print("|---|---|---|")
for m in ORDER:
    r = LX[m]
    print(f"| {m} | {', '.join(f'{g} ({c}, {pct(s,0)} %)' for g, c, s in r['hubs_raw'][:5])} | {', '.join(f'{g} ({c})' for g, c in r['hubs_clean'][:5])} |")

for scope, lab in (("pooled", "all layers pooled"), ("mid", "mid layer")):
    for keyset, title in ((("p3", "p2"), "shared prefix"), (("lcp", "lev"), "prefix length and edit distance")):
        print(f"\n#### T-IE-3{scope[0]}-{keyset[0]}: {title}, {lab} (lexical.json:models.<m>.{scope}.<raw|clean>.metrics.<{'|'.join(keyset)}>)\n")
        hdr = ["model", "variant", "features", "pairs"]
        for k in keyset:
            unit = "%" if k in ("p3", "p2") else ""
            hdr += [f"{k} obs {unit}", f"{k} null {unit}", f"{k} fold", f"{k} z", f"{k} p"]
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for m in ORDER:
            for v in ("raw", "clean"):
                s = LX[m][scope][v]; mt = s["metrics"]; row = [m, v, str(s["n_features"]), str(s["n_pairs"])]
                for k in keyset: row += cell(mt, k, as_pct=k in ("p3", "p2"))
                print("| " + " | ".join(row) + " |")

for scope, lab in (("pooled", "all layers pooled"), ("mid", "mid layer")):
    print(f"\n#### T-IE-4{scope[0]}: order adjacency, {lab} (lexical.json:models.<m>.{scope}.<raw|clean>.metrics.<adj1|adj20|tok1|tok20>)\n")
    hdr = ["model", "variant", "alpha +-1 obs %", "null %", "fold", "z", "p", "alpha +-20 fold", "token +-1 obs %", "null %", "fold", "z", "p", "token +-20 fold"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for m in ORDER:
        for v in ("raw", "clean"):
            mt = LX[m][scope][v]["metrics"]
            a1 = cell(mt, "adj1", nd=3); t1 = cell(mt, "tok1", nd=3)
            print(f"| {m} | {v} | " + " | ".join(a1) + f" | {f_(mt['adj20']['fold'])} | " + " | ".join(t1)
                  + f" | {f_(mt['tok20']['fold']) if isinstance(mt.get('tok20'), dict) else '-'} |")

for scope, lab in (("pooled", "all layers pooled"), ("mid", "mid layer")):
    print(f"\n#### T-IE-5{scope[0]}: same HGNC gene group and UCE input-token (ESM2) similarity, {lab} (lexical.json:models.<m>.{scope}.<raw|clean>.metrics.<hg|esm_cos>)\n")
    hdr = ["model", "variant", "same group obs %", "null %", "fold", "z", "p", "pairs with both genes grouped %", "ESM2 cos obs", "null", "fold", "z", "p"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for m in ORDER:
        for v in ("raw", "clean"):
            mt = LX[m][scope][v]["metrics"]
            print(f"| {m} | {v} | " + " | ".join(cell(mt, "hg")) + f" | {pct(mt.get('hg_both_frac'),1)} | " + " | ".join(cell(mt, "esm_cos", as_pct=False)) + " |")

print("\n#### T-IE-6: per-layer profile (lexical.json:models.<m>.per_layer.<L>)\n")
print("| model | layer | tie->top10 % | p3 fold raw | p3 fold clean | alpha+-1 fold raw | alpha+-1 fold clean | HGNC fold raw | HGNC fold clean | p3 z clean |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in ORDER:
    pl = LX[m].get("per_layer")
    if not pl: continue
    for L, r in sorted(pl.items(), key=lambda x: int(x[0])):
        if not r["raw"]: continue
        a, c = r["raw"]["metrics"], r["clean"]["metrics"]
        print(f"| {m} | {L} | {pct(r['tail_into_top10_share'],1)} | {f_(a['p3']['fold'])} | {f_(c['p3']['fold'])} | "
              f"{f_(a['adj1']['fold'])} | {f_(c['adj1']['fold'])} | {f_(a['hg']['fold'])} | {f_(c['hg']['fold'])} | {f_(c['p3']['z'],1)} |")

print("\n#### T-IE-7: Ensembl ids (lexical.json:models.<m>.ensembl)\n")
print("| model | ENSG ids in index | share of index | with an HGNC symbol | share of top-10 slots | features with >=1 ENSG | ENSG slots in tie tails | share of top-10 slots after removing tails | top ENSG ids (count) |")
print("|---|---|---|---|---|---|---|---|---|")
for m in ORDER:
    e = LX[m]["ensembl"]
    print(f"| {m} | {e['ensg_in_universe']} | {pct(e['universe_share'],1)} % | {e['ensg_in_universe_with_hgnc_symbol']} | {pct(e['share_top10_slots'])} % | "
          f"{pct(e['share_features_with_ensg_in_top10'],1)} % | {pct(e['share_ensg_top10_slots_in_tie_tail'],1) if e['share_ensg_top10_slots_in_tie_tail'] is not None else '-'} % | "
          f"{pct(e['share_top10_slots_after_clean'])} % | {', '.join(f'{g} ({c})' for g, c in e['top_ensg'][:3])} |")

if TR:
    print(f"\n#### T-IE-8: TRRUST test variants (trrust_lexical.json:models.<m>.variants.<v>; {TR['n_perm']} rewirings, seed 0)\n")
    vs = ["published", "no_p3", "no_hgnc", "no_alpha20", "no_lexfam", "clean", "clean_no_lexfam", "published_top8631", "clean_top8631"]
    print("| model | " + " | ".join(vs) + " |"); print("|---|" + "---|" * len(vs))
    for m in ORDER:
        V = TR["models"][m]["variants"]; row = [m]
        for v in vs:
            x = V.get(v)
            row.append("-" if not x else f"{x['hits']} h / {x['n_pairs']} p, {f_(x['fold'])}x, p={f_(x['p_emp'],3)}")
        print("| " + " | ".join(row) + " |")
    print(f"\n#### T-IE-9: TRRUST hits that are lexical / family pairs, and hits lost when tie tails are removed (trrust_lexical.json:models.<m>.<hits|hits_flagged|hits_lost_when_tie_tails_removed|graph_share_flagged>)\n")
    print("| model | graph pairs flagged (any) % | hits | same 3-prefix | same HGNC group | alpha +-20 | any | hits lost without tie tails | lost hits |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        r = TR["models"][m]; h = r["hits_flagged"]
        print(f"| {m} | {pct(r['graph_share_flagged']['any'],1)} | {r['hits']} | {h['p3']} | {h['hgnc']} | {h['alpha20']} | {h['any']} | "
              f"{r['hits_lost_when_tie_tails_removed']} | {', '.join(r['hits_lost_list'][:12])} |")
    rp = TR["released_pairs"]
    print(f"\n#### T-IE-10: the released 4,349 pairs (trrust_lexical.json:released_pairs)\n")
    print("| n | same 3-prefix | same 2-prefix | same HGNC group | alpha +-20 | any lexical/family | still in >= 2 passing models after tie tails removed |")
    print("|---|---|---|---|---|---|---|")
    s, n = rp["share_flagged"], rp["flagged"]
    print(f"| {rp['n']} | {n['p3']} ({pct(s['p3'],1)} %) | {n['p2']} ({pct(s['p2'],1)} %) | {n['hgnc']} ({pct(s['hgnc'],1)} %) | "
          f"{n['alpha20']} ({pct(s['alpha20'],1)} %) | {n['any']} ({pct(s['any'],1)} %) | {rp['still_in_ge2_passing_models_after_tie_tail_removal']} |")
    print("truth pairs flagged:", TR["truth_pairs"], TR["truth_pairs_flagged"])

if AN:
    print("\n#### T-IE-11: calibrated annotator, mid layer, raw vs tie tails removed (annot_clean.json:per_model.<m>, backbone_raw, backbone_clean)\n")
    print("| model | mid layer | features | tie tail reaches top-10 | annotated features raw | clean | terms raw | clean | terms lost |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        p = AN["per_model"][m]
        print(f"| {m} | {p['mid_layer']} | {p['n_features']} | {p['features_tail_into_top10']} | {p['annotated_features_raw']} | "
              f"{p['annotated_features_clean']} | {p['terms_raw']} | {p['terms_clean']} | {p['n_terms_lost']} |")
    print("backbone raw", AN["backbone_raw"], "clean", AN["backbone_clean"], "lost", AN["backbone_ge8_lost"], "gained", AN["backbone_ge8_gained"])

PB = json.load(open(f"{D}/prefix_beyond_family.json"))
print(f"\n#### T-IE-12: shared 3-letter prefix among pairs NOT in the same HGNC group (prefix_beyond_family.json:models.<m>.<mid|pooled>.<raw|clean>.metrics.<p3_nofam|p3_diffgrp>; mid {PB['R_mid']} perms, pooled {PB['R_pool']} perms)\n")
print("| model | mid raw: same prefix, not same group, obs % / null % (fold, z, p) | mid clean: same | mid clean: among pairs in two different groups, obs % / null % (fold, z, p) | pooled clean: same prefix, not same group (fold, z, p) |")
print("|---|---|---|---|---|")
def pb(e): return "-" if not isinstance(e, dict) else f"{100*e['obs']:.3f} / {100*e['null_mean']:.3f} ({f_(e['fold'])}x, z {f_(e['z'],1)}, p {f_(e['p_emp'],4)})"
for m in ORDER:
    r = PB["models"][m]
    print(f"| {m} | {pb(r['mid']['raw']['metrics']['p3_nofam'])} | {pb(r['mid']['clean']['metrics']['p3_nofam'])} | "
          f"{pb(r['mid']['clean']['metrics']['p3_diffgrp'])} | {pb(r['pooled']['clean']['metrics']['p3_nofam'])} |")
PE = json.load(open(f"{D}/prefix_examples.json"))["models"]
print("\n#### T-IE-13: which prefixes make up the 'named alike, different group' pairs, pooled, tie tails removed (prefix_examples.json:models.<m>)\n")
print("| model | pairs | same 3-prefix pairs | of which both genes grouped, groups differ | top prefixes (pairs) |")
print("|---|---|---|---|---|")
for m in ORDER:
    r = PE[m]
    print(f"| {m} | {r['n_pairs']} | {r['same_p3_pairs']} | {r['same_p3_diff_group_pairs']} | {', '.join(f'{p} ({n})' for p, n in r['top_prefixes'][:6])} |")
TO = json.load(open(f"{D}/token_order_check.json"))
print("\n#### T-IE-14: real token-id order vs alphabetical order (token_order_check.json:<m>)\n")
print("| model | token source | featured symbols with a token id | Spearman(alphabetical rank, token id) |")
print("|---|---|---|---|")
for m in ORDER:
    if m in TO:
        r = TO[m]; print(f"| {m} | {r['source']} | {r['with_token']} of {r['n_symbols']} | {r['spearman_alpha_vs_token']} |")
    else:
        print(f"| {m} | not on disk (alphabetical only) | - | - |")
TRp = {v: json.load(open(f"{D}/tie_reproduce_np{v}.json")) for v in ("1.26.4", "2.3.4")}
print("\n#### T-IE-15: re-running the catalogue's argsort on an alphabetical index (tie_reproduce_np<ver>.json:models.<m>)\n")
print("| model | features tested (fired < 10 times) | NumPy 1.26.4: mean share of published tie genes reproduced | share of features with >= 80 % reproduced | exact 20-gene order reproduced | NumPy 2.3.4: mean share reproduced |")
print("|---|---|---|---|---|---|")
for m in ORDER:
    a, b = TRp["1.26.4"]["models"][m], TRp["2.3.4"]["models"][m]
    if a.get("n_tested"):
        print(f"| {m} | {a['n_tested']} | {pct(a['mean_share_of_published_tie_genes_reproduced'],1)} % | {pct(a['share_features_ge80pct_tie_genes_reproduced'],1)} % | {a['exact_order_match']} | {pct(b['mean_share_of_published_tie_genes_reproduced'],1)} % |")
FP = json.load(open(f"{D}/tail_fp_check.json"))
print("\n#### T-IE-16: flagged tails in features that fired >= 100 times (tail_fp_check.json:<m>)\n")
print("| model | features with a tail reaching the top-10 | of which fired >= 100 times | most common genes in those tails |")
print("|---|---|---|---|")
for m in ORDER:
    r = FP[m]; print(f"| {m} | {r['flagged_top10']} | {r['flagged_and_fired_ge100']} | {', '.join(f'{g} ({c})' for g, c in r['genes_in_those_tails'][:6])} |")
