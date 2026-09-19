#!/usr/bin/env python
"""Run every consistency check the paper has, and say plainly whether it is safe to submit.

Three separate things can drift apart, and each has bitten this project:
  the text vs the results        a fold range left at 166x where the source said 166.7
  the supplement vs the results  Table S8 kept 20-permutation folds after the move to 250, and
                                 Table S2 printed unfiltered totals under a "5-500 genes" heading
  the figures vs the results     Fig S2 shipped for four days showing the numbers we had replaced

    python scripts/audit_all.py
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import subprocess, sys

HERE = _os.path.dirname(_os.path.abspath(__file__))   # the audit modules live next to this file
CHECKS = [
    ("Results 3.4-3.5, Discussion, Limitations", "audit_manuscript.py"),
    ("Results 3.1-3.3 and Methods", "audit_manuscript_core.py"),
    ("Numbers derived rather than read off a result", "audit_manuscript_derived.py"),
    ("Supplementary tables against data and code", "audit_supplement.py"),
    ("Figures against the data they were drawn from", "audit_figures.py"),
    ("The public site against the same results", "atlas_hypothesis_block.py --check"),
]

results, width = [], max(len(n) for n, _ in CHECKS)
for name, script in CHECKS:
    cmd, *flags = script.split()
    r = subprocess.run([sys.executable, f"{HERE}/{cmd}", *flags], capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().split("\n") if l.strip()]
    summary = tail[-1] if tail else "(no output)"
    results.append((name, script.split()[0], r.returncode, summary, r.stdout))
    print(f"{'PASS' if r.returncode == 0 else 'FAIL'}  {name:<{width}}  {summary}")

bad = [x for x in results if x[2] != 0]
print()
if bad:
    print(f"{len(bad)} of {len(CHECKS)} checks failed — details:\n")
    for name, script, _, _, out in bad:
        print(f"--- {script} " + "-" * (60 - len(script)))
        for line in out.strip().split("\n"):
            if any(k in line for k in ("MISMATCH", "STALE", "MISSING", "UNSTAMPED", "no such sentence", "needs a look")):
                print("   " + line.strip())
    print("\nFix the source of the disagreement, not the check.")
else:
    print("All checks pass: text, supplement and figures agree with the results they report.")
sys.exit(1 if bad else 0)
