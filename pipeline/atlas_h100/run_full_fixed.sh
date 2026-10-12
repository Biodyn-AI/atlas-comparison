#!/bin/bash
# Corrected all-layer extraction (replaces the out_alllayers run, whose 500,000-position cap read
# 32 cells for AIDO.Cell and all 6,000 for tGPT). Every model reads all 6,000 corpus cells and keeps
# at most 168 positions per cell (seed 0), i.e. about 1M positions per layer, the SAE's training
# budget; stable top-gene order; seeded SAE. Resumable: a model with a DONE marker is skipped.
cd "$(dirname "$0")"                                  # run from pipeline/atlas_h100
VENV="${ATLAS_VENV:-../.venv}"; TAHOE_VENV="${ATLAS_TAHOE_VENV:-../.venv-tahoe}"   # Tahoe-x1 needs its own (torch 2.6, flash-attn)
[ -f env.sh ] && source env.sh                       # optional: cache locations
export CUDA_VISIBLE_DEVICES=0
CORP=external/perturb/tabula_3tissue_6k.h5ad
OUT=out_fixed
mkdir -p $OUT/logs
run(){ m=$1; [ -f $OUT/$m/DONE ] && { echo "skip $m (done)"; return; }
  rm -rf $OUT/$m; echo "=== $m $(date '+%F %H:%M') ==="
  python run_model.py --model $m --corpus $CORP --out $OUT --skip-annotate --geneset-dir data/genesets \
    --all-layers --per-cell 168 --seed 0 > $OUT/logs/$m.log 2>&1 \
    && touch $OUT/$m/DONE && echo "    ok $(date '+%H:%M')" || echo "    FAILED, see $OUT/logs/$m.log"; }
source $VENV/bin/activate
for m in AIDO tGPT scGPT scFoundation GeneCompass MaxToki Geneformer UCE C2S; do run $m; done
deactivate 2>/dev/null || true
source $TAHOE_VENV/bin/activate
run Tahoe
echo ALL_FIXED_DONE $(date '+%F %H:%M')
