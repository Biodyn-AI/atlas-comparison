#!/bin/bash
# SAE-seed control on the corrected activations (out_fixed): retrain with seeds 0, 1, 2 and rebuild the
# concept sets, first at the mid layer (seed_variance.py), then at every layer (seed_variance_alllayer.py).
# Both checkpoint after every unit and resume.
cd "$(dirname "$0")"                                  # run from pipeline/atlas_h100
VENV="${ATLAS_VENV:-../.venv}"; TAHOE_VENV="${ATLAS_TAHOE_VENV:-../.venv-tahoe}"   # Tahoe-x1 needs its own (torch 2.6, flash-attn)
[ -f env.sh ] && source env.sh; source $VENV/bin/activate
export CUDA_VISIBLE_DEVICES=0
M="AIDO C2S Geneformer MaxToki UCE scGPT tGPT scFoundation GeneCompass Tahoe"
python seed_variance.py --out_root out_fixed --genesets data/genesets --bg data/genesets/background.json \
  --models $M --seeds 0 1 2 --dst out_fixed/seed_variance.json --device cuda
echo MID_SEEDS_DONE $(date '+%F %H:%M')
python seed_variance_alllayer.py --out_root out_fixed --genesets data/genesets --bg data/genesets/background.json \
  --models $M --seeds 0 1 2 --dst out_fixed/seed_variance_alllayer.json --device cuda
echo ALL_SEEDS_DONE $(date '+%F %H:%M')
