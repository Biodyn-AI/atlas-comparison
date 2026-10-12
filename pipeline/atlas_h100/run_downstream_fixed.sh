#!/bin/bash
# Every GPU-side analysis that reads the SAEs, their activations or their catalogues, recomputed on the
# corrected run (out_fixed). Steps run in order; a failed step is logged and the next one still runs.
cd "$(dirname "$0")"                                  # run from pipeline/atlas_h100
VENV="${ATLAS_VENV:-../.venv}"; TAHOE_VENV="${ATLAS_TAHOE_VENV:-../.venv-tahoe}"   # Tahoe-x1 needs its own (torch 2.6, flash-attn)
[ -f env.sh ] && source env.sh; export CUDA_VISIBLE_DEVICES=0
OUT=out_fixed; CORP=external/perturb/tabula_3tissue_6k.h5ad
M="AIDO tGPT scGPT scFoundation GeneCompass MaxToki Geneformer UCE C2S Tahoe"
mkdir -p $OUT/logs
step(){ n=$1; shift; [ -f $OUT/logs/$n.ok ] && { echo "skip $n"; return; }
  echo "=== $n $(date '+%F %H:%M') ==="
  "$@" > $OUT/logs/$n.log 2>&1 && { touch $OUT/logs/$n.ok; echo "    ok $(date '+%H:%M')"; } || echo "    FAILED, see $OUT/logs/$n.log"; }
source $VENV/bin/activate
step annotations python -c "from common.annotate import annotate_all; import glob
for m in '$M'.split(): annotate_all(f'$OUT/{m}/feature_catalog_L*.json', 'data/genesets', '$OUT/annotations', 'data/genesets/background.json')"
for m in $M; do
  [ $m = Tahoe ] && { deactivate; source $TAHOE_VENV/bin/activate; }
  bs=16; [ $m = AIDO ] && bs=2                 # AIDO reads all 19,264 genes: its attention mask needs a small batch
  step cell_cka_$m python cell_cka.py --model $m --corpus $CORP --out $OUT --all-layers --batch-size $bs
  [ $m = Tahoe ] && { deactivate; source $VENV/bin/activate; }
done
step cell_cka_matrix python cell_cka.py --cka --out $OUT
step nonlinearity python nonlinearity.py --all --layers all --out $OUT --corpus $CORP
step tissue python tissue_from_emb.py --all --out $OUT --corpus $CORP
step celltype_difficulty python celltype_difficulty.py --out $OUT --corpus $CORP
step hardcell python hardcell_agreement.py --out $OUT --corpus $CORP
step cka_layers python cka_layers.py --all --out $OUT --sample 30000
step cka_svd_null env ATLAS_OUT=$OUT python cka_svd_null.py
step svd_projection_k env ATLAS_OUT=$OUT python svd_projection_k.py
step flow python flow_alllayers.py --out $OUT
step depth python depth_profile.py --all --out $OUT
step circuits_adjacent python circuits_adjacent.py --out $OUT
for m in $M; do step explorer_$m python layer_explorer.py --model $m --out $OUT; done
step tsne python tsne_coords.py --all --out $OUT
step coact_embed python coact_embed.py --all --out $OUT
step modules python modules_alllayers.py --all --out $OUT
step explorer_slim python build_explorer_slim.py --out $OUT --cap 500
echo DOWNSTREAM_DONE $(date '+%F %H:%M')
