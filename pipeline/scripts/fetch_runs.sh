#!/usr/bin/env bash
# Download the per-run outputs and external inputs from the GitHub release and unpack them
# into outputs/atlas/ of this checkout, so the analysis scripts can be re-run with
# ATLAS_BASE pointing at the repository root.
#
#   bash pipeline/scripts/fetch_runs.sh            # everything except the legacy run
#   bash pipeline/scripts/fetch_runs.sh --legacy   # also the superseded first cluster run
#
# Archives, sizes and what they hold are listed in docs/REPRODUCE.md.
set -euo pipefail
REPO="${ATLAS_RELEASE_REPO:-Biodyn-AI/atlas-comparison}"
TAG="${ATLAS_RELEASE_TAG:-runs-v2}"          # v2: catalogues of the corrected extraction
LEGACY_TAG=runs-v1                              # the superseded first cluster run lives only in v1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/outputs/atlas"
BASE="https://github.com/$REPO/releases/download/$TAG"
FILES=(runs-ts3_out.tar.gz runs-alllayer_cat.tar.gz runs-heldout_cat.tar.gz runs-seeds.tar.gz inputs-genesets.tar.gz)

mkdir -p "$DEST"
cd "$DEST"
curl -sSL -o SHA256SUMS "$BASE/SHA256SUMS"
for f in "${FILES[@]}"; do
  echo "== $f"
  curl -sSL -o "$f" "$BASE/$f"
  grep " $f\$" SHA256SUMS | shasum -a 256 -c -
  tar xzf "$f" && rm "$f"
done
if [[ "${1:-}" == "--legacy" ]]; then
  echo "== legacy-cluster-run.tar.gz (from $LEGACY_TAG)"
  L="https://github.com/$REPO/releases/download/$LEGACY_TAG"
  curl -sSL -o SHA256SUMS.legacy "$L/SHA256SUMS"
  curl -sSL -o legacy-cluster-run.tar.gz "$L/legacy-cluster-run.tar.gz"
  grep " legacy-cluster-run.tar.gz\$" SHA256SUMS.legacy | shasum -a 256 -c -
  tar xzf legacy-cluster-run.tar.gz && rm legacy-cluster-run.tar.gz SHA256SUMS.legacy
fi
echo "unpacked into $DEST:"; ls "$DEST"
