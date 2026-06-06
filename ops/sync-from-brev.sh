#!/usr/bin/env bash
# Pull results (run-records, plots, logs) FROM the Brev box to the laptop, to preserve them
# before stopping/deleting the ephemeral instance. Excludes the huge meshes/cache.
#   Usage: bash brev/sync-from-brev.sh <instance-name>
set -euo pipefail

INSTANCE="${1:-${INSTANCE:-}}"
[ -n "$INSTANCE" ] || { echo "Usage: $0 <instance-name>"; exit 1; }
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "→ Refreshing Brev SSH config…"
brev refresh

echo "→ Pulling results/ from ${INSTANCE}…"
mkdir -p "$LOCAL/results"
rsync -avh --progress \
    --exclude='__pycache__/' \
    "${INSTANCE}:aero-cfd-surrogate-bakeoff/results/" "$LOCAL/results/"
# also grab the manifest + the small cached point clouds (.npy) so novelty/plots run locally
rsync -avh "${INSTANCE}:aero-cfd-surrogate-bakeoff/data/manifest.json" "$LOCAL/data/" 2>/dev/null || true
mkdir -p "$LOCAL/data/cache"
rsync -avh --include='*.npy' --exclude='*' \
    "${INSTANCE}:aero-cfd-surrogate-bakeoff/data/cache/" "$LOCAL/data/cache/" 2>/dev/null || true

echo "→ Local results:"
ls -la "$LOCAL/results/" 2>/dev/null
ls -la "$LOCAL/results/plots/" 2>/dev/null || true
