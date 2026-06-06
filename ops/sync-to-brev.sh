#!/usr/bin/env bash
# Sync this repo from the laptop to the Brev instance (into the remote $HOME).
# Excludes data caches, checkpoints, results — those stay on the box / sync back separately.
# Idempotent — rsync transfers diffs only.   Usage: bash brev/sync-to-brev.sh <instance-name>

set -euo pipefail

INSTANCE="${1:-${INSTANCE:-}}"
[ -n "$INSTANCE" ] || { echo "Usage: $0 <instance-name>  (or set INSTANCE=)"; exit 1; }

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"
DST="${INSTANCE}:aero-cfd-surrogate-bakeoff/"   # relative path → remote $HOME, backend-agnostic

echo "→ Refreshing Brev SSH config…"
brev refresh

echo "→ Syncing $SRC → ${DST}…"
rsync -avh --progress \
    --exclude='__pycache__/' --exclude='.git/' --exclude='*.pyc' --exclude='.DS_Store' \
    --exclude='data/cache/' --exclude='results/' --exclude='*.pt' --exclude='*.vtu' --exclude='*.foam' \
    "$SRC" "$DST"

echo "→ Done. Next:"
echo "   brev exec $INSTANCE \"bash -s\" < brev/setup-on-brev.sh"
