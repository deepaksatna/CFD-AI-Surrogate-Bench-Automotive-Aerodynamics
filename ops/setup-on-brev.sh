#!/usr/bin/env bash
# Idempotent env setup on the Brev instance for the Aero CFD Surrogate Bake-Off.
#   Run:  brev exec <instance> "bash -s" < brev/setup-on-brev.sh
# Core stack (Phase 1) is required; PhysicsNeMo/DoMINO (Phase 2+) is best-effort.
set -e

REPO="$HOME/aero-cfd-surrogate-bakeoff"
echo "→ Repo: $REPO"
mkdir -p "$REPO/results" "$REPO/data/cache"

# This Brev image uses an externally-managed system Python (PEP 668). On a disposable
# GPU box we install system-wide with --break-system-packages (simpler than a venv for
# the multi-step exec workflow).
PIP="python3 -m pip install --break-system-packages --quiet"

echo "→ [1/4] Core ML + geometry stack ..."
# torch is usually preinstalled on Brev GPU images; only install if missing.
python3 -c "import torch" 2>/dev/null || $PIP torch torchvision
$PIP numpy scipy pandas pyarrow h5py tqdm pyyaml matplotlib \
    trimesh pyvista "huggingface_hub>=0.23" datasets scikit-learn nvidia-ml-py

echo "→ [2/4] PyTorch Geometric (mesh GNNs) ..."
$PIP torch-geometric || echo "  (torch-geometric install warned; pure ops still work)"

echo "→ [3/4] NVIDIA PhysicsNeMo (DoMINO — Phase 2+, best-effort) ..."
$PIP nvidia-physicsnemo 2>/dev/null \
    && echo "  PhysicsNeMo installed." \
    || echo "  PhysicsNeMo skipped (install later when we reach DoMINO; Phase 1 doesn't need it)."

echo "→ [4/4] Verify GPU + torch ..."
python3 - <<'PYEOF'
import torch, platform
print("python     :", platform.python_version())
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}")
PYEOF

echo "✅ setup complete. Next: python data/get_drivaernet.py --subset cd_pointcloud"
