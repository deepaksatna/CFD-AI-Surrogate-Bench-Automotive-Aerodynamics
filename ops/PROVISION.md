# Provisioning the Brev box for the Aero CFD Bake-Off

## Recommended shape

| Phase | Shape | Why |
|---|---|---|
| **Phase 1 (scalar Cd, point clouds)** | 1× H100 80GB or A100 80GB | RegDGCNN/PointNet on point clouds is light — even A100 is plenty. Cheapest path to a first number. |
| **Phase 2+ (DoMINO / MeshGraphNet, surface fields)** | **1× B200 192GB** (or 2× H100) | 0.5M-point mesh GNN / operator training is memory-bound. B200's 192GB avoids subsampling and lets DoMINO train at full resolution. |

Since GPU isn't a constraint, provisioning **1× B200 192GB** up front avoids re-provisioning
between phases. Pick the cheapest backend offering it in `brev` (MassedCompute / Crusoe /
Lambda typically).

## Spin it up (mirrors the DPMM-Nano flow)

```bash
# 1. Auth (token from https://brev.nvidia.com → Copy CLI Token; ~15-min refresh window)
brev login --token "<JWT>"
brev set Oracle-Brev
brev healthcheck                 # → Healthy!

# 2. Create the instance from the Brev console (pick B200 192GB), then:
brev ls                          # confirm RUNNING, note the <instance-name>

# 3. Open a shell to verify the GPU
brev shell <instance-name>
nvidia-smi                       # confirm B200 / driver / CUDA
exit
```

## Then, from the laptop

```bash
bash brev/sync-to-brev.sh <instance-name>
brev exec <instance-name> "bash -s" < brev/setup-on-brev.sh
```

## Cost discipline

- B200 is ~$3-6/hr depending on backend. **Stop the instance when not training** (`brev stop <name>`); Brev bills by the hour.
- Checkpoints + run-records sync back to the laptop after each phase so a stopped/deleted box loses nothing.
- Token gotcha (bit us in DPMM-Nano): if `brev` errors with `EOF` / `malformed refresh token`, just `brev login --token` again.
