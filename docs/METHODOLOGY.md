# Beyond the Model — Aero Neural-CFD Surrogate Bake-Off

**Pre-registered methodology. Committed before any model is trained.**
Date: 2026-06-06 · Hardware: NVIDIA Brev (B200 / H100 class) · Domain: HPC + AI + Automotive

> **The question:** every automaker spends millions of GPU-hours on CFD to predict a car's
> drag. Neural surrogates promise the same number in milliseconds. We run the leading
> surrogate architectures on the *same* public car dataset, against the *same* CFD ground
> truth, and report the honest speed / accuracy / cost frontier — and where the surrogate
> breaks. Same discipline as the KV-cache and MoE bake-offs: one workload, the architecture
> is the variable, raw data + portable framework in the repo.

---

## 1. Dataset (the ground truth)

| Dataset | Role | What it gives |
|---|---|---|
| **DrivAerNet++** (Elrefaie et al., MIT) | Primary train/val/test | ~8,000 parametric DrivAer cars (fastback / notchback / estate), each with steady RANS (k-ω SST) OpenFOAM CFD: drag Cd, lift Cl, **surface pressure + wall-shear fields**, STL geometry, ~0.5M-cell meshes |
| **DrivAerML** (CAE consortium / NVIDIA-affiliated) | High-fidelity OOD test | 500 hybrid RANS-LES runs — used ONLY as an out-of-distribution / higher-fidelity stress test, never trained on |

Official DrivAerNet++ splits are used verbatim (no re-splitting). Geometry is consumed as the
model's input modality requires (point cloud, surface mesh, or SDF voxel grid) at a fixed
resolution per family, all reported.

## 2. Tasks

- **Task A — Scalar drag regression.** Predict Cd from geometry. Headline metric.
- **Task B — Surface pressure field.** Predict the per-face pressure field on the body
  (the engineering deliverable — drag is an integral of it).

## 3. Surrogates under test (the variable)

| # | Model | Family | Why it's in |
|---|---|---|---|
| 1 | **RegDGCNN** | point-cloud (DrivAerNet paper baseline) | The published baseline — anchors "did we beat the paper" |
| 2 | **PointNet++ / Point-Transformer** | point-cloud | Cheap, fast-to-train reference |
| 3 | **MeshGraphNet (X-MeshGraphNet, PhysicsNeMo)** | mesh GNN | Field prediction on the real mesh |
| 4 | **NVIDIA DoMINO (PhysicsNeMo)** | decomposable multiscale neural operator | The NVIDIA product tie — purpose-built for automotive aero |
| 5 | **Transolver / Geometry-FNO** *(stretch)* | operator-learning transformer | Current SOTA neural operator; included if time allows |

Classical baseline: **OpenFOAM steady RANS** — we report the dataset's published per-run solve
cost AND run **3 cases ourselves on the Brev box** to anchor a real wall-time on our hardware
(credibility: "a CFD run on the same GPU node we trained on takes N hours").

## 4. Metrics (reported for every model)

- **Accuracy (Task A):** Cd MAE in **drag counts** (1 count = 0.001 Cd), R², max abs error, error CDF.
- **Accuracy (Task B):** surface-pressure MAE + relative L2 error; drag recovered by integrating the predicted field vs true Cd.
- **Latency:** ms/car at batch=1 and batched (the inference speedup story).
- **Training cost:** GPU-hours + $ at Brev rates; samples-to-converge.
- **Speedup vs CFD:** surrogate inference vs measured OpenFOAM wall-time → the headline multiplier.
- **Where it breaks (the honest part):** (a) OOD body type (train fastback+notchback → test estate); (b) error vs geometric novelty (distance to nearest train shape); (c) DrivAerML high-fidelity transfer; (d) worst-10 geometries gallery.

## 5. Hardware & protocol

- **Brev shape:** 1× B200 192GB (or 1× H100 80GB) for MVP; multi-GPU noted for full-field DoMINO training. Big VRAM is justified — mesh GNN/operator training on 0.5M-point geometries is memory-bound.
- Fixed seeds; identical train/val/test; same input resolution within a family; matplotlib+numpy plots only; run-recorder JSON schema shared with the other issues (enables a future meta-issue).
- Pre-registered: this file is committed before training. Corrections welcome as repo issues.

## 6. Success criteria for the edition

Ship when we have: a **speed/accuracy/cost Pareto** across ≥3 surrogates + the CFD anchor, AND
at least one non-obvious finding. Hypothesized candidates (to confirm or refute):
- "DoMINO wins surface-pressure fields but RegDGCNN gets scalar Cd within X counts at ~10× lower training cost."
- "Every surrogate degrades sharply on the estate body when trained only on fastback/notchback — the surrogate learned the dataset, not the physics."
- "Integrating the predicted pressure field gives a *better* Cd than the direct scalar head."

## 7. What we will NOT claim

- Surrogates replace CFD for certification / homologation. They don't — this is design-space exploration / early screening.
- These numbers generalize to all geometries, Reynolds numbers, or transient/aeroacoustic cases. They cover steady RANS on DrivAer-family passenger cars.
- A surrogate "beating" CFD on speed means it's right — accuracy envelope and failure modes are reported precisely so readers know when to trust it.

## 8. LinkedIn framing (working hook)

> Three lines before "see more":
> A CFD run to get one car's drag: **~6 GPU-hours**. A neural surrogate: **~0.1 seconds**.
> We trained 4 surrogates on 8,000 CFD'd cars. Best one predicts drag within **X counts** —
> and collapses completely on a body shape it never saw. Here's the data and the framework.

Repo discipline mirrors `kv-cache-bakeoff` and `MOEB`: pre-registered methodology, raw data,
reproducible scripts, portable framework, publication-grade plots.
