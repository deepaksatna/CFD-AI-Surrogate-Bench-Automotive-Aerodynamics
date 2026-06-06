# CFD‑AI Surrogate Bench — Leaderboard

Drag‑coefficient prediction on **DrivAerML** (high‑fidelity HRLES DrivAer cars). Metric: test
**drag‑count MAE** (1 count = 0.001 Cd), lower is better. Submit a PR adding your run‑record
(`results/run_*.json`) and a row here.

## Task A — Scalar drag (Cd), full data (197 train / 25 test)

| Rank | Model | Cd MAE (counts) ↓ | R² ↑ | ms/car ↓ | J/car ↓ | params | by |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | **PointNet** | **4.21** | 0.881 | 0.65 | 0.21 | 1.0 M | baseline |
| 2 | RegDGCNN | 7.16 | 0.683 | 4.67 | 2.63 | 1.8 M | baseline |
| — | *your model* | — | — | — | — | — | PR |

## Data‑efficiency (RegDGCNN, scalar Cd)

| train cars | 20 | 49 | 98 | 148 | 197 |
|---|---:|---:|---:|---:|---:|
| Cd MAE (counts) | 12.4 | 11.5 | 9.9 | 8.9 | 7.2 |
| R² | −0.01 | 0.29 | 0.42 | 0.56 | 0.68 |

## Task B — Surface‑pressure field + drag‑by‑integration

| Model | field rel‑L2 ↓ | Cd‑from‑pred‑field MAE (counts) | train cars | notes |
|---|---:|---:|---:|---|
| PointNet‑Cp (v2) | ~0.55 (0.97→0.61 by ep10, still ↓) | **n/a — negative result** | 88 | the field model learns; **drag‑by‑integration does NOT recover Cd** (see below) |
| NVIDIA DoMINO | — | — | — | planned (PhysicsNeMo) |

> **Negative result (documented):** integrating the predicted surface‑pressure field to recover drag
> does **not** work in this setup — even integrating the **ground‑truth** Cp field misses the reported
> Cd by **~530 counts**. Causes: non‑watertight surface (so ∮ area·normal doesn't cancel), pressure
> convention, and pressure‑only integration ignoring friction drag. **Lesson: for drag, regress the
> scalar directly (Task A, ~4 counts); use the field for visualization/diagnostics, not force recovery.**

## Robustness

| Test | result |
|---|---|
| Geometric‑novelty cliff | novel‑half MAE **1.3×** familiar‑half (corr 0.37) |
| Body‑type holdout (fastback→estate) | pending DrivAerNet++ (multi‑body) |
| RANS→LES cross‑fidelity | pending DrivAerNet++ |

## CFD reference cost
One DrivAerML HRLES case ≈ **61,440 CPU core‑hours** (160 M cells, 1,536 cores × ~40 h);
full 500‑case dataset ≈ 30.7 M core‑hours. Surrogate inference: **0.65 ms / 0.21 J**.

## How to submit
1. Run the harness (any GPU): `python run_bench.py --surrogate <yours> --task cd`.
2. It writes `results/run_<model>_cd_<ts>.json` (schema in `run_bench.py`).
3. Open a PR with that file + your row above. CI validates the schema and recomputes the metric.
