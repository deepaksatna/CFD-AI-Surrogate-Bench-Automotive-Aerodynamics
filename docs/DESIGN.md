# The CFD-AI Surrogate Benchmark — a design built to impress CFD, HPC, and AI at once

**Goal:** not a "my model got 95%" post — a *reproducible, multi-axis benchmark* that answers
the questions all three communities actually argue about, with several genuinely novel,
quotable results nobody has published together. Positioned as a community asset (a mini
"MLPerf for CFD surrogates"), not a one-off.

**Working title for the post:**
> *"How many CFD simulations does an AI surrogate actually need — and when does it lie?
> A reproducible CFD-AI benchmark on 8,000 cars."*

---

## Why each community will care (the three credibility bars)

| Community | What earns respect | What gets you dismissed |
|---|---|---|
| **CFD** | Field accuracy + force balance, validation vs *high-fidelity* (LES) not just RANS-on-RANS, honest generalization to unseen shapes/Reynolds, "this does NOT replace certification" | Reporting only a scalar Cd; RANS treated as truth; cherry-picked geometries |
| **HPC** | Scaling + utilization (MFU), GPU-hours **and energy (kWh/Joules)**, end-to-end cost vs the solver on the *same* hardware, reproducible at scale | Single-GPU toy; no energy; "1000× faster" with no apples-to-apples wall-time |
| **AI** | Fair architecture panel at equal compute, **data-scaling law**, OOD/calibration, uncertainty, released code+weights+splits, multiple seeds + error bars | Leaky splits; one seed; no baselines; unreleased weights |

The design below deliberately hits all three.

---

## The six axes (each is a quotable headline)

### 1. The data-efficiency scaling law — *"how many CFD runs do you actually need?"*
Train each surrogate on 250 / 500 / 1k / 2k / 4k / 8k CFD runs; plot test drag-count MAE vs
training-set size (log-log). Fit the exponent. **This is the single most valuable result for
industry** — CFD runs are the cost, and nobody publishes the curve cleanly. *Headline: "Past
~2,000 runs, accuracy barely improves — you're paying for simulations the model doesn't use."*

### 2. The generalization cliff — *"physics or interpolation?"*
Three held-out stress tests: (a) **body-type holdout** — train fastback+notchback, test estate;
(b) **geometric-novelty** — error vs distance-to-nearest-training-shape; (c) **Reynolds/coverage**
holdout. *Headline: "Every surrogate loses X drag-counts on a body it never saw — it learned the
dataset, not the Navier-Stokes."*

### 3. Uncertainty & active learning — *"does it know when it's wrong, and can that save runs?"*
Equip surrogates with uncertainty (deep ensemble / MC-dropout / evidential). Two results:
(a) **calibration** — does predicted uncertainty track real error? (b) **active learning loop** —
use surrogate uncertainty to pick the next CFD run; show you hit target accuracy with **3-5×
fewer simulations** than random sampling. *This is the killer CFD×AI story* — it turns the
surrogate from a gadget into a simulation-budget optimizer.

### 4. Energy & end-to-end cost on the same hardware — *the HPC clincher*
On the GPU: measure surrogate inference **Joules/design** (via `nvidia-smi`/NVML power
sampling) and a real OpenFOAM solve's GPU-hours + kWh on the *same node*. Project a full
**1,000-design study**: CFD = N GPU-days and M kWh; surrogate-screened = seconds and grams of
CO₂. Report training MFU / GPU-utilization too. *Headline: "A 1,000-car design sweep: 14 GPU-days
and ~300 kWh by CFD, vs 90 seconds and 2 Wh by the surrogate — here's the catch."*

### 5. Field-level physics, not a scalar — *the CFD clincher*
Predict the **surface pressure field**, integrate it to recover Cd, and check **force balance**
(integrated pressure+shear vs reported forces). Prove the model learned a flow field, not a
lookup. Report relative-L2 on pressure + a recovered-vs-direct Cd comparison (often the
integrated field gives a *better* Cd than a direct scalar head — a nice counter-intuitive result).

### 6. Cross-fidelity transfer — *the hardest, most respected axis*
Train on RANS (DrivAerNet++, 8k). Evaluate the best surrogate against **high-fidelity LES**
(DrivAerML, ~500 runs) it never trained on. Quantify the RANS→LES reality gap. *Headline:
"The surrogate matches RANS to 3 counts — but RANS itself is 12 counts off LES. The model is
only as honest as its teacher."* CFD experts will respect this more than any accuracy number.

---

## Architecture panel (AI rigor — same data, same compute budget)

| Model | Family | Role |
|---|---|---|
| PointNet | point MLP | cheap floor |
| RegDGCNN | dynamic graph CNN | published DrivAerNet baseline |
| MeshGraphNet (PhysicsNeMo) | mesh GNN | field prediction |
| **NVIDIA DoMINO (PhysicsNeMo)** | multiscale neural operator | NVIDIA product tie |
| Transolver / GINO | operator-learning transformer | current SOTA neural operator |

All trained under a **fixed compute budget** (equal GPU-hours) so the comparison is about
architecture, not who trained longest. ≥3 seeds, error bars on every number.

## Methodology rigor (the non-negotiables)
- Official DrivAerNet++ splits, **leak-free** (no shared parametric morph across splits).
- Pre-registered (`BENCHMARK_PLAN.md` committed before training); corrections welcomed as issues.
- Drag reported in **counts** (1 count = 0.001 Cd) — the unit engineers decide on.
- Released: code, trained weights, raw run-records (shared JSON schema), and a **leaderboard
  format so others can submit their surrogate**. This is what makes it a benchmark, not a blog.
- Hardware fingerprint + energy logs committed alongside results.

---

## Deliverables
1. **The post / newsletter** — the 6 headlines above, each with one plot.
2. **A public repo** = portable harness + weights + leaderboard (the durable asset).
3. **An interactive leaderboard table** ("submit a PR with your run-record").
4. Optional: a short methods note / arXiv-style writeup for credibility with the CFD/HPC crowd.

## Staged delivery (so we ship, not stall)
| Stage | Scope | Why |
|---|---|---|
| **MVP post (2-3 wks)** | Axes 1 (scaling law) + 2 (generalization cliff) + 4 (energy/cost) on 3 architectures | Already supported by the harness; each is a standalone headline; impressive on its own |
| **v2 (depth)** | Add Axis 5 (fields + force balance) + DoMINO/Transolver | The CFD-credibility upgrade |
| **v3 (the standard)** | Axis 3 (UQ + active learning) + Axis 6 (LES cross-fidelity) + public leaderboard | This is the version that gets cited |

## Headline findings we're chasing (hypotheses — confirm or refute, publish either way)
- Data-scaling saturates surprisingly early (~2k runs) → industry can cut simulation budgets.
- The generalization cliff is real and large → surrogates are screening tools, not oracles.
- Active learning cuts required CFD runs 3-5× → the real ROI story.
- Integrated-field Cd beats direct-scalar Cd → "predict physics, then reduce."
- The RANS→LES gap dwarfs the surrogate's RANS error → honesty that earns CFD trust.

## What we will NOT claim
Surrogates replace CFD for certification/homologation (they don't); numbers generalize beyond
steady RANS on DrivAer-family passenger cars (they don't); "faster" implies "correct" (the
accuracy envelope + failure modes are the whole point).
