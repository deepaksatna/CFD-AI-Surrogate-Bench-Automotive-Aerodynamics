#!/usr/bin/env python3
"""
Geometric-novelty cliff (Axis 2 variant that works on a single-body dataset).

For each TEST car, compute how geometrically novel it is vs the TRAINING set
(distance in shape-descriptor space to the nearest training car), then correlate
that novelty with the surrogate's drag error. The story: error grows with novelty
=> the model interpolates within the training manifold and degrades outside it.

Shape descriptor: coarse voxel-occupancy of the (unit-normalized) cached point
cloud — cheap, scale-robust, captures body shape. Runs locally from data/cache/*.npy
+ results/run_*.json (no GPU, no box).

    python novelty.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = 12  # voxel resolution per axis


def descriptor(pts):
    p = pts - pts.min(0)
    p = p / (p.max() + 1e-9)                      # -> unit cube
    idx = np.clip((p * (GRID - 1)).astype(int), 0, GRID - 1)
    vox = np.zeros((GRID, GRID, GRID), dtype=np.float32)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0    # binary occupancy
    return vox.ravel()


def load_descriptors(ids):
    cache = os.path.join(ROOT, "data", "cache")
    out = {}
    for did in ids:
        hits = glob.glob(os.path.join(cache, f"{did}_*.npy"))
        if hits:
            out[did] = descriptor(np.load(hits[0]))
    return out


def best_record():
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "run_*.json"))]
    recs = [r for r in recs if not r.get("smoke") and r.get("test_samples")
            and r.get("train_frac", 1.0) >= 1.0 and not r.get("holdout")]
    return min(recs, key=lambda r: r["test"]["mae_counts"]) if recs else None


def main():
    manifest = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    train_ids = [m["id"] for m in manifest if m["split"] == "train" and m["mesh"]]
    rec = best_record()
    if not rec:
        print("No full run-record with test_samples found. Run run_bench.py first."); return
    test_samples = rec["test_samples"]
    test_ids = [s["id"] for s in test_samples]

    desc = load_descriptors(train_ids + test_ids)
    train_d = np.stack([desc[i] for i in train_ids if i in desc])
    nov, err = [], []
    for s in test_samples:
        if s["id"] not in desc:
            continue
        d = desc[s["id"]]
        dist = np.sqrt(((train_d - d) ** 2).sum(1)).min()   # nearest train shape
        nov.append(float(dist)); err.append(abs(s["pred"] - s["true"]) * 1000)
    nov, err = np.array(nov), np.array(err)
    if len(nov) < 5:
        print("Too few test descriptors (cache missing?)."); return

    r = float(np.corrcoef(nov, err)[0, 1])
    # split at median novelty -> familiar vs novel halves
    med = np.median(nov)
    fam, new = err[nov <= med], err[nov > med]
    print(f"model={rec['surrogate']}  test cars={len(nov)}")
    print(f"corr(novelty, |error|) = {r:.3f}")
    print(f"MAE familiar half = {fam.mean():.2f} counts | novel half = {new.mean():.2f} counts "
          f"({new.mean()/max(fam.mean(),1e-9):.2f}x)")

    os.makedirs(os.path.join(ROOT, "results", "plots"), exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(nov, err, alpha=0.7)
    if len(nov) >= 8:                              # binned median trend
        bins = np.quantile(nov, np.linspace(0, 1, 5))
        cx, cy = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (nov >= a) & (nov <= b)
            if m.sum():
                cx.append(nov[m].mean()); cy.append(np.median(err[m]))
        plt.plot(cx, cy, "r-o", label="binned median")
        plt.legend()
    plt.xlabel("geometric novelty (distance to nearest training shape)")
    plt.ylabel("drag-count error")
    plt.title(f"Generalization vs novelty — {rec['surrogate']}  (corr={r:.2f})")
    plt.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "plots", "fig7_novelty_cliff.png")
    plt.savefig(out, dpi=150); plt.close()
    json.dump({"model": rec["surrogate"], "corr": r, "mae_familiar": float(fam.mean()),
               "mae_novel": float(new.mean()), "n": len(nov)},
              open(os.path.join(ROOT, "results", "novelty.json"), "w"), indent=2)
    print(f"✅ {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
