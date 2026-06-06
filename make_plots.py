#!/usr/bin/env python3
"""
Build the publication plots from results/run_*.json run-records.

    python make_plots.py

Produces (results/plots/):
  fig1_pareto.png        speed (ms/car) vs accuracy (drag-count MAE), bubble = train cost
  fig2_accuracy_bar.png  test MAE in drag counts per surrogate (+ R^2)
  fig3_error_cdf.png     per-car |error| CDF per surrogate (the 'where it breaks' tail)
  fig4_worst10.png       worst-10 cars for the best surrogate (failure gallery)
Skips --smoke runs. Uses latest run per surrogate.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(ROOT, "results", "plots")


def load_all():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "run_*.json"))):
        r = json.load(open(f))
        if not r.get("smoke"):
            out.append(r)
    return out


def _is_full(r):
    return (r.get("train_frac", 1.0) >= 1.0) and not r.get("holdout")


def load_latest():
    """Latest CANONICAL (full-data, no-holdout) run per surrogate, for the main comparison."""
    recs = {}
    for r in load_all():
        if _is_full(r):
            recs[r["surrogate"]] = r
    return recs


def fig_pareto(recs):
    plt.figure(figsize=(7, 5))
    for name, r in recs.items():
        x = r["latency"]["ms_per_car_bs1"]
        y = r["test"]["mae_counts"]
        s = max(r.get("train_minutes", 1), 1) * 6
        plt.scatter(x, y, s=s, alpha=0.6)
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=10)
    plt.xscale("log")
    plt.xlabel("inference latency  (ms / car, batch=1)  →  faster")
    plt.ylabel("drag-count MAE  (lower = better)")
    plt.title("Aero surrogate bake-off — speed vs accuracy\n(bubble size = training minutes)")
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fig1_pareto.png"), dpi=150); plt.close()


def fig_accuracy_bar(recs):
    names = list(recs); mae = [recs[n]["test"]["mae_counts"] for n in names]
    r2 = [recs[n]["test"]["r2"] for n in names]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, mae)
    for b, m, q in zip(bars, mae, r2):
        plt.text(b.get_x() + b.get_width() / 2, m, f"{m:.1f}\nR²={q:.3f}",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("test drag-count MAE")
    plt.title("Drag prediction accuracy by surrogate")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fig2_accuracy_bar.png"), dpi=150); plt.close()


def fig_error_cdf(recs):
    plt.figure(figsize=(7, 5))
    for name, r in recs.items():
        s = r.get("test_samples") or []
        if not s:
            continue
        err = np.sort(np.abs([x["pred"] - x["true"] for x in s])) * 1000
        cdf = np.arange(1, len(err) + 1) / len(err)
        plt.plot(err, cdf, label=name)
    plt.xlabel("absolute drag-count error"); plt.ylabel("fraction of cars ≤ x")
    plt.title("Per-car error CDF — the tail is where the surrogate lies")
    plt.grid(True, ls=":", alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fig3_error_cdf.png"), dpi=150); plt.close()


def fig_worst10(recs):
    # pick the best surrogate by MAE that has per-sample data
    cand = [(n, r) for n, r in recs.items() if r.get("test_samples")]
    if not cand:
        return
    name, r = min(cand, key=lambda kv: kv[1]["test"]["mae_counts"])
    s = sorted(r["test_samples"], key=lambda x: abs(x["pred"] - x["true"]), reverse=True)[:10]
    ids = [x["id"][:24] for x in s]
    err = [abs(x["pred"] - x["true"]) * 1000 for x in s]
    plt.figure(figsize=(7, 5))
    plt.barh(range(len(ids)), err)
    plt.yticks(range(len(ids)), ids, fontsize=8); plt.gca().invert_yaxis()
    plt.xlabel("absolute drag-count error")
    plt.title(f"Worst-10 cars for {name} — failure gallery")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fig4_worst10.png"), dpi=150); plt.close()


def fig_scaling_law(allrecs):
    """Axis 1: test MAE vs #training CFD runs, per surrogate (full-data lineage, no holdout)."""
    from collections import defaultdict
    series = defaultdict(list)
    for r in allrecs:
        if r.get("holdout"):
            continue
        series[r["surrogate"]].append((r["data"]["train"], r["test"]["mae_counts"]))
    series = {k: sorted(v) for k, v in series.items() if len(v) >= 2}
    if not series:
        return False
    plt.figure(figsize=(7, 5))
    for name, pts in series.items():
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=name)
    plt.xscale("log")
    plt.xlabel("# training CFD runs"); plt.ylabel("test drag-count MAE")
    plt.title("Data-efficiency scaling law — how many CFD runs do you actually need?")
    plt.grid(True, which="both", ls=":", alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "fig5_scaling_law.png"), dpi=150); plt.close()
    return True


def fig_cliff(allrecs):
    """Axis 2: full-data test MAE vs held-out-body-type MAE per surrogate (the cliff)."""
    full = {r["surrogate"]: r["test"]["mae_counts"] for r in allrecs if _is_full(r)}
    held = {r["surrogate"]: r["test"]["mae_counts"] for r in allrecs if r.get("holdout")}
    names = [n for n in full if n in held]
    if not names:
        return False
    import numpy as _np
    x = _np.arange(len(names)); w = 0.38
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w / 2, [full[n] for n in names], w, label="in-distribution")
    plt.bar(x + w / 2, [held[n] for n in names], w, label="held-out body type")
    plt.xticks(x, names); plt.ylabel("test drag-count MAE")
    plt.title("The generalization cliff — physics, or interpolation?")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "fig6_generalization_cliff.png"), dpi=150); plt.close()
    return True


def main():
    os.makedirs(PLOTS, exist_ok=True)
    allrecs = load_all()
    recs = load_latest()
    if not allrecs:
        print("No non-smoke run-records in results/. Run run_bench.py first."); return
    made = []
    if recs:
        print("canonical surrogates:", ", ".join(recs))
        fig_pareto(recs); fig_accuracy_bar(recs); fig_error_cdf(recs); fig_worst10(recs)
        made += ["fig1_pareto", "fig2_accuracy_bar", "fig3_error_cdf", "fig4_worst10"]
    if fig_scaling_law(allrecs): made.append("fig5_scaling_law")
    if fig_cliff(allrecs): made.append("fig6_generalization_cliff")
    print(f"✅ plots -> {os.path.relpath(PLOTS, ROOT)}/  ({', '.join(made) or 'none — need more runs'})")


if __name__ == "__main__":
    main()
