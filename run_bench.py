#!/usr/bin/env python3
"""
Train + evaluate one surrogate on scalar drag (Cd) and write a run-record JSON.

    python run_bench.py --surrogate regdgcnn --task cd
    python run_bench.py --surrogate regdgcnn --task cd --smoke   # 2 epochs, 1k pts, sanity check

Metrics reported (the newsletter numbers):
  - Cd MAE in DRAG COUNTS (1 count = 0.001 Cd), R^2, max abs error (counts)
  - inference latency: ms / car (batch=1 and batched)
Writes results/run_<surrogate>_<task>_<ts>.json with config + hardware + per-epoch curve.
"""
import argparse, json, os, platform, time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, ROOT)
from data.dataset import DrivAerPointClouds, cd_stats
from surrogates import regdgcnn, pointnet

BUILDERS = {"regdgcnn": regdgcnn.build, "pointnet": pointnet.build}


def metrics_counts(pred_cd, true_cd):
    pred_cd, true_cd = np.asarray(pred_cd), np.asarray(true_cd)
    err = pred_cd - true_cd
    mae_counts = float(np.abs(err).mean() * 1000)
    max_counts = float(np.abs(err).max() * 1000)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true_cd - true_cd.mean()) ** 2).sum()) + 1e-12
    return {"mae_counts": mae_counts, "max_counts": max_counts, "r2": 1 - ss_res / ss_tot}


@torch.no_grad()
def evaluate(model, loader, device, cd_mean, cd_std):
    """Returns (metrics, preds, trues). Loader must be unshuffled for preds to align with items."""
    model.eval()
    preds, trues = [], []
    for pts, _, cd_true in loader:
        out = model(pts.to(device)).cpu().numpy() * cd_std + cd_mean
        preds.extend(out.tolist()); trues.extend(cd_true.numpy().tolist())
    return metrics_counts(preds, trues), preds, trues


def measure_latency(model, n_points, device):
    model.eval()
    out = {}
    for bs in (1, 32):
        x = torch.randn(bs, n_points, 3, device=device)
        with torch.no_grad():
            for _ in range(3): model(x)                      # warm
            if device.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10): model(x)
            if device.type == "cuda": torch.cuda.synchronize()
        out[f"ms_per_car_bs{bs}"] = (time.perf_counter() - t) / 10 / bs * 1000
    return out


def measure_energy(model, n_points, device, seconds=3.0):
    """Joules per inference via NVML power sampling — the HPC energy axis. Best-effort."""
    if device.type != "cuda":
        return {}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
    except Exception as e:
        return {"energy_note": f"pynvml unavailable ({e}); pip install nvidia-ml-py"}
    model.eval()
    x = torch.randn(1, n_points, 3, device=device)
    with torch.no_grad():
        for _ in range(3): model(x)
        torch.cuda.synchronize()
        powers, n, t0 = [], 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            model(x); n += 1
            if n % 20 == 0:
                powers.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)  # W
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    avg_w = float(np.mean(powers)) if powers else 0.0
    return {"avg_power_w": avg_w, "joules_per_car_bs1": avg_w * elapsed / max(n, 1),
            "inferences_per_sec": n / elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="regdgcnn", choices=list(BUILDERS))
    ap.add_argument("--task", default="cd", choices=["cd"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=1.0, help="data-scaling law: fraction of train set")
    ap.add_argument("--holdout", default=None, help="generalization cliff: hold out a body type, e.g. E (estate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dcfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "dataset.yaml")))
    scfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "surrogates.yaml")))[args.surrogate]
    device = torch.device(args.device)
    n_points = 1000 if args.smoke else dcfg["n_points"]
    epochs = 2 if args.smoke else (args.epochs or scfg["epochs"])
    bs = 8 if args.smoke else scfg["batch_size"]

    manifest = os.path.join(ROOT, "data", "manifest.json")
    mesh_dir = os.path.join(ROOT, dcfg["mesh_dir"]); cache = os.path.join(ROOT, dcfg["cache_dir"])
    cd_mean, cd_std = cd_stats(manifest)
    print(f"Cd train stats: mean={cd_mean:.4f} std={cd_std:.4f}")

    def mk(split, shuffle):
        ds = DrivAerPointClouds(manifest, mesh_dir, cache, split, n_points,
                                dcfg["normalize"], cd_mean, cd_std,
                                holdout_bodytype=args.holdout, train_frac=args.train_frac, seed=args.seed)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, drop_last=shuffle), len(ds)

    train_dl, n_tr = mk("train", True); val_dl, n_va = mk("val", False); test_dl, n_te = mk("test", False)
    print(f"data: train {n_tr} / val {n_va} / test {n_te}  | n_points={n_points} epochs={epochs} bs={bs}")

    model = BUILDERS[args.surrogate](scfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = torch.nn.SmoothL1Loss()

    best_val, best_state, curve, t0 = 1e9, None, [], time.time()
    for ep in range(epochs):
        model.train(); running = 0.0
        for pts, cd_norm, _ in train_dl:
            opt.zero_grad()
            loss = lossfn(model(pts.to(device)), cd_norm.to(device))
            loss.backward(); opt.step(); running += loss.item()
        sched.step()
        val, _, _ = evaluate(model, val_dl, device, cd_mean, cd_std)
        curve.append({"epoch": ep, "train_loss": running / max(len(train_dl), 1), **val})
        print(f"  ep {ep:3d}  train_loss {curve[-1]['train_loss']:.4f}  "
              f"val_MAE {val['mae_counts']:.2f} counts  R2 {val['r2']:.3f}")
        if val["mae_counts"] < best_val:
            best_val = val["mae_counts"]; best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    test, test_preds, test_trues = evaluate(model, test_dl, device, cd_mean, cd_std)
    test_ids = [it["id"] for it in test_dl.dataset.items]   # aligns: loader unshuffled, drop_last off
    test_samples = [{"id": i, "true": t, "pred": p}
                    for i, t, p in zip(test_ids, test_trues, test_preds)]
    lat = measure_latency(model, n_points, device)
    energy = measure_energy(model, n_points, device)
    train_min = (time.time() - t0) / 60

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    rec = {
        "surrogate": args.surrogate, "task": args.task, "smoke": args.smoke,
        "params": n_params, "n_points": n_points, "epochs": epochs, "batch_size": bs,
        "train_frac": args.train_frac, "holdout": args.holdout, "seed": args.seed,
        "data": {"train": n_tr, "val": n_va, "test": n_te},
        "best_val_mae_counts": best_val, "test": test, "latency": lat, "energy": energy,
        "test_samples": test_samples, "train_minutes": train_min, "config": scfg,
        "hardware": {"gpu": gpu, "python": platform.python_version(), "torch": torch.__version__},
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"run_{args.surrogate}_{args.task}_{int(t0)}.json")
    json.dump(rec, open(out, "w"), indent=2)
    print(f"\n=== TEST: MAE {test['mae_counts']:.2f} drag counts | R2 {test['r2']:.3f} | "
          f"max {test['max_counts']:.1f} counts | {lat['ms_per_car_bs1']:.2f} ms/car ===")
    print(f"train {train_min:.1f} min on {gpu} | run-record -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
