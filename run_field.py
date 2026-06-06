#!/usr/bin/env python3
"""
Phase 2 (v2): train per-cell surface-pressure (Cp) surrogate, then recover drag by
EXACT integration over the full mesh (true per-cell area + outward normal).

Reports three honest numbers:
  - surface-pressure field relative-L2 (prediction quality)
  - Cd-by-integration MAE in counts (drag recovered from the PREDICTED field)
  - sanity: Cd from integrating the TRUE field vs reported Cd (validates the method;
    a gap here = pressure-only integration missing friction drag, not a code bug)

    python run_field.py --smoke
    python run_field.py
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, ROOT)
from data.field_dataset import DrivAerFields, A_REF
from surrogates import pointnet_field


def integ(cp, area, nx, scale):
    return float(scale * (cp * area * nx).sum() / A_REF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-points", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    npts = 2000 if args.smoke else args.n_points
    epochs = 2 if args.smoke else args.epochs
    bs = 4 if args.smoke else 8

    man = os.path.join(ROOT, "data", "manifest.json")
    fd, cache = os.path.join(ROOT, "data", "fields"), os.path.join(ROOT, "data", "cache")
    tr_ds = DrivAerFields(man, fd, cache, "train", npts)
    va_ds = DrivAerFields(man, fd, cache, "val", npts)
    te_ds = DrivAerFields(man, fd, cache, "test", npts)
    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    print(f"field data: train {len(tr_ds)} / val {len(va_ds)} / test {len(te_ds)} | npts {npts} ep {epochs}", flush=True)

    model = pointnet_field.build({}).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.MSELoss()

    @torch.no_grad()
    def evaluate(ds):
        model.eval(); rel, cd_pred_err, cd_true_err = [], [], []
        for i in range(len(ds)):
            pts, cp, area, nx, scale, cd_true, _ = ds.eval_full(i)
            pred = model(pts.unsqueeze(0).to(dev)).squeeze(0).cpu()
            rel.append((torch.linalg.norm(pred - cp) / (torch.linalg.norm(cp) + 1e-6)).item())
            a, x = area.numpy(), nx.numpy()
            cd_pred_err.append(abs(integ(pred.numpy(), a, x, scale) - cd_true) * 1000)
            cd_true_err.append(abs(integ(cp.numpy(), a, x, scale) - cd_true) * 1000)
        return float(np.mean(rel)), float(np.mean(cd_pred_err)), float(np.mean(cd_true_err))

    best, t0 = 1e9, time.time()
    for ep in range(epochs):
        model.train(); tot = 0
        for pts, cp in tr:
            opt.zero_grad(); loss = lossf(model(pts.to(dev)), cp.to(dev))
            loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            rl2, cdp, cdt = evaluate(va_ds); best = min(best, rl2)
            print(f"  ep {ep:3d} loss {tot/max(len(tr),1):.4f} | val relL2 {rl2:.3f} | "
                  f"Cd(pred-field) MAE {cdp:.1f} | Cd(true-field) MAE {cdt:.1f} counts", flush=True)

    rl2, cdp, cdt = evaluate(te_ds)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    rec = {"task": "field", "surrogate": "pointnet_field_v2", "smoke": args.smoke,
           "n_points": npts, "epochs": epochs,
           "data": {"train": len(tr_ds), "val": len(va_ds), "test": len(te_ds)},
           "test_field_relL2": rl2, "test_cd_from_pred_field_mae_counts": cdp,
           "test_cd_from_true_field_mae_counts": cdt, "train_minutes": (time.time()-t0)/60,
           "hardware": {"gpu": gpu}}
    out = os.path.join(ROOT, "results", f"field_{int(t0)}.json")
    json.dump(rec, open(out, "w"), indent=2)
    print(f"\n=== TEST: field relL2 {rl2:.3f} | Cd-from-PRED-field MAE {cdp:.1f} counts | "
          f"Cd-from-TRUE-field MAE {cdt:.1f} counts (method check) | {gpu} ===")
    print(f"record -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
