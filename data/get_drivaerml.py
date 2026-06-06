#!/usr/bin/env python3
"""
Download DrivAerML geometry + drag coefficients from the HuggingFace mirror
(neashton/drivaerml) — HTTPS, no Globus. Per run we fetch ONLY:
  run_N/drivaer_N.stl       (~142 MB geometry)
  run_N/force_mom_N.csv     (tiny; columns: Cd,Cl,Clf,Clr,Cs)
The 26 GB volume_*.vtu files are skipped. Builds a manifest.json compatible with
data/dataset.py (symlinks data/meshes/run_N.stl -> the downloaded STL).

DrivAerML is high-fidelity HRLES on the DrivAer NOTCHBACK (single body type) — great for
the scaling-law / energy / field axes; the body-type cliff needs the multi-body DrivAerNet++.

Usage (on the box):
    HF_TOKEN=hf_xxx python3 data/get_drivaerml.py --n-runs 250
Resumable — hf_hub_download caches, re-runs skip existing files.
"""
import argparse, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO, RTYPE = "neashton/drivaerml", "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=250, help="download runs 1..N (max ~500)")
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import hf_hub_download
    raw = os.path.join(ROOT, "data", "drivaerml_raw")
    meshes = os.path.join(ROOT, "data", "meshes")
    os.makedirs(meshes, exist_ok=True)

    manifest, ok, miss = [], 0, 0
    for n in range(args.start, args.n_runs + 1):
        try:
            csv_path = hf_hub_download(REPO, f"run_{n}/force_mom_{n}.csv", repo_type=RTYPE,
                                       token=tok, local_dir=raw)
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            cd = float(rows[1][0])                      # header Cd,Cl,...  -> first value
            stl_path = hf_hub_download(REPO, f"run_{n}/drivaer_{n}.stl", repo_type=RTYPE,
                                       token=tok, local_dir=raw)
            link = os.path.join(meshes, f"run_{n}.stl")
            if not os.path.exists(link):
                os.symlink(os.path.abspath(stl_path), link)
            h = n % 10
            split = "test" if h == 0 else ("val" if h == 1 else "train")
            manifest.append({"id": f"run_{n}", "cd": cd, "split": split, "mesh": True})
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok} runs ready (last run_{n}, Cd={cd:.4f})", flush=True)
        except Exception as e:
            miss += 1
            if miss <= 10:
                print(f"  skip run_{n}: {str(e)[:80]}", flush=True)

    json.dump(manifest, open(os.path.join(ROOT, "data", "manifest.json"), "w"), indent=0)
    n_tr = sum(m["split"] == "train" for m in manifest)
    n_va = sum(m["split"] == "val" for m in manifest)
    n_te = sum(m["split"] == "test" for m in manifest)
    print(f"\n✅ DrivAerML manifest.json: {ok} runs (train {n_tr} / val {n_va} / test {n_te}); "
          f"{miss} skipped. Cd via force_mom CSV. Ready: python3 run_bench.py --surrogate regdgcnn --task cd")


if __name__ == "__main__":
    main()
