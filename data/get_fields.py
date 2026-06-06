#!/usr/bin/env python3
"""
Phase 2 data: download DrivAerML surface-field files (boundary_N.vtp, ~660 MB each)
from the HF mirror for the first N runs in the manifest. These carry the surface
pressure field used for Task B (field prediction + drag-by-integration).

    HF_TOKEN=hf_xxx python3 data/get_fields.py --n-runs 60
Heavy (~0.66 GB/run). Resumable (hf cache). Writes symlinks data/fields/run_N.vtp.
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO, RTYPE = "neashton/drivaerml", "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=60)
    args = ap.parse_args()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from huggingface_hub import hf_hub_download

    manifest = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    ids = [m["id"] for m in manifest][:args.n_runs]
    raw = os.path.join(ROOT, "data", "fields_raw")
    fields = os.path.join(ROOT, "data", "fields")
    os.makedirs(fields, exist_ok=True)
    ok = 0
    for did in ids:                                  # did like "run_37"
        n = did.split("_")[1]
        try:
            p = hf_hub_download(REPO, f"run_{n}/boundary_{n}.vtp", repo_type=RTYPE,
                                token=tok, local_dir=raw)
            link = os.path.join(fields, f"{did}.vtp")
            if not os.path.exists(link):
                os.symlink(os.path.abspath(p), link)
            ok += 1
            if ok % 10 == 0:
                print(f"  {ok} surface fields ready (last {did})", flush=True)
        except Exception as e:
            print(f"  skip {did}: {str(e)[:80]}", flush=True)
    print(f"✅ {ok} surface-field VTPs in data/fields/. Next: python3 run_field.py --smoke")


if __name__ == "__main__":
    main()
