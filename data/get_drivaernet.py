#!/usr/bin/env python3
"""
Fetch the lightweight parts of DrivAerNet++ (drag-coefficient labels + official
train/val/test splits) and build a manifest the trainer consumes.

The 3D mesh geometry (39 TB on Harvard Dataverse, CC BY-NC, Globus) is NOT downloaded
here — stage the 3D-Meshes subset yourself and point dataset.yaml:mesh_dir at it.
This script only needs the network for the small CSV + split text files.

Usage:
    python data/get_drivaernet.py                 # fetch Cd csv + splits, build manifest
    python data/get_drivaernet.py --require-mesh  # only keep ids whose STL is present

Outputs: data/cd.csv, data/splits/*.txt, data/manifest.json
"""
import argparse, csv, json, os, sys, urllib.request, ssl
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg():
    with open(os.path.join(ROOT, "configs", "dataset.yaml")) as f:
        return yaml.safe_load(f)


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached {os.path.relpath(dest, ROOT)}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(dest, "wb") as o:
            o.write(r.read())
        print(f"  downloaded {os.path.relpath(dest, ROOT)} ({os.path.getsize(dest)} B)")
        return True
    except Exception as e:
        print(f"  WARN could not fetch {url}\n        {e}")
        return False


def norm_id(s):
    return s.strip().replace(".stl", "").replace(".ply", "")


def parse_cd_csv(path):
    """Return {design_id: cd}. Auto-detect id + Cd columns."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lower() for h in rows[0]]
    # id column: contains 'design' or 'name' or 'id'; cd column: contains 'cd' or 'drag'
    id_col = next((i for i, h in enumerate(header) if any(k in h for k in ("design", "name", "id"))), 0)
    cd_col = next((i for i, h in enumerate(header)
                   if ("cd" in h or "drag" in h) and "std" not in h), len(header) - 1)
    print(f"  Cd CSV columns -> id='{rows[0][id_col]}', cd='{rows[0][cd_col]}'")
    out = {}
    for r in rows[1:]:
        if len(r) <= max(id_col, cd_col) or not r[id_col].strip():
            continue
        try:
            out[norm_id(r[id_col])] = float(r[cd_col])
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-mesh", action="store_true",
                    help="only keep ids whose <id>.stl exists in mesh_dir")
    args = ap.parse_args()
    cfg = load_cfg()

    print("→ Cd labels CSV")
    cd_path = os.path.join(ROOT, "data", "cd.csv")
    if not fetch(cfg["cd_csv_url"], cd_path):
        sys.exit("Cd CSV is required. Download it manually from the Dropbox link in dataset.yaml.")
    cd = parse_cd_csv(cd_path)
    print(f"  parsed {len(cd)} Cd labels")

    print("→ Official splits")
    id2split = {}
    for fname, split in cfg["split_files"].items():
        dest = os.path.join(ROOT, "data", "splits", fname)
        if fetch(f"{cfg['splits_base_url']}/{fname}", dest):
            for line in open(dest):
                if line.strip():
                    id2split[norm_id(line)] = split
    if not id2split:
        print("  WARN no split files fetched — will fall back to deterministic 80/10/10 split")

    mesh_dir = os.path.join(ROOT, cfg["mesh_dir"])
    have_meshes = os.path.isdir(mesh_dir) and any(f.endswith(".stl") for f in os.listdir(mesh_dir))
    print(f"→ Mesh dir {cfg['mesh_dir']}: {'found STLs' if have_meshes else 'EMPTY (stage the 3D-Meshes subset here)'}")

    # build manifest
    manifest = []
    ids = sorted(cd.keys())
    for i, did in enumerate(ids):
        mesh_present = have_meshes and os.path.exists(os.path.join(mesh_dir, did + ".stl"))
        if args.require_mesh and not mesh_present:
            continue
        split = id2split.get(did)
        if split is None:  # deterministic fallback by hash of id
            h = sum(map(ord, did)) % 10
            split = "test" if h == 0 else ("val" if h == 1 else "train")
        manifest.append({"id": did, "cd": cd[did], "split": split, "mesh": mesh_present})

    mpath = os.path.join(ROOT, "data", "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=0)
    n_train = sum(m["split"] == "train" for m in manifest)
    n_val = sum(m["split"] == "val" for m in manifest)
    n_test = sum(m["split"] == "test" for m in manifest)
    n_mesh = sum(m["mesh"] for m in manifest)
    print(f"✅ manifest.json: {len(manifest)} cars  (train {n_train} / val {n_val} / test {n_test}; "
          f"{n_mesh} with mesh present)")
    if n_mesh == 0:
        print("   NOTE: 0 meshes present — stage the DrivAerNet++ 3D-Meshes subset into "
              f"{cfg['mesh_dir']} before training (see brev/PROVISION.md).")


if __name__ == "__main__":
    main()
