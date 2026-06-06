"""PyTorch dataset: DrivAerNet++ car surface point cloud -> drag coefficient (Cd).

Loads STL meshes, surface-samples N points (cached as .npy), normalizes to a unit sphere.
Cd is standardized (z-score on the train split) for stable regression; the runner
de-standardizes predictions before computing drag-count errors.
"""
import json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_unit_sphere(pts):
    c = pts.mean(0, keepdims=True)
    pts = pts - c
    scale = np.linalg.norm(pts, axis=1).max()
    return pts / (scale + 1e-9)


def bodytype(design_id):
    """DrivAerNet++ id like 'DrivAer_F_D_WM_WW_0001' -> body-type letter ('F','N','E', ...)."""
    parts = design_id.split("_")
    return parts[1].upper() if len(parts) > 1 and parts[1] else "?"


class DrivAerPointClouds(Dataset):
    """split in {train,val,test}. holdout_bodytype (e.g. 'E') runs the generalization-cliff
    protocol: train/val exclude that body type; test BECOMES that unseen body type.
    train_frac<1 deterministically subsamples the train split for the data-scaling law."""
    def __init__(self, manifest_path, mesh_dir, cache_dir, split, n_points=5000,
                 normalize="unit_sphere", cd_mean=0.0, cd_std=1.0,
                 holdout_bodytype=None, train_frac=1.0, seed=0):
        manifest = json.load(open(manifest_path))
        ho = holdout_bodytype.strip().upper()[0] if holdout_bodytype else None
        if ho:
            if split == "test":   # the unseen body type, drawn from anywhere
                items = [m for m in manifest if m["mesh"] and bodytype(m["id"]) == ho]
            else:                 # train/val: their split, minus the held-out type
                items = [m for m in manifest if m["mesh"] and m["split"] == split
                         and bodytype(m["id"]) != ho]
        else:
            items = [m for m in manifest if m["mesh"] and m["split"] == split]
        if split == "train" and train_frac < 1.0:
            items = sorted(items, key=lambda m: m["id"])
            random.Random(seed).shuffle(items)
            items = items[:max(1, int(round(len(items) * train_frac)))]
        self.items = items
        if not self.items:
            raise RuntimeError(
                f"No '{split}' items (holdout={holdout_bodytype}, frac={train_frac}) with a mesh. "
                f"Stage STLs in {mesh_dir} and re-run data/get_drivaernet.py --require-mesh.")
        self.mesh_dir, self.cache_dir = mesh_dir, cache_dir
        self.n_points, self.normalize = n_points, normalize
        self.cd_mean, self.cd_std = cd_mean, cd_std
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.items)

    def _points_for(self, did):
        cache = os.path.join(self.cache_dir, f"{did}_{self.n_points}.npy")
        if os.path.exists(cache):
            return np.load(cache)
        import trimesh
        # force="mesh" concatenates multi-body STLs (DrivAerML loads as a Scene otherwise)
        mesh = trimesh.load(os.path.join(self.mesh_dir, did + ".stl"), process=False, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, self.n_points)
        pts = np.asarray(pts, dtype=np.float32)
        np.save(cache, pts)
        return pts

    def __getitem__(self, i):
        m = self.items[i]
        pts = self._points_for(m["id"])
        if self.normalize == "unit_sphere":
            pts = _normalize_unit_sphere(pts).astype(np.float32)
        cd_norm = (m["cd"] - self.cd_mean) / (self.cd_std + 1e-9)
        return torch.from_numpy(pts), torch.tensor(cd_norm, dtype=torch.float32), m["cd"]


def cd_stats(manifest_path):
    """Mean/std of Cd over the TRAIN split (for standardization)."""
    manifest = json.load(open(manifest_path))
    vals = np.array([m["cd"] for m in manifest if m["split"] == "train"], dtype=np.float64)
    return float(vals.mean()), float(vals.std())
