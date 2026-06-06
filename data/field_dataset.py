"""Phase 2 dataset (v2, CELL-level): DrivAerML surface -> per-cell pressure (Cp) field.

Fix vs v1: integrate drag over the TRUE per-CELL area + outward normal (not a uniform
per-point approximation), so Cd = sum(Cp_cell * area_cell * nx_cell) / A_ref actually
recovers the pressure-drag. Training samples N cells; evaluation predicts on the full
mesh (or an unbiased subsample, scaled by M/k) and integrates exactly.
"""
import glob, json, os
import numpy as np
import torch
from torch.utils.data import Dataset

A_REF = 2.17           # DrivAer frontal reference area [m^2]
EVAL_MAX_CELLS = 120000  # cap for the eval forward; unbiased scaling applied if subsampled


def _cp_cell_array(m):
    import numpy as _np
    names = list(m.cell_data.keys())
    for cand in ("Cp", "cp", "pMean", "pmean", "p", "pressure", "Pressure"):
        for n in names:
            if n.lower() == cand.lower():
                return _np.asarray(m.cell_data[n], dtype=_np.float32).reshape(-1)
    return _np.asarray(m.cell_data[names[0]], dtype=_np.float32).reshape(-1) if names else None


def parse_vtp(path):
    """All cells -> (centroids[M,3], cp[M], area[M], nx[M]). Cp moved to cells if needed."""
    import pyvista as pv
    m = pv.read(path)
    if len(m.cell_data) == 0 or _cp_cell_array(m) is None:
        m = m.point_data_to_cell_data()
    m = m.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    sized = m.compute_cell_sizes(area=True, volume=False, length=False)
    cen = np.asarray(m.cell_centers().points, dtype=np.float32)
    area = np.asarray(sized.cell_data["Area"], dtype=np.float32).reshape(-1)
    nx = np.asarray(m.cell_data["Normals"][:, 0], dtype=np.float32).reshape(-1)
    cp = _cp_cell_array(m)
    n = min(len(cen), len(area), len(nx), len(cp))
    return cen[:n], cp[:n], area[:n], nx[:n]


def _norm(cen):
    c = cen.mean(0, keepdims=True)
    return ((cen - c) / (np.linalg.norm(cen - c, axis=1).max() + 1e-9)).astype(np.float32)


def bodytype(design_id):
    p = design_id.split("_")
    return p[1].upper() if len(p) > 1 else "?"


class DrivAerFields(Dataset):
    """Training view: sample N cells per car -> (normalized centroids, Cp)."""
    def __init__(self, manifest_path, fields_dir, cache_dir, split, n_points=8000,
                 holdout_bodytype=None, train_frac=1.0, seed=0):
        manifest = json.load(open(manifest_path))
        avail = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(fields_dir, "*.vtp"))}
        ho = holdout_bodytype.strip().upper()[0] if holdout_bodytype else None
        if ho and split == "test":
            items = [m for m in manifest if m["id"] in avail and bodytype(m["id"]) == ho]
        elif ho:
            items = [m for m in manifest if m["id"] in avail and m["split"] == split and bodytype(m["id"]) != ho]
        else:
            items = [m for m in manifest if m["id"] in avail and m["split"] == split]
        if split == "train" and train_frac < 1.0:
            import random; random.Random(seed).shuffle(items)
            items = items[:max(1, int(len(items) * train_frac))]
        self.items, self.fields_dir, self.cache_dir, self.n_points = items, fields_dir, cache_dir, n_points
        os.makedirs(cache_dir, exist_ok=True)
        if not items:
            raise RuntimeError(f"No '{split}' surface fields in {fields_dir} (holdout={holdout_bodytype}).")

    def _full(self, did):
        cache = os.path.join(self.cache_dir, f"{did}_cellfield.npz")
        if os.path.exists(cache):
            z = np.load(cache); return z["cen"], z["cp"], z["area"], z["nx"]
        cen, cp, area, nx = parse_vtp(os.path.join(self.fields_dir, did + ".vtp"))
        np.savez(cache, cen=cen, cp=cp, area=area, nx=nx)
        return cen, cp, area, nx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cen, cp, area, nx = self._full(self.items[i]["id"])
        idx = np.random.choice(len(cen), self.n_points, replace=len(cen) < self.n_points)
        return torch.from_numpy(_norm(cen)[idx]), torch.from_numpy(cp[idx])

    def eval_full(self, i):
        """Full-mesh view for exact drag integration: (pts_n[k,3], cp[k], area[k], nx[k], scale, cd_true)."""
        m = self.items[i]
        cen, cp, area, nx = self._full(m["id"])
        M = len(cen)
        if M > EVAL_MAX_CELLS:                       # unbiased subsample, scale by M/k
            idx = np.random.choice(M, EVAL_MAX_CELLS, replace=False)
            cen, cp, area, nx = cen[idx], cp[idx], area[idx], nx[idx]
        scale = M / len(cen)
        return (torch.from_numpy(_norm(cen)), torch.from_numpy(cp),
                torch.from_numpy(area), torch.from_numpy(nx), scale, m["cd"], m["id"])
