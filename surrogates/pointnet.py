"""PointNet — cheap point-cloud baseline for scalar drag (Cd) regression.

Vanilla PointNet (shared per-point MLP → global max-pool → MLP head). No T-Net:
geometry is already unit-sphere normalized, and for a smooth scalar like Cd the
input transform adds cost without much benefit. This is the "cheap reference" leg
of the bake-off — fast to train, sets the floor the heavier models must beat.
"""
import torch
import torch.nn as nn


def _mlp1d(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Conv1d(dims[i], dims[i + 1], 1, bias=False),
                   nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class PointNet(nn.Module):
    name = "pointnet"

    def __init__(self, feat_dims=(64, 64, 128, 256, 1024), head_dims=(512, 256),
                 dropout=0.3, **_):
        super().__init__()
        self.feat = _mlp1d((3,) + tuple(feat_dims))
        head, prev = [], feat_dims[-1]
        for h in head_dims:
            head += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)

    def forward(self, pts):                  # pts: [B, N, 3]
        x = pts.transpose(2, 1).contiguous() # [B, 3, N]
        x = self.feat(x)                     # [B, 1024, N]
        x = x.max(dim=-1)[0]                 # global max pool -> [B, 1024]
        return self.head(x).squeeze(-1)      # [B]


def build(cfg):
    return PointNet(feat_dims=tuple(cfg.get("feat_dims", (64, 64, 128, 256, 1024))),
                    head_dims=tuple(cfg.get("head_dims", (512, 256))),
                    dropout=cfg.get("dropout", 0.3))
