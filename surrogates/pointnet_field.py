"""Phase 2 model: PointNet segmentation-style per-point pressure (Cp) regressor.

Per-point MLP -> global max-pooled feature -> concat back to each point -> per-point head
predicting a scalar Cp. Lower-risk than DoMINO, reuses the stack; DoMINO is the marquee
upgrade for a later pass.
"""
import torch
import torch.nn as nn


def _mlp1d(dims, act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Conv1d(dims[i], dims[i + 1], 1, bias=False), nn.BatchNorm1d(dims[i + 1])]
        if act or i < len(dims) - 2:
            layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class PointNetField(nn.Module):
    name = "pointnet_field"

    def __init__(self, feat=(64, 128, 1024), head=(512, 256, 128), **_):
        super().__init__()
        self.enc = _mlp1d((3,) + tuple(feat))
        self.head = _mlp1d((feat[-1] + feat[-1],) + tuple(head))
        self.out = nn.Conv1d(head[-1], 1, 1)

    def forward(self, pts):                      # [B, N, 3]
        x = pts.transpose(2, 1).contiguous()     # [B, 3, N]
        f = self.enc(x)                          # [B, C, N]
        g = f.max(dim=-1, keepdim=True)[0].expand(-1, -1, f.shape[-1])  # global feat broadcast
        h = self.head(torch.cat([f, g], dim=1))  # [B, H, N]
        return self.out(h).squeeze(1)            # [B, N]  per-point Cp


def build(cfg):
    return PointNetField(**(cfg or {}))
