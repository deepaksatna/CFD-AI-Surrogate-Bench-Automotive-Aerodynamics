"""RegDGCNN — Dynamic Graph CNN regressor for scalar drag (Cd) from a car point cloud.

Faithful DGCNN backbone (EdgeConv on a dynamic kNN graph) with a regression head.
Hyperparameters live in configs/surrogates.yaml and are to be reconciled with the
official RegDGCNN_SurfaceFields code; the architecture (EdgeConv + global pool + MLP)
matches the published DrivAerNet baseline family.
"""
import torch
import torch.nn as nn


def knn(x, k):
    # x: [B, C, N]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)          # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)              # [B, 1, N]
    pairwise = -xx - inner - xx.transpose(2, 1)              # [B, N, N] (neg sq dist)
    return pairwise.topk(k=k, dim=-1)[1]                     # [B, N, k]


def graph_feature(x, k, idx=None):
    # x: [B, C, N] -> [B, 2C, N, k]
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k)
    base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + base).view(-1)
    x_t = x.transpose(2, 1).contiguous().view(B * N, C)
    feat = x_t[idx, :].view(B, N, k, C)
    x_rep = x.transpose(2, 1).view(B, N, 1, C).repeat(1, 1, k, 1)
    feat = torch.cat((feat - x_rep, x_rep), dim=3)           # edge feature
    return feat.permute(0, 3, 1, 2).contiguous()             # [B, 2C, N, k]


class EdgeConv(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(in_c * 2, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = graph_feature(x, self.k)         # [B, 2C, N, k]
        x = self.net(x)                      # [B, out, N, k]
        return x.max(dim=-1)[0]              # [B, out, N]


class RegDGCNN(nn.Module):
    name = "regdgcnn"

    def __init__(self, k=20, edge_dims=(64, 64, 128, 256), emb_dim=1024,
                 head_dims=(512, 256), dropout=0.3, **_):
        super().__init__()
        self.k = k
        dims, layers = [3] + list(edge_dims), []
        for i in range(len(edge_dims)):
            layers.append(EdgeConv(dims[i], dims[i + 1], k))
        self.edge_convs = nn.ModuleList(layers)
        self.emb = nn.Sequential(
            nn.Conv1d(sum(edge_dims), emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim), nn.LeakyReLU(0.2))
        head, prev = [], emb_dim * 2          # max + mean global pool
        for h in head_dims:
            head += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)

    def forward(self, pts):                   # pts: [B, N, 3]
        x = pts.transpose(2, 1).contiguous()  # [B, 3, N]
        feats = []
        for ec in self.edge_convs:
            x = ec(x)
            feats.append(x)
        x = self.emb(torch.cat(feats, dim=1)) # [B, emb, N]
        x = torch.cat([x.max(dim=-1)[0], x.mean(dim=-1)], dim=1)
        return self.head(x).squeeze(-1)       # [B]


def build(cfg):
    return RegDGCNN(**cfg)
