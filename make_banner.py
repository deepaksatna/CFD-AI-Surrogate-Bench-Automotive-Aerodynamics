#!/usr/bin/env python3
"""Beyond the Model — newsletter banner for the CFD-AI Surrogate Bench.
Size matches the series reference: 1940 x 1100 px."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

C = {"navy": "#0e1630", "navy2": "#16213e", "cyan": "#00d9ff",
     "green": "#76b900", "red": "#e94560", "white": "#ffffff", "grey": "#9fb3c8"}

fig = plt.figure(figsize=(19.4, 11.0), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# vertical gradient background
for i in range(120):
    t = i / 120
    col = (0.055 + 0.04 * t, 0.086 + 0.05 * t, 0.19 + 0.04 * t)
    ax.add_patch(plt.Rectangle((0, 1 - (i + 1) / 120), 1, 1 / 120, color=col, lw=0))

# faint streamlines sweeping across (the "flow")
np.random.seed(7)
for k in range(11):
    y0 = 0.12 + k * 0.072
    xs = np.linspace(0.0, 1.0, 200)
    ys = y0 + 0.05 * np.sin(6 * xs + k) * np.exp(-2 * (xs - 0.5) ** 2)
    ax.plot(xs, ys, color=C["cyan"], alpha=0.10 + 0.02 * (k % 3), lw=1.4)

# stylized car silhouette (bottom-center motif, below the badges) with pressure dots
cx = -0.0  # offset
car = np.array([[0.40,0.115],[0.46,0.115],[0.49,0.165],[0.555,0.19],[0.625,0.19],
                [0.655,0.145],[0.70,0.145],[0.70,0.115],[0.40,0.115]]) + [cx, 0]
ax.add_patch(Polygon(car, closed=True, facecolor=C["navy2"], edgecolor=C["cyan"], lw=2, alpha=0.85))
for wx in (0.485, 0.635):
    ax.add_patch(Circle((wx, 0.115), 0.016, color=C["cyan"], alpha=0.8, zorder=5))
    ax.add_patch(Circle((wx, 0.115), 0.009, color=C["white"], alpha=0.9, zorder=6))
for (x, y, c) in [(0.405,0.12,C["red"]),(0.555,0.195,C["cyan"]),(0.61,0.19,C["cyan"]),(0.69,0.135,C["red"])]:
    ax.add_patch(Circle((x, y), 0.007, color=c, alpha=0.95, zorder=6))

# series tag
ax.text(0.5, 0.925, "B E Y O N D   T H E   M O D E L", color=C["green"],
        fontsize=19, fontweight="bold", ha="center", family="DejaVu Sans")
# title (the name)
ax.text(0.5, 0.83, "CFD-AI Surrogate Bench", color=C["white"], fontsize=66, fontweight="bold", ha="center")
ax.text(0.5, 0.745, "—  Automotive Aerodynamics  —", color=C["cyan"], fontsize=30, fontweight="bold", ha="center")
ax.text(0.5, 0.675, "Where high-fidelity CFD meets neural surrogates  ·  for the CFD, HPC & AI communities",
        color=C["grey"], fontsize=21, ha="center")
ax.text(0.5, 0.615, "“We replaced a wind tunnel with a neural net — the cheap model won.”",
        color=C["green"], fontsize=21, style="italic", ha="center")

# three stat chips
chips = [("0.65 ms / car", "neural surrogate inference", C["cyan"]),
         ("61,440 core-hrs", "one high-fidelity CFD solve", C["red"]),
         ("4.2 drag-counts", "best surrogate accuracy (R² 0.88)", C["green"])]
cw, gap = 0.27, 0.025
x0 = 0.5 - (3 * cw + 2 * gap) / 2
for i, (big, small, col) in enumerate(chips):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch((x, 0.40), cw, 0.16, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=C["navy2"], edgecolor=col, lw=2.5))
    ax.text(x + cw / 2, 0.495, big, color=col, fontsize=30, fontweight="bold", ha="center")
    ax.text(x + cw / 2, 0.435, small, color=C["grey"], fontsize=15, ha="center")

# domain badges row
badges = [("CFD", C["cyan"]), ("HPC", C["red"]), ("AI", C["green"]), ("NVIDIA", C["green"]), ("SURROGATE", C["cyan"])]
bw, bgap = 0.13, 0.015
bx0 = 0.5 - (len(badges) * bw + (len(badges) - 1) * bgap) / 2
for i, (t, col) in enumerate(badges):
    x = bx0 + i * (bw + bgap)
    ax.add_patch(FancyBboxPatch((x, 0.225), bw, 0.07, boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="none", edgecolor=col, lw=2))
    ax.text(x + bw / 2, 0.26, t, color=col, fontsize=18, fontweight="bold", ha="center")

# footer
ax.text(0.5, 0.075, "DrivAerML (HRLES, 500 cars)   ·   single GPU   ·   data-efficiency law · novelty cliff · energy   ·   open benchmark + leaderboard",
        color=C["grey"], fontsize=15.5, ha="center")
ax.text(0.5, 0.035, "github.com/deepaksatna/CFD-AI-Surrogate-Bench-Automotive-Aerodynamics",
        color=C["white"], fontsize=14, ha="center", alpha=0.85)

fig.savefig("results/newsletter_banner.png", dpi=100, facecolor=C["navy"])
print("wrote results/newsletter_banner.png")
