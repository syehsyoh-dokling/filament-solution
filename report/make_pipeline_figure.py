"""Draw the pipeline diagram (fig_pipeline.pdf / .png) for the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 6.4, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.0, 1.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")

BLUE, ORANGE, GREEN, GREY = "#dbe8f6", "#fbe3c8", "#dcefd9", "#eeeeee"


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=color, ec="#555555", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333"))


box(0.5, 7, 10, 8, "H-$\\alpha$\nimage\n2048$^2$", GREY)
box(14, 12.5, 14, 7, "YOLO11s\n@ 1280 px", BLUE)
box(14, 2.5, 14, 7, "YOLO11s\n@ 1920 px", BLUE)
box(31.5, 7, 11, 8, "weighted\nbox fusion", BLUE)
box(46, 7, 16.5, 8, "square crop $\\to$ 512$^2$\nchannels: raw,\nCLAHE, box seed", ORANGE)
box(66, 7, 13.5, 8, "ResNet34 U-Net\n3-flip TTA\nthr 0.5, LCC", ORANGE)
box(83, 7, 16.5, 8, "non-overlapping\nassembly\n$\\to$ RLE", GREEN)
arrow(10.9, 12, 13.6, 15.5); arrow(10.9, 10, 13.6, 6.5)
arrow(28.4, 15.5, 31.1, 12); arrow(28.4, 6.5, 31.1, 10)
arrow(42.9, 11, 45.6, 11); arrow(62.9, 11, 65.6, 11); arrow(79.9, 11, 82.6, 11)
ax.text(21, 21.3, "Stage 1: detection", ha="center", color="#2c5a8a", fontweight="bold")
ax.text(63, 21.3, "Stage 2: box-seeded refinement", ha="center", color="#a35a12", fontweight="bold")
ax.text(91.3, 21.3, "Stage 3", ha="center", color="#3d7a35", fontweight="bold")
plt.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_pipeline.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_pipeline.pdf")
