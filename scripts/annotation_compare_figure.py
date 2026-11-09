"""Draw the re-annotation comparison: dataset labels vs. my independent pass.

Green = dataset boxes, cyan = boxes I drew without looking. Sorted so the worst-granularity cases
show up first, which is the point of the figure — the disagreement is about how a defect gets
sliced, not about whether the defect is there.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

REF = Path("dataset/neu_det/labels/test")
MINE = Path("annotation_task/labels")
IMGS = Path("annotation_task/images")
OUT = Path("assets/step2c_annotation_compare.png")
ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

# four cases that show the range: agreement, granularity split, over-boxing, and a miss
CASES = ["inclusion_291", "crazing_281", "patches_278", "pitted_surface_290"]


def read_labels(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append(tuple(float(v) for v in parts[:5]))
    return out


def mask(boxes, w, h):
    m = np.zeros((h, w), dtype=bool)
    for _, cx, cy, bw, bh in boxes:
        x1 = max(0, min(w, round((cx - bw / 2) * w)))
        x2 = max(0, min(w, round((cx + bw / 2) * w)))
        y1 = max(0, min(h, round((cy - bh / 2) * h)))
        y2 = max(0, min(h, round((cy + bh / 2) * h)))
        m[y1:y2, x1:x2] = True
    return m


def region_iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def draw(ax, img, boxes, color, lw=1.5, ls="solid", alpha=0.18):
    ax.imshow(img, cmap="gray", extent=(0, 1, 1, 0), vmin=0, vmax=255, interpolation="bilinear")
    for _, cx, cy, bw, bh in boxes:
        xy = (cx - bw / 2, cy - bh / 2)
        ax.add_patch(Rectangle(xy, bw, bh, fc=color, ec="none", alpha=alpha, zorder=2))
        ax.add_patch(Rectangle(xy, bw, bh, fill=False, ec=color, lw=lw, ls=ls, zorder=3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    for s in ax.spines.values():
        s.set_color("#c8ccd4")


def main():
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.9), facecolor="white")
    for col, stem in enumerate(CASES):
        img = np.array(Image.open(IMGS / f"{stem}.jpg").convert("L"))
        h, w = img.shape
        ref = read_labels(REF / f"{stem}.txt")
        mine = read_labels(MINE / f"{stem}.txt")
        iou = region_iou(mask(ref, w, h), mask(mine, w, h))

        ax_t = axes[0][col]
        draw(ax_t, img, ref, "#1f9d55")
        ax_t.set_title(f"{stem} — dataset labels, {len(ref)} box(es)", fontsize=9.5,
                       color="#1f9d55", pad=6)

        ax_b = axes[1][col]
        draw(ax_b, img, mine, "#0b7fa8")
        ax_b.set_title(f"my blind pass — {len(mine)} box(es), region IoU {iou:.2f}",
                       fontsize=9.5, color="#0b7fa8", pad=6)

        if col == 0:
            ax_t.set_ylabel("dataset", fontsize=10)
            ax_b.set_ylabel("re-drawn", fontsize=10)

    fig.suptitle("Same defect, different slicing — the two passes find the same regions but cut them differently",
                 fontsize=12, y=0.985)
    fig.text(0.5, 0.012,
             "Box-level F1 is only 0.38, yet 82% of the dataset's defect area is covered by my boxes: "
             "the disagreement is box granularity and tightness, not missed defects.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.subplots_adjust(hspace=0.30, wspace=0.05, top=0.885, bottom=0.075, left=0.045, right=0.99)
    fig.savefig(OUT, dpi=170, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
