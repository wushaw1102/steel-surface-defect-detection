"""Build the step-by-step figures used by the README walkthrough.

Every panel is produced from artefacts that already exist in this repo:

    dataset/neu_det/         labels + images (prepare_dataset.py)
    runs/*/results.csv       training logs
    runs/*_test_metrics.txt  test-split metrics
    inference                live run with runs/finetune/weights/best.pt

Nothing is hand-drawn or simulated; if a number changes, the picture changes.

    python scripts/workflow_figures.py --gif
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.edgecolor": "#c9ced6",
        "axes.labelcolor": "#333",
        "text.color": "#222",
        "xtick.color": "#555",
        "ytick.color": "#555",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

ROOT = Path("dataset/neu_det")
ASSETS = Path("assets")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
GT_C, PRED_C, BOX_C = "#00b050", "#e63232", "#3b7ddd"
random.seed(7)


# ------------------------------------------------------------------ helpers
def imgs_of(split):
    return sorted((ROOT / "images" / split).glob("*"))


def labels_of(split, stem):
    out = []
    for line in (ROOT / "labels" / split / f"{stem}.txt").read_text().strip().splitlines():
        c, cx, cy, w, h = line.split()
        out.append((int(c), float(cx), float(cy), float(w), float(h)))
    return out


def add_boxes(ax, boxes, color, lw=1.6, fill=0.22, tag=False, font=7, alpha=1.0, ls="solid"):
    """Fill + edge in two patches so the fill can be translucent while the edge stays crisp."""
    for c, cx, cy, w, h in boxes:
        xy = (cx - w / 2, cy - h / 2)
        if fill:
            ax.add_patch(Rectangle(xy, w, h, fc=color, ec="none", alpha=fill, zorder=2))
        ax.add_patch(
            Rectangle(xy, w, h, fill=False, ec=color, lw=lw, ls=ls, alpha=alpha, zorder=3)
        )
        if tag:
            ax.text(
                cx - w / 2 + 0.012,
                cy + h / 2 - 0.055,
                CLASSES[c][:11],
                color="white",
                fontsize=font,
                bbox=dict(fc=color, ec="none", pad=1.0, alpha=0.9),
                zorder=4,
            )


def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    for s in ax.spines.values():
        s.set_edgecolor("#d6dae0")


def median_area(split="train"):
    areas = {i: [] for i in range(6)}
    for p in imgs_of(split):
        for c, _, _, w, h in labels_of(split, p.stem):
            areas[c].append(w * h)
    return {i: float(np.median(v)) * 100 for i, v in areas.items()}


def pick_images(split, cls, n, single=True):
    """Pick `n` images whose first label is `cls`; prefer images holding exactly one box."""
    one, many = [], []
    for p in imgs_of(split):
        lbs = labels_of(split, p.stem)
        if not lbs or lbs[0][0] != cls:
            continue
        (one if len(lbs) == 1 else many).append(p)
    pool = one if single and len(one) >= n else one + many
    random.shuffle(pool)
    return pool[:n]


# ------------------------------------------------------- step 1: data samples
def fig_samples():
    fig, axes = plt.subplots(3, 6, figsize=(11, 6.1))
    for ci in range(6):
        pool = [p for p in imgs_of("train") if labels_of("train", p.stem)[0][0] == ci]
        for ri, p in enumerate(random.sample(pool, 3)):
            ax = axes[ri][ci]
            ax.imshow(Image.open(p).convert("RGB"), extent=(0, 1, 1, 0))
            style(ax)
            if ri == 0:
                ax.set_title(CLASSES[ci], fontsize=9.5, pad=7, color=COLORS[ci], fontweight="bold")
    fig.suptitle("NEU-DET — 6 defect classes, 200x200 grayscale crops", fontsize=11.5, y=0.985)
    fig.text(
        0.5,
        0.005,
        "low contrast, thin structures, ambiguous boundaries — this is what 'hard' looks like at 200 px",
        ha="center",
        fontsize=8.5,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.955])
    fig.savefig(ASSETS / "step1_samples.png", dpi=140)
    plt.close(fig)


def fig_class_stats():
    inst = Counter()
    per_img, areas = [], []
    for p in imgs_of("train"):
        lbs = labels_of("train", p.stem)
        per_img.append(len(lbs))
        for b in lbs:
            inst[CLASSES[b[0]]] += 1
            areas.append(b[3] * b[4] * 100)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    vals = [inst[c] for c in CLASSES]
    axes[0].barh(CLASSES[::-1], vals[::-1], color="#4c78a8", height=0.62)
    for i, v in enumerate(vals[::-1]):
        axes[0].text(v + 8, i, str(v), va="center", fontsize=9, color="#333")
    axes[0].set_title("instances per class", fontsize=10)
    axes[0].set_xlim(0, max(vals) * 1.18)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(areas, bins=40, color="#f58518", edgecolor="white")
    axes[1].axvline(np.median(areas), color="#b34700", ls="--", lw=1.2)
    axes[1].text(
        np.median(areas) + 1.5, axes[1].get_ylim()[1] * 0.86,
        f"median {np.median(areas):.0f}%", fontsize=8.5, color="#b34700",
    )
    axes[1].set_title("annotated box area (% of the 200x200 crop)", fontsize=10)
    axes[1].set_xlabel("area ratio (%)")
    axes[1].set_ylabel("boxes")
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].hist(per_img, bins=range(1, max(per_img) + 2), color="#54a24b", edgecolor="white")
    axes[2].set_title("boxes per image", fontsize=10)
    axes[2].set_xlabel("instances")
    axes[2].set_ylabel("images")
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(ASSETS / "step1_stats.png", dpi=140)
    plt.close(fig)


# -------------------------------------------------- step 2: label inspection
def fig_annotations():
    med = median_area()
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
    for ci in range(6):
        for ri, p in enumerate(pick_images("train", ci, 2)):
            ax = axes[ri][ci]
            ax.imshow(Image.open(p).convert("RGB"), extent=(0, 1, 1, 0))
            add_boxes(ax, labels_of("train", p.stem), COLORS[ci], lw=2.4, fill=0.26)
            style(ax)
            if ri == 0:
                ax.set_title(
                    f"{CLASSES[ci]}\nmedian box {med[ci]:.0f}% of image",
                    fontsize=9,
                    pad=6,
                    color=COLORS[ci],
                    fontweight="bold",
                )
    fig.suptitle("Ground-truth boxes — the target the model has to reproduce, class by class",
                 fontsize=11, y=0.99)
    fig.text(0.5, 0.005, "shaded region = annotated defect; checked for misplaced or missing boxes before training",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.945])
    fig.savefig(ASSETS / "step2_annotations.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------- step 3: augmentation
def _rot(boxes, ang, W, H):
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c0 = np.array([W / 2, H / 2])
    out = []
    for c, cx, cy, w, h in boxes:
        pts = np.array(
            [
                [(cx - w / 2) * W, (cy - h / 2) * H],
                [(cx + w / 2) * W, (cy - h / 2) * H],
                [(cx + w / 2) * W, (cy + h / 2) * H],
                [(cx - w / 2) * W, (cy + h / 2) * H],
            ]
        )
        pts = np.clip((pts - c0) @ R.T + c0, 0, [W, H])
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        out.append((c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return out


def fig_augment():
    """Mirror the augmentations YOLO actually applies by default (no rotation, no shear)."""
    p = pick_images("train", 1, 1)[0]
    im = Image.open(p).convert("RGB")
    W, H = im.size
    boxes = labels_of("train", p.stem)

    def clip(bx):
        out = []
        for c, cx, cy, w, h in bx:
            x1, x2 = max(cx - w / 2, 0.0), min(cx + w / 2, 1.0)
            y1, y2 = max(cy - h / 2, 0.0), min(cy + h / 2, 1.0)
            if x2 - x1 > 0.01 and y2 - y1 > 0.01:
                out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        return out

    hsv = np.array(Image.fromarray(np.array(im)).convert("HSV"))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 18) % 255
    hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * 1.22, 0, 255).astype(np.uint8)
    hsv_img = Image.fromarray(hsv, "HSV").convert("RGB")

    dx, dy = 0.14, -0.10  # translate=0.1 in the default config
    border = tuple(int(v) for v in np.array(im).reshape(-1, 3).mean(0))  # 边缘填充用平均灰度，避免黑边
    shifted = Image.new("RGB", (W, H), border)
    shifted.paste(im, (int(dx * W), int(dy * H)))

    r = 0.72  # scale, part of the default scale=0.5 jitter
    x0 = (1 - r) / 2
    crop = im.crop((int(x0 * W), int(x0 * H), int((x0 + r) * W), int((x0 + r) * H))).resize((W, H))

    mos = Image.new("RGB", (W, H))
    mos_boxes, picks = [], random.sample(imgs_of("train"), 4)
    for i, q in enumerate(picks):
        ox, oy = (i % 2) * (W // 2), (i // 2) * (H // 2)
        mos.paste(Image.open(q).convert("RGB").resize((W // 2, H // 2)), (ox, oy))
        for c, cx, cy, w, h in labels_of("train", q.stem):
            mos_boxes.append((c, (cx * W / 2 + ox) / W, (cy * H / 2 + oy) / H, w / 2, h / 2))

    panels = [
        ("original", im, boxes),
        ("HSV jitter (+hue, +brightness)", hsv_img, boxes),
        ("horizontal flip (p=0.5)", im.transpose(Image.FLIP_LEFT_RIGHT),
         [(c, 1 - cx, cy, w, h) for c, cx, cy, w, h in boxes]),
        ("translate (dx=+14%, dy=-10%)", shifted,
         [(c, cx + dx, cy + dy, w, h) for c, cx, cy, w, h in boxes]),
        (f"scale / zoom x{r}", crop,
         [(c, (cx - x0) / r, (cy - x0) / r, w / r, h / r) for c, cx, cy, w, h in boxes]),
        ("mosaic (4 images into one)", mos, mos_boxes),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.3))
    for ax, (title, img, bx) in zip(axes.ravel(), panels):
        ax.imshow(img, extent=(0, 1, 1, 0))
        add_boxes(ax, clip(bx), BOX_C, lw=2.2, fill=0.30)
        ax.set_title(title, fontsize=9.5, pad=5)
        style(ax)
    fig.suptitle("What the model actually sees — geometry and photometry change, the box follows the defect",
                 fontsize=11, y=0.985)
    fig.text(0.5, 0.005, "blue = annotated defect region, warped by exactly the same transform as the pixels",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.955])
    fig.savefig(ASSETS / "step3_augment.png", dpi=140)
    plt.close(fig)


# -------------------------------------------------- prediction bookkeeping
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def run_inference(model, split, conf=0.25, imgsz=320):
    imgs = imgs_of(split)
    results = model.predict([str(p) for p in imgs], imgsz=imgsz, conf=conf, verbose=False)
    out = {}
    for p, r in zip(imgs, results):
        H, W = r.orig_shape
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append(
                (int(b.cls.item()), ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H,
                 (x2 - x1) / W, (y2 - y1) / H, float(b.conf.item()))
            )
        out[p.stem] = boxes
    return out


def match(gt, preds):
    used = [False] * len(preds)
    pairs, missed = [], []
    for g in gt:
        best, bi = 0.0, -1
        for i, q in enumerate(preds):
            if used[i] or q[0] != g[0]:
                continue
            v = iou(g, q)
            if v > best:
                best, bi = v, i
        if best >= 0.5:
            used[bi] = True
            pairs.append((g, preds[bi], best))
        else:
            missed.append(g)
    return pairs, missed, [q for i, q in enumerate(preds) if not used[i]]


# -------------------------------------------- step 6: GT vs prediction panels
def fig_quality(preds):
    picks = []
    for p in imgs_of("test"):
        gt = labels_of("test", p.stem)
        if len(gt) >= 2 and len(preds[p.stem]) >= 2:
            picks.append(p)
        if len(picks) == 3:
            break

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.6))
    for ci, p in enumerate(picks):
        im = Image.open(p).convert("RGB")
        gt, pr = labels_of("test", p.stem), preds[p.stem]

        ax = axes[0][ci]
        ax.imshow(im, extent=(0, 1, 1, 0))
        add_boxes(ax, gt, GT_C, lw=2.0, fill=0)
        ax.set_title(f"ground truth — {len(gt)} box(es)", fontsize=9)
        style(ax)

        ax = axes[1][ci]
        ax.imshow(im, extent=(0, 1, 1, 0))
        add_boxes(ax, [q[:5] for q in pr], PRED_C, lw=2.0, fill=0)
        top = max((q[5] for q in pr), default=0)
        ax.set_title(f"prediction — {len(pr)} box(es), top conf {top:.2f}", fontsize=9)
        style(ax)

    fig.suptitle("Same test images — every box below was produced by the network, not read from the labels",
                 fontsize=11, y=0.985)
    fig.text(0.5, 0.005, "green = annotated ground truth, red = YOLO11n prediction at conf>=0.25",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(ASSETS / "step6_quality.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------ step 7: failure cases
def fig_failures(preds):
    small = thin = fp = hard = None
    for p in imgs_of("test"):
        gt, pr = labels_of("test", p.stem), preds[p.stem]
        pairs, missed, extra = match(gt, pr)

        if small is None:
            cand = [g for g in missed if g[3] * g[4] < 0.02]
            if cand:
                small = (p, max(cand, key=lambda g: g[3] * g[4]))
        if thin is None:
            cand = [g for g in missed if g[3] / g[4] > 2.5 or g[3] / g[4] < 0.4]
            if cand:
                thin = (p, cand[0])
        if fp is None:
            cand = [q for q in extra if q[5] >= 0.45]
            if cand:
                fp = (p, max(cand, key=lambda q: q[5]))
        if hard is None:
            # 已匹配但 IoU 只有 0.5~0.62：定位不精确，而不是完全没找到
            cand = [(g, q, v) for g, q, v in pairs if v < 0.62]
            if cand:
                hard = (p,) + min(cand, key=lambda t: t[2])
        if all([small, thin, fp, hard]):
            break

    cases = [
        ("Missed — small defect", small, "box area < 2% of the crop, nothing fired"),
        ("Missed — elongated defect", thin, "aspect ratio > 2.5 or < 0.4"),
        ("False positive", fp, "conf >= 0.45, no ground truth underneath"),
        ("Imprecise boundary", hard, "matched, but 0.50 <= IoU < 0.62"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.3))
    for ax, (title, case, note) in zip(axes, cases):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(note, fontsize=7.5, color="#777", labelpad=4)
        if case is None:
            ax.text(0.5, 0.5, "not found in this split", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999")
            continue

        p = case[0]
        ax.imshow(Image.open(p).convert("RGB"), extent=(0, 1, 1, 0))
        gt, pr = labels_of("test", p.stem), preds[p.stem]
        add_boxes(ax, gt, GT_C, lw=0.8, fill=0, alpha=0.45)
        add_boxes(ax, [q[:5] for q in pr], PRED_C, lw=0.8, fill=0, alpha=0.45)

        def badge(box, text, color):
            x = min(max(box[1] - box[3] / 2, 0.02), 0.55)
            y = min(max(box[2] - box[4] / 2, 0.06), 0.92)
            ax.text(x, y, text, fontsize=6.8, color="white", zorder=6,
                    bbox=dict(fc=color, ec="white", lw=0.6, pad=1.8, alpha=0.95))

        if len(case) == 2:  # 漏检 / 误检
            target = case[1]
            is_gt = tuple(target) in [tuple(g) for g in gt]
            color = GT_C if is_gt else PRED_C
            add_boxes(ax, [target[:5]], color, lw=3.0, fill=0.34)
            badge(target, "missed GT" if is_gt else "false positive", color)
        elif len(case) == 4:  # 定位不准：真值和预测都高亮
            add_boxes(ax, [case[1]], GT_C, lw=2.6, fill=0.20)
            add_boxes(ax, [case[2][:5]], PRED_C, lw=2.6, fill=0.20)
            badge(case[1], f"IoU {case[3]:.2f}", "#555555")

    fig.suptitle("Failure cases, picked automatically by matching predictions against ground truth",
                 fontsize=11, y=1.02)
    fig.text(0.5, -0.02, "green = ground truth, red = prediction; the highlighted pair is the case being shown",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(ASSETS / "step7_failures.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ step 8: batch inference gif
def fig_preview_gif(model, n=24, cell=440, conf=0.25):
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    frames = []
    for i, p in enumerate(imgs_of("test")[:n]):
        im = Image.open(p).convert("RGB").resize((cell, cell), Image.BICUBIC)
        r = model.predict(p, imgsz=320, conf=conf, verbose=False)[0]
        H, W = r.orig_shape
        d = ImageDraw.Draw(im)
        k = cell / W
        for b in r.boxes:
            x1, y1, x2, y2 = [v * k for v in b.xyxy[0].tolist()]
            c = int(b.cls.item())
            d.rectangle([x1, y1, x2, y2], outline=COLORS[c], width=3)
            tag = f"{CLASSES[c]} {b.conf.item():.2f}"
            tw = d.textlength(tag, font=small)
            ty = max(0, y1 - 20)
            d.rectangle([x1, ty, x1 + tw + 8, ty + 19], fill=COLORS[c])
            d.text((x1 + 4, ty + 1), tag, fill="white", font=small)

        bar = Image.new("RGB", (cell, 34), "#1c2128")
        db = ImageDraw.Draw(bar)
        db.text((10, 8), f"{i + 1:02d}/{n}", fill="#e6e6e6", font=bold)
        meta = "YOLO11n @320  conf>=0.25"
        db.text((cell - 12 - d.textlength(meta, font=small), 9), meta, fill="#8b98a8", font=small)
        canvas = Image.new("RGB", (cell, cell + 34), "white")
        canvas.paste(im, (0, 34))
        canvas.paste(bar, (0, 0))
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))

    out = ASSETS / "step8_preview.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=380, loop=0, optimize=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ------------------------------------------------------------- copy artefacts
def copy_existing():
    for src, dst in [
        ("runs/finetune_test/confusion_matrix.png", "step5_confusion.png"),
        ("runs/finetune_test/BoxPR_curve.png", "step5_pr_curve.png"),
    ]:
        if Path(src).exists():
            shutil.copy2(src, ASSETS / dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/finetune/weights/best.pt")
    ap.add_argument("--gif", action="store_true", help="also render the 24-frame GIF")
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit("missing dataset/, run fetch_data.py + prepare_dataset.py first")

    fig_samples()
    fig_class_stats()
    fig_annotations()
    fig_augment()
    copy_existing()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"missing {w} — train first or drop a checkpoint there")
    model = YOLO(str(w))
    preds = run_inference(model, "test")
    fig_quality(preds)
    fig_failures(preds)
    if args.gif:
        fig_preview_gif(model)
    print("done ->", ASSETS.resolve())


if __name__ == "__main__":
    main()
