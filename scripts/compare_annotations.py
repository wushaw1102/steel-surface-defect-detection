"""Compare a manual re-annotation against the dataset's original labels.

Greedy same-class matching at IoU >= 0.5, the usual detection criterion. Reports how many boxes
were redrawn, where they land relative to the reference, and where the two annotators disagree —
i.e. a small inter-annotator agreement study rather than a self-assessment.

    python scripts/compare_annotations.py
    python scripts/compare_annotations.py --mine annotation_task/labels --markdown audit.md
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def read_labels(path):
    if not path.exists():
        return None
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, cx, cy, w, h = parts[:5]
            out.append((int(float(c)), float(cx), float(cy), float(w), float(h)))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def match(ref, mine, thr=0.5):
    used = [False] * len(mine)
    pairs, missed = [], []
    for r in ref:
        best, bi = 0.0, -1
        for i, m in enumerate(mine):
            if used[i] or m[0] != r[0]:
                continue
            v = iou(r, m)
            if v > best:
                best, bi = v, i
        if best >= thr:
            used[bi] = True
            pairs.append((r, mine[bi], best))
        else:
            missed.append(r)
    return pairs, missed, [m for i, m in enumerate(mine) if not used[i]]


def median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def mask_of(boxes, w, h):
    """Rasterise boxes into a boolean mask so granularity stops mattering."""
    m = np.zeros((h, w), dtype=bool)
    for _, cx, cy, bw, bh in boxes:
        x1 = int(round((cx - bw / 2) * w)); x2 = int(round((cx + bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h)); y2 = int(round((cy + bh / 2) * h))
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        m[y1:y2, x1:x2] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", default="annotation_task/labels", help="YOLO labels drawn by hand")
    ap.add_argument("--ref", default="dataset/neu_det/labels/test", help="reference labels")
    ap.add_argument("--images", default="annotation_task/images", help="images that were annotated")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--exclude", default="",
                    help="comma-separated stems used to calibrate the box convention; excluded from stats")
    ap.add_argument("--markdown", default=None, help="also write a markdown summary here")
    args = ap.parse_args()

    skip = {s.strip() for s in args.exclude.split(",") if s.strip()}
    imgs = sorted(Path(args.images).glob("*"))
    done = [p for p in imgs
            if (Path(args.mine) / f"{p.stem}.txt").exists() and p.stem not in skip]
    if not done:
        raise SystemExit(f"no labels found in {args.mine} for images in {args.images}")

    n_ref = n_mine = n_match = 0
    ious, ref_only, mine_only = [], [], []
    per_class = Counter()
    empty_both = 0
    inter_px = union_px = ref_px = mine_px = 0

    for p in done:
        ref = read_labels(Path(args.ref) / f"{p.stem}.txt") or []
        mine = read_labels(Path(args.mine) / f"{p.stem}.txt") or []
        if not ref and not mine:
            empty_both += 1
        pairs, missed, extra = match(ref, mine, args.iou)
        n_ref += len(ref)
        n_mine += len(mine)
        n_match += len(pairs)
        ious += [v for _, _, v in pairs]
        ref_only += [(p.stem, g) for g in missed]
        mine_only += [(p.stem, m) for m in extra]
        for g, _, _ in pairs:
            per_class[CLASSES[g[0]]] += 1

        with Image.open(p) as im:
            w, h = im.size
        mr, mm = mask_of(ref, w, h), mask_of(mine, w, h)
        inter_px += int((mr & mm).sum())
        union_px += int((mr | mm).sum())
        ref_px += int(mr.sum())
        mine_px += int(mm.sum())

    rec = n_match / n_ref if n_ref else 0.0
    prec = n_match / n_mine if n_mine else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    mask_iou = inter_px / union_px if union_px else 0.0
    mask_rec = inter_px / ref_px if ref_px else 0.0
    mask_prec = inter_px / mine_px if mine_px else 0.0

    img_per_class = Counter(p.stem.rsplit("_", 1)[0] for p in done)
    spread = "/".join(str(img_per_class[c]) for c in CLASSES)
    calib = f"{len(skip)} 张样例先校准框选粒度（crazing 按裂纹簇切分、inclusion 每条夹杂单独成框），" if skip else ""

    lines = [
        "## 手工标注一致性核查",
        "",
        f"从 test 划分抽出 {len(done) + len(skip)} 张（逐类 3 张），{calib}"
        f"其余 {len(done)} 张（逐类 {spread} 张）在看不到原标注的情况下盲标，再逐框比对。",
        "匹配规则：同类且 IoU ≥ %.2f。" % args.iou,
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 核查图像 | {len(done)} 张 |",
        f"| 原标注框 / 我标的框 | {n_ref} / {n_mine} |",
        f"| 匹配上的框 | {n_match} |",
        f"| 漏标（原有、我没标） | {len(ref_only)} |",
        f"| 多标（我标了、原标注没有） | {len(mine_only)} |",
        f"| 匹配框平均 IoU | {sum(ious) / len(ious):.3f} |" if ious else "| 匹配框平均 IoU | — |",
        f"| 匹配框 IoU 中位数 | {median(ious):.3f} |" if ious else "| 匹配框 IoU 中位数 | — |",
        f"| 按原标注计的召回 | {rec:.3f} |",
        f"| 按我的标注计的精度 | {prec:.3f} |",
        f"| F1 | {f1:.3f} |",
        "",
        "框的切分粒度因人而异（同一片缺陷，一个框还是拆成几个框都对），所以上面这组逐框指标会低估一致性。"
        "把两边的框都栅格化成像素掩膜、再算区域重合，就可以把「切法不同」和「真的漏看」分开：",
        "",
        "| 区域级指标（与切分粒度无关） | 值 |",
        "|---|---|",
        f"| 区域 IoU | {mask_iou:.3f} |",
        f"| 原标注区域被覆盖比例 | {mask_rec:.3f} |",
        f"| 我标出的区域里属于缺陷的比例 | {mask_prec:.3f} |",
        "",
        "逐类匹配数：" + "，".join(f"{c} {per_class.get(c, 0)}" for c in CLASSES),
    ]
    text = "\n".join(lines)
    print(text)

    if args.markdown:
        Path(args.markdown).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {args.markdown}")


if __name__ == "__main__":
    main()
