"""Audit the dataset's labels for the defects that actually break training.

No human judgement involved — these are the mechanical failure modes that ruin a detector and
are cheap to detect: boxes outside the image, degenerate boxes, duplicates, two classes fighting
over the same region, boxes that cover the whole crop.

    python scripts/audit_labels.py --markdown annotation_task/label_audit.md
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="dataset/neu_det/labels")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    n_img = n_box = 0
    outside, degen, whole, dup, cross, empty = [], [], [], [], [], []
    per_class = Counter()

    for split in args.splits.split(","):
        for f in sorted((Path(args.labels) / split).glob("*.txt")):
            boxes = []
            for line in f.read_text().strip().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                boxes.append((int(float(p[0])), *map(float, p[1:5])))
            n_img += 1
            n_box += len(boxes)
            if not boxes:
                empty.append(f"{split}/{f.stem}")
            for c, cx, cy, w, h in boxes:
                per_class[CLASSES[c]] += 1
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                if x1 < -0.002 or y1 < -0.002 or x2 > 1.002 or y2 > 1.002:
                    outside.append(f"{split}/{f.stem}")
                if w < 0.01 or h < 0.01:
                    degen.append(f"{split}/{f.stem}")
                if w * h > 0.95:
                    whole.append(f"{split}/{f.stem}")
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    v = iou(a, b)
                    if a[0] == b[0] and v > 0.90:
                        dup.append(f"{split}/{f.stem}")
                    elif a[0] != b[0] and v > 0.80:
                        cross.append(
                            f"{split}/{f.stem} ({CLASSES[a[0]]} vs {CLASSES[b[0]]}, IoU {v:.2f})"
                        )

    def uniq(xs):
        return sorted(set(xs))

    lines = [
        "## 标注质量审查",
        "",
        f"对 `{args.splits}` 全部 **{n_img} 张 / {n_box} 个框**跑了一遍结构性检查"
        "（`scripts/audit_labels.py`，无需人工判读）：",
        "",
        "| 检查项 | 判据 | 命中 |",
        "|---|---|---|",
        f"| 框越过图像边界 | 角点超出 [0,1] | {len(uniq(outside))} 处 |",
        f"| 退化框 | 宽或高 < 1% | {len(uniq(degen))} 处 |",
        f"| 整图框 | 面积 > 95% | {len(uniq(whole))} 处 |",
        f"| 同类重复框 | 同类且 IoU > 0.90 | {len(uniq(dup))} 处 |",
        f"| 跨类重叠 | 异类且 IoU > 0.80 | {len(cross)} 处 |",
        f"| 空标注图 | 无任何框 | {len(uniq(empty))} 张 |",
        "",
        "逐类框数：" + "，".join(f"{c} {per_class[c]}" for c in CLASSES),
    ]
    if cross:
        lines += ["", "跨类重叠（可能是误标，需要人看一眼）："]
        lines += [f"- `{c}`" for c in cross[:10]]

    text = "\n".join(lines)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {args.markdown}")


if __name__ == "__main__":
    main()
