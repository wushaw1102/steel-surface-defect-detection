"""Render test-set predictions: ground truth in green, predictions in red."""
import random
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path("dataset/neu_det")
OUT = Path("assets")
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = Path("runs/finetune/weights/best.pt")
CELL = 300
COLS, ROWS = 4, 2
random.seed(7)


def load_gt(stem):
    lbl = ROOT / "labels" / "test" / f"{stem}.txt"
    boxes = []
    for line in lbl.read_text().strip().splitlines():
        c, cx, cy, w, h = line.split()
        boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def draw(img, boxes, color, width=2, confs=None):
    d = ImageDraw.Draw(img)
    W, H = img.size
    for i, (c, cx, cy, w, h) in enumerate(boxes):
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if confs is not None:
            d.text((x1 + 2, max(0, y1 - 11)), f"{confs[i]:.2f}", fill=color)
    return img


def main():
    if not WEIGHTS.exists():
        raise SystemExit("缺少权重 runs/finetune/weights/best.pt，请先运行 scripts/train.py --exp finetune")
    model = YOLO(str(WEIGHTS))
    imgs = sorted((ROOT / "images" / "test").glob("*"))
    picks = random.sample(imgs, COLS * ROWS)

    canvas = Image.new("RGB", (CELL * COLS, CELL * ROWS), "white")
    for idx, p in enumerate(picks):
        img = Image.open(p).convert("RGB").resize((CELL, CELL))
        gt = load_gt(p.stem)
        res = model.predict(p, imgsz=320, conf=0.25, verbose=False)[0]
        preds, confs = [], []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            W, H = res.orig_shape[1], res.orig_shape[0]
            preds.append(
                (
                    int(b.cls.item()),
                    ((x1 + x2) / 2) / W,
                    ((y1 + y2) / 2) / H,
                    (x2 - x1) / W,
                    (y2 - y1) / H,
                )
            )
            confs.append(float(b.conf.item()))
        img = draw(img, gt, (0, 170, 0), 2)
        img = draw(img, preds, (220, 40, 40), 2, confs)
        canvas.paste(img, ((idx % COLS) * CELL, (idx // COLS) * CELL))

    out = OUT / "result_grid.png"
    canvas.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
