"""Render test-set predictions: ground truth in green, predictions in red."""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path("dataset/neu_det")
OUT = Path("assets")
OUT.mkdir(parents=True, exist_ok=True)
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
    ap = argparse.ArgumentParser(description="GT vs prediction grid on the test split")
    ap.add_argument("--weights", default="runs/finetune/weights/best.pt",
                    help="checkpoint to visualise; a release download works too")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7, help="which 8 images to sample")
    ap.add_argument("--out", default=str(OUT / "result_grid.png"))
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"missing {weights}\n"
            "train it with `python scripts/train.py --exp finetune`, or grab the\n"
            "published checkpoint from the Releases page."
        )
    if not (ROOT / "images" / "test").exists():
        raise SystemExit(f"missing {ROOT}, run scripts/fetch_data.py and scripts/prepare_dataset.py first")

    random.seed(args.seed)
    model = YOLO(str(weights))
    imgs = sorted((ROOT / "images" / "test").glob("*"))
    picks = random.sample(imgs, COLS * ROWS)

    canvas = Image.new("RGB", (CELL * COLS, CELL * ROWS), "white")
    for idx, p in enumerate(picks):
        img = Image.open(p).convert("RGB").resize((CELL, CELL))
        gt = load_gt(p.stem)
        res = model.predict(p, imgsz=320, conf=args.conf, verbose=False)[0]
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
