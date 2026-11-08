"""Sweep the confidence threshold on the test split and report precision / recall.

The mAP numbers in the README are threshold-independent, but the operating point a
production line cares about is not — this prints the trade-off explicitly.

    python scripts/threshold_sweep.py
"""
import argparse
import contextlib
import io

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/finetune/weights/best.pt")
    ap.add_argument("--data", default="dataset/neu_det.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--confs", default="0.25,0.35,0.45,0.60")
    args = ap.parse_args()

    model = YOLO(args.weights)
    print(f"{'conf':>6} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    for c in [float(x) for x in args.confs.split(",")]:
        with contextlib.redirect_stdout(io.StringIO()):
            r = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                          conf=c, batch=16, plots=False, verbose=False)
        p, rec = r.box.mp, r.box.mr
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        print(f"{c:>6.2f} {p:>10.4f} {rec:>8.4f} {f1:>7.4f}")


if __name__ == "__main__":
    main()
