"""NEU-DET 缺陷检测微调 / 从零训练对照实验。

用法:
    python scripts/train.py --exp finetune   # 预训练权重微调
    python scripts/train.py --exp scratch    # 随机初始化从零训练
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DATA = "dataset/neu_det.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["finetune", "scratch"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=320)  # 原图仅 200x200，无需上采样到 640
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help='CUDA index, or "cpu" / "mps"')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if not Path(DATA).exists():
        raise SystemExit(f"missing {DATA}, run scripts/fetch_data.py and scripts/prepare_dataset.py first")

    # finetune: COCO 预训练权重；scratch: 同结构但随机初始化
    model = YOLO("yolo11n.pt") if args.exp == "finetune" else YOLO("yolo11n.yaml")

    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs",
        name=args.exp,
        exist_ok=True,
        seed=args.seed,
        patience=15,
        pretrained=(args.exp == "finetune"),
        plots=True,
        val=True,
        cache="ram",
    )

    # 训练完成后在独立测试集上评估（不参与调参，结果更可信）
    metrics = model.val(data=DATA, split="test", imgsz=args.imgsz, batch=args.batch,
                        device=args.device)
    out = Path("runs") / f"{args.exp}_test_metrics.txt"
    lines = [
        f"experiment: {args.exp}",
        f"epochs: {args.epochs}",
        f"test mAP50:    {metrics.box.map50:.4f}",
        f"test mAP50-95: {metrics.box.map:.4f}",
        f"test precision: {metrics.box.mp:.4f}",
        f"test recall:    {metrics.box.mr:.4f}",
        "",
        "per-class mAP50-95:",
    ]
    for i, name in enumerate(metrics.names.values()):
        lines.append(f"  {name:<16} {metrics.box.maps[i]:.4f}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
