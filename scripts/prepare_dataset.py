"""整理 NEU-DET 为 YOLO 三划分结构，并输出类别分布。"""
import random
import shutil
from collections import Counter
from pathlib import Path

SRC = Path("neu_src/data/NEU-DET")
DST = Path("dataset/neu_det")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)


def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def main():
    if not SRC.exists():
        raise SystemExit(f"source dataset not found: {SRC}\nrun `python scripts/fetch_data.py` first")

    for split in ("train", "val", "test"):
        reset_dir(DST / "images" / split)
        reset_dir(DST / "labels" / split)

    train_imgs = sorted((SRC / "train" / "images").glob("*"))
    test_imgs = sorted((SRC / "test" / "images").glob("*"))
    random.shuffle(train_imgs)
    n_val = int(len(train_imgs) * VAL_RATIO)
    val_imgs, tr_imgs = train_imgs[:n_val], train_imgs[n_val:]

    # 源目录只有 train / test，val 是从 train 里切出来的
    plan = [
        ("train", tr_imgs, "train"),
        ("val", val_imgs, "train"),
        ("test", test_imgs, "test"),
    ]
    dist = {}
    for split, imgs, src_split in plan:
        for img in imgs:
            lbl = SRC / src_split / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copy2(img, DST / "images" / split / img.name)
            shutil.copy2(lbl, DST / "labels" / split / lbl.name)
            for line in lbl.read_text().strip().splitlines():
                cid = int(line.split()[0])
                dist.setdefault(split, Counter())[CLASSES[cid]] += 1
        print(f"{split}: {len(list((DST / 'images' / split).glob('*')))} imgs")

    yaml_text = (
        f"path: {DST.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (Path("dataset") / "neu_det.yaml").write_text(yaml_text, encoding="utf-8")
    print("\n--- yaml ---\n", yaml_text)

    print("\n--- 各类别实例数 ---")
    header = "split".ljust(8) + "".join(c.rjust(16) for c in CLASSES)
    print(header)
    for split, c in dist.items():
        print(split.ljust(8) + "".join(str(c.get(cl, 0)).rjust(16) for cl in CLASSES))


if __name__ == "__main__":
    main()
