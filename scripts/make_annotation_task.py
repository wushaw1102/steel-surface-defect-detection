"""Build the manual re-annotation task used for the label-quality audit.

We take a stratified sample of the **test split** (7 images per class, 42 total), copy the
images into `annotation_task/images/` and deliberately do **not** copy the original labels —
the point is an independent re-draw, so the annotator must not see the reference boxes.

    python scripts/make_annotation_task.py

Then annotate with Label Studio (config: annotation_task/label_studio_config.xml) or labelImg
(classes: annotation_task/classes.txt), export YOLO format into annotation_task/labels/, and run
`python scripts/compare_annotations.py`.
"""
import random
import shutil
from pathlib import Path

ROOT = Path("dataset/neu_det")
TASK = Path("annotation_task")
PER_CLASS = 3
SEED = 2026

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
{labels}  </RectangleLabels>
</View>
"""


def main():
    random.seed(SEED)
    imgs = sorted((ROOT / "images" / "test").glob("*"))

    picks = []
    for ci in range(len(CLASSES)):
        pool = []
        for p in imgs:
            first = (ROOT / "labels" / "test" / f"{p.stem}.txt").read_text().split()
            if first and int(first[0]) == ci:
                pool.append(p)
        picks += random.sample(pool, min(PER_CLASS, len(pool)))

    if (TASK / "images").exists():
        shutil.rmtree(TASK / "images")
    (TASK / "images").mkdir(parents=True)
    (TASK / "labels").mkdir(exist_ok=True)

    for p in picks:
        shutil.copy2(p, TASK / "images" / p.name)

    (TASK / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (TASK / "label_studio_config.xml").write_text(
        LABEL_CONFIG.format(
            labels="".join(
                f'    <Label value="{c}" background="{col}"/>\n'
                for c, col in zip(
                    CLASSES,
                    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                )
            )
        ),
        encoding="utf-8",
    )
    (TASK / "README.txt").write_text(
        "手动标注任务\n"
        f"共 {len(picks)} 张（每类 {PER_CLASS} 张，取自 test 划分，seed={SEED}）\n\n"
        "Label Studio：New Project → Labeling Setup → Code → 粘贴 label_studio_config.xml 的内容\n"
        "              → Import 导入 annotation_task/images/ → 标完后 Export as YOLO\n"
        "labelImg：Open Dir = annotation_task/images/，Change Save Dir = annotation_task/labels/\n"
        "          （按 W 画框，选择类别后保存）\n\n"
        "注意：原标注**没有**放进这个目录，请完全独立地标一遍，不要去看 dataset/ 里的标签。\n",
        encoding="utf-8",
    )
    print(f"images: {len(picks)} -> {TASK / 'images'}")
    print("classes:", ", ".join(CLASSES))


if __name__ == "__main__":
    main()
