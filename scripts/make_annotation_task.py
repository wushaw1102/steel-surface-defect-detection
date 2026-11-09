"""Build the re-annotation task used for the label-quality audit.

Takes a stratified sample of the **test split** (3 images per class, 18 total), copies them into
`annotation_task/images/` and deliberately does **not** copy the original labels — the point is an
independent re-draw, so the annotator must not see the reference boxes. Then it writes the
single-file annotation page (see `build_annotator.py`), so the job needs no installed tool.

    python scripts/make_annotation_task.py
    # draw the boxes in annotation_task/annotate.html, export annotations.json, then
    python scripts/import_annotations_json.py annotation_task/annotations.json
    python scripts/compare_annotations.py --exclude <the 4 calibration stems>
"""
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("dataset/neu_det")
TASK = Path("annotation_task")
PER_CLASS = 3
SEED = 2026

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


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
    (TASK / "README.txt").write_text(
        f"复标任务：{len(picks)} 张（每类 {PER_CLASS} 张，取自 test 划分，seed={SEED}）\n\n"
        "【怎么标】\n"
        "双击 annotate.html（Chrome / Edge 打开，图片已内嵌，不用装任何东西）\n"
        "鼠标拖框 → 按 1~6 选类别 → 回车下一张 → 画完点“导出 JSON”\n\n"
        "【规则】\n"
        "- 原标注不在这个目录里，请完全凭自己的判断画\n"
        "- 先拿 4 张定规矩，其余盲标（这样一致率才有意义）\n"
        "- 不追求完美：标得不准也是有效数据，这正是要测的东西\n\n"
        "【后续命令】\n"
        "python scripts/import_annotations_json.py annotation_task/annotations.json\n"
        "python scripts/compare_annotations.py --exclude crazing_274,inclusion_287,"
        "pitted_surface_288,scratches_294 --markdown annotation_task/annotation_audit.md\n"
        "python scripts/annotation_compare_figure.py\n",
        encoding="utf-8",
    )

    subprocess.run([sys.executable, "scripts/build_annotator.py"], check=False)

    print(f"images: {len(picks)} -> {TASK / 'images'}")
    print("classes:", ", ".join(CLASSES))


if __name__ == "__main__":
    main()
