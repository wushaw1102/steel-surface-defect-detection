"""Turn a Label Studio YOLO export into annotation_task/labels/.

Label Studio prefixes every exported file with a UUID (`05bbbe93-rolled-in_scale_284.txt`), so the
names do not line up with the images until that prefix is removed. Everything else is already in
YOLO format, with `classes.txt` in the same order as the label config — which is also the order the
dataset uses, so no class remapping is needed.

    python scripts/import_label_studio_export.py annotation_task/label_studio_export_yolo.zip
    python scripts/compare_annotations.py --exclude crazing_274,inclusion_287,pitted_surface_288,scratches_294
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

TASK = Path("annotation_task")
UUID_PREFIX = re.compile(r"^[0-9a-f]{8}-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="YOLO export downloaded from Label Studio")
    ap.add_argument("--out", default=str(TASK / "labels"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = boxes = 0
    with zipfile.ZipFile(io.BytesIO(Path(args.zip).read_bytes())) as zf:
        names = [n for n in zf.namelist() if n.startswith("labels/") and n.endswith(".txt")]
        if not names:
            raise SystemExit("no labels/ entries in the export — did you pick the YOLO format?")

        classes = zf.read("classes.txt").decode().split() if "classes.txt" in zf.namelist() else []
        if classes:
            print("export classes:", ", ".join(classes))

        for name in sorted(names):
            stem = UUID_PREFIX.sub("", Path(name).name[:-4])
            text = zf.read(name).decode().strip()
            (out / f"{stem}.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
            files += 1
            boxes += len([ln for ln in text.splitlines() if ln.strip()])

    print(f"wrote {files} label file(s), {boxes} box(es) -> {out}")
    print("\nnext: python scripts/compare_annotations.py "
          "--exclude crazing_274,inclusion_287,pitted_surface_288,scratches_294")


if __name__ == "__main__":
    main()
