"""Convert the browser annotator's JSON into YOLO label files.

The page stores boxes as (class, x, y, w, h) in image fractions with a top-left origin; YOLO wants
centre-origin, so the conversion happens here — in one place, testable.

    python scripts/import_annotations_json.py annotations.json
    python scripts/compare_annotations.py
"""
import argparse
import json
from pathlib import Path

TASK = Path("annotation_task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="annotations.json downloaded from the annotation page")
    ap.add_argument("--out", default=str(TASK / "labels"))
    ap.add_argument("--only-drawn", action="store_true",
                    help="skip images with zero boxes instead of writing empty label files")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = empty = total = 0
    for name, boxes in data["images"].items():
        stem = Path(name).stem
        if not boxes and args.only_drawn:
            empty += 1
            continue
        lines = []
        for c, x, y, w, h in boxes:
            lines.append(f"{int(c)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            total += 1
        (out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    print(f"wrote {written} label files, {total} boxes -> {out}")
    if empty:
        print(f"  skipped {empty} images with no box (not yet annotated)")
    print("\nnext: python scripts/compare_annotations.py")


if __name__ == "__main__":
    main()
