"""Take a Label Studio YOLO export (zip) and drop the label files where the audit expects them.

Label Studio exports a zip shaped like

    classes.txt
    notes.json
    labels/crazing_274.txt
    images/...            (sometimes present)

This script finds every YOLO .txt, copies it into annotation_task/labels/, and reports what it
found — so the folder layout never becomes the reason the audit fails.

    python scripts/import_ls_export.py ~/Downloads/export.zip
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

TASK = Path("annotation_task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="exported zip from Label Studio (YOLO format)")
    ap.add_argument("--out", default=str(TASK / "labels"))
    args = ap.parse_args()

    src = Path(args.zip).expanduser()
    if not src.exists():
        raise SystemExit(f"not found: {src}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = {p.stem for p in (TASK / "images").glob("*")} if (TASK / "images").exists() else set()

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(src) as z:
            z.extractall(td)
        txts = [p for p in Path(td).rglob("*.txt") if p.name != "classes.txt"]
        copied, skipped, empty = 0, 0, 0
        for p in txts:
            if images and p.stem not in images:
                skipped += 1
                continue
            body = p.read_text().strip()
            if not body:
                empty += 1
            shutil.copy2(p, out / f"{p.stem}.txt")
            copied += 1

    print(f"copied {copied} label files -> {out}")
    if empty:
        print(f"  of which {empty} are empty (no box drawn) — that is a valid annotation")
    if skipped:
        print(f"  skipped {skipped} files whose image is not in {TASK / 'images'}")
    print("\nnext: python scripts/compare_annotations.py")


if __name__ == "__main__":
    main()
