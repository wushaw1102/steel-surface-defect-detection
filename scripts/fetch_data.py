"""Clone the source NEU-DET dataset (already in YOLO format) into neu_src/."""
import subprocess
from pathlib import Path

REPO = "https://github.com/Marfbin/NEU-DET-with-yolov8.git"
DST = Path("neu_src")


def main():
    if (DST / "data" / "NEU-DET").exists():
        print(f"already present: {DST}")
        return
    print(f"cloning {REPO} -> {DST} (~145 MB)")
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(DST)], check=True)
    print("done. next: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
