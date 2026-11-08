"""Regenerate the comparison figures used by the READMEs.

Everything here is derived from `runs/*/results.csv` and the two
`runs/*_test_metrics.txt` files, so the pictures can never drift away from the
numbers quoted in the documentation.

    python scripts/make_figures.py
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path("assets")
ASSETS.mkdir(parents=True, exist_ok=True)

EXPS = [("finetune", "Fine-tuned (COCO weights)", "#1f77b4"),
        ("scratch", "From scratch (random init)", "#d62728")]

CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled-in_scale", "scratches"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})


def read_results(exp):
    rows = list(csv.DictReader(open(f"runs/{exp}/results.csv", encoding="utf-8")))
    cols = list(rows[0])
    m50 = next(c for c in cols if c.endswith("mAP50(B)"))
    m5095 = next(c for c in cols if c.endswith("mAP50-95(B)"))
    epoch = [int(float(r["epoch"])) for r in rows]
    return epoch, [float(r[m50]) for r in rows], [float(r[m5095]) for r in rows]


def read_test_metrics(exp):
    """Parse runs/<exp>_test_metrics.txt into (summary dict, per-class list)."""
    text = Path(f"runs/{exp}_test_metrics.txt").read_text(encoding="utf-8")
    head = dict(re.findall(r"^(test [a-zA-Z0-9\-]+):\s+([0-9.]+)$", text, re.M))
    per = dict(re.findall(r"^  (\S+)\s+([0-9.]+)$", text, re.M))
    return head, [float(per[c]) for c in CLASSES]


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)

    for exp, label, colour in EXPS:
        epoch, m50, m5095 = read_results(exp)
        # Push the two annotations apart: fine-tuned above its marker, scratch below.
        offset = (-6, 8) if exp == "finetune" else (-6, -18)
        for ax, series, name in ((axes[0], m50, "mAP@50"), (axes[1], m5095, "mAP@50-95")):
            ax.plot(epoch, series, color=colour, lw=1.6, label=label)
            best_i = max(range(len(series)), key=series.__getitem__)
            ax.scatter([epoch[best_i]], [series[best_i]], s=26, color=colour,
                       zorder=5, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{series[best_i]:.3f} @ ep{epoch[best_i]}",
                        (epoch[best_i], series[best_i]),
                        textcoords="offset points", xytext=offset, ha="right",
                        fontsize=7.5, color=colour, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.85))
            ax.set_title(name, fontsize=10, loc="left")
            ax.set_xlabel("epoch")

    axes[1].axvline(52, color="0.55", lw=0.9, ls="--")
    axes[1].text(52.8, axes[1].get_ylim()[0] + 0.007, "fine-tune plateaus\nat epoch 52",
                 fontsize=7.5, color="0.35", va="bottom")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Validation curves, 60 epochs at imgsz=320", x=0.012, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / "curve_compare.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


def plot_per_class():
    _, ft = read_test_metrics("finetune")
    _, sc = read_test_metrics("scratch")

    x = range(len(CLASSES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    b1 = ax.bar([i - w / 2 for i in x], ft, w, label="Fine-tuned", color="#1f77b4")
    b2 = ax.bar([i + w / 2 for i in x], sc, w, label="From scratch", color="#d62728")

    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_ylabel("mAP@50-95 on the 180-image test split")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where the gain comes from: scratches and rolled-in scale, not crazing",
                 fontsize=10, loc="left")
    fig.tight_layout()
    out = ASSETS / "per_class_map.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    plot_curves()
    plot_per_class()
