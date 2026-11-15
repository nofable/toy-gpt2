"""Plot loss curves (and LR) from one or more runs/<name>/log.csv files.

Usage:  uv run python scripts/plot_run.py tinystories [other_run ...] [--out assets/loss.png]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(name: str) -> dict[str, list[float]]:
    with open(Path("runs") / name / "log.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="assets/loss.png")
    args = p.parse_args()

    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for i, name in enumerate(args.runs):
        log = read_log(name)
        color = f"C{i}"
        ax_loss.plot(log["step"], log["val_loss"], color=color, label=f"{name} val")
        ax_loss.plot(log["step"], log["train_loss"], color=color, linestyle="--", alpha=0.6, label=f"{name} train")
        ax_lr.plot(log["step"], log["lr"], color=color)
        ax_loss.annotate(f"{log['val_loss'][-1]:.2f}", (log["step"][-1], log["val_loss"][-1]),
                         textcoords="offset points", xytext=(4, 0), color=color, fontsize=9)

    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_yscale("log")
    ax_loss.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax_loss.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("step")
    ax_lr.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
