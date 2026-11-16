"""Experiment A plot: final val loss vs non-embedding parameters (log-x), plus the loss curves.

Usage:  uv run python scripts/plot_model_size.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {"size_128": (2, 128), "size_256": (4, 256), "size_384": (6, 384)}  # name: (n_layer, n_embd)
VOCAB, BLOCK = 50257, 256


def non_embedding_params(n_layer: int, C: int) -> int:
    per_block = 2 * (2 * C) + (C * 3 * C + 3 * C) + (C * C + C) + (C * 4 * C + 4 * C) + (4 * C * C + C)
    return n_layer * per_block + 2 * C  # + final LayerNorm


def main() -> None:
    fig, (ax_curve, ax_scale) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs, ys = [], []
    for i, (name, (L, C)) in enumerate(RUNS.items()):
        rows = list(csv.DictReader(open(Path("runs") / name / "log.csv")))
        steps = [int(r["step"]) for r in rows]
        val = [float(r["val_loss"]) for r in rows]
        n = non_embedding_params(L, C)
        total = n + VOCAB * C + BLOCK * C
        ax_curve.plot(steps, val, color=f"C{i}", label=f"{L}L×{C}: {n / 1e6:.1f}M non-emb ({total / 1e6:.1f}M total)")
        xs.append(n)
        ys.append(val[-1])
        ax_scale.annotate(f"{val[-1]:.2f}", (n, val[-1]), textcoords="offset points", xytext=(6, 4))
    ax_scale.plot(xs, ys, "o-", color="C3")
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("non-embedding parameters (log scale)")
    ax_scale.set_ylabel("final val loss")
    ax_scale.set_title("Same 12.3M tokens, bigger model")
    ax_scale.grid(alpha=0.3, which="both")
    ax_curve.set_ylim(top=4)
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("val loss")
    ax_curve.set_title("Val loss during training")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(fontsize=8)
    fig.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/model_size.png", dpi=150)
    for x, y in zip(xs, ys):
        print(f"{x / 1e6:6.2f}M non-emb  val {y:.3f}  tokens/param {12.3e6 / x:5.1f}")


if __name__ == "__main__":
    main()
