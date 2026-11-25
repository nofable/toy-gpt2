"""Plot the GPT-2 124M run: train (smoothed) and val loss against tokens seen, plus the LR schedule.

Usage:  uv run python scripts/plot_gpt2.py [--run gpt2] [--out assets/gpt2_loss.png]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOKENS_PER_STEP = 524_288


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gpt2")
    p.add_argument("--out", default="assets/gpt2_loss.png")
    args = p.parse_args()
    run = Path("runs") / args.run
    steps, evals = read_csv(run / "steps.csv"), read_csv(run / "log.csv")
    hs = json.loads((run / "hellaswag.json").read_text())["acc_norm"]

    k = 20
    smooth = np.convolve(steps["loss"], np.ones(k) / k, mode="valid")
    tok_s = steps["step"][k - 1 :] * TOKENS_PER_STEP / 1e9
    tok_e = evals["step"] * TOKENS_PER_STEP / 1e9

    fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(tok_s, smooth, color="C0", alpha=0.6, label="train (moving avg)")
    ax.plot(tok_e[1:], evals["val_loss"][1:], "o-", color="C1", markersize=3, label="val")
    ax.annotate(f"val {evals['val_loss'][-1]:.3f}\nHellaSwag {hs:.3f} (GPT-2 124M: 0.296)",
                (tok_e[-1], evals["val_loss"][-1]), textcoords="offset points", xytext=(-150, 40),
                fontsize=9, arrowprops={"arrowstyle": "->", "alpha": 0.5})
    ax.set_ylim(2.8, 5.0)
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("GPT-2 124M (RoPE, SwiGLU, RMSNorm) on FineWeb-Edu, 1× A100, 14.9 h, $14.16")
    ax.grid(alpha=0.3)
    ax.legend()
    ax_lr.plot(steps["step"] * TOKENS_PER_STEP / 1e9, steps["lr"], color="C2")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("tokens seen (billions)")
    ax_lr.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
