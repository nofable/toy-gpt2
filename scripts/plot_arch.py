"""Experiments B & C plot: fixed-slice val loss per architecture variant (reads runs/*/eval.json).

Usage:  uv run python scripts/plot_arch.py   (run `python -m toy_gpt2.eval ...` first)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = [
    ("size_384", "GPT-2 baseline\n(pre-LN, LayerNorm,\nlearned pos, GELU)"),
    ("arch_postln", "post-LN"),
    ("arch_rmsnorm", "+ RMSNorm"),
    ("arch_rmsnorm_rope", "+ RMSNorm\n+ RoPE"),
    ("arch_rmsnorm_rope_swiglu", "+ RMSNorm\n+ RoPE\n+ SwiGLU"),
]


def main() -> None:
    evals = {r: json.loads((Path("runs") / r / "eval.json").read_text()) for r, _ in VARIANTS}
    base = evals["size_384"]["val_loss"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#888888", "C1", "C0", "C0", "C0"]
    for i, (run, label) in enumerate(VARIANTS):
        v = evals[run]["val_loss"]
        ax.bar(i, v, color=colors[i])
        delta = "" if run == "size_384" else f"\n({v - base:+.3f})"
        ax.text(i, v + 0.003, f"{v:.3f}{delta}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(VARIANTS)), [l for _, l in VARIANTS], fontsize=8)
    ax.set_ylim(1.9, 2.12)
    ax.axhline(base, color="#888888", linestyle="--", linewidth=1)
    ax.set_ylabel("val loss (fixed 1M-token slice)")
    ax.set_title("Architecture variants: 30M params, 12.3M TinyStories tokens each")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("assets/architecture.png", dpi=150)
    print("saved assets/architecture.png")


if __name__ == "__main__":
    main()
