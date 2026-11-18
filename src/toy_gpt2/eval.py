"""Fixed eval harness: score checkpoints on the same first ~1M tokens of a val split, every time.

  uv run python -m toy_gpt2.eval size_128 size_256 size_384 arch_rmsnorm
Writes runs/<name>/eval.json (val loss and bits per byte).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from toy_gpt2.data import load_shard, shard_paths
from toy_gpt2.gpt import get_device
from toy_gpt2.sample import load_model


def bits_per_byte(mean_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    return mean_loss_nats * n_tokens / math.log(2) / n_bytes


@torch.no_grad()
def evaluate(run: str, n_eval_tokens: int, batch_size: int, device: str, dataset: str = "tinystories") -> dict:
    model, ckpt = load_model(run, device)
    T = model.cfg.block_size
    data = load_shard(shard_paths(dataset, "val")[0])
    n_windows = n_eval_tokens // T
    starts = np.arange(n_windows) * T

    enc = tiktoken.get_encoding("gpt2")
    token_bytes = np.array([len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)])

    total_nats, n_tokens, n_bytes = 0.0, 0, 0
    t0 = time.time()
    for b in range(0, n_windows, batch_size):
        s = starts[b : b + batch_size]
        x = torch.from_numpy(np.stack([data[i : i + T] for i in s]).astype(np.int64)).to(device)
        y_np = np.stack([data[i + 1 : i + 1 + T] for i in s]).astype(np.int64)
        y = torch.from_numpy(y_np).to(device)
        logits, _ = model(x)
        logits = logits[..., : enc.n_vocab]
        total_nats += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        n_tokens += y.numel()
        n_bytes += int(token_bytes[y_np].sum())

    mean_loss = total_nats / n_tokens
    result = {
        "run": run,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "val_loss": round(mean_loss, 4),
        "bits_per_byte": round(bits_per_byte(mean_loss, n_tokens, n_bytes), 4),
        "eval_tokens": n_tokens,
        "eval_bytes": n_bytes,
        "train_step": ckpt["step"],
        "seconds": round(time.time() - t0, 1),
    }
    (Path("runs") / run / "eval.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--tokens", type=int, default=1 << 20)
    p.add_argument("--data", choices=["tinystories", "finewebedu"], default="tinystories")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = get_device()
    print(f"{'run':28s} {'params':>8s} {'val loss':>9s} {'bits/byte':>10s}")
    for run in args.runs:
        r = evaluate(run, args.tokens, args.batch_size, device, args.data)
        print(f"{r['run']:28s} {r['params_M']:7.2f}M {r['val_loss']:9.4f} {r['bits_per_byte']:10.4f}")


if __name__ == "__main__":
    main()
