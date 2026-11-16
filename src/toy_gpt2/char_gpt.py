"""Milestone 3: train the GPT character-level on Tiny Shakespeare (val 1.66 vs bigram 2.49).

Run with:
  uv run python -m toy_gpt2.char_gpt overfit   # sanity check: memorise ONE batch -> loss near 0
  uv run python -m toy_gpt2.char_gpt train     # real training
"""

import argparse
import math
import time
from pathlib import Path

import torch

from toy_gpt2.gpt import GPT, GPTConfig, get_device

text = Path("data/shakespeare/input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


def get_batch(split: str, batch_size: int, block_size: int, device: str):
    src = splits[split]
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i : i + block_size] for i in ix])
    y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, cfg, batch_size, device, iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split, batch_size, cfg.block_size, device))[1].item() for _ in range(iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# --- entry points -------------------------------------------------------------
def main() -> None:
    mode = argparse.ArgumentParser()
    mode.add_argument("mode", choices=["overfit", "train"])
    mode = mode.parse_args().mode

    torch.manual_seed(1337)
    device = get_device()
    cfg = GPTConfig(vocab_size=len(chars))
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  (predict this by hand first!)")

    batch_size = 32
    if mode == "overfit":
        # A correct model + training step can memorise a single batch almost perfectly.
        x, y = get_batch("train", batch_size, cfg.block_size, device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(301):
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"step {step:4d}  loss {loss.item():.4f}")
        return

    # mode == "train"
    max_iters, eval_interval = 2000, 250
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            l = estimate_loss(model, cfg, batch_size, device)
            print(f"step {it:5d}  train {l['train']:.4f}  val {l['val']:.4f}  ({time.time() - t0:.0f}s)")
        x, y = get_batch("train", batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"\nbigram baseline val: 2.49  (ln 65 = {math.log(65):.2f})\n")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(ctx, 500)[0].tolist()))


if __name__ == "__main__":
    main()
