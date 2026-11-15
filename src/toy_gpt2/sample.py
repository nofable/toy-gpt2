"""Generate text from a trained checkpoint.

Usage:
  uv run python -m toy_gpt2.sample --run tinystories --prompt "Once upon a time, a little dragon"
  uv run python -m toy_gpt2.sample --run tinystories --temperature 1.2 --top_k 200 --n 3
"""

import argparse
from pathlib import Path

import tiktoken
import torch

from toy_gpt2.gpt import GPT, GPTConfig, get_device


def load_model(run: str, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(Path("runs") / run / "ckpt.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="tinystories")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--n", type=int, default=1, help="number of samples")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = get_device()
    model, ckpt = load_model(args.run, device)
    print(f"loaded runs/{args.run} (step {ckpt['step']}, val {ckpt['val_loss']:.3f})\n")

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode(args.prompt)] * args.n, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    for i, row in enumerate(out.tolist()):
        text = enc.decode(row).split("<|endoftext|>")[0]  # stop at the end of the story
        print(f"--- sample {i + 1} ---\n{text}\n")


if __name__ == "__main__":
    main()
