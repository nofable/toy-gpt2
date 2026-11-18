"""Generate text from a trained checkpoint.

  uv run python -m toy_gpt2.sample --run gpt2 --prompt "The history of the Roman Empire"
  uv run python -m toy_gpt2.sample --run tinystories --temperature 1.2 --top_k 200 --n 3
"""

import argparse
from pathlib import Path

import tiktoken
import torch

from toy_gpt2.gpt import GPT, GPTConfig, get_device


def load_model(run: str, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(Path("runs") / run / "ckpt.pt", map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    state = {k: v for k, v in ckpt["model"].items() if not k.endswith(".attn.mask")}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gpt2")
    p.add_argument("--prompt", default="The most important thing to know about")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = get_device()
    model, ckpt = load_model(args.run, device)
    print(f"loaded runs/{args.run} (step {ckpt['step']}, val {ckpt.get('val_loss')})\n")

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode(args.prompt)] * args.n, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, vocab_limit=enc.n_vocab)
    for i, row in enumerate(out.tolist()):
        print(f"--- sample {i + 1} ---\n{enc.decode(row).split('<|endoftext|>')[0]}\n")


if __name__ == "__main__":
    main()
