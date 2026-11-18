"""HellaSwag (validation, 10,042 examples): pick the ending the model finds most likely.

  uv run python -m toy_gpt2.hellaswag --run gpt2            # score a checkpoint
  uv run python -m toy_gpt2.hellaswag --run gpt2 --limit 500

Scores acc_norm the way nanoGPT/llm.c do: lowest *mean* loss over the ending tokens wins.
OpenAI's GPT-2 124M scores ~0.295 with this method; random guessing is 0.25.
"""

import argparse
import json
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
import torch
import torch.nn.functional as F

from toy_gpt2.gpt import GPT, get_device

URL = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet"
LOCAL = Path("data/hellaswag/validation.parquet")


def load_examples() -> list[dict]:
    if not LOCAL.exists():
        LOCAL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, LOCAL)
    return pq.read_table(LOCAL, columns=["ctx", "endings", "label"]).to_pylist()


def render(example: dict, enc) -> tuple[torch.Tensor, torch.Tensor, int]:
    ctx = enc.encode(example["ctx"])
    rows = [ctx + enc.encode(" " + ending) for ending in example["endings"]]
    width = max(len(r) for r in rows)
    tokens = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = torch.tensor(row)
        mask[i, len(ctx) : len(row)] = True
    return tokens, mask, int(example["label"])


@torch.no_grad()
def evaluate_model(model: GPT, device: str, out_path: Path | None = None, limit: int | None = None) -> float:
    enc = tiktoken.get_encoding("gpt2")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    examples = load_examples()[:limit]
    model.eval()
    correct = 0
    for n, example in enumerate(examples, 1):
        tokens, mask, label = render(example, enc)
        tokens, mask = tokens.to(device), mask.to(device)
        with autocast:
            logits, _ = model(tokens)
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1), reduction="none"
        ).view(tokens.size(0), -1)
        m = mask[:, 1:]
        mean_loss = (losses * m).sum(1) / m.sum(1)
        correct += int(mean_loss.argmin().item() == label)
        if n % 1000 == 0:
            print(f"{n}/{len(examples)}  acc_norm {correct / n:.4f}", flush=True)
    acc = correct / len(examples)
    if out_path is not None:
        out_path.write_text(json.dumps({"acc_norm": acc, "correct": correct, "n": len(examples)}, indent=2))
    return acc


def main() -> None:
    from toy_gpt2.sample import load_model

    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gpt2")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    device = get_device()
    model, _ = load_model(args.run, device)
    acc = evaluate_model(model, device, Path("runs") / args.run / "hellaswag.json", args.limit)
    print(f"{args.run}: HellaSwag acc_norm {acc:.4f}")


if __name__ == "__main__":
    main()
