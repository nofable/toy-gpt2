"""Milestone 4: train a GPT on TinyStories (GPT-2 tokens) with a proper training loop.

Run:
  uv run python -m toy_gpt2.train --max_iters 50 --name smoke   # quick smoke test
  uv run python -m toy_gpt2.train                               # full run (defaults sized for the M5)

Writes to runs/<name>/:  log.csv (metrics), ckpt.pt (best model), samples.txt

Fill in TODOs W, L and G (and TODO S in gpt.py's `generate`).
"""

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tiktoken
import torch

from toy_gpt2.gpt import GPT, GPTConfig, get_device
from toy_gpt2.schedule import get_lr

DATA = Path("data/tinystories")
PROMPT = "Once upon a time"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="tinystories")
    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--norm_position", choices=["pre", "post"], default="pre")
    p.add_argument("--pos_emb", choices=["learned", "rope"], default="learned")
    p.add_argument("--mlp", choices=["gelu", "swiglu"], default="gelu")
    # optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


# --- data -------------------------------------------------------------------
def load_splits() -> dict[str, np.memmap]:
    # memmap: the 1 GB file stays on disk; we only read the slices we sample
    return {s: np.memmap(DATA / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(model: GPT, splits, args, device) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in splits.items():
        losses = [
            model(*get_batch(data, args.batch_size, args.block_size, device))[1].item()
            for _ in range(args.eval_iters)
        ]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# --- optimiser ----------------------------------------------------------------
def configure_optimizer(model: GPT, args) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    # TODO W: split `params` into two lists:
    #   decay    - the 2D+ tensors (Linear / Embedding weights)
    #   no_decay - the 1D tensors (biases, LayerNorm weight and bias)
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]

    n_decay, n_no = sum(p.numel() for p in decay), sum(p.numel() for p in no_decay)
    print(f"optimizer: {len(decay)} decayed tensors ({n_decay:,} params), "
          f"{len(no_decay)} not decayed ({n_no:,} params)")
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.max_lr, betas=(0.9, 0.95))


# --- main ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    splits = load_splits()
    cfg = GPTConfig(
        vocab_size=enc.n_vocab, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
        norm=args.norm, norm_position=args.norm_position, pos_emb=args.pos_emb, mlp=args.mlp,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.block_size
    print(f"device={device}  params={n_params / 1e6:.2f}M  tokens/step={tokens_per_step:,}  "
          f"total tokens={tokens_per_step * args.max_iters / 1e6:.1f}M")
    opt = configure_optimizer(model, args)

    log_f = open(out_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "lr", "train_loss", "val_loss", "grad_norm", "tok_per_s", "elapsed_s"])
    prompt = torch.tensor([enc.encode(PROMPT)], device=device)
    best_val = float("inf")
    t_start = t_last = time.time()

    for it in range(args.max_iters + 1):
        # TODO L: get this step's learning rate from get_lr and set it on every param group
        #         (optimizers keep their lr in opt.param_groups, a list of dicts)
        lr = get_lr(it, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(splits["train"], args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # TODO G: clip the global gradient norm to args.grad_clip, keeping the norm for logging.
        #         (look at torch.nn.utils.clip_grad_norm_ - what does it return?)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % args.log_interval == 0:
            now = time.time()
            steps = args.log_interval if it else 1
            tok_s = steps * tokens_per_step / (now - t_last)
            t_last = now
            print(f"step {it:5d}  loss {loss.item():.4f}  lr {lr:.2e}  norm {float(grad_norm):.2f}  "
                  f"{tok_s / 1e3:.1f}k tok/s")

        if it % args.eval_interval == 0 or it == args.max_iters:
            l = estimate_loss(model, splits, args, device)
            elapsed = time.time() - t_start
            print(f"--- eval step {it}: train {l['train']:.4f}  val {l['val']:.4f}  ({elapsed / 60:.1f} min)")
            log.writerow([it, lr, l["train"], l["val"], float(grad_norm), round(tok_s), round(elapsed)])
            log_f.flush()
            if l["val"] < best_val:
                best_val = l["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "args": vars(args),
                            "step": it, "val_loss": best_val}, out_dir / "ckpt.pt")
            model.eval()
            sample = enc.decode(model.generate(prompt, 100, temperature=0.8, top_k=50)[0].tolist())
            model.train()
            print(sample.replace("\n", " ") + "\n")
            with open(out_dir / "samples.txt", "a") as f:
                f.write(f"--- step {it} (val {l['val']:.4f}) ---\n{sample}\n\n")

    log_f.close()
    print(f"done. best val {best_val:.4f}. outputs in {out_dir}/")


if __name__ == "__main__":
    main()
