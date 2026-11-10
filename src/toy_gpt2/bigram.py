"""Milestone 1: a character-level bigram language model on Tiny Shakespeare.

Run with:  uv run python -m toy_gpt2.bigram
Fill in every `TODO` (each is 1-3 lines).
"""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# --- hyperparameters --------------------------------------------------------
batch_size = 32  # independent sequences per batch
block_size = 8  # length of each sequence
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
torch.manual_seed(1337)


def get_device() -> str:
    """Prefer an NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()

# --- data -------------------------------------------------------------------
text = Path("data/shakespeare/input.txt").read_text()
chars = sorted(set(text))
vocab_size = len(chars)

# TODO 1: lookups in both directions, then encode/decode using them.
stoi: dict[str, int] = {s: i for i, s in enumerate(chars)}
itos: dict[int, str] = {i: s for s, i in stoi.items()}


def encode(s: str) -> list[int]:
    result = []
    for c in s:
        i = stoi.get(c)
        if i is None:
            raise Exception(f"{c} not in text set")
        else:
            result.append(i)
    return result


def decode(ids: list[int]) -> str:
    result = []
    for i in ids:
        s = itos.get(i)
        if s is None:
            raise Exception(f"{i} not in itos set")
        else:
            result.append(s)
    return "".join(result)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inputs x and targets y, each of shape (batch_size, block_size)."""
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))  # random start positions
    # TODO 2: build x and y from `src` and the start positions in `ix`.
    #         (Remember the "First Cit" inputs/targets rows.)
    x =  torch.stack([src[startp:startp + block_size] for startp in ix])
    y =  torch.stack([src[startp + 1: startp + block_size + 1] for startp in ix])
    return x.to(device), y.to(device)


# --- model ------------------------------------------------------------------
class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO 3: the whole model is one layer.
        self.table = nn.Embedding(vocab_size, vocab_size)
        # Equal logits -> uniform softmax -> initial loss = ln(vocab_size)
        nn.init.zeros_(self.table.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx and targets: (B, T) integer IDs
        logits = self.table(idx)  # TODO 4a: (B, T, vocab_size)
        if targets is None:
            return logits, None
        # TODO 4b: F.cross_entropy wants logits (N, C) and targets (N,).
        #          Reshape so every (batch, time) position is one example.
        loss = torch.nn.functional.cross_entropy(logits.reshape((-1, logits.shape[-1])), targets.reshape(-1)) 
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # TODO 5: sample one next ID per row from `probs` and append it to `idx`.
            sample = torch.multinomial(probs, 1)
            idx = torch.cat((idx, sample), -1)
        return idx


@torch.no_grad()
def estimate_loss(model: nn.Module) -> dict[str, float]:
    """Average loss over several batches, so the printed number isn't noisy."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --- train ------------------------------------------------------------------
def main() -> None:
    print(f"device={device}  vocab_size={vocab_size}  train={len(train_data):,}  val={len(val_data):,}")
    model = BigramLM(vocab_size).to(device)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    print(f"loss before training: {loss.item():.4f}   (you predicted 4.17)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for it in range(max_iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        xb, yb = get_batch("train")
        # TODO 6: one optimisation step (forward, reset grads, backward, update).
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start from ID 0 ('\n')
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
