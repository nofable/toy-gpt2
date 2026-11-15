"""Milestone 3: a GPT (decoder-only transformer), character-level on Tiny Shakespeare.

Run with:
  uv run python -m toy_gpt2.gpt overfit   # sanity check: memorise ONE batch -> loss near 0
  uv run python -m toy_gpt2.gpt train     # real training; beat the bigram's val loss of 2.49

Fill in every `TODO`.

Shape legend (values for the default config):
  B  = batch: sequences processed in parallel        (32)
  T  = time: positions in each sequence              (up to 128)
  C  = channels: width of each position's vector     (128 = n_embd)
  nh = number of attention heads                     (4)
  hs = head size = C // nh                           (32)
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65 # vocab size of embeddings
    block_size: int = 128  # max context length (T)
    n_layer: int = 4 # number of layers in decoder stack
    n_head: int = 4 # number of heads in multihead attention
    n_embd: int = 128  # C, the width of every vector flowing through the model
    dropout: float = 0.0 # dropour ratio


# --- model ------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # Q, K, V for all heads in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular ones: mask[t, s] == 1 means position t may attend to position s.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head  # head size
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, hs): each head attends independently
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # --- my first version (step by step) ---
        # scores = q @ k.transpose(-2, -1)
        # scaled = scores / k.size(dim=-1)**0.5
        # masked =  torch.where(self.mask[:, :, :T, :T] == 1, scaled, -torch.inf)
        # softmaxed = torch.softmax(masked, -1)
        # dropped = self.dropout(softmaxed)
        # y = dropped @ v

        # Scaled dot-product attention with a causal mask.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # hide the future
        att = self.dropout(F.softmax(att, dim=-1))
        y = att @ v  # (B, nh, T, hs)
        # Milestone 6: all four lines become one fused, faster call:
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-join the heads
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO B: Linear (C -> 4C), GELU, Linear (4C -> C), Dropout.  nn.Sequential is fine.
        C = cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C), nn.Dropout(p=cfg.dropout) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO C: two residual sub-layers, PRE-LN style (normalise the input to each
        #         sub-layer, not the output). Two lines.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # position embeddings (learned)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        # TODO D: weight tying - make lm_head and wte share one weight matrix (one line).
        #         Check: why do their shapes line up?

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # GPT-2 style: small random weights (std 0.02), zero biases.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        # TODO E: token emb + position emb -> dropout -> blocks -> final LayerNorm -> lm_head
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]  # Q: why crop?
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            logits = logits / temperature  # T < 1 sharpens, T > 1 flattens
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)  # (B, k), largest first
                logits[logits < v[:, [-1]]] = float("-inf")  # drop everything below the k-th largest
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# --- data (character-level, same as the bigram) -----------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
