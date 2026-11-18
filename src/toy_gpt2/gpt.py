"""GPT model: a decoder-only transformer with switchable GPT-2 / modern components."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    norm: str = "layernorm"
    norm_position: str = "pre"
    pos_emb: str = "learned"
    mlp: str = "gelu"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.eps)


def make_norm(cfg: GPTConfig) -> nn.Module:
    return RMSNorm(cfg.n_embd) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


def rope_tables(head_size: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = base ** (-torch.arange(0, head_size, 2).float() / head_size)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout_p = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj.residual_out = True
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_rope = cfg.pos_emb == "rope"
        if self.use_rope:
            cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.n_embd
        self.net = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C), nn.Dropout(cfg.dropout))
        self.net[2].residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.n_embd
        hidden = int(8 * C / 3)
        self.gate = nn.Linear(C, hidden, bias=False)
        self.up = nn.Linear(C, hidden, bias=False)
        self.down = nn.Linear(hidden, C, bias=False)
        self.down.residual_out = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.mlp == "swiglu" else MLP(cfg)
        self.post_ln = cfg.norm_position == "post"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_ln:
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.mlp(x))
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_emb == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg) if cfg.norm_position == "pre" else nn.Identity()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            std = 0.02
            if getattr(m, "residual_out", False):
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        vocab_limit: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :].float() / temperature
            if vocab_limit is not None:
                logits[:, vocab_limit:] = float("-inf")
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
