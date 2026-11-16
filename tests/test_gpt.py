"""Model checks. Run:  uv run pytest -q"""

import pytest
import torch
import torch.nn.functional as F

from toy_gpt2.gpt import GPT, Block, CausalSelfAttention, GPTConfig, RMSNorm, SwiGLU, MLP, apply_rope, rope_tables


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_attention_matches_pytorch():
    cfg = GPTConfig()
    a = CausalSelfAttention(cfg)
    x = torch.randn(2, 10, cfg.n_embd)
    B, T, C = x.shape
    hs = C // cfg.n_head
    q, k, v = a.qkv(x).split(C, dim=2)
    q, k, v = [t.view(B, T, cfg.n_head, hs).transpose(1, 2) for t in (q, k, v)]
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, C)
    assert torch.allclose(a(x), a.proj(ref), atol=1e-5)


@pytest.mark.parametrize("kw", [{}, {"pos_emb": "rope"}, {"norm": "rmsnorm", "mlp": "swiglu"}, {"norm_position": "post"}])
def test_model_is_causal(kw):
    m = GPT(GPTConfig(**kw)).eval()
    x = torch.randint(0, 65, (2, 10))
    x2 = x.clone()
    x2[:, 7] = (x2[:, 7] + 1) % 65
    assert torch.allclose(m(x)[0][:, :7], m(x2)[0][:, :7], atol=1e-5), "future token changed the past"


@pytest.mark.parametrize("kw", [{}, {"pos_emb": "rope"}])
def test_model_uses_order(kw):
    # "dog bites man" vs "man bites dog": same last token, same SET of earlier tokens.
    # A model with no sense of order would give identical predictions at the last position.
    m = GPT(GPTConfig(**kw)).eval()
    a, _ = m(torch.tensor([[5, 9, 2, 7]]))
    b, _ = m(torch.tensor([[2, 9, 5, 7]]))
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-6)


def test_param_count_and_tying():
    m = GPT(GPTConfig())  # vocab 65, block 128, 4 layers, 128 wide
    assert m.lm_head.weight is m.wte.weight
    assert sum(p.numel() for p in m.parameters()) == 818_048


def test_initial_loss_near_uniform():
    m = GPT(GPTConfig())
    x = torch.randint(0, 65, (4, 32))
    y = torch.randint(0, 65, (4, 32))
    assert abs(m(x, y)[1].item() - torch.log(torch.tensor(65.0)).item()) < 0.2


def test_rmsnorm_matches_pytorch():
    x = torch.randn(2, 5, 16) * 3 + 1
    mine = RMSNorm(16)
    mine.weight.data = torch.rand(16) + 0.5
    ref = torch.nn.RMSNorm(16, eps=1e-6)
    ref.weight.data = mine.weight.data.clone()
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_rope_is_rotation_and_relative():
    hs, T = 8, 6
    cos, sin = rope_tables(hs, T)
    x = torch.randn(1, 1, T, hs)
    assert torch.allclose(x.norm(dim=-1), apply_rope(x, cos, sin).norm(dim=-1), atol=1e-5)
    q, k = torch.randn(hs), torch.randn(hs)
    Q = apply_rope(q.expand(1, 1, T, hs).contiguous(), cos, sin)[0, 0]
    K = apply_rope(k.expand(1, 1, T, hs).contiguous(), cos, sin)[0, 0]
    scores = [(Q[m] * K[n]).sum() for m, n in [(3, 1), (5, 3), (2, 0)]]
    assert torch.allclose(torch.stack(scores), scores[0].expand(3), atol=1e-5)


def test_swiglu_formula_and_param_match():
    cfg = GPTConfig(n_embd=384)
    s = SwiGLU(cfg)
    x = torch.randn(2, 5, 384)
    assert torch.allclose(s(x), s.down(F.silu(s.gate(x)) * s.up(x)), atol=1e-6)
    n_s = sum(p.numel() for p in s.parameters())
    n_m = sum(p.numel() for p in MLP(cfg).parameters())
    assert abs(n_s - n_m) / n_m < 0.01


def test_post_ln_block():
    cfg = GPTConfig(norm_position="post")
    b = Block(cfg)
    x = torch.randn(2, 5, cfg.n_embd)
    ref = b.ln1(x + b.attn(x))
    ref = b.ln2(ref + b.mlp(ref))
    assert torch.allclose(b(x), ref, atol=1e-6)


def test_top_k_sampling_only_picks_top_k():
    m = GPT(GPTConfig()).eval()
    out = m.generate(torch.zeros((4, 1), dtype=torch.long), 5, temperature=1.0, top_k=1)
    greedy = m.generate(torch.zeros((4, 1), dtype=torch.long), 5, temperature=1.0, top_k=1)
    assert torch.equal(out, greedy), "top_k=1 must be deterministic (greedy)"
