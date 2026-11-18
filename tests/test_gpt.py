import pytest
import torch
import torch.nn.functional as F

from toy_gpt2.gpt import GPT, Block, CausalSelfAttention, GPTConfig, MLP, RMSNorm, SwiGLU, apply_rope, rope_tables

MODERN = {"norm": "rmsnorm", "pos_emb": "rope", "mlp": "swiglu"}
VARIANTS = [{}, {"pos_emb": "rope"}, MODERN, {"norm_position": "post"}]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("kw", [{}, {"pos_emb": "rope"}])
def test_attention_matches_manual_masked_softmax(kw):
    cfg = GPTConfig(**kw)
    a = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 10, cfg.n_embd)
    B, T, C = x.shape
    hs = C // cfg.n_head
    q, k, v = a.qkv(x).split(C, dim=2)
    q, k, v = [t.view(B, T, cfg.n_head, hs).transpose(1, 2) for t in (q, k, v)]
    if a.use_rope:
        q, k = apply_rope(q, a.rope_cos[:T], a.rope_sin[:T]), apply_rope(k, a.rope_cos[:T], a.rope_sin[:T])
    att = (q @ k.transpose(-2, -1)) / hs**0.5
    att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf")).softmax(-1)
    ref = a.proj((att @ v).transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(a(x), ref, atol=1e-5)


@pytest.mark.parametrize("kw", VARIANTS)
def test_model_is_causal(kw):
    m = GPT(GPTConfig(**kw)).eval()
    x = torch.randint(0, 65, (2, 10))
    x2 = x.clone()
    x2[:, 7] = (x2[:, 7] + 1) % 65
    assert torch.allclose(m(x)[0][:, :7], m(x2)[0][:, :7], atol=1e-5)


@pytest.mark.parametrize("kw", [{}, {"pos_emb": "rope"}])
def test_model_uses_order(kw):
    m = GPT(GPTConfig(**kw)).eval()
    a, _ = m(torch.tensor([[5, 9, 2, 7]]))
    b, _ = m(torch.tensor([[2, 9, 5, 7]]))
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-6)


def gpt2_small(**kw) -> GPTConfig:
    return GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, **kw)


def n_params(cfg: GPTConfig) -> int:
    with torch.device("meta"):
        return sum(p.numel() for p in GPT(cfg).parameters())


def test_param_count_matches_openai_gpt2():
    assert n_params(gpt2_small()) == 124_439_808


def test_param_count_modern_gpt2():
    assert n_params(gpt2_small(**MODERN)) == 123_588_096


def test_weight_tying():
    m = GPT(GPTConfig())
    assert m.lm_head.weight is m.wte.weight


@pytest.mark.parametrize("kw", [{}, MODERN])
def test_residual_projections_get_scaled_init(kw):
    cfg = GPTConfig(n_layer=8, n_embd=256, **kw)
    m = GPT(cfg)
    block = m.blocks[0]
    out = block.mlp.down if cfg.mlp == "swiglu" else block.mlp.net[2]
    expected = 0.02 / (2 * cfg.n_layer) ** 0.5
    assert abs(block.attn.proj.weight.std().item() - expected) < expected * 0.1
    assert abs(out.weight.std().item() - expected) < expected * 0.1
    assert abs(block.attn.qkv.weight.std().item() - 0.02) < 0.002


def test_initial_loss_near_uniform():
    m = GPT(GPTConfig(**MODERN))
    x, y = torch.randint(0, 65, (4, 32)), torch.randint(0, 65, (4, 32))
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
    scores = torch.stack([(Q[m] * K[n]).sum() for m, n in [(3, 1), (5, 3), (2, 0)]])
    assert torch.allclose(scores, scores[0].expand(3), atol=1e-5)


def test_rope_keeps_bf16():
    cos, sin = rope_tables(8, 4)
    assert apply_rope(torch.randn(1, 1, 4, 8, dtype=torch.bfloat16), cos, sin).dtype == torch.bfloat16


def test_swiglu_formula_and_param_match():
    cfg = GPTConfig(n_embd=384)
    s = SwiGLU(cfg)
    x = torch.randn(2, 5, 384)
    assert torch.allclose(s(x), s.down(F.silu(s.gate(x)) * s.up(x)), atol=1e-6)
    n_s, n_m = sum(p.numel() for p in s.parameters()), sum(p.numel() for p in MLP(cfg).parameters())
    assert abs(n_s - n_m) / n_m < 0.01


def test_post_ln_block():
    b = Block(GPTConfig(norm_position="post"))
    x = torch.randn(2, 5, 128)
    ref = b.ln1(x + b.attn(x))
    assert torch.allclose(b(x), b.ln2(ref + b.mlp(ref)), atol=1e-6)


def test_generate_top_k_1_is_greedy_and_respects_vocab_limit():
    m = GPT(GPTConfig(vocab_size=72)).eval()
    start = torch.zeros((3, 1), dtype=torch.long)
    out = m.generate(start, 20, top_k=1, vocab_limit=65)
    assert torch.equal(out, m.generate(start, 20, top_k=1, vocab_limit=65))
    assert out.max().item() < 65


def test_generate_crops_to_block_size():
    m = GPT(GPTConfig(block_size=8, **MODERN)).eval()
    assert m.generate(torch.zeros((1, 1), dtype=torch.long), 20).shape == (1, 21)
