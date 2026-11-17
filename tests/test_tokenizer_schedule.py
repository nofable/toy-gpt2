"""Tokenizer, LR schedule and eval checks. Run:  uv run pytest -q"""

import math

import pytest

from toy_gpt2.eval import bits_per_byte
from toy_gpt2.schedule import get_lr
from toy_gpt2.tokenizer import BPETokenizer, get_pair_counts, merge


def test_pair_counts():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


@pytest.mark.parametrize(
    "ids, pair, new, expected",
    [([1, 2, 3, 1, 2], (1, 2), 99, [99, 3, 99]), ([5, 5, 5], (5, 5), 9, [9, 5]), ([1, 2, 3], (1, 2), 99, [99, 3]),
     ([], (1, 2), 99, []), ([7], (1, 2), 99, [7])],
)
def test_merge(ids, pair, new, expected):
    assert merge(ids, pair, new) == expected


def test_bpe_round_trip_unseen_text():
    tok = BPETokenizer()
    tok.train("the cat sat on the mat. the dog sat on the log.", 280)
    for s in ["the mat", "Hello, world! 👋 naïve café", "", "def f(x):\n    return x**2"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses():
    tok = BPETokenizer()
    text = "the cat sat on the mat. " * 20
    tok.train(text, 300)
    assert len(tok.encode(text)) < len(text.encode()) / 3


def test_lr_schedule():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=1000)
    assert 0 < get_lr(0, **kw) <= 1e-5 + 1e-12
    assert math.isclose(get_lr(100, **kw), 1e-3)
    assert math.isclose(get_lr(550, **kw), 5.5e-4)
    assert math.isclose(get_lr(1000, **kw), 1e-4)
    assert math.isclose(get_lr(5000, **kw), 1e-4)

def test_lr_schedule_warmup_equals_max():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=100)
    assert math.isclose(get_lr(100, **kw), 1e-4)
    

def test_bits_per_byte():
    # 1 bit per token (ln 2 nats), 1 byte per token -> 1 bit per byte
    assert math.isclose(bits_per_byte(math.log(2), 100, 100), 1.0)
    # same loss, 4 bytes per token -> 0.25
    assert math.isclose(bits_per_byte(math.log(2), 100, 400), 0.25)
