import csv
import math

import numpy as np
import torch

from toy_gpt2 import train
from toy_gpt2.data import HEADER_BYTES, SHARD_MAGIC, ShardLoader, load_shard
from toy_gpt2.eval import bits_per_byte
from toy_gpt2.gpt import GPT, GPTConfig
from toy_gpt2.schedule import get_lr
from toy_gpt2.tokenizer import BPETokenizer, get_pair_counts, merge


def test_pair_counts():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([5, 5, 5], (5, 5), 9) == [9, 5]
    assert merge([], (1, 2), 99) == []
    assert merge([7], (1, 2), 99) == [7]


def test_bpe_round_trip_and_compression():
    tok = BPETokenizer()
    text = "the cat sat on the mat. " * 20
    tok.train(text, 300)
    assert len(tok.encode(text)) < len(text.encode()) / 3
    for s in ["the mat", "Hello, world! 👋 naïve café", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_lr_schedule():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=1000)
    assert 0 < get_lr(0, **kw) <= 1e-5 + 1e-12
    assert math.isclose(get_lr(100, **kw), 1e-3)
    assert math.isclose(get_lr(550, **kw), 5.5e-4)
    assert math.isclose(get_lr(1000, **kw), 1e-4)
    assert math.isclose(get_lr(5000, **kw), 1e-4)
    assert math.isclose(get_lr(100, max_lr=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=100), 1e-4)


def test_bits_per_byte():
    assert math.isclose(bits_per_byte(math.log(2), 100, 100), 1.0)
    assert math.isclose(bits_per_byte(math.log(2), 100, 400), 0.25)


def write_shard(path, tokens, header: bool) -> None:
    with open(path, "wb") as f:
        if header:
            h = np.zeros(HEADER_BYTES // 4, dtype=np.int32)
            h[:3] = [SHARD_MAGIC, 1, len(tokens)]
            f.write(h.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def test_load_shard_with_and_without_header(tmp_path):
    toks = np.arange(100) % 50000
    write_shard(tmp_path / "a.bin", toks, header=True)
    write_shard(tmp_path / "b.bin", toks, header=False)
    assert np.array_equal(load_shard(tmp_path / "a.bin"), toks)
    assert np.array_equal(load_shard(tmp_path / "b.bin"), toks)


def test_loader_is_sequential_shifted_and_resumable(tmp_path):
    paths = []
    for i in range(2):
        write_shard(tmp_path / f"s{i}.bin", np.arange(i * 1000, i * 1000 + 50), header=True)
        paths.append(tmp_path / f"s{i}.bin")
    loader = ShardLoader(paths, batch_size=2, block_size=4, device="cpu")
    x, y = loader.next_batch()
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert torch.equal(y, x + 1)
    batches = [loader.next_batch() for _ in range(10)]
    state = loader.state()
    expected = [loader.next_batch() for _ in range(4)]
    resumed = ShardLoader(paths, batch_size=2, block_size=4, device="cpu")
    resumed.load_state(state)
    for (ex, ey), (rx, ry) in zip(expected, [resumed.next_batch() for _ in range(4)]):
        assert torch.equal(ex, rx) and torch.equal(ey, ry)
    assert any(b[0][0, 0].item() >= 1000 for b in batches), "loader never moved to the second shard"


def run_train(tmp_path, monkeypatch, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/tinystories").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        write_shard(tmp_path / f"data/tinystories/{split}.bin", rng.integers(0, 50257, 20_000), header=False)
    train.main([
        "--name", "t", "--data", "tinystories", "--n_layer", "2", "--n_head", "2", "--n_embd", "32",
        "--block_size", "16", "--batch_size", "4", "--total_batch_tokens", "128", "--warmup_iters", "2",
        "--eval_interval", "4", "--eval_batches", "2", "--log_interval", "1", "--ckpt_interval", "3",
        "--no-compile", "--no-hellaswag", *extra,
    ])
    return tmp_path / "runs/t"


def test_train_end_to_end_with_grad_accum_and_resume(tmp_path, monkeypatch):
    out = run_train(tmp_path, monkeypatch, "--max_iters", "6")
    ckpt = torch.load(out / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 6 and ckpt["max_iters"] == 6
    assert ckpt["loader"] == {"shard": 0, "pos": 6 * 128}
    assert "state" in ckpt["optimizer"] and len(ckpt["optimizer"]["state"]) > 0
    rows = list(csv.DictReader(open(out / "log.csv")))
    assert [r["step"] for r in rows] == ["0", "4", "6"]
    assert all(math.isfinite(float(r["val_loss"])) for r in rows)
    run_train(tmp_path, monkeypatch, "--resume")
    rows = list(csv.DictReader(open(out / "log.csv")))
    assert rows[-1]["step"] == "6"


def read_steps(out):
    return {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(out / "steps.csv"))}


def test_resume_mid_run_matches_uninterrupted_run(tmp_path, monkeypatch):
    straight = read_steps(run_train(tmp_path / "a", monkeypatch, "--max_iters", "8"))
    out = run_train(tmp_path / "b", monkeypatch, "--max_iters", "8", "--stop_after", "4")
    assert torch.load(out / "ckpt.pt", weights_only=False)["step"] == 4
    run_train(tmp_path / "b", monkeypatch, "--resume")
    resumed = read_steps(out)
    assert sorted(resumed) == list(range(8))
    for step in range(8):
        assert math.isclose(resumed[step], straight[step], abs_tol=2e-3), (step, resumed[step], straight[step])


def test_fit_max_iters():
    kw = dict(warmup_iters=700)
    assert train.fit_max_iters(70, 19074, sec_per_step=1.0, remaining_s=10_000, overhead=1.0, **kw) == 10_070
    assert train.fit_max_iters(70, 19074, sec_per_step=0.5, remaining_s=20_000, **kw) == 19074
    assert train.fit_max_iters(70, 19074, sec_per_step=10.0, remaining_s=100, **kw) == 701
    fitted = train.fit_max_iters(70, 19074, sec_per_step=1.13, remaining_s=5.3 * 3600, **kw)
    assert 16_000 < fitted < 17_000


def test_bf16_autocast_forward_backward():
    m = GPT(GPTConfig(norm="rmsnorm", pos_emb="rope", mlp="swiglu", vocab_size=128))
    x, y = torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = m(x, y)
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32 and math.isfinite(loss.item())
    loss.backward()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in m.parameters())
