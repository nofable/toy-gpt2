"""Milestone 4: learning-rate schedule - linear warmup, then cosine decay to a floor.

Run with:  uv run python -m toy_gpt2.schedule
"""

import math


def get_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, max_iters: int) -> float:
    """Learning rate for iteration `it`.

    - it < warmup_iters:         rise linearly from (almost) 0 up to max_lr
    - warmup_iters..max_iters:   cosine curve from max_lr down to min_lr
    - it > max_iters:            stay at min_lr
    """
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    if max_iters > warmup_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)  # 0 -> 1 through the decay
        coeff = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
        return min_lr + coeff * (max_lr - min_lr)
    return min_lr



# --- checks (no need to edit) -------------------------------------------------
def main() -> None:
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=1000)
    assert 0 < get_lr(0, **kw) <= 1e-5 + 1e-12, "step 0 should be tiny but not zero"
    assert math.isclose(get_lr(100, **kw), 1e-3), "end of warmup = max_lr"
    assert math.isclose(get_lr(550, **kw), 5.5e-4), "halfway through decay = midpoint"
    assert math.isclose(get_lr(1000, **kw), 1e-4), "end of decay = min_lr"
    assert math.isclose(get_lr(5000, **kw), 1e-4), "after max_iters = min_lr"
    lrs = [get_lr(i, **kw) for i in range(1200)]
    assert all(a <= b for a, b in zip(lrs[:100], lrs[1:101])), "warmup must rise"
    assert all(a >= b for a, b in zip(lrs[100:1000], lrs[101:1001])), "decay must fall"
    print("all schedule checks passed")

    # tiny ASCII plot
    for i in range(0, 1201, 60):
        lr = get_lr(i, **kw)
        print(f"{i:5d} {lr:.2e} " + "#" * int(lr / 1e-3 * 50))


if __name__ == "__main__":
    main()
