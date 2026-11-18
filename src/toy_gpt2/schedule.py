"""Learning-rate schedule: linear warmup, then cosine decay to a floor."""

import math


def get_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, max_iters: int) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters or max_iters <= warmup_iters:
        return min_lr
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
