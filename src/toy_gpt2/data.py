"""Token shards on disk -> (x, y) batches.

A shard is a flat uint16 array of GPT-2 token ids, either raw (TinyStories `train.bin`) or with the
1024-byte header used by the pre-tokenized FineWeb-Edu shards (magic 20240520, version, n_tokens).
"""

from pathlib import Path

import numpy as np
import torch

SHARD_MAGIC = 20240520
HEADER_BYTES = 1024

DATASETS = {
    "tinystories": ("data/tinystories", "train.bin", "val.bin"),
    "finewebedu": ("data/finewebedu", "finewebedu_train_*.bin", "finewebedu_val_*.bin"),
}


def load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=3)
    if len(header) == 3 and header[0] == SHARD_MAGIC:
        tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
        assert len(tokens) == header[2], f"{path}: header says {header[2]} tokens, file has {len(tokens)}"
        return tokens
    return np.memmap(path, dtype=np.uint16, mode="r")


def shard_paths(dataset: str, split: str) -> list[Path]:
    root, train_glob, val_glob = DATASETS[dataset]
    paths = sorted(Path(root).glob(train_glob if split == "train" else val_glob))
    assert paths, f"no {split} shards for {dataset} in {root}/"
    return paths


class ShardLoader:
    """Reads shards in order, so every token is seen once per epoch. With DDP, `world` processes read
    interleaved B*T chunks: together they consume exactly what one process would in `world` calls."""

    def __init__(self, paths: list[Path], batch_size: int, block_size: int, device: str, rank: int = 0,
                 world: int = 1):
        self.paths, self.B, self.T, self.device = paths, batch_size, block_size, device
        self.rank, self.world = rank, world
        self.reset()

    def reset(self) -> None:
        self.shard, self.pos = 0, 0
        self.tokens = load_shard(self.paths[0])

    def state(self) -> dict:
        return {"shard": self.shard, "pos": self.pos}

    def load_state(self, state: dict) -> None:
        self.shard, self.pos = state["shard"], state["pos"]
        self.tokens = load_shard(self.paths[self.shard])

    def total_tokens(self) -> int:
        return sum(len(load_shard(p)) for p in self.paths)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.B * self.T
        if self.pos + n * self.world + 1 > len(self.tokens):
            self.shard = (self.shard + 1) % len(self.paths)
            self.pos = 0
            self.tokens = load_shard(self.paths[self.shard])
        start = self.pos + self.rank * n
        buf = torch.from_numpy(self.tokens[start : start + n + 1].astype(np.int64))
        self.pos += n * self.world
        x, y = buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
