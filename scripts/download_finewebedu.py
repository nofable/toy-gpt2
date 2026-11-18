"""Download pre-tokenized FineWeb-Edu shards (GPT-2 tokens, 100M per shard) from Hugging Face.

  uv run python scripts/download_finewebedu.py --shards 99    # all ~9.9B train tokens + val (~20 GB)
  uv run python scripts/download_finewebedu.py --shards 1     # val + one train shard, for a smoke test
"""

import argparse
import time
import urllib.request

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2/resolve/main/"
OUT = Path("data/finewebedu")
HEADER_BYTES = 1024


def complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < HEADER_BYTES:
        return False
    magic, _, n_tokens = np.fromfile(path, dtype=np.int32, count=3)
    return magic == 20240520 and path.stat().st_size == HEADER_BYTES + 2 * n_tokens


def fetch(name: str) -> str:
    dest = OUT / name
    if complete(dest):
        return f"[skip] {name}"
    tmp = dest.with_suffix(".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(REPO + name, tmp)
            assert complete(tmp), f"{name}: incomplete ({tmp.stat().st_size} bytes)"
            tmp.rename(dest)
            return f"[done] {name}"
        except Exception as e:
            if attempt == 4:
                raise
            print(f"[retry {attempt + 1}] {name}: {e}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=int, default=99)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["finewebedu_val_000000.bin"] + [f"finewebedu_train_{i:06d}.bin" for i in range(1, args.shards + 1)]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for msg in pool.map(fetch, names):
            print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(names)} files in {OUT}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
