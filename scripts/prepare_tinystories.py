"""Download TinyStories (V2, GPT-4 generated) and tokenize it with the GPT-2 tokenizer.

Usage:  uv run python scripts/prepare_tinystories.py
Output: data/tinystories/{train,val}.bin  (flat uint16 arrays of token IDs)
        Stories are separated by GPT-2's <|endoftext|> token (50256).
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
OUT = Path("data/tinystories")
RAW = OUT / "raw"
SEP = "<|endoftext|>"


def download(name: str) -> Path:
    dest = RAW / name
    if dest.exists():
        print(f"[skip] {dest} exists")
        return dest
    tmp = dest.with_suffix(".part")
    print(f"[download] {name}")
    with urllib.request.urlopen(BASE + name) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, last = 0, 0.0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                last = time.time()
    tmp.rename(dest)
    return dest


def iter_story_batches(path: Path, batch_bytes: int = 64 << 20):
    """Yield lists of stories, reading the file in chunks (never all in memory)."""
    rest = ""
    with open(path, encoding="utf-8") as f:
        while block := f.read(batch_bytes):
            parts = (rest + block).split(SEP)
            rest = parts.pop()  # last piece may be an incomplete story
            yield [s.strip() for s in parts if s.strip()]
    if rest.strip():
        yield [rest.strip()]


def tokenize(src: Path, dst: Path) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    n_tokens = n_stories = 0
    t0 = time.time()
    with open(dst, "wb") as f:
        for stories in iter_story_batches(src):
            out: list[int] = []
            for ids in enc.encode_ordinary_batch(stories, num_threads=8):
                out.extend(ids)
                out.append(eot)
            np.asarray(out, dtype=np.uint16).tofile(f)  # GPT-2 vocab 50,257 < 65,536 fits in uint16
            n_tokens += len(out)
            n_stories += len(stories)
            print(f"  {dst.name}: {n_stories:,} stories, {n_tokens:,} tokens ({time.time() - t0:.0f}s)", flush=True)
    return {"stories": n_stories, "tokens": n_tokens, "bytes": src.stat().st_size}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split in ("val", "train"):  # val first: small, so problems show up fast
        src = download(FILES[split])
        print(f"[tokenize] {split}")
        meta[split] = tokenize(src, OUT / f"{split}.bin")
    meta["vocab_size"] = 50257
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
