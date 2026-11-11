"""Milestone 2: a minimal byte-level BPE tokenizer.

Run with:  uv run python -m toy_gpt2.tokenizer
Fill in every `TODO`.
"""

import time
from pathlib import Path


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count each pair of neighbouring IDs.  [1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}"""
    result = {}
    for l_item, r_item in zip(ids[:-1], ids[1:]):
        if (l_item, r_item) in result:
            result[(l_item, r_item)] += 1
        else:
            result[(l_item, r_item)] = 1
    return result
    

def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`, scanning left to right.
    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    """
    # TODO 2: watch out for the end of the list, and don't reuse an ID that was just merged.
    result = []
    i = 0
    while i < len(ids) - 1:
        l_item = ids[i]
        r_item = ids[i + 1]
        if (l_item, r_item) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(l_item)
            i += 1

    if i == len(ids) - 1:
        result.append(ids[-1])
    return result


class BPETokenizer:
    def __init__(self) -> None:
        # Learned merges, in the order they were learned: (id_a, id_b) -> new_id
        self.merges: dict[tuple[int, int], int] = {}
        # What bytes each ID stands for. IDs 0-255 are the raw bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))  # start: one ID per byte, values 0-255
        for new_id in range(256, vocab_size):
            # TODO 3: count pairs, pick the most frequent, merge it into `ids`,
            #         and record it in self.merges and self.vocab.
            pair_count = get_pair_counts(ids)
            if len(pair_count):
                top_pair = max(pair_count.items(), key=lambda item: item[1])[0]
                ids = merge(ids, top_pair, new_id)
                self.merges[top_pair] = new_id
                self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]] 

            if verbose:
                print(f"merge {new_id - 255:4d}: {self.vocab[new_id]!r}")

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # TODO 4: apply the learned merges to `ids`.
        #         Hint: which merge must be applied first, and how do you find it?
        for merge_item, merge_id in self.merges.items():
            ids = merge(ids, merge_item, merge_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        # TODO 5: IDs -> bytes -> str.
        #         Q: why might bytes.decode need errors="replace"?
        bs = b''
        for id in ids:
            bs += self.vocab[id]
        return bs.decode("utf-8", errors="replace")
        


# --- checks and demo (no need to edit) ---------------------------------------
def main() -> None:
    # 1. Unit checks
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([5, 5, 5], (5, 5), 9) == [9, 5]
    print("unit checks passed")

    # 2. Your hand example
    toy = BPETokenizer()
    print("\nfirst 5 merges on 'the cat sat on the mat':")
    toy.train("the cat sat on the mat", 261, verbose=True)

    # 3. Train on Shakespeare (a slice, since pure Python is slow)
    text = Path("data/shakespeare/input.txt").read_text()
    train_text = text[:100_000]
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(train_text, vocab_size=512)
    print(f"\ntrained 256 merges on {len(train_text):,} chars in {time.time() - t0:.1f}s")
    print("last 10 tokens learned:", [tok.vocab[i].decode("utf-8", "replace") for i in range(502, 512)])

    # 4. Round trip, including text it never saw
    for s in [text[200_000:201_000], "Hello, world! 👋 naïve café", "def f(x):\n    return x**2"]:
        assert tok.decode(tok.encode(s)) == s, f"round trip failed for {s[:30]!r}"
    print("round trips passed")

    # 5. Compression on held-out text
    held_out = text[200_000:300_000]
    n_bytes = len(held_out.encode("utf-8"))
    n_tok = len(tok.encode(held_out))
    print(f"held-out compression: {n_bytes:,} bytes -> {n_tok:,} tokens ({n_bytes / n_tok:.2f} bytes/token)")

    # 6. Compare with GPT-2's tokenizer
    import tiktoken

    gpt2 = tiktoken.get_encoding("gpt2")
    sample = "First Citizen:\nBefore we proceed any further, hear me speak."
    for name, ids, dec in [
        ("ours", tok.encode(sample), tok.vocab),
        ("gpt2", gpt2.encode(sample), None),
    ]:
        pieces = [dec[i].decode("utf-8", "replace") if dec else gpt2.decode([i]) for i in ids]
        print(f"\n{name}: {len(ids)} tokens\n{pieces}")
    print(f"\ngpt2 held-out: {n_bytes / len(gpt2.encode(held_out)):.2f} bytes/token  (vocab {gpt2.n_vocab:,})")


if __name__ == "__main__":
    main()
