"""A minimal byte-level BPE tokenizer (the algorithm behind GPT-2's tokenizer, in pure Python)."""


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for pair in zip(ids[:-1], ids[1:]):
        result[pair] = result.get(pair, 0) + 1
    return result


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    result = []
    i = 0
    while i < len(ids) - 1:
        if (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    if i == len(ids) - 1:
        result.append(ids[-1])
    return result


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            pair_count = get_pair_counts(ids)
            if not pair_count:
                break
            top_pair = max(pair_count.items(), key=lambda item: item[1])[0]
            ids = merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
