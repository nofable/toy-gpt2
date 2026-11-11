# Learnings

## Milestone 1: Bigram baseline (Tiny Shakespeare, characters)

- **A bigram model is a lookup table.** `nn.Embedding(65, 65)`: row *i* holds the logits for the next character, given that the current character is *i*. Multiplying a one-hot vector by a matrix just selects a row, and an embedding layer is exactly that, done without the multiply. A bias is redundant, because each row can absorb it.
- **Character IDs are labels, not quantities.** Use them only to index. Binary or ASCII encodings would force characters that share bits to share predictions.
- **The initial loss should be ln(vocab_size) = ln 65 ≈ 4.17.** I predicted 4.17. The default N(0,1) initialisation gave 4.65, because random logits make the model *confidently wrong*. Zero init (any equal logits make softmax uniform) gave exactly 4.1744, and early training went faster too (step 300: 2.67 vs 2.81).
- **The loss plateaus at about 2.47, and training longer won't help.** That's the best any model can do when it sees only one character of context. The limit is **context**, not optimisation.
- **There was no overfitting (train ≈ val).** 4,225 parameters were estimated from about 1M characters, and the pair statistics are the same in both splits. Overfitting needs more freedom in the model than the data can pin down.
- **Argmax decoding gets stuck in a loop.** It's deterministic, so each character always leads to the same next one, and with 65 states it must repeat. Sampling with `torch.multinomial` gives varied text.
- **The training step:** forward → `zero_grad()` → `loss.backward()` → `step()`. Without `backward()`, the loss stays flat. Gradients *accumulate*, which is why they're zeroed each step (and that's useful on purpose later).
- **Shapes:** B = batch (32), T = time/position (8), C = classes (65), N = B·T (256). `cross_entropy` wants `(N, C)` and `(N,)`, so flatten with `reshape(-1, C)`.

Result: 3,000 steps, about 6 s on the M5 (MPS). Final train 2.47, val 2.49. The samples have the shape of Shakespeare (speaker names, line breaks, punctuation) but no real words.

## Milestone 2: Byte-level BPE tokenizer

- **Why subwords:** characters give a tiny vocabulary but long sequences. Whole words give a huge vocabulary and can't handle unseen words. BPE sits between them. The biggest gain is **context**: a fixed window of T tokens covers more text when tokens are bigger.
- **The BPE training rule:** start from 256 bytes, count neighbouring pairs, merge the most frequent pair into a new token, replace it everywhere, recount, and repeat. Long tokens come from merging merged tokens (`t`+`h` → `th`, `th`+`e` → `the`, `the`+`␣` → `the␣`), with no length cap and no double counting.
- **Encoding replays the merges in the order they were learned.** Order matters: merge 258 = `[th]`+`e` can only apply after merge 257 has made `[th]`. Byte-level BPE never meets an unknown token, because it can always fall back to bytes.
- **Decoding:** IDs → bytes → UTF-8 with `errors="replace"`. A token can hold half a multi-byte character, and the model can produce ID sequences that aren't valid UTF-8.
- **Compression shows diminishing returns.** On held-out Shakespeare, ours (512 tokens) gave **1.84 bytes/token** (estimate ~2), and GPT-2 (50,257 tokens) gave **3.34** (estimate ~4–4.5, lowered by Shakespeare's punctuation and newlines). That's 98× the vocabulary for 1.8× the compression. Every new merge covers a rarer pattern.
- **Pre-tokenization matters.** Ours learned tokens like `'Citizen:\n'` and `'.\n\nV'`, which overfit this text. GPT-2 first splits the text with a regex (words with a leading space, numbers, punctuation, whitespace), so merges never cross those boundaries and `' Citizen'` is reusable everywhere.
- **Why LLMs can't spell:** `' strawberry'` is a single GPT-2 token (41236). The model sees one ID, not letters. `'strawberry'` without the leading space is `st`+`raw`+`berry`, and `' Strawberry'` is yet another ID.
- **Python gotchas I hit:** iterating over a dict gives its *keys* (use `.items()` or `key=d.get`). After a merge, don't skip the next element. Dicts keep insertion order, so `self.merges` is already in the order the merges were learned.
