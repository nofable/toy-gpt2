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
