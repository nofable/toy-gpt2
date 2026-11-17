# Quiz: intuition check

Questions on topics I found tricky. Try to answer each one *before* opening the answer.
Come back to these every few sessions.

**Status:** 🔁 = revisit · ✅ = solid

---

## Losses and what they mean

### 1. 🔁 What does a model learn first?
Training on TinyStories starts at a loss of 10.87. Before it learns any grammar, what's the simplest thing that drives the loss down fast?

<details><summary>Answer</summary>

**Token frequency (a unigram model).** Predicting common tokens like "the", ".", "," and "and", whatever the context, already beats uniform guessing by a lot. The smoke-test samples showed it: *"the the. a to a big day"*. The tokenizer is fixed before training, so the model never learns *which characters form tokens*.
</details>

### 2. 🔁 Can you compare these two losses?
Shakespeare-char GPT: val 1.66. TinyStories GPT: val ≈ 2.0. Which is the better model?

<details><summary>Answer</summary>

**They aren't comparable.**
1. **Different units:** nats per *character* versus nats per *token* (about 4.1 bytes each). Per byte, TinyStories ≈ 2.0 / 4.1 ≈ 0.49, about 3× lower.
2. **Different data:** simple children's stories are far more predictable than Shakespeare.

Fair comparison = the **same eval text, in the same units** (bits per byte). The same goes for "lowercasing gives a lower loss": an easier task, not a better model.
</details>

### 3. ✅ What's the starting loss?
What should an untrained model's loss be with a vocabulary of V tokens, and what makes it come out *higher* than that?

<details><summary>Answer</summary>

**ln(V)**, i.e. uniform guessing (ln 65 = 4.17, ln 50,257 = 10.83). Large random initial logits make the model *confidently wrong*, which costs more than uniform (N(0,1) init gave 4.65). Fix: small init (std 0.02) or zeros.
</details>

---

## Attention

### 4. 🔁 Why divide by √(head size)?
You forget the `/ sqrt(hs)` and hs = 64. What happens?

<details><summary>Answer</summary>

Each score is a sum of 64 products, so its std is about √64 = **8× larger**. Softmax over large scores becomes **nearly one-hot**: each position locks onto one other position, roughly at random. Softmax's gradient there is about 0, so attention **can't learn to shift its focus**. Training stalls. Scaling keeps the scores at a variance of about 1 whatever the head size.
</details>

### 5. 🔁 What if you drop the causal mask?
What happens to the training loss, and to `generate`?

<details><summary>Answer</summary>

Training loss collapses towards 0, but that's **copying, not overfitting**: each position reads its target from the next position. `generate` produces **garbage**, because at generation time the next token doesn't exist yet. The model learned to depend on information that's never available.
</details>

### 6. ✅ What if you drop the position embeddings?
Could the model learn that "the" is common? Tell "dog bites man" from "man bites dog"?

<details><summary>Answer</summary>

Only the first. Without positions, attention treats earlier tokens as an unordered **set**. (The causal mask leaks *a little* order, since later positions see more tokens, but performance still suffers.)
</details>

### 7. 🔁 What exactly does weight tying share?
The embedding turns *the current* token into a vector, and the output predicts *the next* token. So why can they share a matrix?

<details><summary>Answer</summary>

Both use **one shared space of token vectors**. Input: token X → e_X. Output: logit(X) = h · e_X ("how well does the final state match X?"). The blocks' job is to move the residual stream from *current token* towards *next token*. Shapes: Embedding is (V, C), and Linear(C, V) stores (V, C), so no transpose is needed. Trade-off: it saves V×C parameters (about 30% of GPT-2) and helps rare tokens, but it adds a bias towards repeating the current token. Many large models don't tie.
</details>

### 7b. 🔁 Why can't RoPE tell positions apart in "3 3 3 3 3"?
With learned position embeddings, a sequence of identical tokens gives different outputs at each position. With RoPE it doesn't. Why not?

<details><summary>Answer</summary>

RoPE only rotates **q and k**, so position changes *where attention looks* but not *what it retrieves*. With identical tokens, every **v** is identical, so any weighted average of them is the same vector, whatever the attention weights. Learned `wpe` adds position into the residual stream itself, so the vectors differ. (I found this when a test I'd written for RoPE failed. The fix was to test order instead: "dog bites man" vs "man bites dog".)
</details>

---

## Tokenization

### 8. 🔁 What does a bigger vocabulary buy, and what does it cost?
GPT-2 with 100k tokens instead of 50k: what gets better, what gets worse?

<details><summary>Answer</summary>

**Better:** compression, i.e. shorter sequences, so more text per context window. (Not "more words": byte-level BPE can already express any text.)
**Worse:** embedding + output layer double in size and compute (softmax over 100k at every position), and each token is rarer, so **rare-token embeddings get less training**. Diminishing returns: 98× the vocabulary gave only 1.8× the compression on Shakespeare.
</details>

---

## Optimisation

### 9. 🔁 What does AdamW actually store?
And why does resuming with a fresh optimizer hurt?

<details><summary>Answer</summary>

Two running averages **per parameter** (not the gradients themselves):
- **m (first moment):** an average of the gradients, i.e. momentum, which smooths noise.
- **v (second moment):** an average of the *squared* gradients (not a second derivative). Each step is divided by √v, so every parameter gets its own step size.

A fresh optimizer = m and v start at 0 = unreliable step sizes **at a high learning rate**, the very problem warmup prevents, so expect a loss spike. Save `opt.state_dict()`. Memory: m and v are each the size of the model, so training ≈ 4× the model size (weights + grads + m + v).
</details>

### 10. 🔁 Why not apply weight decay to biases and LayerNorm?

<details><summary>Answer</summary>

Weight decay pulls towards **0**. That's a sensible default for weight matrices (small weights mean a simpler function), but **not for a LayerNorm scale, whose neutral value is 1**: pulling it to 0 shrinks the signal into every sub-layer. Biases shift rather than scale, and there are very few of them, so regularising them gains nothing. Rule: decay `p.dim() >= 2`.
</details>

### 11. 🔁 What is gradient clipping protecting against?

<details><summary>Answer</summary>

**Rare gradient spikes** from an unusual batch or a steep region of the loss landscape. One huge step can wreck training. Clipping rescales the *whole* gradient vector to norm ≤ 1.0, so **the direction is kept and only the size is capped**. (Not about activations saturating.) `clip_grad_norm_` returns the norm *before* clipping, so spikes show up in the log.
</details>

### 12. ✅ Why warmup, and why decay?

<details><summary>Answer</summary>

**Warmup:** early gradients are large and noisy, and Adam's moment estimates haven't settled, so full-size steps risk **instability/divergence**.
**Decay:** not mainly to avoid overshooting (on y = x² the gradient shrinks by itself). It's that **mini-batch gradients are noisy even at the minimum**, so a constant learning rate leaves you bouncing around a **noise floor** proportional to the learning rate. Lowering the learning rate lets you settle (annealing).
</details>

### 13. 🔁 Three runs: what do the logs look like?
(a) good LR + warmup + clipping, (b) LR 10× too high, no warmup or clipping, (c) the same high LR *with* clipping.

<details><summary>Answer</summary>

(a) A smooth fall, with grad norm settling. (b) Loss **spikes**, maybe back above ln V or to `NaN` (divergence), with the grad norm exploding. (c) No catastrophic jumps, but noisy, stuck at a **higher noise floor**, and finishing worse than (a).
</details>

---

## Tensor mechanics (quick fire)

### 14. 🔁 `view` vs `reshape` vs `.contiguous()`?
<details><summary>Answer</summary>

`view` = the same memory with a new shape (free), but only if the memory layout allows it. `transpose` swaps strides without moving data, so a `view` afterwards fails. `.contiguous()` copies the data into the new order. `reshape` = view-or-copy automatically.
</details>

### 15. 🔁 Which of these bite?
`torch.Tensor([...])` · `k.T` on a 4D tensor · `for x in some_dict` · calling `module.forward(x)` · `nn.ModuleList(x)`

<details><summary>Answer</summary>

- `torch.Tensor` (capital T) always makes floats and takes no `dtype`. Use `torch.tensor` or `torch.stack`.
- `.T` on 4D reverses **all** dimensions. Use `transpose(-2, -1)`.
- Iterating over a dict gives **keys**. Use `.items()` or `key=d.get`.
- Call `module(x)`, not `.forward(x)` (that skips hooks).
- `ModuleList` has no forward. Loop over it.
</details>

---

## Compression and hardware

### 16. 🔁 Does a perfect model have a loss of 0?

<details><summary>Answer</summary>

**No.** Text has inherent uncertainty: after "there was a little", the next word could be girl, boy or dog. A model that knows the *true* probabilities still has loss = **the entropy of the text** (the Shannon limit), and that's well above 0. Loss 0 means 100% certainty on every token, which happens only by **memorisation** or a **leak** (no causal mask, val data in the training set). Scaling laws: loss = **E** (irreducible) + terms that shrink with more parameters and data.
</details>

### 17. 🔁 Why can a hand-written norm be slower than `nn.LayerNorm` despite less arithmetic? And why did fusing it barely help *our* training speed?

<details><summary>Answer</summary>

Elementwise GPU operations are **memory-bound**: each one reads the whole tensor from memory, does a tiny bit of maths, and writes it all back. Hand-written RMSNorm = ~6 separate operations = ~6 round trips. `nn.LayerNorm` = one **fused kernel** = one read, one write. Moving data costs more than the arithmetic. Fixes: kernel fusion (`F.rms_norm`, `torch.compile`). Flash Attention is the same idea applied to attention.

**But in our runs it didn't matter.** The "RMSNorm was slower" result was laptop noise (the Mac slowed mid-run), and switching to fused `F.rms_norm` gained ~2%, within noise. **Amdahl's law:** if norms are a small fraction of the step, even a 10× faster norm saves only that fraction. Speed up what dominates (matmuls, attention). And measure speed with controlled, back-to-back benchmarks, not the median of one long run.
</details>

### 18. 🔁 Post-LN won at 6 layers. So why does every large model use pre-LN?

<details><summary>Answer</summary>

Look at what gets normalised. **Pre-LN** normalises only each sub-layer's *input*; the residual stream itself is never normalised, so gradients flow straight through (I + f′) to the first layer. That makes very deep stacks train stably, even without careful warmup. **Post-LN** normalises the whole stream after every block, which constrains the representation more tightly and can give a better final loss, but the gradient has to pass through a LayerNorm at every layer, so at depth it becomes unstable early in training (needs long warmup, can diverge). At 6 layers that instability doesn't bite; at 48+ it does. Large models trade a little quality for reliability.
</details>

### 19. 🔁 Mixed precision: why keep the *weights* in fp32 if the matmuls run in bf16?
A weight is 0.02, and this step's update is 0.0003 (later, after LR decay, 0.00003). bf16 has 7 mantissa bits:
the next number after 1.0 is 1.0078. What happens to the update if the weight is stored in bf16?

<details><summary>Answer</summary>

Near 0.02, neighbouring bf16 numbers are about 0.02 × 2⁻⁷ ≈ 0.00016 apart. An update of 0.0003 gets rounded to
roughly one or two steps (a big relative error), and an update of 0.00003 is **less than half a step, so it rounds
to nothing: the weight never changes**. Late in training, learning silently stops. Hence **fp32 master weights +
optimizer state**, with bf16 only for the matmul inputs (accumulated in fp32 inside the tensor cores), and
`autocast` keeping softmax, norms and the loss in fp32. bf16 vs fp16: same 16 bits, but bf16 keeps fp32's 8-bit
exponent (range), so no overflow past 65,504 or underflow of small gradients, and no loss scaling needed.
</details>
