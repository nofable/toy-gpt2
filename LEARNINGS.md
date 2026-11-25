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

## Milestone 3: GPT (decoder-only transformer)

- **Decoder only:** GPT drops the encoder and cross-attention. There's no separate input to condition on, because the prompt is simply the start of the sequence (translation becomes `"French: …\nEnglish:"` + continue).
- **Causal mask:** position t may only attend to positions ≤ t. Without it, training lets each position copy its target, so the loss → 0 and the model is useless at generation. Implementation: scores `masked_fill(mask == 0, -inf)` before softmax (e^−∞ = 0; masking with **0** is wrong, since e⁰ = 1). Store one big `tril` buffer and slice `[:T, :T]`, because its top-left block is itself a valid causal mask.
- **Scale by √(head size)**, not √T. A dot product of `hs`-length vectors has variance ∝ hs, and large scores saturate softmax, which makes gradients vanish.
- **Multi-head = splitting the channels:** `view(B, T, nh, hs).transpose(1, 2)`. No data moves. `view` reinterprets memory, `transpose` swaps strides, and `.contiguous()` copies the data so that `view` works again afterwards.
- **"Attention communicates, the MLP computes."** Attention is the only place positions exchange information. The MLP processes each position independently with shared weights (all B·T positions in one matmul) and holds about 2/3 of each block's parameters.
- **Pre-LN:** `x = x + attn(ln1(x))`. The residual stream is never normalised, only added to, so d(x + f(x))/dx = I + f′ gives the gradient a clean path to the first layer. That's why the model needs a final `ln_f`.
- **Learned position embeddings:** use `wpe(pos)`, NOT `wpe(idx)`. That bug runs silently (token IDs < block_size) and leaves the model unable to tell positions apart. Attention by itself doesn't know about order.
- **Weight tying:** `lm_head.weight = wte.weight`. The shapes match because Linear stores `(out, in)`. One shared space of token vectors: the blocks move the residual stream from "current token" towards "next token", and the logit is h·e_token. Trade-off: it saves vocab×C parameters (about 30% of GPT-2) and helps rare tokens, but it adds a bias towards repeating the current token (an untrained tied model scored 3.80 < ln 65 on targets = inputs). Many larger models don't tie.
- **Parameter count by hand: 818,048**, an exact match. The embeddings are only 3% here (vocab 65), but about 30% of GPT-2 (vocab 50,257).
- **Sanity check:** overfit one batch (4.23 → 0.02 in 300 steps). If a model can't memorise 32 examples, something is broken.
- **Python/PyTorch gotchas:** `.T` on a 4D tensor reverses ALL dims (use `transpose(-2, -1)`). `torch.where` needs a bool condition. `nn.ModuleList` has no forward, so loop over it. Call modules directly (`m(x)`, not `m.forward(x)`).

Result: 2,000 steps, 58 s on the M5. **Val 1.66** (bigram 2.49; my estimate from reasoning was 1.6–1.8, my hunch was 2.0). Real words and character names. Train–val gap 0.20 and growing, the start of overfitting on 1MB of text.

## Milestone 4: Proper training loop and TinyStories

- **Data:** TinyStories V2, 2.7M stories → **541M GPT-2 tokens** (1.08 GB as uint16, memory-mapped so it never fully loads into RAM). 4.1 bytes/token, since simple English suits GPT-2's vocabulary. Tokenizing took about 1 minute.
- **LR schedule:** linear warmup → cosine decay → a `min_lr` floor. Warmup: early gradients are large and noisy, and Adam's moment estimates haven't settled. Decay: mini-batch noise means a constant LR bounces around a **noise floor** ∝ LR. I saw it: val dropped 0.03 in one eval once the LR reached its floor.
- **AdamW details:** decay only `p.dim() >= 2` (a LayerNorm scale's neutral value is 1, not 0). betas (0.9, 0.95). Optimizer state (m, v) is 2× the model size, so save it to resume safely.
- **Gradient clipping** to norm 1.0: insurance against rare spikes. Step 0 had grad norm 5.1, which settled to about 0.7. `clip_grad_norm_` returns the pre-clip norm, which is what you want to log.
- **Sampling:** temperature divides the logits (T < 1 sharper, T > 1 flatter). Top-k: threshold at the k-th largest value with `v[:, [-1]]` (keeps the dim so it broadcasts), set the rest to −∞. It removes the long tail of nonsense tokens.
- **Initial loss = ln(50,257) = 10.83**, predicted, and observed 10.87.
- **Order of learning:** unigram frequencies first ("the the. a to a big day" by step 50), then grammar, then consistent characters and simple plots.
- **No overfitting:** we saw only 41M of 541M tokens (< 8%), none twice, so train ≈ val throughout. The question becomes underfitting: the loss was still falling at the end, so more tokens or a bigger model would help.
- **Eval noise:** final val 1.682 versus 1.674 one eval earlier. That's noise from only 20 eval batches, not getting worse. Use more eval batches for headline numbers.
- **Ops:** stdout redirected to a file is *buffered* (the console log lagged 20 min behind the CSV). Use `python -u` / `flush=True` for runs you'll monitor remotely.

Result: **30M params** (6 layers, 6 heads, 384 dim, 256 context; 19M of it is the embedding), 5,000 steps × 8,192 tokens = 41M tokens, **101 min on the M5** at about 7k tok/s. **Best val 1.674.** The samples are coherent short stories with named characters, dialogue, and the odd logic slip ("I am not a real dragon").

## Milestone 5: Experiments

- **Fair comparisons need a fixed harness:** the same 1M val tokens, non-overlapping windows, for every model. Random-batch evals moved by up to 0.008, which would have hidden the RMSNorm result entirely.
- **Model size (fixed 12.3M tokens):** loss falls roughly linearly in log(params) but with shrinking returns. The 30M model was data-starved (1.2 tokens/param vs Chinchilla's ~20). 3.3× more tokens gained 0.41; 3.4× more params gained 0.17.
- **Post-LN beat pre-LN by 0.05** at 6 layers. It started slower, overtook by step 500. Pre-LN's advantage is *stability at depth*, not final quality at small depth.
- **RoPE was the biggest single win (−0.10)**, more than tripling model size. Relative position is built in (an *inductive bias*), so the model doesn't spend scarce data learning 256 position vectors.
- **RMSNorm: same loss, same speed.** It first looked slower (median 6.8k vs 7.2k tok/s), but the log showed the Mac slowing mid-run (8.3k → 5.3k). Fused `F.rms_norm` gained ~2%, within noise: norms are a small slice of each step (**Amdahl's law**). Lesson: benchmark speed back-to-back, never trust the median of one long laptop run.
- **SwiGLU: small gain (−0.007)**, consistent with the literature.
- **My predictions:** 3/5 right on loss. The two misses (post-LN, RoPE) were the most instructive results.

## Milestone 6: Scale-up prep

- **Size the run by time, not tokens.** Measure seconds per step on the real GPU, then `max_iters = hours / sec_per_step`. The calibration window (steps 20–70) gave 2.87 s/step → 18,472 steps → 9.68B tokens in a 15 h budget.
- **Compute rule of thumb:** training FLOPs ≈ 6N per token (2N forward, 4N backward) plus attention's 12·L·C·T, about 855M FLOPs/token for 124M params at T=1024. **MFU** = achieved FLOPs / peak. 50% on an A100 is good for a single GPU.
- **What made it fast:** bf16 autocast (tensor cores, half the memory traffic), Flash attention via `scaled_dot_product_attention` (never materialises the T×T matrix), `torch.compile` (fuses elementwise ops, so fewer trips to memory), fused AdamW, TF32, vocab padded to 50,304 (a multiple of 128). Most ops are **memory-bound**, not compute-bound, so the wins come from moving fewer bytes.
- **bf16 vs fp16:** bf16 keeps fp32's exponent range (no loss scaling needed) but has fewer mantissa bits. Master weights and the optimizer stay in fp32.
- **Gradient accumulation:** the GPU fits 64×1024 tokens, but the optimizer wants ~0.5M per step, so sum gradients over 8 micro-batches (each loss divided by 8) before one `optimizer.step()`.
- **Scaled init:** residual output projections get std 0.02/√(2·n_layer), so the residual stream's variance doesn't grow with depth.
- **Ops before spending:** a 15-minute dry run on the real box caught a real bug (resume loaded the wrong run). Also checked: the CUDA driver version (torch for CUDA 13 needs driver ≥ 580, or it silently falls back to CPU), atomic checkpoint writes, a NaN guard before saving, and a final eval even when stopped by the time budget.

## Milestone 7: The GPT-2 124M run

- **Setup:** 1× A100 80GB PCIe on Vast.ai at $0.884/h. 9.85B tokens of pre-tokenized FineWeb-Edu (100 shards downloaded in 74 s). 182k tok/s, 47.6 GB memory, flat for the whole run.
- **Result:** 18,472 steps × 524,288 tokens = 9.68B tokens in 14.9 h. **Val loss 3.045. HellaSwag 0.3078 vs OpenAI GPT-2 124M's 0.2955**, with roughly 10× fewer training tokens. Better data (FineWeb-Edu vs WebText) and a modern stack (RoPE, SwiGLU, RMSNorm) beat scale. **Total spend: $14.16** including the dry run.
- **Shape of the curve:** fast drop to ~3.7 in the first 0.8B tokens, then a slow grind. The last third (LR decaying to its floor) gained ~0.1. Train ≈ val throughout: every token seen once, so no overfitting.
- **Train above val is fine:** train loss is measured on whichever shard is loading, and some shards are harder than others. Val pulling *above* train would be the warning sign.
- **What I watched for:** NaN / loss spikes, grad norm climbing past ~1 (steady at 0.2–0.3), throughput or memory drift, and val flattening early.
- **Samples:** fluent, structured (bullet lists), on-topic, but confidently wrong on facts (made up a "UN Convention on the Law of the Land"). Knowledge needs far more parameters than fluency does.
- **Ops:** monitor with one ssh check every 30 min, not a live tail. `pkill -f pattern` over ssh can kill your own ssh session if the pattern appears in the command line.
