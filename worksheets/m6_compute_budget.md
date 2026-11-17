# Worksheet: can we afford GPT-2 for $15?

Fill in every `____`. Each step uses only the step before it. Write the units next to each number.
Hints are folded away: open one only if you're stuck.

---

## Fixed numbers and rules of thumb

**The model and the budget**

| | Value | Where it comes from |
|---|---|---|
| GPT-2 small parameters (N) | 124M | OpenAI's GPT-2 paper (2019), smallest size |
| Budget for the real run | ~$15 | $20 total, minus ~$3 dry run and ~$2 spare |
| H100 rental price | ~$2.50/hr | typical on-demand price on RunPod / Vast.ai |
| A100 rental price | ~$1–1.50/hr | same |

**Token targets**

| Target | Tokens | Where it comes from |
|---|---|---|
| Chinchilla-optimal | ~20 tokens per param ≈ **2.5B** | Hoffmann et al. 2022: best loss for a fixed compute budget |
| GPT-2 parity | **~10B** | Karpathy's GPT-2 124M reproduction (2024): 10B FineWeb-Edu tokens matched or beat OpenAI's GPT-2 124M on HellaSwag |

**Chip speeds (bf16 = 16-bit on tensor cores; fp32 = plain 32-bit)**

| Chip | bf16 peak | fp32 peak (no tensor cores) | Memory | Bandwidth |
|---|---|---|---|---|
| A100 | ~312 TFLOPS | ~19.5 TFLOPS | 40/80 GB | ~2 TB/s |
| H100 (SXM) | ~990 TFLOPS | ~67 TFLOPS | 80 GB | ~3.35 TB/s |

**Rules of thumb**

| Rule | Value | Meaning |
|---|---|---|
| Training cost per token | **6 × N FLOPs** | 2 forward + 4 backward per parameter (see step 2) |
| MFU (model FLOPs utilisation) | **30–50%**, use 40% | the fraction of peak speed real training achieves |
| Attention's extra FLOPs | ~15% on top of 6N at T = 1024 | the T×T score matmuls, which 6N leaves out |
| Training memory | ~4× model size + activations | weights + grads + Adam's m and v |
| Text per token (GPT-2 tokenizer, English) | ~4 bytes | so 10B tokens ≈ 40 GB of text |

---

## 0. Units cheat sheet

| Prefix | Means | Example |
|---|---|---|
| k | 10³ (thousand) | 8k tok/s = 8,000 tokens per second |
| M | 10⁶ (million) | 124M params |
| B / G | 10⁹ (billion) | 10B tokens |
| T | 10¹² (trillion) | 990 TFLOPS |

- **FLOP** = one floating-point operation (one multiply, or one add). It's a *count* of work.
- **FLOPS** (or FLOP/s) = FLOPs **per second**. It's a *speed*.

Rule: `work ÷ speed = time`. Same as `distance ÷ speed = time`.

---

## 1. How fast do we need to go?

Budget: about **$15** at **~$2.50/hr** for an H100.

| | |
|---|---|
| Hours we can afford: `15 ÷ 2.5` | 6 hours |
| ... in seconds (× 3,600) | 21,600 s |
| Tokens we want | 10,000,000,000 (10B) |
| **Tokens per second needed:** `tokens ÷ seconds` | ~463,000 tok/s |

---

## 2. How much work is one token?

Rule of thumb: **training costs 6 FLOPs per parameter per token.**

<details><summary>Why 6?</summary>

- **Forward pass = 2 per parameter.** Every weight takes part in one multiply and one add for each token
  (`y = w·x + ...`).
- **Backward pass = 4 per parameter**, about twice the forward. For each weight you compute the gradient
  for the *weight* and the gradient for the *input*, and each of those is another multiply-add.
- Total: 2 + 4 = **6**. Bytes per token don't matter: inside the model every token is one vector of size C.
</details>

| | |
|---|---|
| Parameters N (GPT-2 small) | 124,000,000 |
| **FLOPs per token:** `6 × N` | 744M FLOPs |

---

## 3. How fast is the chip, really?

The H100's advertised bf16 speed is **990 TFLOPS** = 990 × 10¹² FLOPs per second. That's a peak.
Real training achieves a fraction of it, called **MFU** (model FLOPs utilisation). Assume **40%**.

| | |
|---|---|
| **Real FLOPs per second:** `990 × 10¹² × 0.4` | 396 TFLOPS |

---

## 4. So how many tokens per second do we get?

`(FLOPs per second the chip gives) ÷ (FLOPs each token needs) = tokens per second`

| | |
|---|---|
| **Our tok/s:** step 3 ÷ step 2 | 532K tok/s |
| Needed (step 1) |  463K tok/s |
| **Do we make it?** | yes |
| Hours for 10B tokens at our speed: `10B ÷ tok/s ÷ 3,600` | 5.22 hours |
| Cost at $2.50/hr | $13.05 |

---

## 5. Sanity check against your Mac

Reverse the formula: `tok/s × FLOPs per token = FLOPs per second actually achieved`.

| | |
|---|---|
| Your M5 measured | ~8,000 tok/s |
| Model size in the M5 runs | 30,000,000 params |
| **M5's achieved speed:** `8,000 × 6 × 30M` | TFLOPS: 1.44 |
| H100 achieved speed from step 3, in TFLOPS | |
| **How many times faster is the H100?** | 275 × |

<details><summary>Hint: converting to TFLOPS</summary>

Divide by 10¹² (1,000,000,000,000).
</details>

---

## 6. What if we forget bf16?

Without bf16, the matmuls don't use the tensor cores. The H100 does about **67 TFLOPS** in plain fp32.
Redo steps 3 and 4 with 67 instead of 990 (keep 40% MFU).

| | |
|---|---|
| Real FLOPs per second | 26.8 TFLOPS |
| tok/s | 36K |
| Hours for 10B tokens | 77 |
| Cost | $192.9 |
| **How much more expensive than with bf16?** | ~15 × |

---

## 7. Ranking the speed-ups

For each one, ask: *which part of the step does it speed up, and how much of the step is that part?*
Roughly, at T = 1024 for GPT-2 small:

| Part of a training step | Share of the FLOPs |
|---|---|
| Linear layers (qkv, proj, MLP, lm_head) | ~85% |
| Attention scores and `att @ v` (the T×T part) | ~15% |
| Elementwise ops (norms, activations, residual adds, RoPE) | tiny in FLOPs, but memory-bound, so they still cost time |

| Speed-up | Which part does it speed up? | How big is that part? | My rank (1 = biggest) |
|---|---|---|---|
| bf16 mixed precision | ____ | ____ | ____ |
| Flash Attention | ____ | ____ | ____ |
| `torch.compile` | ____ | ____ | ____ |

<details><summary>Hint</summary>

Step 6 already tells you what bf16 is worth on its own. Can the other two come anywhere near that?
</details>

---

## 8. Decision

| | |
|---|---|
| Tokens we can afford in ~6 hours (if step 4 says we fall short) | ____ B |
| One sentence: is the plan realistic, and what has to go right? | ____ |
