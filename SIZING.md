# Sizing: parameters, compute and cost

GPT-2 small: 12 layers, 12 heads, C = 768, context T = 1024, vocab V = 50,257.
Every count below is checked by `tests/test_gpt.py`.

## 1. Parameters per component

| Component | Shape | GPT-2 (original) | Ours (RMSNorm + RoPE + SwiGLU) |
|---|---|---:|---:|
| Token embedding `wte` (shared with `lm_head`) | V × C | 38,597,376 | 38,597,376 |
| Position embedding `wpe` | T × C | 786,432 | 0 (RoPE has no weights) |
| **Per block** | | | |
| ln1 | C (+ C bias for LayerNorm) | 1,536 | 768 |
| attn.qkv | C × 3C + 3C | 1,771,776 | 1,771,776 |
| attn.proj | C × C + C | 590,592 | 590,592 |
| ln2 | C (+ C bias) | 1,536 | 768 |
| MLP up / fc | C × 4C + 4C | 2,362,368 | — |
| MLP down / proj | 4C × C + C | 2,360,064 | — |
| SwiGLU gate, up, down | 3 × (C × 2048), no bias | — | 4,718,592 |
| Block total | | 7,087,872 | 7,082,496 |
| **× 12 blocks** | | 85,054,464 | 84,989,952 |
| Final norm `ln_f` | C (+ C bias) | 1,536 | 768 |
| `lm_head` | tied to `wte` | 0 | 0 |
| **Total** | | **124,439,808** | **123,588,096** |

- The embedding is **31%** of the model. The 12 blocks are 69%, and inside each block the MLP is 2/3 and attention 1/3.
- SwiGLU's hidden size of 2048 (= 8C/3) keeps it within 0.1% of the GELU MLP's size, so the comparison is fair.
- For training the vocab is padded to **50,304** (a multiple of 64, which GPU matmul kernels prefer). That adds
  36,096 parameters: **123,624,192** in total. The 47 padding tokens never appear in the data. `generate` masks them out.

## 2. Compute and cost

| Quantity | Formula | Value |
|---|---|---:|
| Matmul FLOPs per token | 6 × N | 742M |
| Attention-score FLOPs per token | 12 × layers × C × T | 113M |
| **FLOPs per token** | sum | **855M** |
| Tokens per optimizer step | 64 seqs × 1024 × 8 accumulation steps | 524,288 |
| Steps for 10B tokens | 10B ÷ 524,288 | 19,074 |
| H100 SXM bf16 peak | | 989 TFLOPS |
| Achieved at 40% MFU | 989 × 0.4 | 396 TFLOPS |
| **Throughput** | 396 TFLOPS ÷ 855M | **~463k tok/s** (1.13 s per step) |
| **Time for 10B tokens** | 10B ÷ 463k | **6.0 h** |
| **Cost** at $2.50/h | | **$15.00** |

That's exactly the budget with no margin, so the run is sized by **time, not tokens**: `--max_hours 5.5` measures
real speed after ~70 steps and shortens the schedule to fit (the LR still decays fully).

| MFU actually achieved | tok/s | Hours for 10B | Tokens in 5.5 h | Cost of 5.5 h |
|---|---:|---:|---:|---:|
| 30% | 347k | 8.0 | 6.9B | $13.75 |
| **40%** | **463k** | **6.0** | **9.2B** | $13.75 |
| 50% | 578k | 4.8 | 10B (capped) in 4.8 h | $12.00 |

Every row is well above Chinchilla-optimal (20 tokens/param ≈ 2.5B), so even the bad case is a properly trained model.

## 3. Memory (one H100, 80 GB)

| Item | Size |
|---|---:|
| Weights (fp32) | 0.5 GB |
| Gradients (fp32) | 0.5 GB |
| AdamW m and v (fp32) | 1.0 GB |
| Activations, micro-batch 64 × 1024 in bf16 (largest: the 64×1024×50,304 logits) | ~20–40 GB |
| Checkpoint on disk (weights + optimizer) | ~1.5 GB |

If it runs out of memory: `--batch_size 32`. Gradient accumulation doubles to 16, and the optimizer sees exactly the same batch.

## 4. The budget

| Item | Cost |
|---|---:|
| Dry run (download one shard, ~15 min of training, check MFU, test `--resume`) | ~$1 |
| Data download (~20 GB, ~5–10 min on the GPU box) | ~$0.40 |
| Main run (`--max_hours 5.5`) | ~$13.75 |
| HellaSwag at the end (~2 min) | ~$0.10 |
| Spare | ~$4.75 |
| **Total** | **≤ $20** |
