# toy-gpt2: Project Plan

Build a GPT from scratch, train it, and publish it, as a learning project with coaching.

## Goals
- **Understand** how a decoder-only LLM is built, trained, and run, end to end.
- **Show it** with a clean public GitHub repo: readable code, experiments, plots, and a write-up that will hold up in an interview.
- **Enjoy it.** Make visible progress every session and avoid rabbit holes.

## Constraints and decisions
| Topic | Decision |
|---|---|
| Hardware | Apple M5 with 24 GB RAM (PyTorch `mps` backend) for development and the small model |
| Budget | $0 locally. At most **$20** in total on a rented GPU, used only for the final run |
| Final target | A small model trained locally first, then one **GPT-2 small (124M)** run on a rented GPU |
| Who codes what | **William writes the core:** the model, the tokenizer, and the training loop. Claude can scaffold boilerplate on request: downloads, logging, plotting, CLI |
| Tokenizer | Write a simple BPE to learn how it works, then use `tiktoken` (the GPT-2 tokenizer) for real training |
| Pace | About 5–10 hours a week, so roughly 4–6 weeks |
| Stack | Python with `uv` and PyTorch |

## How the coaching works
- Claude asks questions and gives hints before answers. When stuck, ask for the hint level you want: *nudge*, then *bigger hint*, then *show me*.
- Before running anything, predict the result (for example the initial loss, or whether it will overfit), then check.
- Each milestone has a **"done when"** check, so we know when to move on.
- Write what you learn in `LEARNINGS.md` as you go. It becomes the README write-up and your interview notes.

## Progress
| # | Milestone | Status |
|---|---|---|
| 0 | Setup | ✅ Done, 2026-11-09 |
| 1 | Bigram baseline | ▶️ Next |
| 2 | Tokenizer | |
| 3 | GPT model | |
| 4 | Training loop + TinyStories | |
| 5 | Experiments | |
| 6 | Scale-up prep | |
| 7 | GPT-2 124M cloud run | |
| 8 | Polish & publish | |

## Milestones

### 0. Setup (≈1 hr)
Set up git, a `uv` project, and PyTorch, and confirm `mps` works. Create the repo layout.
*Done when:* a matrix multiply runs on `mps` and the first commit is made.

### 1. Baseline: bigram model on Shakespeare characters (≈2 hrs)
*Think about:* what loss should an untrained model have, and why? What can a bigram model never learn?
*Done when:* the loss drops from your predicted starting value and samples look roughly like Shakespeare.

### 2. Tokenizer (≈3–4 hrs)
Write a minimal byte-level BPE (train, encode, decode), then compare it with `tiktoken`.
*Think about:* the tradeoffs of vocabulary size, why byte-level, and odd tokenization behavior (numbers, whitespace).
*Done when:* `decode(encode(x)) == x` holds for any string, and you can explain why GPT models struggle to spell.

### 3. The GPT model (≈4–6 hrs)
Build a decoder-only transformer. You have already built the original transformer, so focus on **what is different** and why.
*Done when:* the model overfits a single batch to near-zero loss, and the parameter count matches your hand calculation.

### 4. A proper training loop, and TinyStories on the M5 (≈5–8 hrs)
Add AdamW, a learning-rate schedule, gradient clipping, train/val evaluation, checkpointing, and sampling (temperature and top-k).
*Done when:* a ~10–30M parameter model writes coherent short stories, and you have a loss-curve plot.

### 5. Experiments: the fun part (≈4–6 hrs)
Run small, cheap ablations, for example model size against loss, learning-rate sweeps, and removing one component to see what breaks.
*Done when:* you have 2–3 plots, each with a one-paragraph takeaway, for the README.

### 6. Prepare for the scale-up (≈4–6 hrs)
Work on speed (mixed precision, fused attention, `torch.compile`), data (shard FineWeb-Edu), and evaluation (HellaSwag).
**Measure throughput first, then calculate the cost**, and do a short dry run on a cheap GPU.
*Done when:* you have a written budget (tokens, hours, $) that comes to ≤ $20, with a safety margin.

### 7. The GPT-2 124M run (≈1 day, mostly waiting)
Train on rented hardware, with regular checkpoints and logging. Have a plan for what to do if the run fails halfway.
*Done when:* the model is trained, its validation loss and HellaSwag score are compared with GPT-2 124M, and the GPU instance is **shut down**.

### 8. Polish and publish (≈3–4 hrs)
Write the README (results, plots, what you learned), and optionally put the weights on Hugging Face with a small demo.
*Done when:* the repo is public and you could walk an interviewer through any file in it.

## Guardrails
- Set a spending cap or alert with the GPU provider before renting anything.
- If something takes more than about 2× its estimate, stop and re-scope together.
- No secrets and no large files (weights, data) in git. Set up `.gitignore` on day one.
