#!/usr/bin/env bash
# Usage: ./scripts/exp_model_size.sh   (EXPERIMENTS.md A; numbers there come from commit 0d6b861)
set -euo pipefail
COMMON="--data tinystories --vocab_size 50257 --block_size 256 --batch_size 32 --total_batch_tokens 8192
  --max_iters 1500 --warmup_iters 100 --max_lr 1e-3 --min_lr 1e-4 --eval_interval 250 --eval_batches 50
  --ckpt_interval 100000 --no-hellaswag --norm layernorm --norm_position pre --pos_emb learned --mlp gelu"

SNAP=runs/_snapshot_model_size
rm -rf "$SNAP" && mkdir -p "$SNAP" && cp -R src/toy_gpt2 "$SNAP"/

run() {
  local name=$1; shift
  mkdir -p runs/"$name"
  echo "=== $name: $* ==="
  PYTHONPATH="$SNAP" uv run python -u -m toy_gpt2.train --name "$name" $COMMON "$@" > runs/"$name"/stdout.log 2>&1
  tail -1 runs/"$name"/stdout.log
}

run size_128 --n_layer 2 --n_head 4 --n_embd 128
run size_256 --n_layer 4 --n_head 4 --n_embd 256
run size_384 --n_layer 6 --n_head 6 --n_embd 384
