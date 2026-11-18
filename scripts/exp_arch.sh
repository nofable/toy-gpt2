#!/usr/bin/env bash
# Usage: ./scripts/exp_arch.sh modern|postln   (EXPERIMENTS.md B and C; numbers there come from commit 0d6b861)
set -euo pipefail
COMMON="--data tinystories --vocab_size 50257 --block_size 256 --batch_size 32 --total_batch_tokens 8192
  --max_iters 1500 --warmup_iters 100 --max_lr 1e-3 --min_lr 1e-4 --eval_interval 250 --eval_batches 50
  --ckpt_interval 100000 --no-hellaswag --n_layer 6 --n_head 6 --n_embd 384
  --norm layernorm --norm_position pre --pos_emb learned --mlp gelu"

SNAP=runs/_snapshot_$1
rm -rf "$SNAP" && mkdir -p "$SNAP" && cp -R src/toy_gpt2 "$SNAP"/

run() {
  local name=$1; shift
  mkdir -p runs/"$name"
  echo "=== $name: $* ==="
  PYTHONPATH="$SNAP" uv run python -u -m toy_gpt2.train --name "$name" $COMMON "$@" > runs/"$name"/stdout.log 2>&1
  tail -1 runs/"$name"/stdout.log
}

case $1 in
  modern)
    run arch_rmsnorm --norm rmsnorm
    run arch_rmsnorm_rope --norm rmsnorm --pos_emb rope
    run arch_rmsnorm_rope_swiglu --norm rmsnorm --pos_emb rope --mlp swiglu
    ;;
  postln)
    run arch_postln --norm_position post
    ;;
esac
