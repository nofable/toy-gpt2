#!/usr/bin/env bash
# Experiment A: model size vs loss. Same data, steps, schedule, seed - only width/depth change.
# 1,500 steps x 8,192 tokens = 12.3M tokens per run.
#
# Runs from a frozen snapshot of the code, so editing src/ while it runs can't break it.
# Architecture flags are pinned to GPT-2 so these runs reproduce now that train.py defaults to the modern stack.
set -euo pipefail
COMMON="--max_iters 1500 --warmup_iters 100 --eval_interval 250 --eval_iters 50 --norm layernorm --norm_position pre --pos_emb learned --mlp gelu"

SNAP=runs/_snapshot_model_size
rm -rf "$SNAP" && mkdir -p "$SNAP" && cp -R src/toy_gpt2 "$SNAP"/

run() {  # name, then model flags
  local name=$1; shift
  mkdir -p runs/"$name"
  echo "=== $name: $* ==="
  PYTHONPATH="$SNAP" uv run python -u -m toy_gpt2.train --name "$name" $COMMON "$@" \
    > runs/"$name"/stdout.log 2>&1
  tail -1 runs/"$name"/stdout.log
}

run size_128 --n_layer 2 --n_head 4 --n_embd 128
run size_256 --n_layer 4 --n_head 4 --n_embd 256
run size_384 --n_layer 6 --n_head 6 --n_embd 384   # = default model; also the baseline for B and C
