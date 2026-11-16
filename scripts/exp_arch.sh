#!/usr/bin/env bash
# Experiments B & C: architecture variants vs the GPT-2 baseline (runs/size_384).
# Same as size_384 in every way except the flags given. Usage:
#   ./scripts/exp_arch.sh modern     # C: +RMSNorm, +RoPE, +SwiGLU (cumulative)
#   ./scripts/exp_arch.sh postln     # B: post-LN
#
# Runs from a frozen snapshot of the code, so editing src/ while it runs can't break it.
set -euo pipefail
COMMON="--max_iters 1500 --warmup_iters 100 --eval_interval 250 --eval_iters 50 --n_layer 6 --n_head 6 --n_embd 384"

SNAP=runs/_snapshot_$1
rm -rf "$SNAP" && mkdir -p "$SNAP" && cp -R src/toy_gpt2 "$SNAP"/

run() {
  local name=$1; shift
  mkdir -p runs/"$name"
  echo "=== $name: $* ==="
  PYTHONPATH="$SNAP" uv run python -u -m toy_gpt2.train --name "$name" $COMMON "$@" \
    > runs/"$name"/stdout.log 2>&1
  tail -1 runs/"$name"/stdout.log
}

case $1 in
  modern)
    run arch_rmsnorm         --norm rmsnorm
    run arch_rmsnorm_rope    --norm rmsnorm --pos_emb rope
    run arch_rmsnorm_rope_swiglu --norm rmsnorm --pos_emb rope --mlp swiglu
    ;;
  postln)
    run arch_postln --norm_position post
    ;;
esac
