#!/usr/bin/env bash
# Usage on a fresh 1x H100 box: bash scripts/cloud_run.sh <hours> [shards]   e.g. bash scripts/cloud_run.sh 5.5 99
set -euo pipefail
HOURS=${1:?hours}
SHARDS=${2:-99}
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv sync
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python scripts/download_finewebedu.py --shards "$SHARDS"
mkdir -p runs/gpt2
RESUME=""
[ -f runs/gpt2/ckpt.pt ] && RESUME="--resume"
nohup uv run python -u -m toy_gpt2.train --name gpt2 --max_hours "$HOURS" $RESUME >> runs/gpt2/stdout.log 2>&1 &
echo "training in background (pid $!). follow with: tail -f runs/gpt2/stdout.log"
