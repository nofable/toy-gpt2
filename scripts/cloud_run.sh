#!/usr/bin/env bash
# Usage on a fresh 1x H100 box: bash scripts/cloud_run.sh <hours> [shards]   e.g. bash scripts/cloud_run.sh 5.5 99
set -euo pipefail
HOURS=${1:?hours}
SHARDS=${2:-99}
export UV_PYTHON_PREFERENCE=only-managed
command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv python install 3.12
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), f'CUDA not usable: torch built for CUDA {torch.version.cuda} needs NVIDIA driver >= 580'; print(torch.cuda.get_device_name(), 'cuda', torch.version.cuda)"
df -h . | tail -1
uv run python scripts/download_finewebedu.py --shards "$SHARDS"
mkdir -p runs/gpt2
RESUME=""
[ -f runs/gpt2/ckpt.pt ] && RESUME="--resume"
nohup uv run python -u -m toy_gpt2.train --name gpt2 --max_hours "$HOURS" $RESUME >> runs/gpt2/stdout.log 2>&1 &
echo "training in background (pid $!). follow with: tail -f runs/gpt2/stdout.log"
