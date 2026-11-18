#!/usr/bin/env bash
# Usage: bash scripts/cloud_run.sh <hours> <shards> <name> [extra train args]
#   dry run:  bash scripts/cloud_run.sh 0.25 1 dry --ckpt_interval 100
#   real run: bash scripts/cloud_run.sh 5.3 99 gpt2
set -euo pipefail
HOURS=${1:?hours}
SHARDS=${2:?shards}
NAME=${3:?name}
shift 3
export UV_PYTHON_PREFERENCE=only-managed
command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv python install 3.12
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), f'CUDA not usable: torch built for CUDA {torch.version.cuda} needs NVIDIA driver >= 580'; print(torch.cuda.get_device_name(), 'cuda', torch.version.cuda)"
df -h . | tail -1
uv run python scripts/download_finewebedu.py --shards "$SHARDS"
mkdir -p "runs/$NAME"
if [ -f "runs/$NAME/ckpt.pt" ]; then
  set -- --name "$NAME" --resume
else
  set -- --name "$NAME" --max_hours "$HOURS" "$@"
fi
nohup uv run python -u -m toy_gpt2.train "$@" >> "runs/$NAME/stdout.log" 2>&1 &
echo "training in background (pid $!). follow with: tail -f runs/$NAME/stdout.log"
