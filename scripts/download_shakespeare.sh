#!/usr/bin/env bash
# Download Tiny Shakespeare (~1.1 MB) into data/shakespeare/
set -euo pipefail
mkdir -p data/shakespeare
curl -fsSL -o data/shakespeare/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "Saved data/shakespeare/input.txt ($(wc -c < data/shakespeare/input.txt) bytes)"
