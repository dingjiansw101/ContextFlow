#!/usr/bin/env bash
#SBATCH --job-name=norm_pi0_libero_split0
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00

set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi_libero/openpi}"
cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

uv run scripts/compute_norm_stats.py --config-name=pi0_libero_split0

find assets/pi0_libero_split0 -maxdepth 4 -type f -name 'norm_stats*' -print
