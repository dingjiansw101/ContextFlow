#!/usr/bin/env bash
# Usage: sbatch-pi jobs/orix/train_reproduction_pi1.sh PREPARED_ROOT EXPERIMENT
#SBATCH --job-name=cf-repro-pi1
#SBATCH --partition=pi-elhosemh
#SBATCH --qos=pi-elhosemh
#SBATCH --account=pi-elhosemh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT="${1:?prepared reproduction root}"
EXPERIMENT="${2:?new experiment name}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_OFFLINE=1
ulimit -n 65536
cd "$ROOT/source"
exec "$ROOT/bootstrap/bin/uv" run --no-project --python "$ROOT/env/server/bin/python" python \
    jobs/orix/train_reproduction_pi1.py "$ROOT" "$EXPERIMENT"
