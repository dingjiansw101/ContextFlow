#!/usr/bin/env bash
#SBATCH --job-name=train_fast_seq_raw_images
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_v18_raw_images

set -euo pipefail

export CONFIG="pi0_fast_incontext_prompt_action_7_state_8_train_split"  # on-disk exp/ckpt name (pre-rename)
export POLICY_CONFIG="ContextAR"  # get_config lookup (renamed)
export ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_fastincontext/openpi/assets"
export ASSETS_NAME="debug_pi0_fast_libero_incontext_inference"

exec bash jobs/orix/raw_images/_train_common.sh
