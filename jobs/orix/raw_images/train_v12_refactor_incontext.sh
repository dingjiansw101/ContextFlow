#!/usr/bin/env bash
#SBATCH --job-name=train_v12_raw_images
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

export CONFIG="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor"
export ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_libero/openpi/assets"
export ASSETS_NAME="$CONFIG"

exec bash jobs/orix/raw_images/_train_common.sh
