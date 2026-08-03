#!/usr/bin/env bash
#SBATCH --job-name=eval_v12_raw_images
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_v18_raw_images

set -euo pipefail

export CONFIG="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor"
export POLICY_CONFIG="ContextFlow_Plain"  # served config (renamed); CONFIG stays old for the on-disk ckpt path
export JOB_TAG="v12"
export ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_libero/openpi/assets"

exec bash jobs/orix/raw_images/_eval_common.sh
