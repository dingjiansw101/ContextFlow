#!/usr/bin/env bash
#SBATCH --job-name=eval_v18_sf8_raw_images
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_v18_raw_images

set -euo pipefail

export CONFIG="pi0_libero_incontextv18_low_mem_finetune_sample_frames8"
export POLICY_CONFIG="$CONFIG"
export JOB_TAG="v18_sf8"
export ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_libero/openpi/assets"

exec bash jobs/orix/raw_images/_eval_common.sh
