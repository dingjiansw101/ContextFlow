#!/usr/bin/env bash
#SBATCH --job-name=eval_fast_seq_900m_raw_images
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_v18_raw_images

set -euo pipefail

export CONFIG="pi0_fast_incontext_prompt_action_7_state_8_train_split_900m"
export POLICY_CONFIG="pi0_fast_incontext_prompt_action_7_state_8_inference_900m"
export JOB_TAG="fast_seq_900m"
export ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_fastincontext/openpi/assets"

exec bash jobs/orix/raw_images/_eval_common.sh
