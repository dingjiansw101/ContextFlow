#!/usr/bin/env bash
#SBATCH --job-name=pi0_libero_incontextv18_low_mem_finetune_sample_frames8_orix_eval
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --chdir=/home/dingj0b/code/openpi_libero/openpi

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-$HOME/nvidia-egl}"
export LD_LIBRARY_PATH="$HOME/nvidia-egl/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

CONFIG="pi0_libero_incontextv18_low_mem_finetune_sample_frames8"
EXP_NAME="${CONFIG}_orix"
INFERENCE_CONFIG="${CONFIG}_inference"
ITER="${ITER:-19999}"
CHECKPOINT_DIR="checkpoints/${CONFIG}/${EXP_NAME}/${ITER}"
RUN_ID="${RUN_ID:-orix_$(date +%Y%m%d)}"

bash jobs/local/eval_pi0_libero_incontext_unseen.sh \
    "$EXP_NAME" "$INFERENCE_CONFIG" "$CHECKPOINT_DIR" "$RUN_ID"
