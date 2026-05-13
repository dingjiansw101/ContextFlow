#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
SUBMIT_CMD="${SUBMIT_CMD:-sbatch}"
EVAL_SUBMIT_CMD="${EVAL_SUBMIT_CMD:-sbatch}"
PARTITION="${PARTITION:-batch-h100}"
QOS="${QOS:-batch}"
EVAL_PARTITION="${EVAL_PARTITION:-batch-h100}"
EVAL_QOS="${EVAL_QOS:-batch}"
RUN_ID="${RUN_ID:-orix_refactor_merge_$(date +%Y%m%d)}"
NUM_WORKERS="${NUM_WORKERS:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
MODE="${1:-submit}"

if [ "$MODE" != "submit" ] && [ "$MODE" != "--test-only" ]; then
    echo "usage: $0 [submit|--test-only]" >&2
    exit 64
fi

cd "$REPO"
mkdir -p logs errs

configs=(
    "pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor"
    "pi0_libero_incontextv18_low_mem_finetune_sample_frames8"
    "pi0_fast_incontext_prompt_action_7_state_8_train_split"
    "pi0_fast_incontext_prompt_action_7_state_8_train_split_900m"
)
labels=(
    "v12"
    "v18_sf8"
    "fast_seq"
    "fast_seq_900m"
)
policy_configs=(
    "pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor"
    "pi0_libero_incontextv18_low_mem_finetune_sample_frames8"
    "pi0_fast_incontext_prompt_action_7_state_8_inference"
    "pi0_fast_incontext_prompt_action_7_state_8_inference_900m"
)

parse_job_id() {
    awk '/Submitted batch job/ {print $4}' <<<"$1"
}

submit_train() {
    local label="$1"
    local config="$2"
    local exp_name="${config}_refactor_merge"
    local job_name="train_${label}_refactor_merge"
    local args=(
        --job-name="$job_name"
        --output="logs/%x-%j.log"
        --error="errs/%j-%x.err"
        --gres=gpu:4
        --cpus-per-gpu=16
        --mem=400G
        --time=24:00:00
        --chdir="$REPO"
        --export=ALL,REPO="$REPO",CONFIG="$config",EXP_NAME="$exp_name",NUM_WORKERS="$NUM_WORKERS",FSDP_DEVICES="$FSDP_DEVICES",ASSETS_BASE_DIR="$ASSETS_BASE_DIR"
    )
    case "$SUBMIT_CMD" in
        sbatch) args=(--partition="$PARTITION" --qos="$QOS" "${args[@]}") ;;
        sbatch-pi) ;;
        sbatch-free) args=(--requeue --signal=B:SIGTERM@60 "${args[@]}") ;;
        *) echo "unsupported SUBMIT_CMD=$SUBMIT_CMD" >&2; exit 64 ;;
    esac
    if [ "$MODE" = "--test-only" ]; then
        args=(--test-only "${args[@]}")
    fi
    echo "Submitting train ${label}: ${SUBMIT_CMD} ${args[*]} jobs/orix/refactor_merge/train_one.sh"
    "$SUBMIT_CMD" "${args[@]}" jobs/orix/refactor_merge/train_one.sh
}

submit_eval() {
    local label="$1"
    local config="$2"
    local policy_config="$3"
    local train_job_id="$4"
    local exp_name="${config}_refactor_merge"
    local checkpoint_dir="checkpoints/${config}/${exp_name}/19999"
    local job_name="eval_${label}_refactor_merge"
    local args=(
        --dependency="afterok:${train_job_id}"
        --job-name="$job_name"
        --output="logs/%x-%j.log"
        --error="errs/%j-%x.err"
        --gres=gpu:1
        --cpus-per-gpu=16
        --mem=120G
        --time=24:00:00
        --chdir="$REPO"
        --export=ALL,REPO="$REPO",CONFIG="$config",POLICY_CONFIG="$policy_config",EXP_NAME="$exp_name",CHECKPOINT_DIR="$checkpoint_dir",RUN_ID="$RUN_ID",TASK_SPLIT="$TASK_SPLIT",SUITE_LIST="$SUITE_LIST",ASSETS_BASE_DIR="$ASSETS_BASE_DIR",SKIP_LOG_TO_SHEET=1
    )
    case "$EVAL_SUBMIT_CMD" in
        sbatch) args=(--partition="$EVAL_PARTITION" --qos="$EVAL_QOS" "${args[@]}") ;;
        sbatch-pi) ;;
        *) echo "unsupported EVAL_SUBMIT_CMD=$EVAL_SUBMIT_CMD; eval must use sbatch or sbatch-pi" >&2; exit 64 ;;
    esac
    if [ "$MODE" = "--test-only" ]; then
        args=(--test-only "${args[@]}")
    fi
    echo "Submitting eval ${label}: ${EVAL_SUBMIT_CMD} ${args[*]} jobs/orix/refactor_merge/eval_libero_unseen_one.sh"
    "$EVAL_SUBMIT_CMD" "${args[@]}" jobs/orix/refactor_merge/eval_libero_unseen_one.sh
}

for i in "${!configs[@]}"; do
    label="${labels[$i]}"
    config="${configs[$i]}"
    policy_config="${policy_configs[$i]}"
    train_out="$(submit_train "$label" "$config")"
    echo "$train_out"
    if [ "$MODE" = "--test-only" ]; then
        continue
    fi
    train_job_id="$(parse_job_id "$train_out")"
    if [ -z "$train_job_id" ]; then
        echo "Could not parse train job id for ${label}" >&2
        exit 65
    fi
    eval_out="$(submit_eval "$label" "$config" "$policy_config" "$train_job_id")"
    echo "$eval_out"
done
