#!/usr/bin/env bash
# Evaluate all five public non-90 checkpoints, one policy server per GPU.
# Usage: bash jobs/local/eval_release_all.sh RUN_ID CHECKPOINT_ROOT FOURGPU_CHECKPOINT [TRIALS] [SCOPE]
set -euo pipefail
RUN_ID="${1:?unique run identifier}"
CHECKPOINT_ROOT="${2:?downloaded ContextFlow directory}"
FOURGPU_CHECKPOINT="${3:?verified 4gpu checkpoint directory}"
TRIALS="${4:-50}"
SCOPE="${5:-unseen}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
NAMES=(ContextFlow_4gpu ContextFlow_726c829_2gpu_defaultprec_run1 ContextFlow_defaultprec_2gpu_seed42_run1 ContextFlow_refactor_merge_nw16 ContextFlow_repro_nw16_seed3)
[[ ! -e "logs/release_eval/$RUN_ID" ]] || { echo 'Run output already exists'; exit 1; }
for NAME in "${NAMES[@]}"; do
    CHECKPOINT="$CHECKPOINT_ROOT/$NAME/19999"
    [[ "$NAME" != ContextFlow_4gpu ]] || CHECKPOINT="$FOURGPU_CHECKPOINT"
    [[ -d "$CHECKPOINT/params" && -d "$CHECKPOINT/assets" ]] || { echo "Missing checkpoint: $CHECKPOINT"; exit 1; }
done
PIDS=()
for GPU in "${!NAMES[@]}"; do
    NAME="${NAMES[$GPU]}"
    CHECKPOINT="$CHECKPOINT_ROOT/$NAME/19999"
    [[ "$NAME" != ContextFlow_4gpu ]] || CHECKPOINT="$FOURGPU_CHECKPOINT"
    bash jobs/local/eval_release_checkpoint.sh "$CHECKPOINT" "$NAME" "$GPU" "$RUN_ID" "$TRIALS" "$SCOPE" &
    PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
uv run --no-sync python jobs/local/summarize_release_eval.py "logs/release_eval/$RUN_ID" || FAILED=1
exit "$FAILED"
