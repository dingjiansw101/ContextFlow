#!/usr/bin/env bash
# Queue the four downloaded checkpoints; ContextFlow_4gpu runs separately on GPU 0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN_ID="${1:?unique run identifier}"
NAMES=(ContextFlow_726c829_2gpu_defaultprec_run1 ContextFlow_defaultprec_2gpu_seed42_run1 ContextFlow_refactor_merge_nw16 ContextFlow_repro_nw16_seed3)
PIDS=()
for INDEX in "${!NAMES[@]}"; do
    NAME="${NAMES[$INDEX]}"
    uv run --no-sync python jobs/local/eval_release_when_ready.py "$NAME" "$((INDEX + 1))" "$RUN_ID" 50 unseen \
        > "logs/release_wait_${NAME}.log" 2>&1 &
    PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
# The already-running primary checkpoint may take longer than the staged runs.
for ((attempt=0; attempt<480; attempt++)); do
    COMPLETE=1
    for SUITE in libero_spatial libero_object libero_goal libero_10; do
        [[ -s "logs/release_eval/$RUN_ID/ContextFlow_4gpu/$SUITE.json" ]] || COMPLETE=0
    done
    [[ "$COMPLETE" == 0 ]] || break
    tmux has-session -t "${PRIMARY_SESSION:-contextflow-eval-4gpu}" 2>/dev/null || break
    sleep 30
done
uv run --no-sync python jobs/local/summarize_release_eval.py "logs/release_eval/$RUN_ID" || FAILED=1
exit "$FAILED"
