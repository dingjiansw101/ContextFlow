#!/bin/bash
# Regression eval for the demo-norm fix (commit 5445332): C FAST in-context (non-900m).
# The fix changes C's eval-time demo conditioning raw -> normalized; the old-pipeline orix
# checkpoint (trained on normalized demos) should now reproduce its old-eval numbers
# through the migrated path. Checkpoint: orix openpi_refactor_fast .../train_split/19999
# (refactor row 6 / Libero Experiments row 14). Baselines (held-out unseen):
# spatial 0.37 (0.72 / 0.02), object 0.43 (0.80 / 0.06), goal 0.22 (0.00 / 0.44), 10 0.00.
#
# Mirrors jobs/local/eval_unseen_D_900m_orix_normdemo_fix.sh and jobs/ibex/eval_unseen_C_dropcache.sh.
set -euo pipefail

cd "$(dirname "$0")/../.."

export Name=pi0_fast_incontext_prompt_action_7_state_8_train_split
EXP=$Name
STEP="${STEP:-19999}"
RUN=orix_normdemo_fix
CKPT_DIR="checkpoints/$Name/$EXP/$STEP"
GPUS=(${GPUS:-0 1 2 3})

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint directory missing: $CKPT_DIR"
    exit 2
fi

source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD/packages/openpi-client/src:$PWD/third_party/libero"
export MUJOCO_GL=egl

BUNDLED_CUDA_NVCC="$PWD/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$BUNDLED_CUDA_NVCC"
export PATH="$BUNDLED_CUDA_NVCC/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION=float32

mkdir -p "logs/$EXP/$RUN" "data/libero_incontext/$EXP/$RUN"

SUITES=(libero_spatial libero_object libero_goal libero_10)
PIDS=()
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

for i in "${!SUITES[@]}"; do
    SUITE=${SUITES[$i]}
    GPU=${GPUS[$i]}
    PREFIX=${SUITE#libero_}_unseen
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    echo "[$SUITE] GPU $GPU port $PORT"
    (
        CUDA_VISIBLE_DEVICES=$GPU uv run scripts/serve_policy.py --loader=INCONTEXT --port "$PORT" policy:checkpoint \
            --policy.inference-dtype=float32 --policy.config="ContextAR" --policy.dir="$CKPT_DIR" \
            > "logs/$EXP/$RUN/${PREFIX}_server.log" 2>&1 &
        SERVER_PID=$!
        trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
        sleep 20
        # Simulator must NOT see CUDA_VISIBLE_DEVICES (robosuite substring-parses it)
        env -u CUDA_VISIBLE_DEVICES python examples/libero/main_incontext_unseen.py \
            --args.host 127.0.0.1 \
            --args.port "$PORT" \
            --args.task_suite_name "$SUITE" \
            --args.video_out_path "data/libero_incontext/$EXP/$RUN" \
            --args.results_out_path "logs/$EXP/$RUN/${SUITE}_incontext_unseen_results.json" \
            > "logs/$EXP/$RUN/${PREFIX}_weight_float32.log" 2>&1
        echo "[$SUITE] client done"
    ) &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "=== results ==="
for SUITE in "${SUITES[@]}"; do
    F="logs/$EXP/$RUN/${SUITE}_incontext_unseen_results.json"
    if [ -f "$F" ]; then
        python -c "import json;d=json.load(open('$F'));print('$SUITE', d['summary'])"
    else
        echo "$SUITE MISSING"
        FAIL=1
    fi
done

# Sync to experiment tracker (set SKIP_SHEET_SYNC=1 to leave the sheet untouched)
if [ "${SKIP_SHEET_SYNC:-0}" != "1" ]; then
    claude -p "/log-to-sheet Read eval results from logs/$EXP/$RUN/*_results.json and sync to https://docs.google.com/spreadsheets/d/16It_o0GO_eYTpek65dSKr3sB0TOc_4FXZ5Uqwp9gKjU/edit?gid=184562329#gid=184562329 tab 'refactor'. Do not overwrite existing rows. Include config name ($Name), checkpoint path ($CKPT_DIR, row-6 orix-trained 19999), eval log path (logs/$EXP/$RUN), Training Machine: orix 4xH100 (pre-migration pipeline), Eval Machine: local 4xH100 (dropcache branch + demo-norm fix regression)." || echo "(log-to-sheet failed; results JSONs are in logs/$EXP/$RUN/)"
fi

exit $FAIL
