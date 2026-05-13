#!/usr/bin/env bash
set -euo pipefail

# Submit pi0_fast_libero_split{0..2} training jobs on ORIX.
#
# Examples:
#   bash jobs/orix/submit_pi0_fast_libero_splits.sh --test-only
#   SUBMIT_CMD=sbatch PARTITION=batch-h100 QOS=batch bash jobs/orix/submit_pi0_fast_libero_splits.sh submit
#   SUBMIT_CMD=sbatch-free bash jobs/orix/submit_pi0_fast_libero_splits.sh submit 0 2

MODE="${1:-submit}"
if [ "$MODE" != "submit" ] && [ "$MODE" != "--test-only" ]; then
    echo "usage: $0 [--test-only|submit] [split ...]" >&2
    exit 64
fi
shift || true

SPLITS=("$@")
if [ "${#SPLITS[@]}" -eq 0 ]; then
    SPLITS=(0 1 2)
fi

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi_libero/openpi}"
SUBMIT_CMD="${SUBMIT_CMD:-sbatch}"
PARTITION="${PARTITION:-batch-h100}"
QOS="${QOS:-batch}"
GPUS="${GPUS:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-16}"
MEM="${MEM:-160G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
STATS_PATH="assets/pi0_fast_libero_split0/physical-intelligence/libero/norm_stats.json"

cd "$REPO"
mkdir -p logs errs

if [ ! -f "$STATS_PATH" ]; then
    echo "Missing required norm stats: $REPO/$STATS_PATH" >&2
    exit 66
fi

for split in "${SPLITS[@]}"; do
    case "$split" in
        0|1|2) ;;
        *) echo "unsupported split: $split" >&2; exit 64 ;;
    esac

    job_name="pi0_fast_libero_split${split}"
    sbatch_args=(
        --job-name="$job_name"
        --output="logs/%x-%j.log"
        --error="errs/%j-%x.err"
        --gres="gpu:${GPUS}"
        --cpus-per-gpu="$CPUS_PER_GPU"
        --mem="$MEM"
        --time="$TIME_LIMIT"
        --chdir="$REPO"
        --export=ALL,SPLIT="$split"
    )

    case "$SUBMIT_CMD" in
        sbatch)
            sbatch_args=(--partition="$PARTITION" --qos="$QOS" "${sbatch_args[@]}")
            ;;
        sbatch-pi)
            ;;
        sbatch-free)
            sbatch_args=(--requeue --signal=B:SIGTERM@60 "${sbatch_args[@]}")
            ;;
        *)
            echo "unsupported SUBMIT_CMD=$SUBMIT_CMD" >&2
            exit 64
            ;;
    esac

    if [ "$MODE" = "--test-only" ]; then
        sbatch_args=(--test-only "${sbatch_args[@]}")
    fi

    echo "Submitting split ${split} with ${SUBMIT_CMD}: ${sbatch_args[*]}"
    "$SUBMIT_CMD" "${sbatch_args[@]}" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi_libero/openpi}"
cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

STATS_PATH="assets/pi0_fast_libero_split0/physical-intelligence/libero/norm_stats.json"
if [ ! -f "$STATS_PATH" ]; then
    echo "Missing required norm stats: $REPO/$STATS_PATH" >&2
    exit 66
fi

NAME="pi0_fast_libero_split${SPLIT}"

trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"' SIGTERM
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$NAME" --project-name=openpi --exp-name="$NAME" --resume &
pid=$!
wait "$pid"
SBATCH
done
