#!/usr/bin/env bash
set -euo pipefail

# Submit the pi0_libero_heldout training job on ORIX.
#
# Examples:
#   bash jobs/orix/submit_pi0_libero_heldout.sh --test-only
#   SUBMIT_CMD=sbatch PARTITION=batch-h100 QOS=batch bash jobs/orix/submit_pi0_libero_heldout.sh submit
#   SUBMIT_CMD=sbatch-free bash jobs/orix/submit_pi0_libero_heldout.sh submit

MODE="${1:-submit}"
if [ "$MODE" != "submit" ] && [ "$MODE" != "--test-only" ]; then
    echo "usage: $0 [--test-only|submit] [config ...]" >&2
    exit 64
fi
shift || true

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    CONFIGS=(pi0_libero_heldout)
fi

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi_libero/openpi}"
SUBMIT_CMD="${SUBMIT_CMD:-sbatch}"
PARTITION="${PARTITION:-batch-h100}"
QOS="${QOS:-batch}"
GPUS="${GPUS:-4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-16}"
MEM="${MEM:-400G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

cd "$REPO"
mkdir -p logs errs

for job_name in "${CONFIGS[@]}"; do
    case "$job_name" in
        pi0_libero_heldout) ;;
        *) echo "unsupported config: $job_name" >&2; exit 64 ;;
    esac

    sbatch_args=(
        --job-name="$job_name"
        --output="logs/%x-%j.log"
        --error="errs/%j-%x.err"
        --gres="gpu:${GPUS}"
        --cpus-per-gpu="$CPUS_PER_GPU"
        --mem="$MEM"
        --time="$TIME_LIMIT"
        --chdir="$REPO"
        --export=ALL,CONFIG_NAME="$job_name"
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

    echo "Submitting ${job_name} with ${SUBMIT_CMD}: ${sbatch_args[*]}"
    "$SUBMIT_CMD" "${sbatch_args[@]}" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi_libero/openpi}"
cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

if ! find assets/pi0_libero_heldout -maxdepth 4 -type f -name 'norm_stats*' | grep -q .; then
    echo "Missing assets/pi0_libero_heldout norm stats; run jobs/orix/compute_pi0_libero_heldout_norm_stats.sh first." >&2
    exit 66
fi

NAME="${CONFIG_NAME}"

trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"' SIGTERM
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$NAME" --project-name=openpi --exp-name="$NAME" --resume &
pid=$!
wait "$pid"
SBATCH
done
