#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-kw60803_20260501_seq1}"
SPLITS="${SPLITS:-0 1 2}"

for split in $SPLITS; do
    name="pi0_fast_libero_split${split}"
    checkpoint="checkpoints/${name}/${name}/19999"
    echo "=== starting ${name} with run id ${RUN_ID} ==="
    bash jobs/local/eval_pi0_fast_libero_split_unseen.sh "$split" "$checkpoint" "$RUN_ID"
    echo "=== completed ${name} with run id ${RUN_ID} ==="
done
