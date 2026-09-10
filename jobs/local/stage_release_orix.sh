#!/usr/bin/env bash
# Source locations are recorded in the original release inventory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
BASE=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge/checkpoints/pi0_libero_incontextv18_low_mem_finetune_sample_frames8
for SUFFIX in defaultprec_2gpu_seed42_run1 refactor_merge_nw16; do
    DEST="checkpoints/ContextFlow/ContextFlow_$SUFFIX/19999"
    mkdir -p "$DEST"
    rsync -a --checksum --info=progress2 \
        --include='/params/***' --include='/assets/***' --include='/_CHECKPOINT_METADATA' --exclude='*' \
        "orix:$BASE/pi0_libero_incontextv18_low_mem_finetune_sample_frames8_$SUFFIX/19999/" "$DEST/"
done
