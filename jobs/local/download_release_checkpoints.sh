#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# ContextFlow_4gpu is already present locally and verified against the release.
EXCLUDES=()
if [[ "${SKIP_ORIX:-0}" == 1 ]]; then
    EXCLUDES+=(--exclude '/ContextFlow_defaultprec_2gpu_seed42_run1/**' --exclude '/ContextFlow_refactor_merge_nw16/**')
fi
rclone copy 'contextflow,root_folder_id=1Bf5j90lifJ9kPy2YSQG1bp5FKWZwzTES:ContextFlow' \
    checkpoints/ContextFlow --exclude '/ContextFlow_4gpu/**' \
    "${EXCLUDES[@]}" \
    --checksum --transfers 4 --checkers 4 --retries 3 --stats 30s --stats-log-level NOTICE
