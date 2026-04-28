# openpi — Pre-flight & Runtime Checks

## Normalization Stats

Training requires precomputed normalization stats. Without them, `data_loader.py` raises:
```
ValueError: Normalization stats not found.
```

**Check:**
```bash
ls assets/<config_name>/*/norm_stats* 2>/dev/null \
  || echo "MISSING: run 'uv run scripts/compute_norm_stats.py --config-name=<config_name>'"
```

**Generate:**
```bash
uv run scripts/compute_norm_stats.py --config-name=<config_name>
```

**Reuse existing stats via `assets_repo_override`:**
Configs sharing the same dataset and action representation (e.g., same `repo_id`, same `use_delta_joint_actions`) can reuse another config's norm stats instead of recomputing. Add to the `TrainConfig`:
```python
assets_repo_override="<config_with_existing_stats>",
```
This redirects `assets_dirs` from `./assets/<this_config>/` to `./assets/<other_config>/`. Example: `pi0_libero_split1` through `split4` all set `assets_repo_override="pi0_libero_split0"` to share one set of norm stats.

**Pre-flight check:** When a config lacks `assets_repo_override`, verify `./assets/<config_name>/*/norm_stats*` exists. If missing, either add `assets_repo_override` pointing to a compatible config or run `compute_norm_stats.py`.

## Eval Checkpoint

If the job script includes `serve_policy.py` or an eval phase, the inference server loads a checkpoint at a specific step. If that checkpoint doesn't exist, the server crashes silently and the eval client loops forever:
```
INFO:root:Still waiting for server...
```

**Check:**
```bash
ls <checkpoint_dir>/<exp_name>/<step> 2>/dev/null \
  || echo "MISSING: checkpoint at <checkpoint_dir>/<exp_name>/<step>"
```

## Cascade Failure Pattern

A common failure mode in openpi jobs that combine training + eval in one script:

1. Training crashes (e.g., missing norm stats) → no checkpoint is saved
2. `serve_policy.py` tries to load a non-existent checkpoint → server crashes
3. Eval client connects to `ws://0.0.0.0:<port>` → server never comes up → infinite `Still waiting for server...`
4. The SLURM job keeps running, burning GPU hours with no useful work

**Prevention:** Always verify norm stats and checkpoint paths before submitting combined train+eval jobs.

## Eval Parameter Consistency

Eval scripts often have default parameter values that silently produce wrong results when they don't match the training config's data split.

**Example:** `main_incontext_unseen.py` defaults to `--args.task_split split0`. If the training config uses `DEFAULT_LIBERO_TEST_TASK_V3` (split2), but the job script omits `--args.task_split`, the eval uses split0's seen/unseen task assignments — producing plausible but incorrect results.

**Mapping (openpi LIBERO):**

| Training config `remove_task_list` | Eval `--args.task_split` |
|-----------------------------------|--------------------------|
| `DEFAULT_LIBERO_TEST_TASK`    | `split0` (default) |
| `DEFAULT_LIBERO_TEST_TASK_V2` | `split1` |
| `DEFAULT_LIBERO_TEST_TASK_V3` | `split2` |
| `DEFAULT_LIBERO_TEST_TASK_V4` | `split3` |
| `DEFAULT_LIBERO_TEST_TASK_V5` | `split4` |
| `DEFAULT_LIBERO_TEST_TASK_V6` | `split5` |
| `DEFAULT_LIBERO_TEST_TASK_V7` | `split6` |
| `DEFAULT_LIBERO_TEST_TASK_V8` | `split7` |

**Check:** For every eval command in the job script:
1. Read the eval script's arg defaults (look for dataclass fields)
2. Identify which `remove_task_list` the training config uses
3. Verify the eval command explicitly passes the matching `--args.task_split`
4. If omitted, flag it before submitting

## Log-to-Sheet Metadata Paths

When logging results to the experiment tracking sheet, always include these metadata columns alongside metrics:

- **Checkpoint path**: `checkpoints/<config>/<exp_name>/<step>` — resolve from the job script's `serve_policy` command or checkpoint directory listing
- **Log path**: `logs/<config>/test1` (or whichever `<run_id>` was used) — the directory containing the eval `.log` files

These go in the columns after Config Name. Check existing rows in the sheet to confirm which columns they occupy.

## Structured Eval Results (JSON)

All eval scripts (`examples/libero/main*.py`) write a structured JSON results file at the end of evaluation. This is the **preferred source** for reading eval results programmatically — no log parsing needed.

- **Default path**: `logs/eval_results/<task_suite_name>_results.json` (in the `logs/` hierarchy, with task-suite-specific filenames to avoid collisions)
- **Override**: pass `--results-out-path /custom/path.json` to the eval script
- **Schema**: `config` (eval parameters), `per_task_results` (per-task success rates), `summary` (aggregated metrics including seen/unseen splits where applicable)

To read results:
```python
import json
with open("logs/eval_results/libero_spatial_results.json") as f:
    results = json.load(f)
print(results["summary"]["total_success_rate"])
```

In job scripts, override to save alongside eval logs:
```bash
python examples/libero/main_incontext_unseen.py \
  --args.results_out_path logs/${Name}/test1/goal_unseen_results.json \
  ...
```

## Post-Eval Result Sync

Eval scripts now produce a structured JSON file (`eval_results.json`) alongside videos. Use this as the primary data source for syncing results:

```bash
# Preferred: read structured JSON (no log parsing needed)
claude -p "/log-to-sheet Read eval results from logs/${Name}/test1/*_results.json and sync to https://docs.google.com/spreadsheets/d/16It_o0GO_eYTpek65dSKr3sB0TOc_4FXZ5Uqwp9gKjU/edit?gid=499236864#gid=499236864 tab Libero Experiments"
```

Fallback: eval logs are also written to `logs/${Name}/<run_id>/` and can still be parsed if the JSON file is unavailable.

`${Name}` is the experiment name variable already defined in the job script (e.g., `pi0_fast_libero_split0`).

## ORIX-specific failure modes (LIBERO+MuJoCo)

These failure modes are openpi+LIBERO-specific and only manifest on ORIX. They live in this project ref because they're not generic to ORIX.

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: Cannot initialize a EGL device display` (MuJoCo/robosuite) | ORIX nodes only have the kernel-side NVIDIA driver; the user-space EGL ICD is not on the default library path. Also: `.bashrc` is **never sourced** in SLURM batch jobs, so env vars set there are invisible. | Add these lines directly in the job script, **before** launching the simulator: `export __EGL_VENDOR_LIBRARY_DIRS=$HOME/nvidia-egl`, `export LD_LIBRARY_PATH=$HOME/nvidia-egl/lib:$LD_LIBRARY_PATH`, `export MUJOCO_GL=egl`. Run eval clients as `env -u CUDA_VISIBLE_DEVICES python ...` — note bare `python`, not `uv run`, because the simulator venv (`examples/libero/.venv`, Python 3.8) must be activated first via `source examples/libero/.venv/bin/activate`. The policy server still uses `uv run` (project root `.venv`, Python 3.11). Stripping `CUDA_VISIBLE_DEVICES` is needed because robosuite parses it as a substring and fails on multi-digit GPU IDs set by JAX. |
| LIBERO/MuJoCo client `Aborted (core dumped)` on H200 *only*, immediately after `[InjectDemoIndexes] Test task: …` (one crash per `libero_*` suite, no Python traceback) — same script runs fine on H100 | User-space EGL ICD shim at `~/nvidia-egl/lib` is H100-built and ABI-incompatible with the H200 driver; native segfault inside MuJoCo's EGL backend before any episode runs | **Pin the eval job script to H100:** add `#SBATCH --partition=batch-h100` and submit with `sbatch -q batch <script>`. Avoid `-p batch-h100,batch-h200` for any libero eval — comma-list partitions can land on H200. Use `scontrol update jobid=<N> Partition=batch-h100` to repin already-pending jobs without losing queue position. (Confirmed 2026-04-27 across splits 1, 3, 5 of `pi0_libero_incontextv18_normstats_fix` — all crashed identically on `orix-worker-h200-1`; splits 2, 4 with the same script worked on `orix-worker-h100-0`.) |
| `Normalization stats not found` | Missing `assets/<config>/.../norm_stats.json` | Check `assets_repo_override` (see § Normalization Stats above) OR run `compute_norm_stats.py` |
| `FileNotFoundError` on a metadata path | Precomputed cache missing | Run the corresponding `build_*_cache.py` script |
