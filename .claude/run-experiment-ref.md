# openpi — Pre-flight & Runtime Checks

## Python venvs

openpi jobs may touch up to two uv venvs:

| Venv | Python | Used by |
|------|--------|---------|
| Project root `.venv` | 3.11 | Training (`uv run scripts/train.py`, or `scripts/train_incontext.py` for `pi0_incontext*` configs — see "Train entry point" below), norm stats (`uv run scripts/compute_norm_stats.py`), policy server (`uv run scripts/serve_policy.py`) — every `uv run` from the project root. |
| `examples/libero/.venv` | 3.8 | LIBERO/MuJoCo eval client (`examples/libero/main*.py`). Pinned to 3.8 for the simulator. Job scripts activate it with `source examples/libero/.venv/bin/activate` and run a bare `python`, not `uv run`. |

If the job script never `source`s `examples/libero/.venv/bin/activate` and has no `cd examples/libero`, only the project root venv is needed.

**Pre-flight (covers both, when both apply):**
```bash
bash "$SKILL_DIR/scripts/check_python_env.sh" .venv examples/libero/.venv
```

If a venv is missing or broken, rebuild with `uv sync` in that venv's parent directory (project root for `.venv`, `examples/libero/` for the LIBERO venv). Do not submit until the script reports `OK` for every venv the job references.

## Train entry point: `train.py` vs `train_incontext.py`

For configs with `use_custom_dataloader=True` (any `pi0_incontext*` family — v12, v18, etc.), the training command must be:

```bash
uv run scripts/train_incontext.py <config> --exp-name=...
```

`scripts/train.py` always calls `_data_loader.create_data_loader` and silently ignores `config.use_custom_dataloader`. The standard repack transform then expects `dem_prompt_actions` (produced only by `CustomLeRobotDataset` via `scripts/train_incontext.py` → `create_custom_incontext_data_loader`), so training crashes ~2 min in with `KeyError: 'dem_prompt_actions'`.

**Pre-flight:** When generating a new training job script, grep the config in `src/openpi/training/config*.py` for `use_custom_dataloader=True`. If true (or the `model=` line names a `pi0_incontext*` class), use `scripts/train_incontext.py`. Reference: `jobs/ibex/contextflow_libero.sh`.

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
This redirects `assets_dirs` from `./assets/<this_config>/` to `./assets/<other_config>/`. Example: `pi0_libero_low_mem_finetune_split_train` sets `assets_repo_override="pi0_libero_heldout"` to reuse that config's norm stats.

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

On this branch there is a single held-out task set, so the split mismatch this section used to warn
about cannot occur: the training configs exclude `DEFAULT_LIBERO_TEST_TASK`
(`src/openpi/training/config.py`), and the eval clients classify against the matching `UNSEEN_TASKS`
constant. There is no `--args.task_split` flag any more.

**Check:** the two lists are duplicated across the train/eval process boundary (the clients run in a
separate Python 3.8 environment and cannot import `openpi`). If you change one, change the other —
nothing enforces it automatically. A silent divergence would train on a task that is then scored as
unseen.

## Log-to-Sheet Metadata Paths

When logging results to the experiment tracking sheet, always include these metadata columns alongside metrics:

- **Checkpoint path**: `checkpoints/<config>/<exp_name>/<step>` — resolve from the job script's `serve_policy` command or checkpoint directory listing
- **Log path**: `logs/<config>/test1` (or whichever `<run_id>` was used) — the directory containing the eval `.log` files

These go in the columns after Config Name. Check existing rows in the sheet to confirm which columns they occupy.

## Structured Eval Results (JSON)

All eval scripts (`examples/libero/main*.py`) write a structured JSON results file at the end of evaluation. This is the **preferred source** for reading eval results programmatically — no log parsing needed.

- **Default path**: `logs/eval_results/<task_suite_name>_<variant>_results.json` (e.g. `libero_spatial_incontext_results.json`; `<variant>` is `base` for `main.py`, `incontext` for `main_incontext.py`, `incontext_unseen` for `main_incontext_unseen.py`)
- **Override**: pass `--args.results_out_path /custom/path.json` — typical usage is to save alongside the eval logs:
  ```bash
  python examples/libero/main_incontext_unseen.py \
    --args.results_out_path logs/${Name}/test1/goal_unseen_results.json \
    ...
  ```
- **Schema**: `config` (eval parameters), `per_task_results` (per-task success rates), `summary` (aggregated metrics including seen/unseen splits where applicable)

To sync to the experiment tracking sheet:
```bash
claude -p "/log-to-sheet Read eval results from logs/${Name}/test1/*_results.json and sync to https://docs.google.com/spreadsheets/d/16It_o0GO_eYTpek65dSKr3sB0TOc_4FXZ5Uqwp9gKjU/edit?gid=499236864#gid=499236864 tab Libero Experiments"
```

Fallback: eval logs at `logs/${Name}/<run_id>/` can still be parsed if the JSON file is unavailable. `${Name}` is the experiment name variable defined in the job script (e.g. `pi0_fast_libero_heldout`).

## ORIX-specific failure modes (LIBERO+MuJoCo)

These failure modes are openpi+LIBERO-specific and only manifest on ORIX. They live in this project ref because they're not generic to ORIX.

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: Cannot initialize a EGL device display` (MuJoCo/robosuite) | ORIX nodes only have the kernel-side NVIDIA driver; the user-space EGL ICD is not on the default library path. Also: `.bashrc` is **never sourced** in SLURM batch jobs, so env vars set there are invisible. | Add these lines directly in the job script, **before** launching the simulator: `export __EGL_VENDOR_LIBRARY_DIRS=$HOME/nvidia-egl`, `export LD_LIBRARY_PATH=$HOME/nvidia-egl/lib:$LD_LIBRARY_PATH`, `export MUJOCO_GL=egl`. Run eval clients as `env -u CUDA_VISIBLE_DEVICES python ...` — note bare `python`, not `uv run`, because the simulator venv (`examples/libero/.venv`, Python 3.8) must be activated first via `source examples/libero/.venv/bin/activate`. The policy server still uses `uv run` (project root `.venv`, Python 3.11). Stripping `CUDA_VISIBLE_DEVICES` is needed because robosuite parses it as a substring and fails on multi-digit GPU IDs set by JAX. |
| LIBERO/MuJoCo client `Aborted (core dumped)` on H200 *only*, immediately after `[InjectDemoIndexes] Test task: …` (one crash per `libero_*` suite, no Python traceback) — same script runs fine on H100 | User-space EGL ICD shim at `~/nvidia-egl/lib` is H100-built and ABI-incompatible with the H200 driver; native segfault inside MuJoCo's EGL backend before any episode runs | **Pin the eval job script to H100:** add `#SBATCH --partition=batch-h100` and submit with `sbatch -q batch <script>`. Avoid `-p batch-h100,batch-h200` for any libero eval — comma-list partitions can land on H200. Use `scontrol update jobid=<N> Partition=batch-h100` to repin already-pending jobs without losing queue position. (Confirmed 2026-04-27 across splits 1, 3, 5 of `pi0_libero_incontextv18_normstats_fix` — all crashed identically on `orix-worker-h200-1`; splits 2, 4 with the same script worked on `orix-worker-h100-0`.) |
| `Normalization stats not found` | Missing `assets/<config>/.../norm_stats.json` | Check `assets_repo_override` (see § Normalization Stats above) OR run `compute_norm_stats.py` |

## Ibex-specific failure modes (LIBERO+MuJoCo)

Don't port the ORIX EGL workaround — Ibex compute has full system EGL.

| Symptom | Fix |
|---|---|
| Eval client `ModuleNotFoundError: torch` / `robosuite` / `libero` | `examples/libero/.venv` not built. Bootstrap once: `uv venv --python 3.8 examples/libero/.venv && source examples/libero/.venv/bin/activate && uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt packages/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match` (3-file sync matches Dockerfile). |
| `ModuleNotFoundError: openpi_client` after libero imports succeed | `export PYTHONPATH="${PYTHONPATH:-}:$PWD:$PWD/packages/openpi-client/src:$PWD/third_party/libero"`. |
| `ImportError: Cannot initialize a EGL device display` | Drop ORIX EGL exports (`__EGL_VENDOR_LIBRARY_DIRS`, `LD_LIBRARY_PATH=$HOME/nvidia-egl/lib:...`) — the user-space libs pin the wrong driver version. Just `MUJOCO_GL=egl`. |
