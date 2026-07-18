# Config name mapping

The in-context configs, their **model classes**, the **module files**, and the
**assets keys** were all renamed to the paper method names (`ContextFlow` /
`ContextFlow_Plain` / `ContextAR`, plus `_Aloha` variants). Use the **new** config names
when launching training/eval (`uv run python scripts/train.py <new_name> ...`). This file
records the old → new mapping so existing checkpoints, job scripts, and experiment logs
(which still use the old names) can be cross-referenced.

## Config-name mappings

### LIBERO

| New name | Old name | Defined in | What it is |
|---|---|---|---|
| `ContextFlow` | `pi0_libero_incontextv18_low_mem_finetune_sample_frames8` | `config_libero.py` | LIBERO in-context v18, `sample_frames=8`, `sample_actions=128` (gemma_300m_v2 + gemma_300m_lora) |
| `ContextFlow_Plain` | `pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor` | `config_libero.py` | LIBERO in-context v12, `sample_frames=2`, `sample_actions=32`, no delta actions |
| `ContextAR` | `pi0_fast_incontext_prompt_action_7_state_8_train_split` | `config_sequence.py` | pi0-FAST in-context, prompt-action 7 / state 8, LIBERO train split (base variant only) |

### LIBERO — `+libero90` sweep & `900m` (second batch)

Renamed to `<paper-base>_plus_libero90[_splitN]` / `ContextAR_900m`. `splitN` configs are evaluated on
the matching `--task_split splitN`; the base `_plus_libero90` and `900m` on `split0`. **Name-only** —
assets keys are unchanged (`ContextFlow_Plain` for v18/v12, `debug_pi0_fast_libero_incontext_inference`
for FAST), so existing checkpoints + baked norm stats are unaffected.

| New name | Old name | Defined in |
|---|---|---|
| `ContextFlow_plus_libero90` | `pi0_libero_incontextv18_low_mem_finetune_sample_frames8_plus_libero90` | `config_libero.py` |
| `ContextFlow_plus_libero90_split1` | `…_sample_frames8_plus_libero90_split1` | `config_libero.py` |
| `ContextFlow_plus_libero90_split2` | `…_sample_frames8_plus_libero90_split2` | `config_libero.py` |
| `ContextFlow_plus_libero90_split3` | `…_sample_frames8_plus_libero90_split3` | `config_libero.py` |
| `ContextFlow_Plain_plus_libero90` | `pi0_libero_refactor_incontextv12_…_dataset_refactor_plus_libero90` | `config_libero.py` |
| `ContextFlow_Plain_plus_libero90_split1` | `…_dataset_refactor_plus_libero90_split1` | `config_libero.py` |
| `ContextFlow_Plain_plus_libero90_split2` | `…_dataset_refactor_plus_libero90_split2` | `config_libero.py` |
| `ContextFlow_Plain_plus_libero90_split3` | `…_dataset_refactor_plus_libero90_split3` | `config_libero.py` |
| `ContextAR_plus_libero90` | `pi0_fast_incontext_prompt_action_7_state_8_train_split_plus_libero90` | `config_sequence.py` |
| `ContextAR_plus_libero90_split1` | `…_train_split_plus_libero90_split1` | `config_sequence.py` |
| `ContextAR_plus_libero90_split2` | `…_train_split_plus_libero90_split2` | `config_sequence.py` |
| `ContextAR_plus_libero90_split3` | `…_train_split_plus_libero90_split3` | `config_sequence.py` |
| `ContextAR_900m` | `pi0_fast_incontext_prompt_action_7_state_8_train_split_900m` | `config_sequence.py` (f-string loop, `suffix==""` special-case) |

### ALOHA (`aloha_data_unique`)

| New name | Old name | Defined in | What it is |
|---|---|---|---|
| `ContextFlow_Aloha` | `pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_no_test` | `config_aloha.py` | ALOHA in-context v18, `sample_frames=8`, `sample_actions=128`; test tasks excluded (train) |
| `ContextFlow_Aloha_Inference` | `pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_inference` | `config_aloha.py` | Same, inference variant (no test-task filtering) |
| `ContextAR_Aloha` | `pi0_fast_aloha_data_unique_incontext_train_split_v1` | `config_aloha.py` | ALOHA pi0-FAST in-context, `sample_frames=2`, `sample_actions=4`; test tasks excluded (train) |
| `ContextAR_Aloha_Inference` | `pi0_fast_aloha_data_unique_incontext_inference` | `config_aloha.py` | Same, inference variant (no test-task filtering) |
| `Pi0_Aloha` | `pi0_aloha_data_unique_low_mem_finetune_no_test` | `config_aloha.py` | ALOHA plain-pi0 baseline (paper "pi0" row); not an in-context method. Uses `pi0_base` assets (no `assets_repo_override`). Renamed in the second batch. |

**Note:** the **LIBERO** plain-pi0 baseline is still left untouched (descriptive name); only the
**ALOHA** pi0 baseline was renamed (→ `Pi0_Aloha`) in the second batch.

## Model class / module / file renames

| Old class(es) | New class(es) | Module file (old → new) |
|---|---|---|
| `Pi0IncontextConfigv18` / `Pi0Incontextv18` | `ContextFlowConfig` / `ContextFlow` | `models/pi0_incontextv18.py` → `models/contextflow.py` |
| `Pi0IncontextConfigv12` / `Pi0Incontextv12` | `ContextFlowPlainConfig` / `ContextFlowPlain` | `models/pi0_incontextv12.py` → `models/contextflow_plain.py` |
| `Pi0FASTIncontextConfig` / `Pi0FASTIncontext` | `ContextARConfig` / `ContextAR` | `models/pi0_fast_incontext.py` → `models/contextar.py` |
| (unit test) — | — | `models/pi0_incontextv18_test.py` → `models/contextflow_test.py` |
| `SequenceDebugLeRobotLiberoIncontextDataConfig` | `SequenceLeRobotLiberoIncontextDataConfig` | `training/config_sequence_debug.py` → `training/config_sequence.py` |
| `MultiCustomSequenceDebugLeRobotLiberoIncontextDataConfig` | `MultiCustomSequenceLeRobotLiberoIncontextDataConfig` | (same file) |

In `config.py` the module import aliases were renamed to match
(`pi0_incontextv18`→`contextflow`, `pi0_incontextv12`→`contextflow_plain`,
`pi0_fast_incontext`→`contextar`); every `api.<module>.<Class>` call site in
`config_libero.py` / `config_aloha.py` and the direct importers
(`policy_config.py`, `data_loader.py`, `model_test.py`, `config_libero_test.py`,
`contextflow_test.py`) were updated. `config_sequence.py` stays glob-discovered
(`config_*.py`), so no import changed for it.

**Intentionally NOT renamed:**
- `PerceiverCompressor` (a generic compression module, not method-specific).
- `Pi0FASTIncontextSeq` / `Pi0FASTIncontextSeqConfig` and `models/pi0_fast_incontext_seq.py`
  — an experimental debug sibling, not a paper method (see the ⚠ note below).
- `ModelType` enum + its string values (`model.py`) — enum values, unrelated to class names.
- The shared plumbing classes `LeRobot*IncontextDataConfig`, `*IncontextInputs/Outputs`,
  `PolicyIncontext`/`PolicyFASTIncontext` — architecture reused across many configs.

> ⚠ **LIBERO `ContextAR` uses the Seq class, not `ContextARConfig`.** The renamed
> `ContextARConfig` (ex-`Pi0FASTIncontextConfig`) backs the **ALOHA** `ContextAR_Aloha`
> configs. The **LIBERO** `ContextAR` config (built by `make_config` in
> `config_sequence.py`) uses `Pi0FASTIncontextSeqConfig`, which was left unrenamed. This is
> a pre-existing architecture split between the LIBERO and ALOHA ContextAR configs, not
> introduced by this rename — don't read `ContextARConfig` as "the LIBERO ContextAR model".

## File renames (scripts)

| Old path | New path | Why |
|---|---|---|
| `scripts/v18_dataloader_check.py` | `scripts/contextflow_dataloader_check.py` | Loads the `ContextFlow` config by default; filename now reflects the method. Its `--config` default and the docstring reference in `scripts/dataloader_spawn_smoke.py` were updated too. |

## Assets-key (`assets_repo_override`) renames

The `assets_repo_override` values were renamed from the old descriptive strings to the
paper names, so the assets key now matches the config name.

| Old assets key | New assets key | Shared by |
|---|---|---|
| `pi0_libero_refactor_incontextv12_…_dataset_refactor` (the anchor) | `ContextFlow_Plain` | `ContextFlow`, `ContextFlow_Plain`, **+ 43 non-paper LIBERO split/eval configs** — all 45 `assets_repo_override=` lines repointed |
| `pi0_aloha_data_unique_incontextv18_…_no_test` | `ContextFlow_Aloha` | `ContextFlow_Aloha` |
| `pi0_aloha_data_unique_incontextv18_…_inference` | `ContextFlow_Aloha_Inference` | `ContextFlow_Aloha_Inference` |
| `pi0_fast_aloha_data_unique_incontext_train_split_v1` | `ContextAR_Aloha` | `ContextAR_Aloha` + `ContextAR_Aloha_Inference` (shared) |

**⚠ Requires a one-time on-disk move per training filesystem.** `./assets/` is **gitignored**
(not in the commit) and norm stats are read from `./assets/<assets_key>/<repo_id>/`. After
pulling this change, rename the local assets dir(s) so the new keys resolve — otherwise
`config.py` silently logs "Norm stats not found … skipping" and trains with **no** norm
stats (no hard error):

```bash
# Only the anchor exists locally (as a symlink); do this in each checkout's ./assets:
mv ./assets/pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor \
   ./assets/ContextFlow_Plain

# On kw61077 (ALOHA training box), if these dirs exist:
mv ./assets/pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_no_test  ./assets/ContextFlow_Aloha
mv ./assets/pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_inference ./assets/ContextFlow_Aloha_Inference
mv ./assets/pi0_fast_aloha_data_unique_incontext_train_split_v1                          ./assets/ContextAR_Aloha
```

(The ALOHA keys have no local dir in this repo — they resolve from trained checkpoints,
which carry their own baked-in norm stats and are unaffected.) Left unchanged (out of
scope): the non-paper `pi0_aloha_objects_task_suite_incontextv18_…` key and the
`debug_pi0_fast_libero_incontext_inference` key used by the LIBERO ContextAR debug configs.

## Things to know

- **Checkpoints are keyed on `config.name`, not on the assets key.** Eval/resume builds
  `./checkpoints/<config_name>/<exp_name>/`, so runs trained before the rename live under
  `./checkpoints/<old name>/…`. To evaluate them under the new name, rename the on-disk
  checkpoint dir on the cluster, e.g.:
  ```bash
  mv checkpoints/pi0_libero_incontextv18_low_mem_finetune_sample_frames8 checkpoints/ContextFlow
  mv checkpoints/pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_no_test checkpoints/ContextFlow_Aloha
  mv checkpoints/pi0_fast_aloha_data_unique_incontext_train_split_v1 checkpoints/ContextAR_Aloha
  ```
  Renaming classes/modules/asset-keys does **not** break existing checkpoints: orbax keys
  on nnx pytree attribute paths, and norm stats are baked into each checkpoint's `assets/`.

- **`jobs/**/*.sh` launch wrappers were not touched** — they still pass the old config
  names, so they will fail `get_config` until updated to the new names. The ALOHA launch
  scripts live on `kw61077`, not in-tree.

- **`ContextAR` family renames (two batches).** The FAST configs are generated by f-strings /
  helpers in `config_sequence.py`. **Renamed to paper names:** the bare base (`ContextAR`), the
  `900m` base (`ContextAR_900m`, via a `suffix==""` special-case in the 900m loop), and the four
  `..._train_split_plus_libero90[_split1/2/3]` configs (`ContextAR_plus_libero90[_splitN]`).
  **Still descriptive (out of scope):** the numbered `..._train_split2` … `_train_split8` (and
  their `_900m` forms), the `..._900m_avg_demo_img_plus_libero90` avg-image variants, and the
  `_inference*` eval configs. The ALOHA `ContextAR_Aloha` configs are standalone (not
  f-string-generated) and have no siblings.

## Job-script mapping (old → new)

New-name eval job scripts were added for the renamed LIBERO configs. The old scripts still pass
the old config names (they fail `get_config` on the rename branch), so use the new ones. All serve
the **new** config name via `scripts/serve_policy.py --loader=INCONTEXT` (the `serve_policy_incontext.py`
in older docs does not exist).

### orix SLURM eval wrappers (unseen, split0)

| Config | Old job script | New job script |
|---|---|---|
| `ContextFlow` | `jobs/orix/refactor_merge/eval_v18_sample_frames8_unseen.sh` | `jobs/orix/refactor_merge/eval_contextflow_unseen.sh` |
| `ContextFlow_Plain` | `jobs/orix/refactor_merge/eval_v12_refactor_incontext_unseen.sh` | `jobs/orix/refactor_merge/eval_contextflow_plain_unseen.sh` |
| `ContextAR` | `jobs/orix/refactor_merge/eval_fast_prompt_action7_state8_unseen.sh` | `jobs/orix/refactor_merge/eval_contextar_unseen.sh` |

The FAST eval historically served the untouched sibling `pi0_fast_incontext_prompt_action_7_state_8_inference`;
the new script serves `ContextAR` directly. This is serve-identical — same model dims, same assets key
`debug_pi0_fast_libero_incontext_inference`, and the policy dataset spans all episodes (`episodes=None`),
so the train-time `remove_task_list` difference between the two configs never affects inference.

### New machine-specific eval scripts (all three configs)

| Machine | New job script | How it runs |
|---|---|---|
| visioncair (local) | `jobs/local/eval_incontext_unseen_local.sh <run-name> <policy-config> <ckpt-dir> [run-id]` | borrowed venvs + `PYTHONPATH=src` (renamed code) + system nvidia EGL; server doesn't preallocate GPU, so unseen suites pack across GPUs (env `GPUS`). Supersedes the per-config `eval_unseen_{A_v18,C_orix,D_900m}_*_normdemo_fix.sh` for the renamed configs. |
| ibex (SLURM) | `jobs/ibex/eval_incontext_unseen_ibex.sh` (`sbatch --export=ALL,NAME=…,POLICY_CONFIG=…,CKPT_DIR=…`) | mirrors the proven `openpi_repro_nw16/jobs/ibex/eval_v18_sf8_repro_nw16_unseen.sh`: one server, sequential suites, ibex EGL env (cuda-12.1 + mujoco210 + `~/nvidia-egl`). |

The shared orchestrator `jobs/local/eval_pi0_libero_incontext_unseen.sh` is unchanged; the orix wrappers
still call it. The new visioncair/ibex scripts bypass it to inject `PYTHONPATH=src` for the borrowed-venv
setup (the rename worktree has no synced `.venv`).
