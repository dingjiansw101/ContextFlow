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

### ALOHA (`aloha_data_unique`)

| New name | Old name | Defined in | What it is |
|---|---|---|---|
| `ContextFlow_Aloha` | `pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_no_test` | `config_aloha.py` | ALOHA in-context v18, `sample_frames=8`, `sample_actions=128`; test tasks excluded (train) |
| `ContextFlow_Aloha_Inference` | `pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_inference` | `config_aloha.py` | Same, inference variant (no test-task filtering) |
| `ContextAR_Aloha` | `pi0_fast_aloha_data_unique_incontext_train_split_v1` | `config_aloha.py` | ALOHA pi0-FAST in-context, `sample_frames=2`, `sample_actions=4`; test tasks excluded (train) |
| `ContextAR_Aloha_Inference` | `pi0_fast_aloha_data_unique_incontext_inference` | `config_aloha.py` | Same, inference variant (no test-task filtering) |

**Not renamed (kept descriptive):** `pi0_aloha_data_unique_low_mem_finetune_no_test` is the
plain pi0 baseline, not one of the paper in-context methods (mirrors LIBERO, where the pi0
baseline was left untouched).

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

- **`ContextAR` is the bare variant only.** Its config is generated by an f-string in
  `config_sequence.py` that also produces `..._train_split2` … `_train_split8`,
  `..._train_split_900m`, and `..._train_split_plus_libero90*`. Those siblings keep their
  original names; only the empty-suffix base config became `ContextAR`. The ALOHA
  `ContextAR_Aloha` configs are standalone (not f-string-generated) and have no siblings.
