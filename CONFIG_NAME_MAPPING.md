# Config name mapping

Training configs were renamed to their paper method names (`ContextFlow` /
`ContextFlow_Plain` / `ContextAR`, plus `_Aloha` variants). Use the **new** names
when launching training/eval (`uv run python scripts/train.py <new_name> ...`).
This file records the old → new mapping so existing checkpoints, job scripts, and
experiment logs (which still use the old names) can be cross-referenced.

## Config mappings

### LIBERO

| New name | Old name | Defined in | What it is |
|---|---|---|---|
| `ContextFlow` | `pi0_libero_incontextv18_low_mem_finetune_sample_frames8` | `src/openpi/training/config_libero.py` | LIBERO in-context v18, `sample_frames=8`, `sample_actions=128` (gemma_300m_v2 + gemma_300m_lora) |
| `ContextFlow_Plain` | `pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor` | `src/openpi/training/config_libero.py` | LIBERO in-context v12, `sample_frames=2`, `sample_actions=32`, no delta actions |
| `ContextAR` | `pi0_fast_incontext_prompt_action_7_state_8_train_split` | `src/openpi/training/config_sequence_debug.py` | pi0-FAST in-context, prompt-action 7 / state 8, LIBERO train split (base variant only) |

### ALOHA (`aloha_data_unique`)

| New name | Old name | Defined in | What it is |
|---|---|---|---|
| `ContextFlow_Aloha` | `pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_no_test` | `src/openpi/training/config_aloha.py` | ALOHA `aloha_data_unique` in-context v18, `sample_frames=8`, `sample_actions=128`; test tasks excluded (train) |
| `ContextFlow_Aloha_Inference` | `pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_inference` | `src/openpi/training/config_aloha.py` | Same as above, inference variant (no test-task filtering) |
| `ContextAR_Aloha` | `pi0_fast_aloha_data_unique_incontext_train_split_v1` | `src/openpi/training/config_aloha.py` | ALOHA `aloha_data_unique` pi0-FAST in-context, `sample_frames=2`, `sample_actions=4`; test tasks excluded (train) |
| `ContextAR_Aloha_Inference` | `pi0_fast_aloha_data_unique_incontext_inference` | `src/openpi/training/config_aloha.py` | Same as above, inference variant (no test-task filtering) |

**Not renamed (kept descriptive):** `pi0_aloha_data_unique_low_mem_finetune_no_test`
is the plain pi0 baseline, not one of the three paper in-context methods, so it keeps
its original name (mirrors LIBERO, where the pi0 baseline was likewise left untouched).

## File renames

| Old path | New path | Why |
|---|---|---|
| `scripts/v18_dataloader_check.py` | `scripts/contextflow_dataloader_check.py` | Method-specific consistency-check script; it loads the `ContextFlow` config by default, so its filename now reflects the paper method. Its `--config` default and the docstring reference in `scripts/dataloader_spawn_smoke.py` were updated accordingly. |

Other files that carry `v18`/`v12`/`incontext` in their names
(`src/openpi/models/pi0_incontextv18.py`, `pi0_incontextv12.py`,
`pi0_fast_incontext.py`, the `examples/**/main_incontext*.py`, etc.) were **not**
renamed: those name *model architectures / shared infrastructure*, not a single paper
method — see the class-name note below.

## Class names — checked, none renamed

The paper-method identity lives at the **config-name level** (`TrainConfig.name`), not
at the class level. Every candidate class is architecture/plumbing that is reused by
many configs (paper methods *and* non-paper ablations/splits), so renaming any of them
to `ContextFlow` / `ContextAR` would be semantically wrong and high-blast-radius:

| Class | Kind | Reused by |
|---|---|---|
| `Pi0IncontextConfigv18` / `Pi0Incontextv18` | model (v18 architecture) | `ContextFlow`, `ContextFlow_Aloha`, and ~69 v18 config sites (splits, sample-actions, ablations) |
| `Pi0IncontextConfigv12` / `Pi0Incontextv12` | model (v12 architecture) | `ContextFlow_Plain` and ~40 v12 config sites |
| `Pi0FASTIncontextConfig` / `Pi0FASTIncontext` | model (pi0-FAST in-context) | `ContextAR`, `ContextAR_Aloha`, and other FAST in-context configs |
| `LeRobotAlohaMobileIncontextDataConfig` / `LeRobotAlohaMobileFASTIncontextDataConfig` | data-config factory | 12 / 3 ALOHA config sites |
| `LeRobotLiberoIncontextDataConfig` / `CustomLeRobotLiberoIncontextDataConfig` | data-config factory | 53 / 47 LIBERO config sites |
| `AlohaMobileIncontextInputs/Outputs`, `LiberoIncontextInputs/Outputs`, `PolicyIncontext`, `PolicyFASTIncontext` | transforms / policy | shared inference/training plumbing |

Conclusion: **no class was renamed** — the config-name rename above is the complete,
correct surface for the paper-method rename.

## Things to know

- **Scope of this rename (intentional).** Only the config definitions and the Python
  tests/scripts that call `get_config()` were updated. The `jobs/**/*.sh` launch
  wrappers were **not** touched — they still pass the old names, so they will fail with
  a `get_config` "did you mean" error until you update them to the new names. Job
  wrappers to update (exact-name references): the `*_v18_sample_frames8*`,
  `*_v12_refactor_incontext*`, and `*_fast_prompt_action7_state8*` scripts under
  `jobs/orix/**` and `jobs/local/**`. Their `_900m` / `_plus_libero90` / numbered-split
  siblings reference *other* configs and are unaffected. No ALOHA `jobs/**` wrappers
  reference the four renamed ALOHA configs by name (the launch scripts live on the
  `kw61077` training box, not in-tree).

- **Assets are preserved — nothing to recompute.** Every renamed config keeps its
  **old** name as its own `assets_repo_override`, so norm-stats resolution is
  byte-identical after the rename:
  - `ContextFlow_Plain` is the shared assets anchor: 44 other configs point at it via
    `assets_repo_override`, plus its own self-reference (45 total). `ContextFlow`
    already borrowed those same assets and is unchanged.
  - The four `_Aloha` configs each carry `assets_repo_override=<their old name>`. The
    `ContextAR_Aloha_Inference` config already pointed its override at the train
    config's original name (`pi0_fast_aloha_data_unique_incontext_train_split_v1`), so
    both `ContextAR_Aloha` and its inference sibling resolve to the same on-disk assets.

- **Existing checkpoints live under the old-name directories.** Eval/resume builds the
  path `./checkpoints/<config_name>/<exp_name>/`, so runs trained before the rename are
  under `./checkpoints/<old name>/…`. To evaluate them under the new name, rename the
  on-disk dir on the cluster, e.g.:
  ```bash
  mv checkpoints/pi0_libero_incontextv18_low_mem_finetune_sample_frames8 checkpoints/ContextFlow
  mv checkpoints/pi0_aloha_data_unique_incontextv18_low_mem_finetune_sample_frames8_no_test checkpoints/ContextFlow_Aloha
  mv checkpoints/pi0_fast_aloha_data_unique_incontext_train_split_v1 checkpoints/ContextAR_Aloha
  ```
  (Do the same for the other renamed configs if you have checkpoints under their old
  names — the ALOHA `aloha_data_unique` checkpoints live on `kw61077`.) Assets do
  **not** need moving — see above.

- **`ContextAR` is the bare variant only.** Its config is generated by an f-string in
  `config_sequence_debug.py` that also produces `..._train_split2` … `_train_split8`,
  `..._train_split_900m`, and `..._train_split_plus_libero90*`. Those siblings keep
  their original names; only the empty-suffix base config became `ContextAR`. The ALOHA
  `ContextAR_Aloha` configs are standalone (not f-string-generated) and have no such
  siblings.
