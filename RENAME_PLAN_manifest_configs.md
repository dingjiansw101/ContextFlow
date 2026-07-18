# Rename plan — remaining manifest configs → paper names

**Status:** PLANNED, awaiting go-ahead. No code/Drive/sheet changes made yet.

Follow-up to the first rename (`CONFIG_NAME_MAPPING.md`), which renamed the three core paper
methods + their ALOHA twins. This plan renames the **14 remaining old-named configs that appear
in the Drive backup `MANIFEST.json`** (the `+libero90` sweep variants, the FAST `900m` variant, and
the ALOHA plain-pi0 baseline) to paper names.

## Decisions (locked)

- **Naming scheme:** `Base_plus_libero90[_splitN]` — keep the meaningful `+libero90` / `900m` /
  `splitN` markers, matching the existing suffix convention.
- **Google Drive:** overlay-only. Add new-named shortcuts in `ContextFlow_Data` + update
  `MANIFEST.json`; **never** rename folders inside the source `ContextFlow` folder (same policy as
  the first rename).

## Why this is low-risk (what does NOT change)

- **Assets keys are already correct.** All 8 v18/v12 `+libero90` configs use
  `assets_repo_override="ContextFlow_Plain"`; all FAST configs use the kept
  `debug_pi0_fast_libero_incontext_inference`. → **no assets-key rename, no norm-stat recompute.**
- **Model classes / modules / imports:** unchanged (the variants reuse the already-renamed
  `contextflow` / `contextflow_plain` / `contextar` classes).
- **Existing checkpoints do not break:** orbax keys on nnx pytree paths and norm stats are baked
  into each checkpoint's `assets/`; renaming a `config.name` doesn't touch either.
- **No data movement** anywhere — Drive uses shortcut pointers; cluster checkpoint-dir renames are
  optional (see step 7).

## Scope — the 14 configs

| # | Old config name | Defined at | **New name** | Assets key (unchanged) |
|---|---|---|---|---|
| 1 | `pi0_libero_incontextv18_low_mem_finetune_sample_frames8_plus_libero90` | config_libero.py:2728 | `ContextFlow_plus_libero90` | `ContextFlow_Plain` |
| 2 | `…_sample_frames8_plus_libero90_split1` | config_libero.py:2980 | `ContextFlow_plus_libero90_split1` | `ContextFlow_Plain` |
| 3 | `…_sample_frames8_plus_libero90_split2` | config_libero.py:3043 | `ContextFlow_plus_libero90_split2` | `ContextFlow_Plain` |
| 4 | `…_sample_frames8_plus_libero90_split3` | config_libero.py:3106 | `ContextFlow_plus_libero90_split3` | `ContextFlow_Plain` |
| 5 | `pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor_plus_libero90` | config_libero.py:2665 | `ContextFlow_Plain_plus_libero90` | `ContextFlow_Plain` |
| 6 | `…_dataset_refactor_plus_libero90_split1` | config_libero.py:2791 | `ContextFlow_Plain_plus_libero90_split1` | `ContextFlow_Plain` |
| 7 | `…_dataset_refactor_plus_libero90_split2` | config_libero.py:2854 | `ContextFlow_Plain_plus_libero90_split2` | `ContextFlow_Plain` |
| 8 | `…_dataset_refactor_plus_libero90_split3` | config_libero.py:2917 | `ContextFlow_Plain_plus_libero90_split3` | `ContextFlow_Plain` |
| 9 | `pi0_fast_incontext_prompt_action_7_state_8_train_split_plus_libero90` | config_sequence.py:562 | `ContextAR_plus_libero90` | `debug_pi0_fast_libero_incontext_inference` |
| 10 | `…_train_split_plus_libero90_split1` | config_sequence.py:566 | `ContextAR_plus_libero90_split1` | `debug_pi0_fast_libero_incontext_inference` |
| 11 | `…_train_split_plus_libero90_split2` | config_sequence.py:570 | `ContextAR_plus_libero90_split2` | `debug_pi0_fast_libero_incontext_inference` |
| 12 | `…_train_split_plus_libero90_split3` | config_sequence.py:574 | `ContextAR_plus_libero90_split3` | `debug_pi0_fast_libero_incontext_inference` |
| 13 | `pi0_fast_incontext_prompt_action_7_state_8_train_split_900m` | config_sequence.py:546 (f-string loop, `suffix==""`) | `ContextAR_900m` | `debug_pi0_fast_libero_incontext_inference` |
| 14 | `pi0_aloha_data_unique_low_mem_finetune_no_test` | config_aloha.py:1353 | `Pi0_Aloha` | *(verify at execution — plain-pi0 key, keep as-is)* |

### Out of scope (stay old-named, same rule as the first rename)
Non-manifest siblings sharing the same generators: FAST `_train_split2..8`,
`_900m_avg_demo_img_plus_libero90`, `_split2_900m_avg_demo_img_plus_libero90`, the `_inference*`
variants; and the many v18/v12 ablations in `config_libero.py` (`_gemma2b*`, `_sample_actions*`,
`_split_v2/v3/v4`, `_without_img`, `_avg_current_img`, `_wo_compress_state`, `_without_text`, …).

## Steps, by surface

### 1. Code (~14 name-only edits)
- **config_libero.py** — 8 literal `name="…"` edits (rows 1–8).
- **config_sequence.py** — 4 name-arg edits to `_fast_split_plus_libero90_config(...)` (rows 9–12);
  1 special-case in the `_900m` loop so `suffix==""` emits `"ContextAR_900m"`
  (mirroring the existing `cfg_name = "ContextAR" if suffix == "" else …` for the base).
- **config_aloha.py** — 1 edit (row 14).
- No changes to model classes, assets keys, imports, or `config.py`.

### 2. Job scripts (~11 checked-in wrappers)
Scripts under `jobs/**` that pass these old names will fail `get_config` on the branch:
`train_v18_plus_libero90.sh`, `eval_v18_plus_libero90_unseen.sh`, `train_v12_plus_libero90.sh`,
`eval_v12_plus_libero90_unseen.sh`, `train_fast_prompt_action7_state8_900m.sh`,
`eval_fast_prompt_action7_state8_900m_unseen.sh`, `eval_unseen_D_900m_orix_normdemo_fix.sh`, and the
`*_900m_avg_demo_img_plus_libero90*` scripts. Add new-named wrappers (mirroring the first rename's
`jobs/orix/refactor_merge/eval_context*_unseen.sh`).

### 3. Google Drive `ContextFlow_Data` overlay (source untouched)
Extend `restructure_contextflow_data.py`: create real folders for the 13 libero configs
(`ContextFlow_plus_libero90/`, `…_split1/2/3/`, `ContextFlow_Plain_plus_libero90*`,
`ContextAR_plus_libero90*`, `ContextAR_900m/`) + `Pi0_Aloha/`, each holding one new-named exp
shortcut per manifest exp, pointing at the untouched old source folders. No bytes move.

### 4. `MANIFEST.json`
Rewrite each entry's `config`/`exp` to the new names, adding an `old_name`/`old_exp` field per entry
for provenance. Re-upload to the source `ContextFlow` folder (this edits the manifest file only, not
the checkpoint folders).

### 5. Google Sheets
Update/annotate the `+libero90` and `900m` rows in the refactor / ECCV-rebuttal / rename tabs with
the new names (add a mapping column rather than destroying the old names).

### 6. `CONFIG_NAME_MAPPING.md`
Append these 14 to the mapping tables (LIBERO `+libero90`/`900m` section + the ALOHA baseline).

### 7. (Optional) cluster checkpoint dirs
Only if you want to eval these under the new name in future:
`mv checkpoints/<old>/ checkpoints/<new>/` on each source machine (per `MANIFEST.json.source_machine`
— mostly orix `refactor_merge` exps + a few ibex `5b23ef9` exps). Existing eval via explicit
`--policy.dir` works without this.

## Validation
- `get_config(<new>)` resolves for all 14; each old name is gone from `_CONFIGS_DICT`.
- `uv run pytest -m "not manual" src/openpi/training` green.
- Optional: one smoke unseen-eval under a new name (e.g. `ContextAR_900m`) reproduces its reference SR.

## Order & rollback
Code → validate → job scripts → Drive overlay → manifest → sheets → mapping doc.
Fully reversible: changes are name-strings + shortcut pointers only; nothing overwrites checkpoint
bytes or the source `ContextFlow` folder.
