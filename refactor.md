# Refactor / Roadmap TODO

Updated: 2026-04-28 — statuses verified against the codebase. Legend: `[x]` done, `[~]` partial / needs follow-up, `[ ]` not started.

## Extension

High priority:
- [ ] ContextAR (600M) — no code in `src/openpi/` yet
- [ ] ContextAR (600M) + q-former — no code yet
- [ ] ContextFlow 2B — no code yet

Low priority:
- [ ] Sequence training — no `sequence_training` / `seq_train` code in `src/openpi/`
- [ ] Light version, light-version inherit from the base version — no `_light` variants found

## Refactor the data pipeline

- [x] refactor the custom dataset and custom dataset v2 — only `src/openpi/training/custom_dataset.py` remains; v2 merged in (commits `597fc42`, `e9ddc6c`)
- [ ] fix the padding bug in custom dataset — revisit if still reproduces

## Optimize the code

- [ ] can we simplify the `reindex_filtered_dict` function (in `src/openpi/transforms.py`)? — still present; simplification not yet assessed
- [ ] clean the configs
- [ ] organize the unit testing files
- [~] remove third image for libero dataset — `src/openpi/policies/libero_policy.py:54` zeros out `right_wrist_0_rgb` in the transform instead of dropping it from the model input; full removal still pending
- [ ] merge `pi0_incontextv12` and `pi0_incontextv18` — both files still in `src/openpi/models/`
- [ ] merge the light versions — n/a, no light versions exist yet
- [x] simplify the data preparation process — addressed by `edf2f30 remove generate_cache.py + document Episode Caches workflow`

## Misc

- [x] remove old incontext data loader — no "old" loader found in `src/openpi/training/`
- [x] update aloha with custom incontext data loader — `src/openpi/training/config_aloha.py` routes through `aloha_mobile_incontext_policy` and `Pi0IncontextConfigv12` / `v18`
- [ ] modify the inference code to use `customdataconfig` — no `CustomDataConfig` class found in `src/openpi/`; intent unclear, may need scoping first
