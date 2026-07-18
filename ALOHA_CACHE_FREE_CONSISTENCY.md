# Aloha cache-free port: consistency verification vs `aloha-dev`

Branch `feat/aloha-cache-free` (cut from `feat/rename-configs-contextflow` @ `e63059d`)
ports the four aloha in-context configs — `ContextFlow_Aloha(_Inference)` and
`ContextAR_Aloha(_Inference)` — from the legacy JSON state/action cache pipeline
(`InjectDemoIndexes` + `AddImagePromptTransform` + `AddStatesActionsPromptTransform`
reading `metadata/aloha_data_unique/episode_{states,actions}_cache.json`, ~510 MB) to the
cache-free `CustomLeRobotDataset` path (`use_custom_dataloader=True`; serving via
`InjectDemoFromCustomDataset` + `provides_incontext_demos=True`). The only metadata the
new path needs is `metadata/aloha_data_unique/task_to_episode.json` (14 KB, now in-repo).

All checks ran on kw61077 (the aloha training box: `vo2yager/aloha_data_unique` local,
legacy caches present, all venvs jax 0.5.0 / torch 2.6.0 / PIL 11.0.0, RTX A6000).
Three code states were compared:

- **aloha-dev**: `~/code/openpi` @ `origin/aloha-dev` (reference; legacy loader, old config names)
- **legacy**: worktree @ `e63059d` (rename-branch tip; legacy cache loader)
- **cache-free**: worktree @ this branch

Artifacts: `kw61077:~/aloha_consistency/20260719a/` (harness JSONs, per-leaf hashes,
action tensors). Harness: in-tree `scripts/consistency_check.py` semantics, extended with
`--no-random-select` (the copy used is `scripts/consistency_check_x.py` in each worktree;
`aloha-dev` has no `seed_base` plumbing, so cross-branch selection determinism uses
`--no-random-select`, which both `InjectDemoIndexes` and `CustomLeRobotDataset` implement
as "first candidate episode").

## Training batch (seed 12345, batch_size 1, num_workers 0, `--deterministic-data`)

SHA-256 over every tensor (values + shapes + dtypes) of the first batch:

| Check | aloha-dev | legacy @ e63059d | cache-free |
|---|---|---|---|
| ContextFlow, seeded `random_select=True` | n/a (no seed_base on aloha-dev) | `69e7dcf8…` | `69e7dcf8…` ✅ |
| ContextFlow, `--no-random-select` | `7775e190…` | `7775e190…` | `7775e190…` ✅ |
| ContextAR, `--no-random-select` | `f6c78ad9…` | `f6c78ad9…` | `f6c78ad9…` ✅ |

The seeded ContextFlow match exercises `demo_selection_seed_compat`: with `seed_base`
set, `CustomLeRobotDataset` reproduces `InjectDemoIndexes`' exact seeded draw.

## One-step train (ContextFlow, same batch, pi0_base weights)

| Metric | aloha-dev | legacy @ e63059d | cache-free |
|---|---|---|---|
| loss | 4.00637149810791 | 4.001909255981445 | **4.001909255981445** ✅ |
| grad_norm | 118.43104553222656 | 118.4654312133789 | **118.4654312133789** ✅ |
| param_norm | 947.1429443359375 | 947.1429443359375 | 947.1429443359375 ✅ |

cache-free == legacy **bitwise**. The ~0.1% loss delta vs aloha-dev is a pre-existing
cross-branch model-code float difference (`pi0_incontextv18` vs the renamed `contextflow`
module): identical batch, identical initial `param_norm`, and each side reproduces its own
values exactly under `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, so it is stable
numerics, not data or randomness.

## Inference (shared checkpoints, fixed synthetic observation, `split="test"`, task 5, float32 matmul + deterministic XLA ops)

Fixture hash `79c51f17…` identical on all sides. Direct `create_trained_policy_incontext`
call, no server. Checkpoints: the kw61077 trained runs (`…sample_frames8_no_test/19999`
for ContextFlow, `…train_split_v1/29999` for ContextAR).

| Check | aloha-dev | legacy @ e63059d | cache-free |
|---|---|---|---|
| ContextFlow actions [50,16] | `7ddc3bd7…` (norm 16.905480) | `6a4c72bf…` (norm 16.910723) | `6a4c72bf…` ✅ bitwise == legacy |
| ContextAR actions [10,16] | `0660745f…` (norm 7.818520) | — | `0660745f…` ✅ **bitwise == aloha-dev** |

ContextFlow cache-free serving is bitwise-identical to the legacy cache-based serving on
the same branch; the ≤2.3e-4/element residual vs aloha-dev is the same pre-existing
cross-branch numeric difference as in training (aloha-dev is self-deterministic — its
det-ops rerun reproduces `7ddc3bd7…` exactly). ContextAR matches aloha-dev bitwise
(discrete FAST token decoding absorbs float-level noise).

## The ContextAR demo-normalization quirk (root-caused and reproduced)

First ContextAR batch comparison mismatched (`d3f4ddd8…` vs `f6c78ad9…`), isolated by
per-leaf hashing to `incontext_actions` (+ its FAST tokens) only. Root cause: the shared
legacy cache was built once under the v18/pi0 pipeline, so its rows are normalized with
the **pi0_base** `trossen_mobile` stats; legacy ContextAR consumed them as-is even though
its own normalization uses **pi0_fast_base** stats (action mean/std differ; state stats
are bitwise identical between the two asset sets — which is why demo states matched).
Verified numerically: `denorm(cache, pi0_base) → renorm(pi0_fast_base)` reproduces the
divergence digit-for-digit.

To stay faithful to what the ECCV ContextAR checkpoints were trained on, the config now
declares `demo_norm_stats_assets` (pi0_base/trossen_mobile); those stats are injected
into `norm_stats` as `demo_ref_state` / `demo_ref_actions` and the demo aliases point at
them, in training and serving. `ContextFlow_Aloha` is unaffected (its own stats are the
cache-builder stats).

## Verdict

Dropping the JSON state/action caches is behavior-preserving: training batches, a full
optimizer step, and checkpoint inference are **bitwise identical** to the legacy cache
pipeline on the rename branch, and identical to `aloha-dev` up to the pre-existing,
quantified cross-branch float noise (ContextAR inference matches aloha-dev exactly).
