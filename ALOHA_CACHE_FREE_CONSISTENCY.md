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
`--no-random-select` (`aloha-dev` has no `seed_base` plumbing, so cross-branch selection
determinism uses `--no-random-select`, which both `InjectDemoIndexes` and
`CustomLeRobotDataset` implement as "first candidate episode").

## Training batch (seed 12345, batch_size 1, num_workers 0, `--deterministic-data`)

SHA-256 over every tensor (values + shapes + dtypes) of the first batch:

| Check | aloha-dev | legacy @ e63059d | cache-free |
|---|---|---|---|
| ContextFlow, seeded `random_select=True` | n/a (no seed_base on aloha-dev) | `69e7dcf8…` | `69e7dcf8…` ✅ |
| ContextFlow, `--no-random-select` | `7775e190…` | `7775e190…` | `7775e190…` ✅ |
| ContextAR, `--no-random-select` (†) | `f6c78ad9…` | `f6c78ad9…` | `f6c78ad9…` ✅ |

The seeded ContextFlow match exercises `demo_selection_seed_compat`: with `seed_base`
set, `CustomLeRobotDataset` reproduces `InjectDemoIndexes`' exact seeded draw.

(†) This ContextAR row was produced by an earlier variant of the port that **pinned demo
normalization to the pi0_base stats** to byte-reproduce the legacy cache. That pinning has
since been intentionally removed — see **ContextAR demo normalization** below — so current
ContextAR cache-free batches deliberately differ from aloha-dev on the demo-action tensor.
The row is retained as evidence that the legacy pipeline was reproduced exactly before the
deliberate correction.

## One-step train (ContextFlow, same batch, pi0_base weights)

| Metric | aloha-dev | legacy @ e63059d | cache-free |
|---|---|---|---|
| loss | 4.00637149810791 | 4.001909255981445 | **4.001909255981445** ✅ |
| grad_norm | 118.43104553222656 | 118.4654312133789 | **118.4654312133789** ✅ |
| param_norm | 947.1429443359375 | 947.1429443359375 | 947.1429443359375 ✅ |

cache-free == legacy **bitwise**. The ~0.1% loss delta vs aloha-dev is a pre-existing
cross-branch model-code difference, **not** data or randomness (identical batch, identical
initial `param_norm`, each side self-deterministic under
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`).

**Root cause of the aloha-dev vs rename-branch delta (found by full model-module diff):**
the rename branch wraps every SigLIP image-encode in a `jax.lax.cond` short-circuit
(`contextflow.py:406-417` `_encode_image_tokens` = `lax.cond(any(mask), encode, zeros)`,
call sites `:428/:465/:476`), which aloha-dev's `pi0_incontextv18.py` does not have
(it always calls `PaliGemma.img(...)` inline at `:412/:448/:459`). With `dtype="bfloat16"`,
wrapping the encode in `lax.cond` changes XLA's op-fusion / reduction ordering, shifting the
bf16 low-order bits deterministically → the ~1e-3 relative loss delta. Everything else in the
two modules is byte-identical on the numeric path: matmul precision (both
`jax.lax.Precision.HIGHEST`, attention `preferred_element_type=float32`), compute dtype
(both bf16), SigLIP/LoRA (0-line diff), and the loss / noise / time-sampling / attention tail
(empty diff). Matmul-precision and dtype were explicitly ruled out. This delta is a property
of the rename branch itself (present before the cache drop); the cache drop contributes
exactly zero (cache-free == legacy bitwise).

## Inference (shared checkpoints, fixed synthetic observation, `split="test"`, task 5, float32 matmul + deterministic XLA ops)

Fixture hash `79c51f17…` identical on all sides. Direct `create_trained_policy_incontext`
call, no server.

| Check | aloha-dev | legacy @ e63059d | cache-free |
|---|---|---|---|
| ContextFlow actions [50,16] | `7ddc3bd7…` (norm 16.905480) | `6a4c72bf…` (norm 16.910723) | `6a4c72bf…` ✅ bitwise == legacy |
| ContextAR actions [10,16] (†) | `0660745f…` (norm 7.818520) | — | `0660745f…` (pi0_base-pinned variant) |

ContextFlow cache-free serving is bitwise-identical to legacy cache-based serving on the
same branch; the ≤2.3e-4/element residual vs aloha-dev is the same rename-branch numeric
difference as in training. The (†) ContextAR inference match to aloha-dev was, again, the
pi0_base-pinned variant; the current port uses pi0_fast_base demo stats and so differs by
design (see below).

## ContextAR demo normalization: legacy quirk, and the corrected port

**The legacy inconsistency.** In-context demos contribute demo states/actions to the model.
The legacy shared cache (`episode_{states,actions}_cache.json`) was built once under the
v18/pi0 pipeline, so its rows are normalized with the **pi0_base** `trossen_mobile` stats.
`ContextFlow` (pi0 diffusion) uses pi0_base stats for its current frame too, so for it the
cache is native and consistent. `ContextAR` (pi0_fast) uses **pi0_fast_base** stats for its
current frame, whose **action** mean/std differ from pi0_base (std ~25–30% larger on the 12
arm-joint dims; state stats are bitwise identical between the two asset bundles). Because the
legacy loader consumed the cached demo values as-is, ContextAR's *demo* actions were
normalized in pi0_base space while its *current-frame* actions were normalized in
pi0_fast_base space — an internal normalization mismatch baked into the legacy pipeline (and
into any ContextAR checkpoints trained on it).

**Decision: the cache-free port uses the correct pi0_fast stats for ContextAR demos.**
Rather than reproduce the legacy mismatch, the port normalizes ContextAR demo states/actions
with the config's **own pi0_fast_base** `trossen_mobile` stats — identical to how the current
frame is normalized. This is the default factory behavior (`norm_stats_aliases`
`dem_prompt_all_* → state/actions`); no per-config override, no injected `demo_ref_*` keys.
Consequence: ContextAR cache-free batches/inference **deliberately differ** from the legacy
cache / aloha-dev, by exactly the demo-action normalization (the ~25–30% arm-joint scale
difference above). `ContextFlow` is unaffected — its own stats already are the cache-builder
stats, so it stays consistent with the legacy pipeline.

⚠️ **Existing ContextAR checkpoints:** the released ContextAR checkpoint
(`pi0_fast_aloha_data_unique_incontext_train_split_v1`) was trained with the legacy
pi0_base-normalized demos. Serving it under this corrected config feeds pi0_fast-normalized
demos (a train/inference distribution shift), so ContextAR should be **retrained** from the
base checkpoint under the corrected normalization for a self-consistent pipeline. New runs
are correct by construction.

## Verdict

Dropping the JSON state/action caches is behavior-preserving for the loading mechanics:
`ContextFlow` training batches, a full optimizer step, and checkpoint inference are **bitwise
identical** to the legacy cache pipeline on the rename branch, and identical to `aloha-dev`
up to the pre-existing, root-caused cross-branch `lax.cond` float difference. For
`ContextAR`, the port additionally **corrects** the legacy demo-normalization mismatch: demos
now use the same pi0_fast_base stats as the current frame, a deliberate, documented divergence
from aloha-dev that calls for retraining the ContextAR checkpoint.
