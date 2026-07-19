# Plan: paper-consistent ALOHA dataset (`aloha_contextflow`)

Goal: produce a re-organized ALOHA dataset whose task list matches the ContextFlow paper
exactly — in both training and testing — and update the code to consume it.

The existing release
([`vo2yager/aloha_data_unique`](https://huggingface.co/datasets/vo2yager/aloha_data_unique),
1,487 episodes / 49 tasks) stays as the raw archive. The new dataset is a derived,
metadata-level reorganization of it.

> **Counts below are taken from the dataset statistics notes and the paper appendix.**
> Phase 1 re-derives every number from the actual `meta/episodes.jsonl` before anything is
> built. Treat them as the expected values to assert against, not as ground truth.

---

## 0. Two defects found in the current setup

These are why the reorganized dataset cannot simply reproduce the existing checkpoints.

### 0.1 `kiwi/<right>` leaked into training

`src/openpi/training/config_aloha.py:46` contains:

```python
"pick_up_the_kiwi_and_place_it_in_the_basket_with_right_hand add to test tasks"
```

Exclusion is exact string membership (`src/openpi/training/config.py:351`:
`if not any(task in exclude_task_language for task in tasks)`), so the trailing
` add to test tasks` prevents this entry from ever matching. `kiwi/<right>` (2 episodes)
was therefore **trained on**, while the paper reports it as an *unseen* configuration.

### 0.2 Actual training set was 1,400 episodes, not 1,318

Of the 16 entries in `ALOHA_DATA_UNIQUE_TEST_TASK`, 15 match and exclude 87 episodes:

| Excluded | Episodes |
| --- | ---: |
| `pen_uncap_red_right_b5` | 22 |
| `pen_uncap_blue_left_b5` | 25 |
| `put_red_egg_close_box` | 3 |
| `separate_cups_big_right` | 25 |
| 11 pick-and-place single-demo tasks | 12 |
| **Total** | **87** |

1,487 − 87 = **1,400** episodes actually trained on. The paper reports **1,318**. The
82-episode gap is exactly:

- separate cups `big_left` + `small_left` + `small_right` = 75 (only `big_right` was held out)
- `bottle/<right>` 3 + `onion/<right>` 2 + `kiwi/<right>` 2 = 7

So the separate-cups suite — absent from the paper in both training and testing — supplied
75 training episodes, and 7 further pick-and-place episodes were trained on without being
counted.

**Consequence:** the reorganized dataset changes the training distribution. Norm stats
change, retraining is required, and reported numbers may shift. This is a re-run, not a
relabeling.

---

## 1. Target specification

**31 task configurations / 1,349 episodes** = 25 seen (1,318) + 6 unseen (31).

### 1.1 Merges (many collected tasks -> one paper configuration)

| Paper configuration | Source tasks | Episodes |
| --- | --- | ---: |
| gray pen (single config) | `pen_uncap_gray_left_b5` (91) + `pen_uncap_gray_left_b9` (52) | 143 |
| red pen (single config) | `pen_uncap_red_left_b9` (39) + `pen_uncap_red_left_b5` (30) | 69 |
| handover | `handover_b9` (54) + `handover_b5` (50) | 104 |

`b5` / `b9` are scene batches of the same configuration, so merging them is
straightforward. **`handover` needs confirmation**: the original statistics describe
`handover_b9` as a *bottle* handover and `handover_b5` as a *generic object* handover, yet
the paper counts them as one 104-episode `handover` configuration.

### 1.2 Training configurations (25 configs, 1,318 episodes)

| Suite | Configs | Episodes |
| --- | ---: | ---: |
| Pick & Place | 14 | 527 |
| Pen Uncap | 6 | 338 |
| Put Egg in Box (white egg) | 1 | 199 |
| Extra bimanual (handover, cup stack, stir, water wipe) | 4 | 254 |
| **Total** | **25** | **1,318** |

### 1.3 Test configurations (6 configs, 31 episodes)

| Paper configuration | Source task | Episodes |
| --- | --- | ---: |
| pear / `<left>` | `pick_up_the_pear_..._with_left_hand` | 1 |
| orange juice / `<left>` | `pick_up_the_orange_juice_..._with_left_hand` | 1 |
| kiwi / `<right>` | `pick_up_the_kiwi_..._with_right_hand` | 2 |
| banana / `<right>` | `pick_up_the_banana_..._with_right_hand` | 2 |
| red pen (paper `<left>`) | `pen_uncap_red_right_b5` | 22 |
| red egg | `put_red_egg_close_box` | 3 |
| **Total** | | **31** |

### 1.4 Drops (15 tasks, 138 episodes)

| Dropped | Episodes | Reason |
| --- | ---: | --- |
| `separate_cups_{big,small}_{left,right}` | 100 | absent from the paper entirely |
| `pen_uncap_blue_left_b5` | 25 | in neither the seen (6) nor unseen (1) pen sets |
| 10 pick-and-place single-demo tasks | 13 | not among the paper's 4 unseen configs |

Dropped single-demo tasks: apple/R (1), blue milk/L (1), bottle/L (1), bottle/R (3),
cucumber/L (1), gluten flour/L (1), gluten flour/R (1), kiwi/L (1), onion/L (1),
onion/R (2).

Arithmetic check: 1,487 − 138 = **1,349** = 1,318 + 31.

---

## 1.5 Disposition of all 49 released tasks

Every task in `aloha_data_unique` is accounted for as **train**, **test**, or **drop**.
Nothing is left unclassified — a task absent from the paper in *both* training and testing
is dropped.

### Pen uncap (10 tasks, 385 episodes)

| Source task | Ep | Disposition | Paper configuration (corrected convention) |
| --- | ---: | --- | --- |
| `pen_uncap_gray_left_b5` | 91 | train | gray pen / `<right>` *(merged)* |
| `pen_uncap_gray_left_b9` | 52 | train | gray pen / `<right>` *(merged)* |
| `pen_uncap_gray_right_b5` | 51 | train | gray pen / `<left>` |
| `pen_uncap_red_left_b9` | 39 | train | red pen / `<right>` *(merged)* |
| `pen_uncap_red_left_b5` | 30 | train | red pen / `<right>` *(merged)* |
| `pen_uncap_blue_right_b5` | 25 | train | blue pen / `<left>` |
| `pen_uncap_blue2_left_b5` | 25 | train | blue pen v2 / `<right>` |
| `pen_uncap_blue2_right_b5` | 25 | train | blue pen v2 / `<left>` |
| `pen_uncap_red_right_b5` | 22 | **test** | red pen / `<left>` |
| `pen_uncap_blue_left_b5` | 25 | **drop** | — (in neither seen nor unseen) |

### Pick and place (28 tasks, 546 episodes)

| Source task | Ep | Disposition |
| --- | ---: | --- |
| apple/L 25, corn/L 25, gray milk/L 25, carrot/L 25, chips/L 25 | 125 | train (5 configs) |
| pear/R 50, orange juice/R 25, gray milk/R 26, cucumber/R 50, corn/R 50, red apple/R 25, chips/R 25, blue milk/R 76, carrot/R 75 | 402 | train (9 configs) |
| pear/L 1, orange juice/L 1, kiwi/R 2, banana/R 2 | 6 | **test** (4 configs) |
| apple/R 1, blue milk/L 1, bottle/L 1, bottle/R 3, cucumber/L 1, gluten flour/L 1, gluten flour/R 1, kiwi/L 1, onion/L 1, onion/R 2 | 13 | **drop** (10 configs) |

### Remaining suites (11 tasks, 556 episodes)

| Source task | Ep | Disposition | Paper configuration |
| --- | ---: | --- | --- |
| `put_white_egg_close_box` | 199 | train | put egg in box (white) |
| `put_red_egg_close_box` | 3 | **test** | put egg in box (red) |
| `handover_b9` | 54 | train | handover *(merged)* |
| `handover_b5` | 50 | train | handover *(merged)* |
| `cup_stack` | 50 | train | cup stack |
| `stir` | 50 | train | stir |
| `water_wipe` | 50 | train | water wipe |
| `separate_cups_big_left` | 25 | **drop** | — (absent from paper) |
| `separate_cups_big_right` | 25 | **drop** | — (absent from paper) |
| `separate_cups_small_left` | 25 | **drop** | — (absent from paper) |
| `separate_cups_small_right` | 25 | **drop** | — (absent from paper) |

### Ledger

| | Source tasks | Output configs | Episodes |
| --- | ---: | ---: | ---: |
| train | 28 | 25 *(3 merges)* | 1,318 |
| test | 6 | 6 | 31 |
| drop | 15 | — | 138 |
| **Total** | **49** | **31** | **1,487** |

---

## 2. Decisions

| # | Decision | Resolution |
| --- | --- | --- |
| D1 | Hand convention for new task names | **RESOLVED — corrected paper convention** (picking hand), per `ALOHA_DATASET_NAMING.md`. The merged gray batch is `gray pen / <right>`, printed in Table S2 as `gray pen / <left>`. |
| D2 | Merge `handover_b9` (bottle) with `handover_b5` (generic object)? | **RESOLVED — merge**, per the paper's single 104-episode `handover` row. |
| D3 | Fix the `kiwi/<right>` leak? | **Yes.** Makes the unseen split honest; the kiwi number will change. |
| D4 | Natural-language task strings? | **RESOLVED — yes**, applied uniformly (the current release mixes sentences and folder names). |
| D5 | Publish as a new HF repo? | **Yes**, e.g. `vo2yager/aloha_contextflow`. Leave `aloha_data_unique` untouched as the raw archive. |

### Instruction templates (D1 + D4)

From the paper, with `<left>`/`<right>` naming the **picking** hand:

- Pick and place — `Pick up the {object} and place it in the basket with the {left|right} hand.`
- Pen uncap — `Pick up the {pen} with the {left|right} hand, grasp the cap with the other hand and uncap it.`
- Put egg in box — `Pick up the {object} with the right hand, place it in the box, and close the box.`

The paper gives no template for the four extra bimanual configurations (handover, cup
stack, stir, water wipe). Their instructions must be **authored** in Phase 1 and flagged as
such, since they are not quotable from the paper.

---

## 3. Execution phases

### Phase 1 — Manifest and verification (no data written)
1. Pull `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/info.json` from `aloha_data_unique`.
2. Compute the true per-task episode counts and average lengths.
3. Diff against Section 1. **Every count must reconcile before proceeding**; investigate any mismatch rather than adjusting the target.
4. Emit `metadata/aloha_contextflow/manifest.json`: for each of the 31 output configs, its paper label, natural-language instruction, source task name(s), and source episode indices.

Deliverable: a reviewed manifest. This is the single source of truth for later phases.

### Phase 2 — Build the dataset (metadata-level)
LeRobot layout means this is a **metadata operation, not a re-encode**: episode parquet
files are copied or hardlinked unchanged.

1. Select the 1,349 source episodes from the manifest.
2. Renumber episodes 0..1348; place into `data/chunk-000` (1000) and `chunk-001` (349).
3. Write `meta/tasks.jsonl` with the 31 merged configs and remap every `task_index`.
4. Write `meta/episodes.jsonl` (new episode index, remapped task index, preserved length).
5. Write `meta/info.json` (`total_episodes`, `total_frames`, `total_tasks`, unchanged fps / features / camera specs).
6. Recompute `meta/stats.json` over the new subset.
7. Emit `train_episodes.json` / `test_episodes.json` index lists for the 25/6 split.

Cost: a copy of ~215 GB (the 1,349 kept episodes), no video re-encoding.

### Phase 3 — Verify the built dataset
1. Assert totals: 1,349 episodes / 31 tasks / 25 train + 6 test configs.
2. Assert per-config counts match the manifest.
3. Assert every dropped task is absent and no test episode appears in the train split.
4. Spot-check a merged config (e.g. gray pen) — confirm both source batches present and frame counts preserved.
5. Checksum a sample of copied episodes against the source to prove the copy is faithful.

### Phase 4 — Code revision
1. New config entries with `repo_id="vo2yager/aloha_contextflow"`.
2. Replace `ALOHA_DATA_UNIQUE_TEST_TASK` with the 6-entry paper test list in the new naming; delete the malformed entry.
3. **Add validation to `get_kept_episode_indices`**: raise if any string in `remove_task_list` matches no task in `episodes.jsonl`. This would have caught 0.1 and is the highest-value code change here.
4. Regenerate `metadata/aloha_pen_uncap/*.json` (`task_to_episode`, `episode_to_indexes`, state/action caches) against the new dataset — all are keyed by task name and episode index, both of which change.
5. Recompute norm stats: `uv run scripts/compute_norm_stats.py --config-name <new_config>`.
6. Update `ALOHA_DATASET_NAMING.md` to cover the new dataset alongside the raw release.

### Phase 5 — Retrain and re-evaluate
1. Retrain ContextFlow (and the baselines used in the real-robot table) on the new split.
2. Re-run the real-robot evaluations, including the now-genuinely-unseen `kiwi/<right>`.
3. Record which paper numbers move. Expect movement: 82 episodes leave the training set and the kiwi contamination is removed.

---

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| Published checkpoints are invalidated (norm stats + training distribution both change) | Keep `aloha_data_unique` and existing configs intact so old results stay reproducible; the new dataset is additive |
| Reported numbers shift after retraining | Expected, not a defect — the current numbers reflect a training set the paper does not describe |
| Merged `handover` may combine two distinct tasks | D2 — confirm before building |
| Metadata regeneration desyncs from the new episode indices | Phase 3 assertions run before any training |
| ~215 GB copy | Hardlink where source and destination share a filesystem |
