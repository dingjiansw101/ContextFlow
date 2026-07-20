# Plan: paper-consistent ALOHA dataset (`aloha_incontext`)

Goal: produce a re-organized ALOHA dataset whose task list matches the ContextFlow paper
exactly — in both training and testing — and update the code to consume it.

The existing release
([`vo2yager/aloha_data_unique`](https://huggingface.co/datasets/vo2yager/aloha_data_unique),
1,487 episodes / 49 tasks) stays as the raw archive, untouched. The new dataset is a
derived regrouping of it, published as a separate HF repo: no episode is invented and no
image is re-encoded, but the parquet index columns are rewritten (see Phase 2).

> **All counts below are VERIFIED against the actual `meta/episodes.jsonl`** of
> `aloha_data_unique` (1,487 episodes / 49 tasks / 661,200 frames, matching `meta/info.json`).
> Every task in the dataset is classified exactly once, no classified name is missing from
> the dataset, and train + test + drop = 1,487. See Section 5.

### Scope of the code change

**Minimal: correct the task names and the paths that point at the dataset. Nothing else.**
No new modules, no new config entries, no validation or refactoring. Existing aloha configs
are re-pointed in place at the new dataset; the previous behaviour stays reachable through
git history rather than through parallel configs. See Phase 4.

### After the cache-free merge (`9658ef7`)

The merge dropped the ~510 MB JSON state/action caches for the aloha in-context path —
demos now come straight from `CustomLeRobotDataset`, so **there is no cache to build for
the new dataset**.

One small derived file survives: `metadata/aloha_data_unique/task_to_episode.json`
(14 KB, tracked in-repo). `CustomLeRobotDataset` still requires it
(`src/openpi/training/custom_dataset.py:143`), and it maps `task_index` -> episode indices —
both of which are renumbered by the reorganization. It must therefore be regenerated, but
it is a direct derivation from `meta/episodes.jsonl`, not a cache build.

---

## 0. Two defects found in the current setup

These are why the reorganized dataset cannot simply reproduce the existing checkpoints.

### 0.1 `kiwi/<right>` leaked into training — FIXED

The test-task entry in `config_aloha.py` used to read:

```python
"pick_up_the_kiwi_and_place_it_in_the_basket_with_right_hand add to test tasks"
```

Exclusion is exact string membership (`src/openpi/training/config.py:351`:
`if not any(task in exclude_task_language for task in tasks)`), so the trailing
` add to test tasks` prevented this entry from ever matching. `kiwi/<right>` (2 episodes)
was therefore **trained on**, while the paper reports it as an *unseen* configuration.

**Corrected** — the trailing text has been removed, so the entry now matches. Note this is
a forward fix only: the published checkpoints and the reported `kiwi/<right>` number
predate it and were produced with the leak in place.

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

With the 0.1 fix in place, the current code on the current dataset would train on 1,398
episodes (`kiwi/<right>`'s 2 now excluded) — still not 1,318, because separate cups and the
uncounted `bottle/<right>` / `onion/<right>` episodes remain. Only the reorganized dataset
closes the gap.

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
| D4 | Natural-language task strings? | **RESOLVED — yes for the three suites the paper gives templates for** (pick and place, pen uncap, put egg in box). The four extra bimanual configs keep their short folder-style names. |
| D5 | Publish as a new HF repo? | **Yes**, e.g. `vo2yager/aloha_incontext`. Leave `aloha_data_unique` untouched as the raw archive. |
| D6 | Scope of the code change | **RESOLVED — minimal.** Correct the task names and the dataset paths only; re-point the existing configs in place. No new modules, no parallel configs, no validation or refactoring. See Phase 4b. |
| D7 | Demo caches for the new dataset | **RESOLVED — none needed.** The cache-free merge (`9658ef7`) removed the JSON state/action caches. Only the 14 KB `task_to_episode.json` is regenerated. |

### Instruction templates (D1 + D4)

From the paper, with `<left>`/`<right>` naming the **picking** hand:

- Pick and place — `Pick up the {object} and place it in the basket with the {left|right} hand.`
- Pen uncap — `Pick up the {pen} with the {left|right} hand, grasp the cap with the other hand and uncap it.`
- Put egg in box — `Pick up the {object} with the right hand, place it in the box, and close the box.`

The four extra bimanual configurations keep their existing short names — `handover`,
`cup_stack`, `stir`, `water_wipe`. The paper gives no instruction template for them, and
inventing one would put text in the dataset that is not quotable from the paper, so no
natural-language string is generated for these.

---

## 3. Execution phases

### Phase 1 — Manifest and verification (no data written)
1. Pull `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/info.json` from `aloha_data_unique`.
2. Compute the true per-task episode counts and average lengths.
3. Diff against Section 1. **Every count must reconcile before proceeding**; investigate any mismatch rather than adjusting the target.
4. Emit `metadata/aloha_incontext/manifest.json`: for each of the 31 output configs, its paper label, natural-language instruction, source task name(s), and source episode indices.

Deliverable: a reviewed manifest. This is the single source of truth for later phases.

### Phase 2 — Build the dataset

> **CORRECTION.** An earlier draft of this plan called the build "metadata-only" and assumed
> episode parquet files could be copied or hardlinked unchanged. **That is wrong.** Verified
> by reading `data/chunk-000/episode_000005.parquet` on kw61077: every parquet carries three
> index columns *inside* it —
>
> ```
> cols: [observation.state, action, observation.velocity, observation.effort,
>        observation.images.cam_{high,left_wrist,right_wrist},
>        timestamp, frame_index, episode_index, index, task_index]
> episode_index[:3] = [5, 5, 5]   task_index[:3] = [0, 0, 0]   index[:3] = [1000, 1001, 1002]
> ```
>
> `index` is a **global row counter** across the whole dataset, so dropping any earlier
> episode shifts it for every later one. Renumbering episodes and remapping task indices
> therefore requires **rewriting all three columns in all 1,349 kept parquets**.

The three image columns pass through as opaque binary, so **no image is re-encoded** — the
rewrite touches three `int64` columns and is I/O-bound, not CPU-bound. But it is a genuine
215 GB read + write, not a hardlink.

1. Select the 1,349 source episodes from the manifest.
2. Renumber episodes 0..1348; place into `data/chunk-000` (1000) and `chunk-001` (349).
3. For each kept episode, rewrite `episode_index` (new number), `task_index` (merged/remapped), and `index` (recomputed running offset). All other columns copied through untouched.
4. Write `meta/tasks.jsonl` with the 31 merged configs.
5. Write `meta/episodes.jsonl` (new episode index, remapped task index, preserved length).
6. Write `meta/info.json` (`total_episodes` 1349, `total_frames` 595900, `total_tasks` 31, `splits: {"train": "0:1349"}`; fps / features / camera specs unchanged).
7. Recompute `meta/stats.json` over the new subset.
8. Derive `metadata/aloha_incontext/task_to_episode.json` from the new `meta/episodes.jsonl` (Phase 4b edit 5).

No train/test index lists are emitted: the split is expressed only as the 6 names in
`remove_task_list`, so an episode-index list would be an unused second source of truth.

#### Rejected alternative: keep the original numbering

Dropping episodes *without* renumbering would leave the parquets byte-identical and make the
build nearly free. Rejected because (a) the three merges still require a `task_index`
rewrite on 316 episodes (gray 143 + red 69 + handover 104, ~50 GB), and (b) it leaves gaps in
`episode_index` and a non-contiguous global `index`, which `splits: {"train": "0:N"}` and
LeRobot's episode-boundary computation assume are dense. Worth revisiting only if the full
rewrite proves impractical, and only after checking LeRobot's tolerance for gaps directly.

### Phase 2b — Publishing to Hugging Face

Where the data is (verified):

| Box | `aloha_data_unique` | Free space |
| --- | --- | --- |
| visioncair (local) | `meta/` only, 160 KB — **no parquet** | 920 GB of 14 TB (94% used) |
| kw61077 | full 220 GB (`data/` + `meta/`) | 297 GB of 14 TB (**98% used**) |

kw61077 has the data but almost no headroom, and per prior findings its `/home` also holds
the **only** copy of the ALOHA checkpoints. Writing a 215 GB output tree there would leave
~82 GB — too close to the edge to risk.

**Chosen approach: stream on kw61077, one episode at a time, never materializing the full
output tree.**

1. Authenticate: `huggingface_hub` 0.28.1 is present and `~/.cache/huggingface/token` exists. Confirm the token has **write** scope on the `vo2yager` org before starting.
2. `HfApi().create_repo("vo2yager/aloha_incontext", repo_type="dataset", private=True)` — keep it private until Phase 3 verification passes. Name checked and free: `dataset_info` 404s for `vo2yager/aloha_incontext` while resolving `vo2yager/aloha_data_unique`, so the token has org read access and the 404 is a genuine absence rather than a permissions artifact.
3. Upload `meta/` first (small), so the repo is inspectable early.
4. For each kept episode, in new-index order: read the source parquet, rewrite the three index columns, write to a local temp file, add to a pending commit batch, delete the temp.
5. Commit in **batches of ~50 episodes** via `HfApi.create_commit` with `CommitOperationAdd`. One commit per episode would make 1,349 commits; a single commit of 1,349 files risks a timeout.
6. Record the last successfully committed episode index to a resume file, so an interrupted upload restarts at the batch boundary rather than from zero.

Peak local disk: a few GB. Cost is network — ~215 GB uploaded — with no 220 GB download and
no large intermediate tree on the box that holds the checkpoints.

7. After Phase 3 passes, flip the repo public and add a dataset card stating: derived from `vo2yager/aloha_data_unique`, the three merges, the 15 dropped tasks, and the corrected picking-hand convention (linking `ALOHA_DATASET_NAMING.md`).

`vo2yager/aloha_data_unique` is **never modified** — the new dataset is a separate repo.

### Phase 3 — Verify the built dataset
1. Assert totals: 1,349 episodes / 31 tasks / 25 train + 6 test configs.
2. Assert per-config counts match the manifest.
3. Assert every dropped task is absent and no test episode appears in the train split.
4. Spot-check a merged config (e.g. gray pen) — confirm both source batches present and frame counts preserved.
5. Prove the rewrite is faithful on a sample of episodes: every column **except** `episode_index`, `task_index`, and `index` must be bitwise equal to the source, and the image columns must be byte-identical (confirming no re-encode).
6. Assert the three rewritten columns are internally consistent: `episode_index` constant within a file and matching the filename, `task_index` matching `meta/episodes.jsonl`, and `index` densely increasing 0..595,899 across the dataset in episode order.

### Phase 4a — The 31 output task names

These strings are written into the new `meta/tasks.jsonl` in Phase 2. The six test names
are the ones that must appear verbatim in `ALOHA_DATA_UNIQUE_TEST_TASK` (Phase 4b, edit 1).

*Pick and place (18).* Hand convention unchanged — paper and dataset agree here.
`Pick up the {object} and place it in the basket with the {left|right} hand.`
14 train (apple/L, corn/L, gray milk/L, carrot/L, chips/L, pear/R, orange juice/R,
gray milk/R, cucumber/R, corn/R, red apple/R, chips/R, blue milk/R, carrot/R) and
4 test (pear/L, orange juice/L, kiwi/R, banana/R).

*Pen uncap (7).* Hand is **flipped** relative to the source folder names, since output uses
the corrected picking-hand convention.
`Pick up the {pen} with the {left|right} hand, grasp the cap with the other hand and uncap it.`

| Output configuration | Source task(s) | Ep | Split |
| --- | --- | ---: | --- |
| gray pen / `<right>` | `pen_uncap_gray_left_b5` + `_b9` | 143 | train |
| gray pen / `<left>` | `pen_uncap_gray_right_b5` | 51 | train |
| red pen / `<right>` | `pen_uncap_red_left_b9` + `_b5` | 69 | train |
| blue pen / `<left>` | `pen_uncap_blue_right_b5` | 25 | train |
| blue pen v2 / `<right>` | `pen_uncap_blue2_left_b5` | 25 | train |
| blue pen v2 / `<left>` | `pen_uncap_blue2_right_b5` | 25 | train |
| red pen / `<left>` | `pen_uncap_red_right_b5` | 22 | **test** |

*Put egg in box (2).*
`Pick up the {white|red} egg with the right hand, place it in the box, and close the box.`

*Extra bimanual (4).* `handover`, `cup_stack`, `stir`, `water_wipe` — folder-style, no
natural-language string (D4).

**Open naming question:** how "blue pen v2" should read inside an instruction sentence
(`the blue pen v2` vs `the second blue pen`). Needs a call before Phase 2. It affects two
*training* names only — none of the six test names — so it does not block Phase 4b.

#### The six test names, verbatim

This is the exact replacement content for `ALOHA_DATA_UNIQUE_TEST_TASK` (Phase 4b, edit 1).
Each must match its `meta/tasks.jsonl` string byte-for-byte, since exclusion is exact string
membership (`config.py:351`).

```python
ALOHA_DATA_UNIQUE_TEST_TASK = [
    "Pick up the pear and place it in the basket with the left hand.",
    "Pick up the orange juice and place it in the basket with the left hand.",
    "Pick up the kiwi and place it in the basket with the right hand.",
    "Pick up the banana and place it in the basket with the right hand.",
    # Source task is `pen_uncap_red_right_b5`: <left> here is the PICKING hand
    # (paper convention), which is the opposite of the source folder's name.
    "Pick up the red pen with the left hand, grasp the cap with the other hand and uncap it.",
    "Pick up the red egg with the right hand, place it in the box, and close the box.",
]
```

The 16 current entries collapse to 6 because the 10 dropped pick-and-place single-demo
tasks no longer exist in the dataset at all — they are excluded by omission from
`meta/tasks.jsonl`, not by exclusion at load time. Keeping them listed would be harmless
today but would silently rot, and (absent the rejected validation) nothing would report it.

### Phase 4b — Code edits (task names and dataset paths only)

Five edits, all string swaps, all in `src/openpi/training/config_aloha.py` except the last.
Existing configs are re-pointed in place — no parallel config entries.

| # | Target | Change |
| --- | --- | --- |
| 1 | `config_aloha.py:33` `ALOHA_DATA_UNIQUE_TEST_TASK` | replace the 16 source-name entries with the **6 paper test names** from Phase 4a |
| 2 | `config_aloha.py:1372,1417,1455,1505,1543` | `repo_id` -> `"vo2yager/aloha_incontext"` (5 sites) |
| 3 | `config_aloha.py:1385,1480,1518` | `episode_json_path` -> `.../aloha_incontext/meta/episodes.jsonl` (3 sites) |
| 4 | `config_aloha.py:300` | `task_to_episode_path` -> `"metadata/aloha_incontext/task_to_episode.json"` |
| 5 | `metadata/aloha_incontext/task_to_episode.json` *(new, ~14 KB)* | derive from the new `meta/episodes.jsonl` and commit, mirroring the existing `metadata/aloha_data_unique/` file |

Edits 2 and 3 both target the four configs `ContextFlow_Aloha`, `ContextFlow_Aloha_Inference`,
`ContextAR_Aloha`, `ContextAR_Aloha_Inference` plus the `pi0` aloha baseline.

**Explicitly out of scope** (would be extra modifications):

- no `aloha_paper_tasks.py` or any new module
- no `get_kept_episode_indices` validation change in `config.py` — see "The unguarded defect
  class" below. Deferred to a separate follow-up, not part of this change
- no renaming of the config names themselves (`ContextFlow_Aloha` etc. keep their names)
- no changes to `ALOHA_OBJECT_TEST_TASK`, the `objects_pickup_place` / `object_task_suite`
  configs, or `metadata/aloha_pen_uncap/` — a different dataset family, untouched by this reorg
- no change to `examples/aloha_mobile_real/main_incontext.py`; its `--prompt` / `--task_json`
  defaults point at the separate `objects_pickup_place_right_hand` metadata, and the new task
  string is passed on the command line at eval time

#### The unguarded defect class

`get_kept_episode_indices` (`config.py:282-356`) keeps an episode when none of its task
strings appear in `remove_task_list`:

```python
if not any(task in exclude_task_language for task in tasks):
    kept.append(int(entry["episode_index"]))
```

Membership is exact. **An entry in `remove_task_list` that matches no task in the dataset
does nothing at all** — no error, no warning, no log line. The episodes it was meant to hold
out simply stay in training. That is defect 0.1 exactly: one entry carried a trailing
` add to test tasks`, matched nothing, and `kiwi/<right>` trained for months while being
reported as unseen.

The guard would be about two lines: after loading the task set, raise if any
`remove_task_list` entry is absent from it.

**Why it is deferred, not just "extra work":** the change lives in `config.py`, which is
shared by *every* config that uses `remove_task_list` — LIBERO included, not just aloha. If
any existing config anywhere in the repo carries a stale or misspelled entry, adding a raise
converts a config that loads today into a hard failure at load time. That is a repo-wide
behavioural change hiding inside a two-line diff, and it deserves to be evaluated against all
configs on its own rather than riding along with a task-name migration.

**What deferring costs:** this migration replaces all six test names with natural-language
strings that must match `meta/tasks.jsonl` byte-for-byte — capitalization and trailing period
included. That is precisely the condition under which a silent mismatch is most likely, and
the failure mode is invisible: training proceeds on 1,349 episodes instead of 1,318 and
nothing says so.

**Mitigation in place:** Phase 4c step 2 asserts the kept-episode count is 1,318. It catches
the same failure once, manually, without touching shared code.

**Recommended follow-up:** land the guard as its own commit after this migration, so its
blast radius across all configs can be assessed separately. A warning rather than a raise, or
a check scoped to the aloha configs, are both safer intermediate options.

### Phase 4c — After the edits
1. Recompute norm stats: `uv run scripts/compute_norm_stats.py --config-name ContextFlow_Aloha` (and the other three).
2. Confirm the exclusion actually bites: load each config's data config and assert the kept-episode count is **1,318**, not 1,349. This is the manual stand-in for the validation rejected above — without it, a typo in edit 1 silently trains on 1,349 episodes.
3. Update `ALOHA_DATASET_NAMING.md` to cover the new dataset alongside the raw release.

### Phase 5 — Retrain and re-evaluate
1. Retrain ContextFlow (and the baselines used in the real-robot table) on the new split.
2. Re-run the real-robot evaluations, including the now-genuinely-unseen `kiwi/<right>`.
3. Record which paper numbers move. Expect movement: 82 episodes leave the training set and the kiwi contamination is removed.

---

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| Published checkpoints are invalidated (norm stats + training distribution both change) | `vo2yager/aloha_data_unique` stays published and untouched, so old results remain reproducible by checking out the pre-change commit. Note the configs themselves are re-pointed in place (minimal-change scope), so reproducing old results requires git history, not a config switch |
| Reported numbers shift after retraining | Expected, not a defect — the current numbers reflect a training set the paper does not describe |
| Merged `handover` may combine two distinct tasks | D2 — confirm before building |
| A typo in the 6 test names silently disables exclusion (defect 0.1 recurring) | Validation was deliberately left out of scope; Phase 4c step 2 asserts the kept-episode count is 1,318 instead |
| `task_to_episode.json` desyncs from the new episode indices | Regenerated in Phase 4b edit 5 from the new `episodes.jsonl`; Phase 3 assertions run before any training |
| ~215 GB rewrite, not a hardlinkable copy (parquets carry `episode_index` / `task_index` / `index`) | Stream episode-at-a-time and upload directly to HF (Phase 2b); peak local disk stays a few GB |
| kw61077 `/home` is 98% full and holds the only ALOHA checkpoint copy | Never materialize the output tree there; the streaming build writes only a small rolling temp |
| A 215 GB upload is interrupted | Batch commits of ~50 episodes with a resume file, so a restart resumes at a batch boundary |
| Token lacks write scope on the `vo2yager` org | Verified before any data is rewritten (Phase 2b step 1) |

---

## 5. Verification against the real metadata

Source: `~/.cache/huggingface/lerobot/vo2yager/aloha_data_unique/meta/{episodes,tasks}.jsonl`.

Dataset totals read back: **1,487 episodes / 49 tasks / 661,200 frames**, matching
`meta/info.json`.

Integrity of the classification in Section 1.5:

- every classified task name exists in the dataset (0 missing)
- every dataset task is classified (0 unclassified)
- no task is classified twice (0 duplicates)
- train + test + drop = 1,487 = the full dataset

| Group | Source tasks | Episodes | Frames |
| --- | ---: | ---: | ---: |
| train | 28 | **1,318** | 582,800 |
| test | 6 | **31** | 13,100 |
| drop | 15 | **138** | 65,300 |
| **Total** | **49** | **1,487** | **661,200** |

The training total of 1,318 matches the paper's reported figure exactly.

### Merged configurations, measured

| Paper configuration | Episodes | Avg length | Source tasks |
| --- | ---: | ---: | --- |
| gray pen / `<right>` | 143 | 603.5 | `pen_uncap_gray_left_b5` (91) + `_b9` (52) |
| gray pen / `<left>` | 51 | 500.0 | `pen_uncap_gray_right_b5` |
| red pen / `<right>` | 69 | 497.1 | `pen_uncap_red_left_b9` (39) + `_b5` (30) |
| blue pen / `<left>` | 25 | 500.0 | `pen_uncap_blue_right_b5` |
| blue pen v2 / `<right>` | 25 | 500.0 | `pen_uncap_blue2_left_b5` |
| blue pen v2 / `<left>` | 25 | 500.0 | `pen_uncap_blue2_right_b5` |
| handover | 104 | 641.3 | `handover_b9` (54) + `handover_b5` (50) |
| red pen / `<left>` *(test)* | 22 | 427.3 | `pen_uncap_red_right_b5` |

Per-suite training subtotals also reconcile with the paper appendix:

| Suite | Episodes | Avg length | Paper |
| --- | ---: | ---: | --- |
| Pick & Place | 527 | 235 | 527 / 235 |
| Pen Uncap | 338 | 543 | 338 / 543 |
| Put Egg in Box | 199 | 734.7 | 199 / 735 |
| Extra bimanual | 254 | 508.7 | 254 / 509 |

Every episode count and average length in the paper's Table S2 is reproduced from the
released metadata. The reorganization is therefore a pure regrouping of the existing
release — no episode is invented, and the only quantities that change are which tasks are
merged and which are excluded.
