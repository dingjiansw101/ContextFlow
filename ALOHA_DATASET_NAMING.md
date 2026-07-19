# ALOHA Pen-Task Hand Naming Convention

Two different `<left>` / `<right>` conventions are in use for the pen-uncap tasks — one in
the ContextFlow paper, one in the collected data and the released dataset
([`vo2yager/aloha_data_unique`](https://huggingface.co/datasets/vo2yager/aloha_data_unique)).
This note documents both, gives the mapping between them, and provides a corrected
statistics table.

## The two conventions

| | Definition of `<right>` | Labels the... |
| --- | --- | --- |
| **Paper** (ContextFlow) | "Pick up the pen with the `<right>` hand, grasp the cap with the other hand and uncap it." | **picking** hand |
| **Collected data / released dataset** | Uncap with the `<right>` hand. | **uncapping** hand |

Because one hand picks up the pen while the *other* grasps and removes the cap, the two
conventions are exact mirror images. For every pen configuration:

```
paper <right>  ==  dataset <left>
paper <left>   ==  dataset <right>
```

**Pick-and-place tasks are unaffected** — the paper and the dataset use the same
`<left>` / `<right>` convention there. This note applies only to the pen-uncap suite.

## Note on Table S2 of the paper appendix

In Table S2 of the appendix, **only the `red pen / <right>` row followed the paper's
definition**. The remaining pen rows were named following the collected-data convention.
A corrected table is given below.

Only the hand *labels* change. Every episode count and average length is unchanged, and
the training/held-out split is unchanged, so **no experimental result is affected** — this
is a labeling correction only.

## Corrected pen statistics (paper convention)

| Table S2 as published | Corrected (paper convention) | Released task name | Episodes | Avg length |
| --- | --- | --- | ---: | ---: |
| gray pen / `<left>` | gray pen / **`<right>`** | `pen_uncap_gray_left_b5` + `_b9` | 143 | 604 |
| gray pen / `<right>` | gray pen / **`<left>`** | `pen_uncap_gray_right_b5` | 51 | 500 |
| red pen / `<right>` &nbsp;*(already correct)* | red pen / `<right>` | `pen_uncap_red_left_b9` + `_b5` | 69 | 497 |
| blue pen / `<right>` | blue pen / **`<left>`** | `pen_uncap_blue_right_b5` | 25 | 500 |
| blue pen v2 / `<left>` | blue pen v2 / **`<right>`** | `pen_uncap_blue2_left_b5` | 25 | 500 |
| blue pen v2 / `<right>` | blue pen v2 / **`<left>`** | `pen_uncap_blue2_right_b5` | 25 | 500 |
| **Subtotal** | | | **338** | **543** |

Corresponding corrections to Table S1 (seen / unseen pen configurations):

- Seen: `<left & right>`: gray pen, blue pen v2 — **unchanged** (both hands appear either way)
- Seen: `<right>`: red pen — **unchanged**
- Seen: `<right>`: blue pen → **`<left>`: blue pen**
- Unseen: `<left>`: red pen — **unchanged** (red already followed the paper convention)

The unseen pen configuration, `red pen / <left>` in paper convention, is
`pen_uncap_red_right_b5` in the released dataset (22 episodes, avg length 427). The
Table S7 note that the pen-uncap test "uses the left hand to pick up the pen" is correct
as published.

## Full pen inventory in the released dataset

The release contains 385 pen episodes; 338 of them make up the training subtotal above.

| Released task name | Episodes | Avg length | Paper label | Split |
| --- | ---: | ---: | --- | --- |
| `pen_uncap_gray_left_b5` | 91 | 557 | gray pen / `<right>` | train |
| `pen_uncap_gray_left_b9` | 52 | 685 | gray pen / `<right>` | train |
| `pen_uncap_gray_right_b5` | 51 | 500 | gray pen / `<left>` | train |
| `pen_uncap_red_left_b9` | 39 | 562 | red pen / `<right>` | train |
| `pen_uncap_red_left_b5` | 30 | 413 | red pen / `<right>` | train |
| `pen_uncap_blue_right_b5` | 25 | 500 | blue pen / `<left>` | train |
| `pen_uncap_blue2_left_b5` | 25 | 500 | blue pen v2 / `<right>` | train |
| `pen_uncap_blue2_right_b5` | 25 | 500 | blue pen v2 / `<left>` | train |
| `pen_uncap_blue_left_b5` | 25 | 500 | blue pen / `<right>` | held out |
| `pen_uncap_red_right_b5` | 22 | 427 | red pen / `<left>` | held out (unseen eval) |

## Convention used in this repository

Task-name strings in code follow the **dataset (collected-data) convention**, because they
are exact keys into the released dataset's `meta/tasks.jsonl` and into
`metadata/aloha_pen_uncap/*.json`. They are deliberately *not* renamed to the paper
convention — see `ALOHA_DATA_UNIQUE_TEST_TASK` in
`src/openpi/training/config_aloha.py`.

When reading those lists, translate with the mapping above. In particular, the two
held-out pen entries are:

```python
"pen_uncap_red_right_b5",   # paper: red pen / <left>  (the unseen eval configuration)
"pen_uncap_blue_left_b5",   # paper: blue pen / <right>
```
