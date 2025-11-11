# Steps for Creating Data for ICFM (In-Context Fine-tuning Models)

This guide provides the workflow for preparing data and training in-context learning models.

## Step 1: Convert HDF5 to Parquet

Convert the raw HDF5 data to LeRobot parquet format using:

```bash
python examples/aloha_mobile_real/convert_aloha_mobile_data_to_lerobot_multi.py
```

## Step 2: Compute Normalization Statistics

Compute normalization statistics for actions and states:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8
```

**Note:** This must be done before generating task-to-index metadata, as the generation script tests the full transform pipeline.

## Step 3: Create Empty Placeholder Metadata Files

Create placeholder JSON files to allow the config to load:

```bash
mkdir -p metadata/object_task_suite
echo '{}' > metadata/object_task_suite/task_to_episode.json
echo '{}' > metadata/object_task_suite/episode_to_indexes.json
```



**Why:** The config tries to load these files during initialization. Creating empty placeholders bypasses the chicken-and-egg problem.

## Step 4: Generate Task-to-Index Metadata

Generate `task_to_episode.json` and `episode_to_indexes.json` files with all tasks, using the Pi0 config:

```bash
uv run src/openpi/training/generate_task_to_index.py \
  --config pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8 \
  --output_dir metadata/object_task_suite
```

This script is located at `openpi/src/openpi/training/generate_task_to_index.py` and will overwrite the placeholder files with actual task-to-episode and episode-to-frame mappings.

## Step 5: Run Config with All Tasks to Generate Cache Files

Run the training config with all tasks included to generate the following cache files:
- `episode_actions_without_delta_cache.json`
- `episode_states_without_delta_cache.json`

These cache files are automatically generated during the first training run.

Or use the script to build cache
`
/home/dingj0b/code/openpi/src/openpi/training/build_episode_cache.py
`

## Step 6: Run Config and Remove Unseen Tasks

For actual training with train-test split, run the config with `remove_task_list` enabled to exclude unseen/test tasks from training.

Update the config to uncomment:
```python
remove_task_list=ALOHA_OBJECT_TEST_TASK
```

This will exclude the 9 test tasks defined in `ALOHA_OBJECT_TEST_TASK` from training, allowing you to evaluate zero-shot generalization.

---

## Evaluation

Once training is complete, you can evaluate the model using a server-client setup.

### Terminal 1: Start Policy Server

Start the policy server with your trained checkpoint:

```bash
uv run scripts/serve_policy_incontext.py --port 8001 policy:checkpoint \
  --policy.config=pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_80k_inference \
  --policy.dir=checkpoints/pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_80k/pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_80k/79999
```

```bash
export JAX_DEFAULT_MATMUL_PRECISION=float32

uv run scripts/serve_policy_incontext.py --port 8001 policy:checkpoint \
  --policy.config=pi0_aloha_objects_all_incontextv18_low_mem_finetune_sample_frames8_inference \
  --policy.dir=checkpoints/pi0_aloha_objects_all_incontextv18_low_mem_finetune_sample_frames8/pi0_aloha_objects_all_incontextv18_low_mem_finetune_sample_frames8/10000
```

**Note:** Adjust the checkpoint path and config name to match your trained model.

### Terminal 2: Run Evaluation Client

1. Modify the task prompt in `examples/aloha_mobile_real/main_incontext.py` to specify the task you want to evaluate:

```python
# Example task prompt
task_prompt = "pick_up_the_pear_and_place_it_in_the_basket_with_left_hand"
```

2. Activate the environment and run the client:

```bash
source examples/aloha_mobile_real/.venv/bin/activate
python examples/aloha_mobile_real/main_incontext.py
```

The client will connect to the policy server and execute the specified task using the trained model.

---

## Troubleshooting and Testing

### Check Correctness of Prompt During Real-World Testing

#### Check Pi0
- Verify that prompts are correctly formatted and sent to the model

#### Check ICFM
- When sending data to the model, ensure that `prompt` and `task_index` are consistent
- During debugging, all `task_index` and `prompt` values were printed in `transforms.py` and confirmed to be consistent

### Important Notes

**Task JSON File Consistency:**
Make sure the task JSON files are correct and identical on both local and remote machines. Inconsistent files may cause bugs during distributed training or inference.
