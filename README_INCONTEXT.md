steps for create data of ICFM
1. convert hdf5 to parquet
``convert_aloha_mobile_data_to_lerobot_multi.py``
1.2 compute norm stats
``uv run scripts/compute_norm_stats.py
  --config-name pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8 ``
2. generate "task_to_episode.json" and "episode_to_indexes.json", with all tasks, and use pi0 config to generate
``openpi/src/openpi/training/generate_task_to_index.py``   
3. run the config with all tasks to generate "episode_actions_without_delta_cache.json" and "episode_states_without_delta_cache.json"
4. run the config and remove the unseen tasks


check the correctness of prompt during real-world testing
check pi0

check icfm
to send data to the model, to make sure prompt and task_index are consistent
printed all the task_index and prompt in transforms.py they are consistent. 

need to make sure the task json file is correct in both local and remote machine, otherwies, it may cause bug
