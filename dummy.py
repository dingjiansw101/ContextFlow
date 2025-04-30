'''
uv run dummy.py pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split --exp_name dummy
uv run dummy.py pi0_rlbench_joint_low_mem_finetune_train --exp_name dummy

'''
from typing import Any
import numpy as np
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi import transforms as _transforms
import random
import dataclasses
import enum
import logging
import socket
from tqdm import tqdm

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

def describe_field(name, field):
    import numpy as np
    print(f"\n {name}:")
    print(f"  type: {type(field)}")

    if isinstance(field, np.ndarray):
        print(f"  shape: {field.shape}")
        print(f"  dtype: {field.dtype}")
        print(f"  first values: {field.ravel()[:5]}")
    elif isinstance(field, list):
        print(f"  len: {len(field)}")
        if len(field) > 0:
            print(f"  item[0] type: {type(field[0])}")
            if isinstance(field[0], dict):
                print(f"  item[0] keys: {field[0].keys()}")
    elif isinstance(field, dict):
        print(f"  keys: {field.keys()}")
    else:
        print(f"  value: {field}")

def inspect_transforms(dataset):
    layer = 0
    all_transforms = []

    while isinstance(dataset, _data_loader.TransformedDataset):
        layer += 1
        transforms = dataset._transform_list  
        print(f"Layer {layer}: {[t.__class__.__name__ for t in transforms]}")
        all_transforms = transforms + all_transforms  
        dataset = dataset._dataset

    print(f"\nTotal layers: {layer}")
    print("Final transform order (first applied → last applied):")
    for i, t in enumerate(all_transforms, 1):
        print(f"{i}. {t.__class__.__name__}")

def get_raw_dataset(dataset):
    while isinstance(dataset, _data_loader.TransformedDataset):
        dataset = dataset._dataset
    return dataset

def compare_keys(d1: dict, d2: dict):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    only_in_d1 = keys1 - keys2
    only_in_d2 = keys2 - keys1

    if only_in_d1:
        print(f"! Keys only in first dict: {only_in_d1}")
    if only_in_d2:
        print(f"! Keys only in second dict: {only_in_d2}")

    if not only_in_d1 and not only_in_d2:
        print(" Key sets match exactly.")

    return list(keys1 & keys2)

def deep_compare(v1, v2, prefix="", atol=1e-5):
    import numpy as np

    # Type mismatch
    if type(v1) != type(v2):
        print(f"! {prefix} type mismatch: {type(v1)} vs {type(v2)}")
        return False

    # Dict: recursive
    if isinstance(v1, dict):
        keys1 = set(v1.keys())
        keys2 = set(v2.keys())
        if keys1 != keys2:
            print(f"! {prefix} dict key mismatch: {keys1 ^ keys2}")
            return False
        all_match = True
        for k in sorted(keys1):
            all_match &= deep_compare(v1[k], v2[k], prefix + f"{k}.", atol)
        return all_match

    # List: element-wise compare
    elif isinstance(v1, list):
        if len(v1) != len(v2):
            print(f"! {prefix} list length mismatch: {len(v1)} vs {len(v2)}")
            return False
        all_match = True
        for i, (a, b) in enumerate(zip(v1, v2)):
            all_match &= deep_compare(a, b, prefix + f"[{i}].", atol)
        return all_match

    # NumPy array / number
    elif isinstance(v1, np.ndarray) or np.isscalar(v1):
        try:
            a1, a2 = np.asarray(v1), np.asarray(v2)
            if np.issubdtype(a1.dtype, np.number):
                if not np.allclose(a1, a2, atol=atol):
                    print(f"! {prefix} numerical mismatch")
                    return False
            else:
                if not np.array_equal(a1, a2):
                    print(f"! {prefix} array mismatch")
                    return False
            return True
        except Exception as e:
            print(f"!!  {prefix} array comparison error: {e}")
            return False

    # Other scalar types
    else:
        if v1 != v2:
            print(f"! {prefix} value mismatch: {v1} != {v2}")
            return False
        return True


def compare_values(d1: dict, d2: dict, keys: list[str], atol=1e-5):
    for k in keys:
        v1 = d1[k]
        v2 = d2[k]
        print(f"\n Comparing key: '{k}'")

        if not deep_compare(v1, v2, prefix=f"{k}.", atol=atol):
            print(f"! Mismatch in key: '{k}'")
        else:
            print(f" Match in key: '{k}'")
    
def compare_transformed_dicts(policy_output: dict, transformed_sample: dict):
    print("Comparing key sets...")
    shared_keys = compare_keys(policy_output, transformed_sample)

    print("\n Comparing values...")
    compare_values(policy_output, transformed_sample, shared_keys)

def inspect_composite_transform(transform):
    if not isinstance(transform, _transforms.CompositeTransform):
        print("Not a CompositeTransform")
        return

    print("Input Transform List (in execution order):")
    for i, t in enumerate(transform.transforms):
        print(f"{i+1}. {t.__class__.__name__}")

def main(config: _config.TrainConfig):
    # print(config)
    random.seed(42)
    data_loader = _data_loader.create_data_loader(
        config,
        num_workers=config.num_workers,
        shuffle=False,
    )

    # the private attribute  (_dataset) of data_loader is the outer-most dataset,
    # which is TransformedDataset embedded in another TransformedDataset etc
    # therefore, we get the inner-most dataset which can be used to extract raw data sample
    # then we use outer-most dataset to get transformed data sample
    dataset = data_loader._dataset

    '''
    Expected output:
    1. PromptFromLeRobotTask
    2. RepackTransform
    3. InjectDemoIndexes
    4. LiberoIncontextInputs
    5. DeltaActions
    6. Normalize
    7. InjectDefaultPrompt
    8. ResizeImages
    9. TokenizePrompt
    10. AddDemoPromptTransform
    '''
    inspect_transforms(dataset)

    inner_most_dataset = get_raw_dataset(dataset)
    # this step is to check that the loaded LeRobotDataset contains only training tasks
    # task_indices = set()

    # for s in tqdm(inner_most_dataset):
    #     task_indices.add(s["task_index"])

    # print(f"Unique task indices ({len(task_indices)}): {sorted(task_indices)}")

    test_sample_index = 0
    raw_sample = inner_most_dataset[test_sample_index]
    transformed_sample = dataset[test_sample_index]

    print("raw_sample:")
    print(f"  index: {raw_sample['index']}")
    print(f"  frame_index: {raw_sample['frame_index']}")
    print(f"  episode_index: {raw_sample['episode_index']}")
    print(f"  task_index: {raw_sample['task_index']}")

    # print("transformed_sample:")
    # print(f"  index: {transformed_sample['index']}")
    # print(f"  selected_episode: {transformed_sample['selected_episode']}")
    # print(f"  dem_prompt_indexes: {transformed_sample['dem_prompt_indexes']}")

    '''
    Expected output:
    dict_keys([
        'image', 
        'wrist_image', 
        'state', 
        'actions', 
        'timestamp', 
        'frame_index', 
        'episode_index', 
        'index', 
        'task_index', 
        'actions_is_pad'])
    '''
    print("Test sample before any transform:")
    print(raw_sample.keys())
    print(raw_sample['actions'].shape)
    print(raw_sample['actions'][0])
    print(raw_sample['action_joint_velocity'].shape)
    print(raw_sample['action_joint_velocity'][0])
    '''
    Expected output:
    dict_keys([
        'state', 
        'image', 
        'image_mask', 
        'actions', 
        'dem_prompt_indexes',   (from InjectDemoIndexes)
        'selected_episode',     (from InjectDemoIndexes)
        'index', 
        'tokenized_prompt', 
        'tokenized_prompt_mask', 
        'dem_prompt_items', 
        'dem_prompt_all_states', 
        'dem_prompt_all_states_mask', 
        'dem_prompt_all_actions', 
        'dem_prompt_all_actions_mask'])
    '''
    print("Test sample after all transform:")
    print(transformed_sample.keys())
    print(transformed_sample['actions'].shape)
    print(transformed_sample['actions'][0])

    # Now we create pi0 in context policy
    # which will be used to get all input transform(s)
    # and compare the transformed inputs with outer-most dataset
    # test_policy_config=r"pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split"
    # test_policy_dir="/ibex/tmp/c2090/openpi_explore_storage//checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split/19999"
    
    # random.seed(42)
    # test_policy = _policy_config.create_trained_policy_incontext(
    #             _config.get_config(test_policy_config), test_policy_dir, default_prompt=None
    #         )
    
    '''
    Expected output:
    1. InjectDefaultPrompt
    2. InjectDemoIndexes
    3. LiberoIncontextInputs
    4. DeltaActions
    5. Normalize
    6. InjectDefaultPrompt
    7. ResizeImages
    8. TokenizePrompt
    9. AddDemoPromptTransform
    '''
    # inspect_composite_transform(test_policy._input_transform)

    # # in order to compare, we align the first and second of training transform
    # dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_loader._data_config.repo_id, local_files_only=data_loader._data_config.local_files_only)
    # first_align_transform = _transforms.PromptFromLeRobotTask(dataset_meta.tasks)
    # second_align_transform = _transforms.RepackTransform(
    #                 {
    #                     "observation/image": "image",
    #                     "observation/wrist_image": "wrist_image",
    #                     "observation/state": "state",
    #                     "actions": "actions",
    #                     "prompt": "prompt",
    #                     "episode_index": "episode_index",
    #                     "index": "index",
    #                     "task_index": "task_index",
    #                 }
    # )
    # aligned_raw_sample = first_align_transform(raw_sample)
    # aligned_raw_sample = second_align_transform(aligned_raw_sample)

    # # get the transformed input from policy
    # random.seed(42)
    # policy_output = test_policy._input_transform(aligned_raw_sample)

    # # the following commented out funcs are used to inspect inner structure of "image" & "dem_prompt_items"
    # # describe_field("image", policy_output["image"])
    # # describe_field("dem_prompt_items", policy_output["dem_prompt_items"])

    # # thorough compare between key and value of transformed input (which is of type dict)
    # compare_transformed_dicts(policy_output, transformed_sample)

if __name__ == "__main__":
    main(_config.cli())