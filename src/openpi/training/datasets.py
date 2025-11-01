"""Custom dataset classes that extend LeRobotDataset functionality.

This module contains custom dataset implementations that inherit from
lerobot.common.datasets.lerobot_dataset.LeRobotDataset and add or modify
functionality for specific use cases.
"""

from typing import Any, Dict, SupportsIndex

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class CustomLeRobotDataset(LeRobotDataset):
    """Custom dataset that extends LeRobotDataset with modified data loading.

    This class inherits from LeRobotDataset and allows you to customize how
    individual samples are loaded and processed. Override the __getitem__ method
    to implement your custom data loading logic.

    Example usage:
        dataset = CustomLeRobotDataset(
            repo_id="your/dataset",
            episodes=[0, 1, 2],  # Optional: specific episodes to load
            delta_timestamps={"action": [0.0, 0.1, 0.2]},  # Optional: action horizon
            local_files_only=True,  # Optional: don't download from hub
        )

    Args:
        Same as LeRobotDataset parent class.
    """

    def __getitem__(self, idx: SupportsIndex) -> Dict[str, Any]:
        """Get a single sample from the dataset with custom processing.
        Return:
        (1) n consecutive frames from the dataset, each frame at time step t includes: 
            (a) image, state, and action at time step t.
            (b) state, and action in [t, t + h - 1], h is the action horizon.
        make sure the n + h time steps are consecutive in the same episode.
        if t + h - 1 is greater than the episode length, how to handle the situation? 
        Check the original implementation of __getitem__ method of LeRobotDataset.
        (2) m subsampled frames as a in-context demonstration, it's from another episode but same task:
            (a) each time step t includes image, state, and action at time step t.
        m is the number of subsampled frames for an in-context demonstration episode.
        n and m are hyperparameters, they are set in the initialization of class, you can set them in the config file.

        TODO: check how is the LeRobotDataset used in create_incontext_data_loader of data_loader.py 
        what are the transforms applied to the LeRobotDataset?

        If we use the CustomLeRobotDataset, we will need to write a new create_data_loader_incontextv2 function in data_loader.py     
        The transform AddImagePromptTransform, AddStatesActionsPromptTransform, AddCurrentFramesSequenceTransform are not needed anymore with CustomLeRobotDataset.
        But we want the data loading and processing of CustomLeRobotDataset + create_data_loader_incontextv2 is the same as the LeRobotDataset + create_incontext_data_loader.
        Make sure the CustomLeRobotDataset is compatible with the create_data_loader_incontextv2 function, and the transforms.
        
        optional, if can speed up the data loading, you can use the following cache files:
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"
     
        """
        pass
        # TODO: 
        # return data


