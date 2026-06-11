"""Shared per-dataset spec for multi-dataset in-context data config factories."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset settings for concatenated in-context LeRobot datasets.

    Each spec is loaded as its own CustomLeRobotDataset; the factory carrying
    ``dataset_specs`` concatenates them at training time (see
    ``data_loader.create_custom_dataset``). Demo retrieval is within-dataset
    only, so each spec must point at its own task_to_episode.json keyed by that
    dataset's task indices.

    Defined in a leaf module (no openpi imports) so config fragments can bind it
    at module scope, letting tyro resolve the forward reference
    ``tuple[DatasetSpec, ...]`` against module globals when introspecting the
    multi-dataset factories.
    """

    repo_id: str
    episode_json_path: str
    task_to_episode_path: str
    remove_task_list: list[str] | None = None
    local_files_only: bool = False
