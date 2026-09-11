"""Shared per-dataset spec for multi-dataset in-context data config factories."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset settings for concatenated in-context LeRobot datasets.

    Each spec is loaded as its own CustomLeRobotDataset; the factory carrying
    ``dataset_specs`` concatenates them at training time (see
    ``data_loader.create_custom_dataset``). Demo retrieval is within-dataset
    only, and each sub-dataset derives its own task-to-episode table from
    ``repo_id``'s metadata, so the table is always keyed by that dataset's task
    indices (task 0 in one dataset != task 0 in another).

    Defined in a leaf module (no openpi imports) so config fragments can bind it
    at module scope, letting tyro resolve the forward reference
    ``tuple[DatasetSpec, ...]`` against module globals when introspecting the
    multi-dataset factories.
    """

    repo_id: str
    episode_json_path: str
    remove_task_list: list[str] | None = None
    local_files_only: bool = False
