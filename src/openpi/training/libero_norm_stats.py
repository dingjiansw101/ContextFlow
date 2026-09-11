"""Numeric-only normalization for the ContextFlow LIBERO configuration.

Keep the legacy accumulator order, including its every-fourth-frame sampling
and repeated passes, so regenerating assets does not change existing models.
Parquet data is read once; images and in-context demonstrations are never decoded.
"""

from concurrent.futures import ThreadPoolExecutor

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import pyarrow.parquet as pq
import tqdm

from openpi import transforms
from openpi.shared import normalize


class SparseRunningStats(normalize.RunningStats):
    """Update the same histograms without allocating 5,000 bins per observation."""

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i, (hist, edges) in enumerate(zip(self._histograms, self._bin_edges, strict=True)):
            values = batch[:, i]
            # Padded dimensions become constant zero once the range is adjusted.
            if edges[0] == edges[-1] and np.all(values == edges[0]):
                hist[-1] += len(values)
                continue
            indices = np.searchsorted(edges, values, side="right") - 1
            # Like np.histogram, include the rightmost edge in the final bin.
            indices[values == edges[-1]] = len(hist) - 1
            valid = (values >= edges[0]) & (values <= edges[-1])
            np.add.at(hist, indices[valid], 1)


class NumericDataset:
    def __init__(self, states, actions, episode_lengths, *, action_dim, action_horizon, use_delta_joint_actions):
        if states.shape != (sum(episode_lengths), 8) or actions.shape != (len(states), 7):
            raise ValueError("Expected LIBERO states [frames, 8] and actions [frames, 7]")
        self.states = transforms.pad_to_dim(states, action_dim)
        self.actions = transforms.pad_to_dim(actions, action_dim)
        self.ends = np.repeat(np.cumsum(episode_lengths) - 1, episode_lengths)
        self.offsets = np.arange(action_horizon)
        self.delta = transforms.DeltaActions(transforms.make_bool_mask(6, -1) if use_delta_joint_actions else None)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # LeRobot repeats the final action at episode boundaries.
        indices = np.minimum(index + self.offsets, self.ends[index])
        return self.delta({"state": self.states[index], "actions": self.actions[indices]})


def load_dataset(config, data_config, *, use_delta_joint_actions):
    meta = LeRobotDatasetMetadata(data_config.repo_id, local_files_only=data_config.local_files_only)
    episodes = data_config.train_episode
    if episodes is None:
        episodes = [row["episode_index"] for row in meta.episodes]
    paths = [meta.root / meta.get_data_file_path(ep) for ep in episodes]
    missing = [str(path.relative_to(meta.root)) for path in paths if not path.is_file()]
    if missing:
        meta.pull_from_repo(allow_patterns=missing)

    def read(path):
        table = pq.read_table(path, columns=["state", "actions"], use_threads=False)
        return tuple(np.asarray(table[key].to_pylist(), dtype=np.float32) for key in ("state", "actions"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(tqdm.tqdm(pool.map(read, paths), total=len(paths), desc="Reading numeric columns"))
    if not rows:
        raise ValueError("No episodes selected for normalization")
    return NumericDataset(
        np.concatenate([row[0] for row in rows]),
        np.concatenate([row[1] for row in rows]),
        [len(row[0]) for row in rows],
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
        use_delta_joint_actions=use_delta_joint_actions,
    )


def compute(dataset, *, progress=True):
    size = len(dataset)
    batches_per_pass = size // 4
    if batches_per_pass == 0:
        raise ValueError("At least four frames are required, matching the legacy loader")
    stats = {key: SparseRunningStats() for key in ("state", "actions")}
    for batch_index in tqdm.trange(size, desc="Computing stats (numeric)", disable=not progress):
        sample = dataset[(batch_index % batches_per_pass) * 4]
        for key, accumulator in stats.items():
            values = sample[key]
            accumulator.update(values.reshape(-1, values.shape[-1]))
    return {key: accumulator.get_statistics() for key, accumulator in stats.items()}
