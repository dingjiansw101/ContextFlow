from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.shared import normalize
from openpi.training import data_loader
from openpi.training import libero_norm_stats as fast


def test_loading_reads_only_selected_numeric_columns(tmp_path, monkeypatch):
    for episode in [0, 2]:
        pq.write_table(
            pa.table({"state": [[float(episode)] * 8] * 4, "actions": [[float(episode)] * 7] * 4}),
            tmp_path / f"{episode}.parquet",
        )
    # No image columns exist: decoding/repacking images would fail here.
    # Missing episode 1 must not be accessed when it is excluded from the split.
    meta = SimpleNamespace(root=tmp_path, get_data_file_path=lambda ep: f"{ep}.parquet")
    monkeypatch.setattr(fast, "LeRobotDatasetMetadata", lambda *args, **kwargs: meta)
    dataset = fast.load_dataset(
        SimpleNamespace(model=SimpleNamespace(action_dim=32, action_horizon=5)),
        SimpleNamespace(repo_id="physical-intelligence/libero", local_files_only=True, train_episode=[2, 0]),
        use_delta_joint_actions=False,
    )
    assert len(dataset) == 8
    np.testing.assert_array_equal(dataset[0]["state"][:8], 2)
    np.testing.assert_array_equal(dataset[4]["state"][:8], 0)
    np.testing.assert_array_equal(dataset[3]["actions"][:, :7], 2)


@pytest.mark.parametrize("batch_size", [1, 50])
def test_sparse_histograms_match_legacy_with_expanding_ranges(batch_size):
    rng = np.random.default_rng(42)
    old, new = normalize.RunningStats(), fast.SparseRunningStats()
    for i in range(25):
        batch = np.zeros((batch_size, 32), dtype=np.float32)
        batch[:, :7] = rng.normal(size=(batch_size, 7)) * (1 + i)
        old.update(batch)
        new.update(batch)
        for a, b in zip(old._histograms, new._histograms, strict=True):  # noqa: SLF001
            np.testing.assert_array_equal(a, b)
    assert normalize.serialize_json({"x": old.get_statistics()}) == normalize.serialize_json(
        {"x": new.get_statistics()}
    )


@pytest.mark.parametrize("size", [8, 11])
@pytest.mark.parametrize("delta", [False, True])
def test_numeric_stats_match_legacy_loader(size, delta):
    rng = np.random.default_rng(42)
    states = rng.normal(size=(size, 8)).astype(np.float32)
    actions = rng.normal(size=(size, 7)).astype(np.float32)
    dataset = fast.NumericDataset(
        states, actions, [3, size - 3], action_dim=32, action_horizon=5, use_delta_joint_actions=delta
    )
    expected = np.repeat(actions[2:3], 5, axis=0)
    if delta:
        expected[:, :6] -= states[2, :6]
    np.testing.assert_array_equal(dataset[2]["actions"][:, :7], expected)
    np.testing.assert_array_equal(dataset[2]["actions"][:, 7:], 0)
    # Calling the transform repeatedly must not mutate the source data.
    np.testing.assert_array_equal(dataset[2]["actions"][:, :7], expected)
    old = {key: normalize.RunningStats() for key in ("state", "actions")}
    loader = data_loader.TorchDataLoader(dataset, 4, num_workers=0, num_batches=size)
    for batch in loader:
        for key, accumulator in old.items():
            value = np.asarray(batch[key][0])
            accumulator.update(value.reshape(-1, value.shape[-1]))
    expected_stats = {key: accumulator.get_statistics() for key, accumulator in old.items()}
    assert normalize.serialize_json(expected_stats) == normalize.serialize_json(fast.compute(dataset, progress=False))
