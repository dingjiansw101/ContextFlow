import json
import random

import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
import jax

def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_inject_demo_indexes_seed_base_is_deterministic(tmp_path):
    task_to_episode_path = tmp_path / "task_to_episode.json"
    episode_to_indexes_path = tmp_path / "episode_to_indexes.json"
    task_to_episode_path.write_text(json.dumps({"0": [0, 1, 2, 3]}))
    episode_to_indexes_path.write_text(
        json.dumps(
            {
                "0": [0, 1, 2],
                "1": [3, 4, 5],
                "2": [6, 7, 8],
                "3": [9, 10, 11],
            }
        )
    )

    transform = _transforms.InjectDemoIndexes(
        task_to_episode=str(task_to_episode_path),
        episode_to_indexes=str(episode_to_indexes_path),
        sample_frames=2,
        sample_episodes=2,
        random_select=True,
        seed_base=123,
    )
    item = {
        "task_index": np.array(0),
        "index": np.array(17),
        "frame_index": np.array(2),
        "episode_index": np.array(0),
    }

    random.seed(1)
    first = transform(dict(item))
    random.seed(999)
    second = transform(dict(item))

    assert np.array_equal(first["selected_episode"], second["selected_episode"])
    assert first["dem_prompt_indexes"] == second["dem_prompt_indexes"]


def test_inject_demo_indexes_without_seed_base_uses_global_random(tmp_path):
    task_to_episode_path = tmp_path / "task_to_episode.json"
    episode_to_indexes_path = tmp_path / "episode_to_indexes.json"
    task_to_episode_path.write_text(json.dumps({"0": [0, 1, 2, 3]}))
    episode_to_indexes_path.write_text(
        json.dumps(
            {
                "0": [0, 1, 2],
                "1": [3, 4, 5],
                "2": [6, 7, 8],
                "3": [9, 10, 11],
            }
        )
    )

    transform = _transforms.InjectDemoIndexes(
        task_to_episode=str(task_to_episode_path),
        episode_to_indexes=str(episode_to_indexes_path),
        sample_frames=2,
        sample_episodes=2,
        random_select=True,
    )
    item = {
        "task_index": np.array(0),
        "index": np.array(17),
        "frame_index": np.array(2),
        "episode_index": np.array(0),
    }

    random.seed(123)
    result = transform(dict(item))

    expected = random.Random(123).sample([0, 1, 2, 3], 2)
    assert np.array_equal(result["selected_episode"], np.array(expected, dtype=np.int32))


# def test_injectdemoindexes():
#     item = {"task_index": 0}
#     transform = _transforms.InjectDemoIndexes(random_select=False, train_episode_index_list=[0, 1, 2])
#     transform_refactor = _transforms.InjectDemoIndexes_refactor(random_select=False, train_episode_index_list=[0, 1, 2])
#     data = transform(item)
#     data_refactor = transform_refactor(item)
#     assert jax.tree_util.tree_all(jax.tree_map(np.allclose, data, data_refactor))
#     # import ipdb; ipdb.set_trace()

# def test_adddemo():


# if __name__ == "__main__":
#     test_injectdemoindexes()


# ============================================================================
# Tests for AddStatesActionsPromptTransform padding behavior
# ============================================================================

def create_mock_episode_data(num_frames: int, state_dim: int = 8, action_dim: int = 7):
    """Create synthetic episode data for testing."""
    # Use predictable values for easy debugging
    states = np.arange(num_frames * state_dim).reshape(num_frames, state_dim).astype(np.float32)
    actions = np.arange(num_frames * action_dim).reshape(num_frames, action_dim).astype(np.float32)
    return states, actions


def simulate_custom_dataset_padding(states, actions, max_len):
    """Simulate CustomLeRobotDataset + CustomLeRobotLiberoIncontextInputs behavior.

    This mimics the training path:
    1. CustomLeRobotDataset: linspace sampling + repeat last frame
    2. CustomLeRobotLiberoIncontextInputs: all masks True
    """
    num_frames = len(states)
    num_samples = min(num_frames, max_len)

    # Linspace sampling (matching CustomLeRobotDataset logic)
    indices = np.linspace(0, num_frames - 1, num=num_samples, dtype=int)
    sampled_states = states[indices]
    sampled_actions = actions[indices]

    # Repeat last frame if needed (matching CustomLeRobotDataset padding)
    if num_samples < max_len:
        pad_count = max_len - num_samples
        sampled_states = np.concatenate([sampled_states, np.repeat(sampled_states[-1:], pad_count, axis=0)])
        sampled_actions = np.concatenate([sampled_actions, np.repeat(sampled_actions[-1:], pad_count, axis=0)])

    # CustomLeRobotLiberoIncontextInputs creates all-True masks
    states_mask = np.ones(max_len, dtype=bool)
    actions_mask = np.ones(max_len, dtype=bool)

    return sampled_states, sampled_actions, states_mask, actions_mask


def test_padding_mode_matches_custom_dataset_fewer_frames():
    """Test that linspace_repeat mode matches CustomLeRobotDataset when num_frames < max_len.

    This is the CRITICAL test case where padding behavior matters most.
    """
    max_len = 32
    num_frames = 20  # Less than max_len
    state_dim = 8
    action_dim = 7

    # Create mock episode data
    episode_states, episode_actions = create_mock_episode_data(num_frames, state_dim, action_dim)

    # Simulate training path (CustomLeRobotDataset + CustomLeRobotLiberoIncontextInputs)
    training_states, training_actions, training_states_mask, training_actions_mask = \
        simulate_custom_dataset_padding(episode_states, episode_actions, max_len)

    # Simulate inference path using _linspace_sample_and_pad directly
    # (We test the core method directly to avoid needing a full dataset setup)
    # Create a mock transform instance
    import tempfile
    import json
    from pathlib import Path

    # Create temporary cache files
    with tempfile.TemporaryDirectory() as tmpdir:
        states_cache = Path(tmpdir) / "states.json"
        actions_cache = Path(tmpdir) / "actions.json"

        # Write mock cache data
        cache_data = {
            "0": episode_states.tolist(),  # episode_id=0
        }
        with states_cache.open("w") as f:
            json.dump(cache_data, f)

        cache_data = {
            "0": episode_actions.tolist(),  # episode_id=0
        }
        with actions_cache.open("w") as f:
            json.dump(cache_data, f)

        # Create transform with training-style parameters
        # We'll call _linspace_sample_and_pad directly to test the core logic
        transform = _transforms.AddStatesActionsPromptTransform(
            dataset=None,  # Not needed for direct method call
            max_len=max_len,
            states_cache_path=str(states_cache),
            actions_cache_path=str(actions_cache),
            padding_mode="linspace_repeat",
            mask_padding_as_valid=True,
        )

        # Test the _linspace_sample_and_pad method directly
        inference_states, inference_states_mask = transform._linspace_sample_and_pad(
            episode_states, s=0, e=num_frames
        )
        inference_actions, inference_actions_mask = transform._linspace_sample_and_pad(
            episode_actions, s=0, e=num_frames
        )

    # Assertions: inference should match training
    assert inference_states.shape == training_states.shape, \
        f"States shape mismatch: {inference_states.shape} vs {training_states.shape}"
    assert inference_actions.shape == training_actions.shape, \
        f"Actions shape mismatch: {inference_actions.shape} vs {training_actions.shape}"

    # Values should match exactly
    assert np.array_equal(inference_states, training_states), \
        "States values don't match between training and inference paths"
    assert np.array_equal(inference_actions, training_actions), \
        "Actions values don't match between training and inference paths"

    # Masks should match (all True)
    assert np.array_equal(inference_states_mask, training_states_mask), \
        "States masks don't match"
    assert np.array_equal(inference_actions_mask, training_actions_mask), \
        "Actions masks don't match"
    assert np.all(inference_states_mask == True), \
        "States mask should be all True (training behavior)"
    assert np.all(inference_actions_mask == True), \
        "Actions mask should be all True (training behavior)"


def test_padding_mode_matches_custom_dataset_exact_frames():
    """Test that linspace_repeat mode works correctly when num_frames == max_len."""
    max_len = 32
    num_frames = 32  # Exactly max_len
    state_dim = 8
    action_dim = 7

    # Create mock episode data
    episode_states, episode_actions = create_mock_episode_data(num_frames, state_dim, action_dim)

    # Simulate training path
    training_states, training_actions, training_states_mask, training_actions_mask = \
        simulate_custom_dataset_padding(episode_states, episode_actions, max_len)

    # Test inference path
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        states_cache = Path(tmpdir) / "states.json"
        actions_cache = Path(tmpdir) / "actions.json"

        with states_cache.open("w") as f:
            json.dump({"0": episode_states.tolist()}, f)
        with actions_cache.open("w") as f:
            json.dump({"0": episode_actions.tolist()}, f)

        transform = _transforms.AddStatesActionsPromptTransform(
            dataset=None,
            max_len=max_len,
            states_cache_path=str(states_cache),
            actions_cache_path=str(actions_cache),
            padding_mode="linspace_repeat",
            mask_padding_as_valid=True,
        )

        inference_states, inference_states_mask = transform._linspace_sample_and_pad(
            episode_states, s=0, e=num_frames
        )
        inference_actions, inference_actions_mask = transform._linspace_sample_and_pad(
            episode_actions, s=0, e=num_frames
        )

    # No padding needed, should match exactly
    assert np.array_equal(inference_states, training_states)
    assert np.array_equal(inference_actions, training_actions)
    assert np.all(inference_states_mask == True)
    assert np.all(inference_actions_mask == True)


def test_padding_mode_matches_custom_dataset_more_frames():
    """Test that linspace_repeat mode works correctly when num_frames > max_len."""
    max_len = 32
    num_frames = 50  # More than max_len
    state_dim = 8
    action_dim = 7

    # Create mock episode data
    episode_states, episode_actions = create_mock_episode_data(num_frames, state_dim, action_dim)

    # Simulate training path (downsampling via linspace)
    training_states, training_actions, training_states_mask, training_actions_mask = \
        simulate_custom_dataset_padding(episode_states, episode_actions, max_len)

    # Test inference path
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        states_cache = Path(tmpdir) / "states.json"
        actions_cache = Path(tmpdir) / "actions.json"

        with states_cache.open("w") as f:
            json.dump({"0": episode_states.tolist()}, f)
        with actions_cache.open("w") as f:
            json.dump({"0": episode_actions.tolist()}, f)

        transform = _transforms.AddStatesActionsPromptTransform(
            dataset=None,
            max_len=max_len,
            states_cache_path=str(states_cache),
            actions_cache_path=str(actions_cache),
            padding_mode="linspace_repeat",
            mask_padding_as_valid=True,  # Note: doesn't matter when downsampling (no padding occurs)
        )

        inference_states, inference_states_mask = transform._linspace_sample_and_pad(
            episode_states, s=0, e=num_frames
        )
        inference_actions, inference_actions_mask = transform._linspace_sample_and_pad(
            episode_actions, s=0, e=num_frames
        )

    # Downsampling, should match linspace sampling
    assert np.array_equal(inference_states, training_states)
    assert np.array_equal(inference_actions, training_actions)
    assert np.all(inference_states_mask == True)
    assert np.all(inference_actions_mask == True)


def test_default_padding_mode_unchanged():
    """Test that default parameters preserve backward compatibility.

    With padding_mode="keep_all" and mask_padding_as_valid=False (defaults),
    the behavior should be different from training (this is expected).
    """
    max_len = 32
    num_frames = 20  # Less than max_len
    state_dim = 8
    action_dim = 7

    # Create mock episode data
    episode_states, episode_actions = create_mock_episode_data(num_frames, state_dim, action_dim)

    # Test with default parameters (backward compatible behavior)
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        states_cache = Path(tmpdir) / "states.json"
        actions_cache = Path(tmpdir) / "actions.json"

        with states_cache.open("w") as f:
            json.dump({"0": episode_states.tolist()}, f)
        with actions_cache.open("w") as f:
            json.dump({"0": episode_actions.tolist()}, f)

        # Use default parameters
        transform = _transforms.AddStatesActionsPromptTransform(
            dataset=None,
            max_len=max_len,
            states_cache_path=str(states_cache),
            actions_cache_path=str(actions_cache),
            # padding_mode="keep_all",  # default
            # mask_padding_as_valid=False,  # default
        )

        inference_states, inference_states_mask = transform._window_sample_and_pad(
            episode_states, s=0, e=num_frames
        )
        inference_actions, inference_actions_mask = transform._window_sample_and_pad(
            episode_actions, s=0, e=num_frames
        )

    # With default "keep_all" mode:
    # - First 20 frames should be original data
    # - Last 12 frames should be padded with last frame
    # - First 20 masks should be True, last 12 should be False

    assert inference_states.shape == (max_len, state_dim)
    assert inference_actions.shape == (max_len, action_dim)

    # First num_frames should be original data
    assert np.array_equal(inference_states[:num_frames], episode_states)
    assert np.array_equal(inference_actions[:num_frames], episode_actions)

    # Remaining should be last frame repeated (broadcast comparison)
    assert np.all(inference_states[num_frames:] == episode_states[-1])
    assert np.all(inference_actions[num_frames:] == episode_actions[-1])

    # Masks: first num_frames True, rest False (default behavior)
    assert np.all(inference_states_mask[:num_frames] == True)
    assert np.all(inference_states_mask[num_frames:] == False)
    assert np.all(inference_actions_mask[:num_frames] == True)
    assert np.all(inference_actions_mask[num_frames:] == False)