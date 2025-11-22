from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_incontextv12
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as train_config


def print_freeze_report(config, *, max_items: int = 20) -> None:
    """Utility to print frozen vs. trainable parameter paths for a given config.

    Args:
        config: Either a model config, a TrainConfig, or the string name of a TrainConfig.
        max_items: Number of entries to show for each set before truncating the output.
    """
    if isinstance(config, str):
        cfg = train_config.get_config(config)
        model_cfg = cfg.model
        label = config
    elif hasattr(config, "model") and hasattr(config, "freeze_filter"):
        cfg = config
        model_cfg = cfg.model
        label = getattr(cfg, "name", cfg.__class__.__name__)
    else:
        cfg = None
        model_cfg = config
        label = model_cfg.__class__.__name__

    freeze_filter = model_cfg.get_freeze_filter()
    abstract_model = nnx.eval_shape(model_cfg.create, jax.random.key(0))

    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(freeze_filter))).flat_state()

    def summarize(paths: dict, title: str) -> None:
        print(f"{title} ({len(paths)}):")
        for path in list(paths)[:max_items]:
            print("  ", "/".join(str(part) for part in path))
        if len(paths) > max_items:
            print(f"  ... ({len(paths) - max_items} more)")

    print(f"=== Freeze report for {label} ===")
    print(f"Filter type: {type(freeze_filter)}")
    summarize(frozen, "Frozen params")
    summarize(trainable, "Trainable params")
    print()


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

@pytest.mark.manual
def test_model_PaliGemma():
    key = jax.random.key(0)
    config = pi0.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

def test_pi0_lora_model_params():
    key = jax.random.key(0)
    config = pi0.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    # import ipdb; ipdb.set_trace()
    import flax.nnx as nnx

    for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Param):           # only params
            print(".".join(map(str, path)), node.value.shape, node.value.dtype)

    import ipdb; ipdb.set_trace()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

def test_pi0_lora_model_v12_num_params():
    def list_trainable_paths(cfg: pi0_incontextv12.Pi0IncontextConfigv12):
        model = cfg.create(jax.random.key(0))

        freeze_filter    = cfg.get_freeze_filter()
        trainable_filter = nnx.Not(freeze_filter)

        return [
            "/".join(map(str, path))
            for path, node in nnx.iter_graph(model)
            if isinstance(node, nnx.Param) and trainable_filter(path, node)
        ]

    # -------------------------------------------------------------
    # 1.  Enumerate the four (action_has_lora, prompt_has_lora) cases
    # -------------------------------------------------------------
    cases = {
        "(0, 0)  no-LoRA": dict(
            prompt_expert_variant  ="gemma_300m_v2",
            action_expert_variant  ="gemma_300m",
        ), # number of keys: 54, with input_embedding
        "(1, 0)  action-LoRA only": dict(
            prompt_expert_variant  ="gemma_300m_v2",
            action_expert_variant  ="gemma_300m_lora",
        ), # number of keys: 56, with input_embedding

    }

    # -------------------------------------------------------------
    # 2.  Run each case and print results
    # -------------------------------------------------------------
    for title, variants in cases.items():
        cfg = pi0_incontextv12.Pi0IncontextConfigv12(
            **variants,
            sample_frames   = 2,
            sample_actions  = 32,
            random_select   = True,
            use_image_prompts = False,
        )

        trainable_paths = list_trainable_paths(cfg)

        print(f"\n=== {title} ===")
        print(f"Trainable parameter keys: {len(trainable_paths)}")
        for p in trainable_paths:
            print("  ", p)


import collections

# Helper function (can be defined once at the top or within the test)
def _list_trainable_paths(cfg):
    model = cfg.create(jax.random.key(0))
    freeze_filter = cfg.get_freeze_filter()
    trainable_filter = nnx.Not(freeze_filter)
    return sorted([
        "/".join(map(str, path))
        for path, node in nnx.iter_graph(model)
        if isinstance(node, nnx.Param) and trainable_filter(path, node)
    ])

def _compare_and_assert(paths1, paths2, label1, label2):
    """Helper function to compare two lists of paths and assert equality."""
    assert len(paths1) == len(paths2), \
        f"Parameter counts differ: {label1} has {len(paths1)}, {label2} has {len(paths2)}"

    if paths1 != paths2:
        set1 = set(paths1)
        set2 = set(paths2)
        
        diff1_only = sorted(list(set1 - set2))
        diff2_only = sorted(list(set2 - set1))
        
        error_message = f"Trainable parameter paths (names) do not match between {label1} and {label2}.\n"
        if diff1_only:
            error_message += f"  Only in {label1} ({len(diff1_only)}):\n    " + "\n    ".join(diff1_only) + "\n"
        if diff2_only:
            error_message += f"  Only in {label2} ({len(diff2_only)}):\n    " + "\n    ".join(diff2_only) + "\n"
            
        raise AssertionError(error_message)


if __name__ == "__main__":
    # test_pi0_lora_model_v12_num_params()
    pass
