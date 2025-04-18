from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_incontextv7
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


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

def test_pi0_lora_model_v7_params():
    key = jax.random.key(0)
    # config = pi0.Pi0Config(paligemma_variant="gemma_2b_lora")
    config = pi0_incontextv7.Pi0IncontextConfigv7(paligemma_variant="gemma_2b_lora",
                                                   prompt_expert_variant="gemma_300m", 
                                                  action_expert_variant="gemma_300m_lora",
                                                    sample_frames=2, sample_actions=32, random_select=True,)
    model = config.create(key)

    # import ipdb; ipdb.set_trace()
    import flax.nnx as nnx

    from openpi.shared import nnx_utils            # where PathRegex lives

    action_filter = nnx_utils.PathRegex(".*llm.*_1.*")  # already defined in your code
    for path, node in nnx.iter_graph(model):      # walk the whole graph  :contentReference[oaicite:1]{index=1}
        if isinstance(node, nnx.Param) and action_filter(path, node):
            print("/".join(map(str, path)), node.value.shape)

    # gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
    # for path, node in nnx.iter_graph(model):      # walk the whole graph  :contentReference[oaicite:1]{index=1}
    #     if isinstance(node, nnx.Param) and gemma_params_filter(path, node):
    #         print("/".join(map(str, path)), node.value.shape)

    # prompt_expert_params_filter = nnx_utils.PathRegex(".*llm.*_prompt_expert.*")
    # for path, node in nnx.iter_graph(model):      # walk the whole graph  :contentReference[oaicite:1]{index=1}
    #     if isinstance(node, nnx.Param) and prompt_expert_params_filter(path, node):
    #         print("/".join(map(str, path)), node.value.shape)

    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    test_pi0_lora_model_v7_params()