import dataclasses
import os
import pathlib
import signal
import types

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as _model
import openpi.shared.array_typing as at
from openpi.training import config as _config

from . import train


class _FakeWeightLoader:
    def __init__(self, params):
        self._params = params

    def load(self, params):
        del params
        return self._params


@dataclasses.dataclass(frozen=True)
class _TinyModelConfig(_model.BaseModelConfig):
    action_dim: int = 2
    action_horizon: int = 3
    max_token_len: int = 4

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    def create(self, rng: jax.Array) -> "_TinyModel":
        return _TinyModel(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, 1, 1, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation = _model.Observation(
                images={"base_0_rgb": image_spec},
                image_masks={"base_0_rgb": image_mask_spec},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            )
        actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation, actions


class _TinyModel(_model.BaseModel):
    def __init__(self, config: _TinyModelConfig, rngs: nnx.Rngs):
        del rngs
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.scale = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        del rng, observation, train
        return jnp.mean((actions * self.scale.value) ** 2, axis=-1)

    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation) -> _model.Actions:
        del rng, observation
        return jnp.zeros((self.action_horizon, self.action_dim), dtype=jnp.float32)


def test_create_train_data_loader_routes_by_model_type(monkeypatch):
    calls = []
    create_train_data_loader = train._create_train_data_loader  # noqa: SLF001
    data_loader = train._data_loader  # noqa: SLF001

    def fake_standard(config, **kwargs):
        calls.append(("standard", config, kwargs))
        return "standard"

    def fake_incontext(config, **kwargs):
        calls.append(("incontext", config, kwargs))
        return "incontext"

    def fake_custom(config, **kwargs):
        calls.append(("custom", config, kwargs))
        return "custom"

    monkeypatch.setattr(data_loader, "create_data_loader", fake_standard)
    monkeypatch.setattr(data_loader, "create_incontext_data_loader", fake_incontext)
    monkeypatch.setattr(data_loader, "create_custom_incontext_data_loader", fake_custom)

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(model_type=_model.ModelType.PI0_FAST_INCONTEXT),
        use_custom_dataloader=False,
    )
    assert create_train_data_loader(config, sharding="sharding", num_workers=3, shuffle=True) == "incontext"

    config.model.model_type = _model.ModelType.PI0
    assert create_train_data_loader(config, sharding="sharding", num_workers=3, shuffle=True) == "standard"

    config.model.model_type = _model.ModelType.PI0_INCONTEXT
    config.use_custom_dataloader = True
    assert create_train_data_loader(config, sharding="sharding", num_workers=3, shuffle=True) == "custom"

    assert [call[0] for call in calls] == ["incontext", "standard", "custom"]
    assert all(call[2] == {"sharding": "sharding", "num_workers": 3, "shuffle": True} for call in calls)


def test_register_preemption_handlers(monkeypatch):
    registered = {}

    def fake_signal(signum, handler):
        registered[signum] = handler

    monkeypatch.setattr(train.signal, "signal", fake_signal)

    train._register_preemption_handlers()  # noqa: SLF001

    assert registered == {
        signal.SIGTERM: train._on_preempt,  # noqa: SLF001
        signal.SIGUSR1: train._on_preempt,  # noqa: SLF001
    }


def test_on_preempt_sets_requested_flag():
    train._preempt_requested = False  # noqa: SLF001

    train._on_preempt(signal.SIGTERM, None)  # noqa: SLF001

    assert train._preempt_requested is True  # noqa: SLF001


def test_load_weights_accepts_allowlisted_missing_subtrees():
    load_weights_and_validate = train._load_weights_and_validate  # noqa: SLF001
    params_shape = {
        "PaliGemma": {"img": {"kernel": jax.ShapeDtypeStruct((1,), jnp.float32)}},
        "demo_state_proj": {
            "kernel": jax.ShapeDtypeStruct((8, 16), jnp.float32),
            "bias": jax.ShapeDtypeStruct((16,), jnp.float32),
        },
    }
    loaded_params = {"PaliGemma": {"img": {"kernel": np.ones((1,), dtype=np.float32)}}}

    result = load_weights_and_validate(_FakeWeightLoader(loaded_params), params_shape)

    assert "demo_state_proj" not in result
    np.testing.assert_array_equal(result["PaliGemma"]["img"]["kernel"], np.ones((1,), dtype=np.float32))


def test_load_weights_rejects_unexpected_missing_subtrees():
    load_weights_and_validate = train._load_weights_and_validate  # noqa: SLF001
    params_shape = {
        "PaliGemma": {"img": {"kernel": jax.ShapeDtypeStruct((1,), jnp.float32)}},
        "unexpected_proj": {"kernel": jax.ShapeDtypeStruct((1,), jnp.float32)},
    }
    loaded_params = {"PaliGemma": {"img": {"kernel": np.ones((1,), dtype=np.float32)}}}

    with pytest.raises(ValueError, match="PyTrees have different structure"):
        load_weights_and_validate(_FakeWeightLoader(loaded_params), params_shape)


def test_train(tmp_path: pathlib.Path):
    config = _config.TrainConfig(
        name="debug",
        model=_TinyModelConfig(),
        batch_size=2,
        checkpoint_base_dir=tmp_path / "checkpoint",
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        num_workers=0,
        wandb_enabled=False,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
