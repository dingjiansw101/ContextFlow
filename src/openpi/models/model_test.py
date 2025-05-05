from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_incontext
from openpi.models import pi0_incontextv7
from openpi.models import pi0_incontextv9
from openpi.models import pi0_incontextv12
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

    # action_filter = nnx_utils.PathRegex(".*llm.*_1.*")  # already defined in your code
    # for path, node in nnx.iter_graph(model):      # walk the whole graph  :contentReference[oaicite:1]{index=1}
    #     if isinstance(node, nnx.Param) and action_filter(path, node):
    #         print("/".join(map(str, path)), node.value.shape)

    gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
    for path, node in nnx.iter_graph(model):      # walk the whole graph  :contentReference[oaicite:1]{index=1}
        if isinstance(node, nnx.Param) and gemma_params_filter(path, node):
            print("/".join(map(str, path)), node.value.shape)

    # prompt_expert_params_filter = nnx_utils.PathRegex(".*llm.*_prompt_expert.*")
    # for path, node in nnx.iter_graph(model):      # walk the whole graph  :contentReference[oaicite:1]{index=1}
    #     if isinstance(node, nnx.Param) and prompt_expert_params_filter(path, node):
    #         print("/".join(map(str, path)), node.value.shape)

    import ipdb; ipdb.set_trace()

def test_pi0_lora_model_v7_num_params():
    key = jax.random.key(0)
    config = pi0_incontextv7.Pi0IncontextConfigv7(
        paligemma_variant="gemma_2b_lora",
        prompt_expert_variant="gemma_300m",
        action_expert_variant="gemma_300m_lora",
        sample_frames=2,
        sample_actions=32,
        random_select=True,
        use_image_prompts=False,
    ) # trainable parameters excludes input_embedding
    model = config.create(key)

    # 2) build filters
    freeze_filter    = config.get_freeze_filter()
    trainable_filter = nnx.Not(freeze_filter)

    # 3) collect all trainable keys
    trainable_paths = [
        "/".join(map(str, path))
        for path, node in nnx.iter_graph(model)
        if isinstance(node, nnx.Param) and trainable_filter(path, node)
    ]

    for p in trainable_paths:
        print("  ", p)    # 4) print the count
    print(f"Number of trainable parameter keys: {len(trainable_paths)}")

def test_pi0_lora_model_v7_4_params():
    key = jax.random.key(0)
    config = pi0_incontextv7.Pi0IncontextConfigv7(
        paligemma_variant="gemma_2b_lora",
        prompt_expert_variant="gemma_300m",
        action_expert_variant="gemma_300m_lora",
        sample_frames=2,
        sample_actions=32,
        random_select=True,
    )
    model = config.create(key)

    # 2) build filters
    freeze_filter    = config.get_freeze_filter()
    trainable_filter = nnx.Not(freeze_filter)

    # 3) collect all trainable keys
    trainable_paths = [
        "/".join(map(str, path))
        for path, node in nnx.iter_graph(model)
        if isinstance(node, nnx.Param) and trainable_filter(path, node)
    ]

    # 4) print the count
    print(f"Number of trainable parameter keys: {len(trainable_paths)}")


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


def test_pi0_lora_model_v9_num_params():
    def list_trainable_paths(cfg: pi0_incontextv9.Pi0IncontextConfigv9):
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
            paligemma_variant="gemma_2b_lora",
            prompt_expert_variant  ="gemma_300m_v2",
            action_expert_variant  ="gemma_300m",
        ), # number of keys: 53
        "(0, 1)  action-LoRA only": dict(
            paligemma_variant="gemma_2b_lora",
            prompt_expert_variant  ="gemma_300m_v2",
            action_expert_variant  ="gemma_300m_lora",
        ), # number of keys: 55
        "(1, 0)  prompt-LoRA only": dict(
            paligemma_variant="gemma_2b",
            prompt_expert_variant  ="gemma_300m_v2",
            action_expert_variant  ="gemma_300m_lora",
        ), # number of keys: 56, with an additional input embedding
        # "(1, 1)  both LoRA": dict(
        #     action_expert_variant  ="gemma_300m_lora",
        #     prompt_expert_variant  ="gemma_300m_v2_lora",
        # ),
    }

    # -------------------------------------------------------------
    # 2.  Run each case and print results
    # -------------------------------------------------------------
    for title, variants in cases.items():
        cfg = pi0_incontextv9.Pi0IncontextConfigv9(
            **variants,
            sample_frames   = 2,
            sample_actions  = 32,
            random_select   = True,
            use_image_prompts = False,
        )

        trainable_paths = list_trainable_paths(cfg)

        print(f"\n=== {title} ===")
        print(f"Trainable parameter keys: {len(trainable_paths)}")
        import ipdb; ipdb.set_trace()
        for p in trainable_paths:
            print("  ", p)

def test_pi0_lora_model_num_params():
    def list_trainable_paths(cfg: pi0_incontext.Pi0IncontextConfig):
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
    # TODO: note, input_embedding is considered as part of gemma_2b base
    # if set "gemma_2b", "input_embedding" is trainable
    # if set "gemma_2b_lora", "input_embedding" is not trainable
    cases = {
        "(0, 0)  no-LoRA": dict(
            paligemma_variant="gemma_2b",
            action_expert_variant  ="gemma_300m",
        ), # number of keys: 52, with input_embedding
        # "(1, 0)  action-LoRA only": dict(
        #     paligemma_variant="gemma_2b",
        #     action_expert_variant  ="gemma_300m_lora",
        # ), # number of keys: 54, with input_embedding
        # "(1, 1)  action-LoRA only": dict(
        #     paligemma_variant="gemma_2b_lora",
        #     action_expert_variant  ="gemma_300m_lora",
        # ), # number of keys: 55
    }

    

    # -------------------------------------------------------------
    # 2.  Run each case and print results
    # -------------------------------------------------------------
    for title, variants in cases.items():
        cfg = pi0_incontext.Pi0IncontextConfig(
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

def test_compare_v12_v9_trainable_params():
    """
    Tests if the trainable parameters for specific v12 and v9 configurations match.
    Case 1: v12 (1, 0) action-LoRA only vs v9 (1, 0) prompt-LoRA only
    Case 2: v12 (0, 0) no-LoRA vs v9 (0, 0) no-LoRA (with gemma_2b base for v9)
    """
    # --- Case 1: LoRA comparison ---
    print("\n--- Comparing Case 1: v12 (1,0) vs v9 (1,0) ---")
    # Config for v12 case "(1, 0) action-LoRA only"
    v12_variants_lora = dict(
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m_lora",
    )
    cfg_v12_lora = pi0_incontextv12.Pi0IncontextConfigv12(
        **v12_variants_lora,
        sample_frames=2,
        sample_actions=32,
        random_select=True,
        use_image_prompts=False,
    )
    trainable_paths_v12_lora = _list_trainable_paths(cfg_v12_lora)
    print("=== Trainable Params V12 (1, 0) ===")
    print(f"Count: {len(trainable_paths_v12_lora)}")

    # Config for v9 case "(1, 0) prompt-LoRA only"
    v9_variants_lora = dict(
        paligemma_variant="gemma_2b",
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m_lora",
    )
    cfg_v9_lora = pi0_incontextv9.Pi0IncontextConfigv9(
        **v9_variants_lora,
        sample_frames=2,
        sample_actions=32,
        random_select=True,
        use_image_prompts=False,
    )
    trainable_paths_v9_lora = _list_trainable_paths(cfg_v9_lora)
    print("=== Trainable Params V9 (1, 0) ===")
    print(f"Count: {len(trainable_paths_v9_lora)}")

    # Compare Case 1
    _compare_and_assert(trainable_paths_v12_lora, trainable_paths_v9_lora, "v12 (1,0)", "v9 (1,0)")
    print("Case 1 comparison passed.")

    # --- Case 2: No-LoRA comparison ---
    print("\n--- Comparing Case 2: v12 (0,0) vs v9 (0,0) ---")
    # Config for v12 case "(0, 0) no-LoRA"
    v12_variants_no_lora = dict(
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m",
    )
    cfg_v12_no_lora = pi0_incontextv12.Pi0IncontextConfigv12(
        **v12_variants_no_lora,
        sample_frames=2,
        sample_actions=32,
        random_select=True,
        use_image_prompts=False,
    )
    trainable_paths_v12_no_lora = _list_trainable_paths(cfg_v12_no_lora)
    print("=== Trainable Params V12 (0, 0) ===")
    print(f"Count: {len(trainable_paths_v12_no_lora)}")

    # Config for v9 case "(0, 0) no-LoRA" (using gemma_2b base)
    v9_variants_no_lora = dict(
        paligemma_variant="gemma_2b", # Note: v9 needs paligemma specified
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m",
    )
    cfg_v9_no_lora = pi0_incontextv9.Pi0IncontextConfigv9(
        **v9_variants_no_lora,
        sample_frames=2,
        sample_actions=32,
        random_select=True,
        use_image_prompts=False,
    )
    trainable_paths_v9_no_lora = _list_trainable_paths(cfg_v9_no_lora)
    print("=== Trainable Params V9 (0, 0) ===")
    print(f"Count: {len(trainable_paths_v9_no_lora)}")

    # Compare Case 2
    _compare_and_assert(trainable_paths_v12_no_lora, trainable_paths_v9_no_lora, "v12 (0,0)", "v9 (0,0)")
    print("Case 2 comparison passed.")


if __name__ == "__main__":
    # test_pi0_lora_model_v7_params()
    # test_pi0_lora_model_v7_num_params()
    # test_pi0_lora_model_v7_num_params()
    # test_pi0_lora_model_v7_4_params()
    # test_pi0_lora_model_v12_num_params()
    test_pi0_lora_model_v9_num_params()
    # test_pi0_lora_model_num_params()

    # test_compare_v12_v9_trainable_params()