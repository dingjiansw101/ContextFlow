# Implementation Guide: Pi0 IncontextV17 - Hierarchical 3-Expert Model

This document explains how to extend the openpi model from 2 experts (v12) to 3 experts (v17) for hierarchical action prediction.

**For detailed explanation of how the multi-expert system works internally**, see [Gemma Multi-Expert Architecture Guide](gemma_multi_expert_architecture.md).

## Table of Contents
1. [How Current 2-Expert System Works (v12)](#how-current-2-expert-system-works-v12)
2. [Extending to 3 Experts for v17](#extending-to-3-experts-for-v17)
3. [Implementation Details](#implementation-details)
4. [Architecture Overview](#architecture-overview)

---

## How Current 2-Expert System Works (v12)

**Note**: For a deep dive into expert relationships, parameter sharing, and attention mechanics, see [gemma_multi_expert_architecture.md](gemma_multi_expert_architecture.md).

### 1. Expert Initialization

The v12 model uses **2 experts**:
- **Prompt Expert**: Processes images, text, and demonstration data
- **Action Expert**: Processes state and predicts actions

**Code Location**: `src/openpi/models/pi0_incontextv12.py:235-263`

```python
def __init__(self, config: Pi0IncontextConfigv12, rngs: nnx.Rngs):
    # Create separate configs for each expert
    action_expert_config = _gemma.get_config(config.action_expert_variant, "action_expert")
    prompt_expert_config = _gemma.get_config(config.prompt_expert_variant, "prompt_expert")

    # Pass configs as a LIST to _gemma.Module
    llm = nnx_bridge.ToNNX(
        _gemma.Module(
            configs=[prompt_expert_config, action_expert_config],  # <-- 2 configs
            embed_dtype=config.dtype,
        )
    )
    llm.lazy_init(rngs=rngs, method="init")
    # ...
```

**Key Insight**: The `_gemma.Module` class (`src/openpi/models/gemma.py:407`) accepts `configs: Sequence[Config]`, allowing multiple experts to share the same transformer layers while having different input/output projections.

### 2. Forward Pass Mechanics

When calling the model, you pass a **list of token tensors**, one for each expert (or `None` to skip):

**Training (both experts active)** - `compute_loss()` at line 608:
```python
(midfix_out, suffix_out), _ = self.PaliGemma.llm(
    [midfix_tokens, suffix_tokens],  # <-- List with 2 elements
    mask=attn_mask,
    positions=positions
)
# midfix_tokens → expert 0 (prompt expert) → midfix_out
# suffix_tokens → expert 1 (action expert) → suffix_out
```

**Inference (skip action expert initially)** - `sample_actions()` at line 638:
```python
_, kv_cache = self.PaliGemma.llm(
    [midfix_tokens, None],  # <-- None means skip expert 1
    mask=midfix_attn_mask,
    positions=positions
)
# Only compute prompt expert, cache results for later use
```

### 3. Attention Mask Structure

The attention mask is **SHARED** across all experts. The transformer concatenates tokens from all experts and applies a single attention mask.

**For a complete step-by-step explanation of how attention works with multiple experts**, see the [Attention Mechanism Deep Dive](gemma_multi_expert_architecture.md#attention-mechanism-deep-dive) section in the Gemma guide.

**Summary from `src/openpi/models/gemma.py:266-267, 298-310`:**

```python
class Attention(nn.Module):
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # 1. Compute Q, K, V for each expert separately
        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            # ... compute q, k, v for this expert
            qkvs.append((q, k, v))

        # 2. CONCATENATE Q, K, V from all active experts
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 3. Compute attention with SINGLE SHARED MASK
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)

        # 4. SPLIT outputs back to individual experts
        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)
```

**Attention Mask Shape**: `[B, T, S]` where:
- `B` = batch size
- `T` = total sequence length (sum of all expert token lengths)
- `S` = total sequence length (same as T)
- Tokens are arranged: `[expert0_tokens | expert1_tokens | ...]`

**Example from v12**:
```python
# Token layout: [image_tokens | text_tokens | state_token | action_tokens]
#               |<----- prompt expert ----->|<---- action expert ---->|

# ar_mask controls causal structure:
ar_mask = [
    False, False, ..., False,  # images: bidirectional attention
    False, False, ..., False,  # text: bidirectional attention
    True,                       # state: new causal block
    True, False, False, ...     # actions: causal within block
]

# make_attn_mask converts 1D ar_mask to 2D attention mask
attn_mask = make_attn_mask(input_mask, ar_mask)
```

**`make_attn_mask` Function** (`src/openpi/models/pi0_incontextv12.py:20-45`):
```python
def make_attn_mask(input_mask, mask_ar):
    """
    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

---

## Extending to 3 Experts for v17

### Goal: Future State Prediction + Action Generation

Create a model with **3 experts**:
1. **Prompt Expert**: Process images, text, demonstrations (same as v12)
2. **State Prediction Expert**: Predict downsampled future state trajectory
3. **Action Expert**: Predict actions conditioned on predicted future states

### Architecture Design

**Two-stage prediction with flow matching:**
1. **First stage (Planning)**: Predict future state trajectory (downsampled by factor K)
   - Learn where the robot will be in the future
   - Coarse-grained temporal planning
2. **Second stage (Control)**: Generate actions to reach predicted states
   - Conditioned on predicted future state trajectory
   - Fine-grained action generation

**Key insight**: Hierarchical decomposition of "what to achieve" (states) vs "how to achieve it" (actions).

**Loss computation**: Both experts contribute to loss
```
total_loss = state_loss_weight * loss_future_states + loss_actions
```

### Data Requirements

**Important**: This architecture requires **future state trajectories** as training labels.

**Training data format** (prepared by dataloader):
```python
{
    "observation": {
        "images": ...,
        "state": [s_t],  # Current state at time t
        ...
    },
    "actions": [a_t, a_{t+1}, ..., a_{t+H}],  # H = action_horizon (e.g., 50)
    "future_states": [s_{t+K}, s_{t+2K}, ..., s_{t+H}],  # Downsampled future states (e.g., 10 states)
}
```

**Dataloader Responsibility**: The dataloader must **downsample** future states before passing to the model:
- Model expects `future_states` with shape `[B, future_state_horizon, state_dim]` (e.g., 10 states)
- Raw trajectory has `action_horizon` states (e.g., 50 states)
- Dataloader downsamples by factor K = action_horizon // future_state_horizon (e.g., K=5)

**Downsampling strategies**:
1. **Strided sampling** (simpler):
   ```python
   # Select every Kth state
   K = action_horizon // future_state_horizon
   future_states_downsampled = future_states_raw[::K]  # [50, D] → [10, D]
   ```

2. **Average pooling** (smoother):
   ```python
   # Average over windows of size K
   K = action_horizon // future_state_horizon
   future_states_reshaped = future_states_raw.reshape(future_state_horizon, K, state_dim)
   future_states_downsampled = future_states_reshaped.mean(axis=1)  # [10, K, D] → [10, D]
   ```

**Recommendation**: Use strided sampling for simplicity unless your state trajectories are very noisy.


## Implementation Details

### 1. Config Changes

**Create new config**: `src/openpi/models/pi0_incontextv17.py`

```python
@dataclasses.dataclass(frozen=True)
class Pi0IncontextConfigv17(_model.BaseModelConfig):
    dtype: str = "bfloat16"

    # THREE expert variants
    prompt_expert_variant: _gemma.Variant = "gemma_300m_v2"
    state_expert_variant: _gemma.Variant = "gemma_300m_lora"     # NEW: Predicts future states
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"    # NEW: Predicts actions

    # Model dimensions
    action_dim: int = 32       # Action dimension
    state_dim: int = 32        # State dimension (often same as action_dim)
    action_horizon: int = 50   # Number of actions to predict
    max_token_len: int = 48

    # Future state prediction configuration (NEW)
    future_state_horizon: int = 10        # Number of future states expected from dataloader
    state_loss_weight: float = 0.5        # Weight for future state prediction loss

    # NOTE: Future states are downsampled in the dataloader, not in the model.
    # The dataloader should provide future_states with shape [B, future_state_horizon, state_dim].

    # In-context learning params (from v12)
    sample_frames: int = 16
    sample_actions: int = 32
    random_select: bool = True
    avg_current_img: bool = False
    causal_attention: bool = False

    # Note: future_state_horizon is directly specified as a config parameter
    # Typically: future_state_horizon = action_horizon // state_prediction_downsample

    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_INCONTEXT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Incontextv17":
        return Pi0Incontextv17(self, rngs=nnx.Rngs(rng))
```

### 2. Model Initialization

```python
class Pi0Incontextv17(_model.BaseModel):
    def __init__(self, config: Pi0IncontextConfigv17, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # Create THREE expert configs
        prompt_expert_config = _gemma.get_config(
            config.prompt_expert_variant, "prompt_expert"
        )
        state_expert_config = _gemma.get_config(
            config.state_expert_variant, "state_expert"  # Predicts future states
        )
        action_expert_config = _gemma.get_config(
            config.action_expert_variant, "action_expert"  # Predicts actions
        )

        # Initialize shared transformer with 3 experts
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[prompt_expert_config, state_expert_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

        # Image encoder (shared)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=prompt_expert_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Current state projection (shared)
        self.state_proj = nnx.Linear(config.state_dim, action_expert_config.width, rngs=rngs)

        # STATE EXPERT projections (predicts future states)
        self.future_state_in_proj = nnx.Linear(
            config.state_dim, state_expert_config.width, rngs=rngs
        )
        self.state_time_mlp_in = nnx.Linear(
            2 * state_expert_config.width, state_expert_config.width, rngs=rngs
        )
        self.state_time_mlp_out = nnx.Linear(
            state_expert_config.width, state_expert_config.width, rngs=rngs
        )
        self.future_state_out_proj = nnx.Linear(
            state_expert_config.width, config.state_dim, rngs=rngs
        )

        # ACTION EXPERT projections (predicts actions)
        self.action_in_proj = nnx.Linear(
            config.action_dim, action_expert_config.width, rngs=rngs
        )
        self.action_time_mlp_in = nnx.Linear(
            2 * action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_time_mlp_out = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_out_proj = nnx.Linear(
            action_expert_config.width, config.action_dim, rngs=rngs
        )

        # Future state conditioning projection (for action expert)
        self.future_state_conditioning_proj = nnx.Linear(
            config.state_dim, action_expert_config.width, rngs=rngs
        )

        # Demo projections (from v12)
        if config.use_action_state_prompts:
            self.demo_action_proj = nnx.Linear(
                config.action_dim, prompt_expert_config.width, rngs=rngs
            )
            self.demo_state_proj = nnx.Linear(
                config.action_dim, prompt_expert_config.width, rngs=rngs
            )

        # Store config
        self.config = config
```

### 3. Embedding Methods

**Reuse from v12**: `embed_midfix()`
- Embeds: demo images, demo states/actions, current images, text
- Output: midfix_tokens, midfix_mask, midfix_ar_mask
- Used by prompt expert

**NEW**: `embed_suffix_state()` (for state prediction expert)
```python
def embed_suffix_state(
    self,
    obs: _model.ObservationIncontext,
    noisy_future_states: at.Float[at.Array, "b future_horizon state_dim"],  # Noisy future state trajectory
    timestep: at.Float[at.Array, " b"]
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    """Embed current state + noisy future states + timestep for state prediction expert."""
    input_mask = []
    ar_mask = []
    tokens = []

    # Current state token
    state_token = self.state_proj(obs.state)[:, None, :]
    tokens.append(state_token)
    input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
    ar_mask += [True]  # New causal block

    # Timestep embedding
    time_emb = posemb_sincos(
        timestep,
        self.future_state_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0
    )

    # Future state + time tokens
    future_state_tokens = self.future_state_in_proj(noisy_future_states)
    time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.config.future_state_horizon)
    state_time_tokens = jnp.concatenate([future_state_tokens, time_tokens], axis=-1)
    state_time_tokens = self.state_time_mlp_in(state_time_tokens)
    state_time_tokens = nnx.swish(state_time_tokens)
    state_time_tokens = self.state_time_mlp_out(state_time_tokens)

    tokens.append(state_time_tokens)
    input_mask.append(jnp.ones(state_time_tokens.shape[:2], dtype=jnp.bool_))
    ar_mask += [True] + ([False] * (self.config.future_state_horizon - 1))  # Causal

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)

    return tokens, input_mask, ar_mask
```

**NEW**: `embed_suffix_action()` (for action prediction expert)
```python
def embed_suffix_action(
    self,
    obs: _model.ObservationIncontext,
    future_states: at.Float[at.Array, "b future_horizon state_dim"],  # Predicted future states - CONDITIONING
    noisy_actions: _model.Actions,  # [B, action_horizon, action_dim]
    timestep: at.Float[at.Array, " b"]
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    """Embed current state + predicted future states (conditioning) + noisy actions + timestep for action expert."""
    input_mask = []
    ar_mask = []
    tokens = []

    # Current state token
    state_token = self.state_proj(obs.state)[:, None, :]
    tokens.append(state_token)
    input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
    ar_mask += [True]  # New causal block

    # Future states as conditioning tokens (NO GRADIENTS during action expert training)
    future_state_conditioning = self.future_state_conditioning_proj(future_states)  # [B, future_horizon, emb]
    tokens.append(future_state_conditioning)
    input_mask.append(jnp.ones(future_state_conditioning.shape[:2], dtype=jnp.bool_))
    ar_mask += [False] * self.config.future_state_horizon  # Bidirectional within conditioning

    # Timestep embedding
    time_emb = posemb_sincos(
        timestep,
        self.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0
    )

    # Action + time tokens
    action_tokens = self.action_in_proj(noisy_actions)
    time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
    action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
    action_time_tokens = self.action_time_mlp_in(action_time_tokens)
    action_time_tokens = nnx.swish(action_time_tokens)
    action_time_tokens = self.action_time_mlp_out(action_time_tokens)

    tokens.append(action_time_tokens)
    input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
    ar_mask += [True] + ([False] * (self.action_horizon - 1))  # Causal

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)

    return tokens, input_mask, ar_mask
```

### 4. Training: compute_loss()

```python
@override
def compute_loss(
    self,
    rng: at.KeyArrayLike,
    observation: _model.ObservationIncontext,
    actions: _model.Actions,  # [B, action_horizon, action_dim]
    future_states: at.Float[at.Array, "b future_state_horizon state_dim"],  # Pre-downsampled by dataloader
    *,
    train: bool = False,
) -> at.Float[at.Array, "*b"]:
    """Compute loss for both state prediction and action generation experts.

    Args:
        future_states: Pre-downsampled future states from dataloader.
                      Shape: [B, future_state_horizon, state_dim] (NOT action_horizon!)
                      Dataloader is responsible for downsampling.
    """

    # Split RNG
    preprocess_rng, noise_rng_state, noise_rng_action, time_rng_state, time_rng_action = jax.random.split(rng, 5)

    # Preprocess observation
    observation = _model.preprocess_observation_incontext(preprocess_rng, observation, train=train)

    # === STAGE 1: Future State Prediction ===

    # future_states is already downsampled by dataloader: [B, future_state_horizon, state_dim]

    # Sample noise and time for state expert
    batch_shape = future_states.shape[:-2]
    noise_state = jax.random.normal(noise_rng_state, future_states.shape)
    time_state = jax.random.beta(time_rng_state, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_state_expanded = time_state[..., None, None]

    # Flow matching: x_t = t * noise + (1-t) * data
    x_t_state = time_state_expanded * noise_state + (1 - time_state_expanded) * future_states
    u_t_state = noise_state - future_states  # Target velocity

    # === STAGE 2: Action Prediction ===

    # Sample noise and time for action expert
    noise_action = jax.random.normal(noise_rng_action, actions.shape)
    time_action = jax.random.beta(time_rng_action, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_action_expanded = time_action[..., None, None]

    # Flow matching for actions
    x_t_action = time_action_expanded * noise_action + (1 - time_action_expanded) * actions
    u_t_action = noise_action - actions  # Target velocity

    # === FORWARD PASS: All 3 Experts ===

    # Embed midfix (prompt expert input)
    midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)

    # Embed suffix for state expert
    suffix_state_tokens, suffix_state_mask, suffix_state_ar_mask = self.embed_suffix_state(
        observation, x_t_state, time_state
    )

    # Embed suffix for action expert (conditioned on GT future states)
    suffix_action_tokens, suffix_action_mask, suffix_action_ar_mask = self.embed_suffix_action(
        observation, future_states, x_t_action, time_action
    )

    # Concatenate all masks
    input_mask = jnp.concatenate([midfix_mask, suffix_state_mask, suffix_action_mask], axis=1)
    ar_mask = jnp.concatenate([midfix_ar_mask, suffix_state_ar_mask, suffix_action_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    # Forward through all 3 experts
    (midfix_out, state_out, action_out), _ = self.PaliGemma.llm(
        [midfix_tokens, suffix_state_tokens, suffix_action_tokens],  # <-- 3 experts!
        mask=attn_mask,
        positions=positions
    )

    # === COMPUTE LOSSES ===

    # Future state prediction loss
    v_t_state = self.future_state_out_proj(state_out[:, -self.config.future_state_horizon:])
    loss_state = jnp.mean(jnp.square(v_t_state - u_t_state), axis=-1)

    # Action prediction loss
    v_t_action = self.action_out_proj(action_out[:, -self.action_horizon:])
    loss_action = jnp.mean(jnp.square(v_t_action - u_t_action), axis=-1)

    # Weighted combination
    total_loss = self.config.state_loss_weight * loss_state + loss_action

    return total_loss
```

### 5. Inference: sample_actions()

```python
@override
def sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.ObservationIncontext,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
) -> _model.Actions:
    """Sequential diffusion: predict future states → generate actions."""

    observation = _model.preprocess_observation_incontext(None, observation, train=False)
    rng_state, rng_action = jax.random.split(rng)
    batch_size = observation.state.shape[0]
    dt = -1.0 / num_steps

    # === STAGE 1: Generate Future State Trajectory ===

    # Prepare midfix embeddings (shared by all experts)
    midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)

    midfix_attn_mask = make_attn_mask(midfix_mask, midfix_ar_mask)
    positions_midfix = jnp.cumsum(midfix_mask, axis=1) - 1

    # Pre-compute KV cache for midfix (reused in all diffusion steps)
    _, kv_cache = self.PaliGemma.llm(
        [midfix_tokens, None, None],  # Only compute prompt expert
        mask=midfix_attn_mask,
        positions=positions_midfix
    )

    # Initialize with noise
    noise_state = jax.random.normal(
        rng_state,
        (batch_size, self.config.future_state_horizon, self.config.state_dim)
    )

    # Diffusion loop for future states
    def step_state(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_state(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )

        # Build attention mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        midfix_attn_mask_repeat = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([midfix_attn_mask_repeat, suffix_attn_mask], axis=-1)
        positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Forward pass with KV cache
        (_, state_out, _), _ = self.PaliGemma.llm(
            [None, suffix_tokens, None],  # Only compute state expert
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache
        )

        # Predict velocity
        v_t = self.future_state_out_proj(state_out[:, -self.config.future_state_horizon:])

        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    future_states, _ = jax.lax.while_loop(cond, step_state, (noise_state, 1.0))

    # === STAGE 2: Generate Actions (conditioned on predicted future states) ===

    # Initialize with noise
    noise_action = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))

    # Diffusion loop for actions
    def step_action(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_action(
            observation, future_states, x_t, jnp.broadcast_to(time, batch_size)
        )

        # Build attention mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        midfix_attn_mask_repeat = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([midfix_attn_mask_repeat, suffix_attn_mask], axis=-1)
        positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Forward pass with KV cache
        (_, _, action_out), _ = self.PaliGemma.llm(
            [None, None, suffix_tokens],  # Only compute action expert
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache
        )

        # Predict velocity
        v_t = self.action_out_proj(action_out[:, -self.action_horizon:])

        return x_t + dt * v_t, time + dt

    actions, _ = jax.lax.while_loop(cond, step_action, (noise_action, 1.0))

    return actions  # Final output
```

---

## Architecture Overview

### Token Layout During Training

```
Total Sequence: [Prompt Expert Tokens | State Expert Tokens | Action Expert Tokens]
                |<------- midfix ------>|<--- suffix_state --->|<---- suffix_action ---->|

Positions:      [0    ...    P-1       | P    ...    P+S-1    | P+S  ...  P+S+A-1     ]

Expert IDs:     [      0                |        1             |          2             ]
```

**Example dimensions** (action_horizon=50, state_prediction_downsample=5, future_state_horizon=10):
- Midfix length (P): ~800 tokens (images + text + demos)
- State suffix (S): 11 tokens (1 current state + 10 future states)
- Action suffix (A): 61 tokens (1 current state + 10 future state conditioning + 50 actions)
- **Total**: ~872 tokens

### Attention Pattern (ar_mask)

```python
ar_mask = [
    # Midfix (Prompt Expert): bidirectional/causal based on config
    False, False, ..., False, True, False, ..., False,  # P tokens

    # State Expert: causal (predicts future states)
    True,           # current state token (starts new causal block)
    False, ...,     # future state tokens (causal within)

    # Action Expert: causal with access to future state conditioning
    True,           # current state token (starts new causal block)
    False, ...,     # future state conditioning tokens (bidirectional)
    True,           # first action token
    False, ...      # remaining action tokens (causal)
]
```

**Attention behavior**:
- State expert attends to: midfix + current state + previous future state tokens
- Action expert attends to: midfix + current state + all future state conditioning + previous action tokens
- Prompt expert: depends on `causal_attention` config flag

### Training vs Inference

**Training**:
- Single forward pass through all 3 experts
- Ground truth future states (downsampled) used for conditioning action expert
- Both losses computed in one pass: `state_loss_weight * loss_state + loss_action`

**Inference**:
1. Pre-compute midfix + KV cache (once)
2. Run state prediction diffusion loop (10 steps) → predicted future states
3. Run action generation diffusion loop (10 steps) conditioned on predicted states

**Total inference cost**: ~21 forward passes (1 + 10 + 10)

---

## Key Implementation Notes

### 1. Independent Time Sampling

During training, state and action experts use **different timesteps**:
```python
time_state = jax.random.beta(time_rng_state, 1.5, 1, batch_shape) * 0.999 + 0.001
time_action = jax.random.beta(time_rng_action, 1.5, 1, batch_shape) * 0.999 + 0.001
```

This allows both experts to learn the full diffusion process independently - crucial because they predict different modalities (states vs actions).

### 2. Stop-Gradient for Future States

During action expert training, predicted future states are treated as **fixed conditioning**:
```python
# In embed_suffix_action():
future_state_conditioning = self.future_state_conditioning_proj(future_states)  # future_states = GT
```

No gradient flows from action expert loss to future states during training. This keeps the two experts decoupled.

### 3. Dataloader Handles Downsampling

**Important**: Future state downsampling is done in the **dataloader**, not in the model.

The model expects `future_states` to already be downsampled:
- Input shape: `[B, future_state_horizon, state_dim]` (e.g., 10 states)
- NOT: `[B, action_horizon, state_dim]` (e.g., 50 states)

**Why dataloader-side downsampling?**
- **Cleaner separation**: Data preprocessing stays in data pipeline
- **Flexibility**: Easy to switch downsampling strategies without changing model code
- **Efficiency**: Downsample once during data loading, not every training step

See [Data Requirements](#data-requirements) section above for downsampling strategies (strided sampling vs average pooling).

### 4. KV Cache Reuse

The midfix KV cache is computed once and reused:
- During state prediction diffusion loop (10 times)
- During action generation diffusion loop (10 times)

This saves significant computation vs recomputing midfix each step.

### 5. Shared vs Separate Experts

**Current design**: 3 experts share the same transformer layers but have:
- Separate input projections (future_state_in_proj vs action_in_proj)
- Separate output projections (future_state_out_proj vs action_out_proj)
- Separate attention heads (configured via expert configs)
- Different dimensions: state_dim for state expert, action_dim for action expert

**Alternative**: Could use completely separate Gemma modules, but this would be much more expensive.

---

## Freeze Filter and LoRA Configuration

The freeze filter system controls which parameters are frozen (not updated during training) and which are trainable. This is essential for:
- **LoRA fine-tuning**: Freeze base model weights, train only lightweight LoRA adapters
- **Transfer learning**: Load pretrained weights and selectively fine-tune specific experts
- **Memory efficiency**: Frozen parameters can use lower precision (bfloat16) and require less optimizer state

### Expert Parameter Naming Convention

The `_namev2` function in `src/openpi/models/gemma.py` generates unique parameter names for each expert:

```python
def _namev2(name, i, expert_names=["paligemma", "action_expert"]):
    assert expert_names[i] in ["paligemma", "action_expert", "prompt_expert", "state_expert"]
    if expert_names[i] == "paligemma":
        return name  # No suffix
    elif expert_names[i] == "action_expert":
        return f"{name}_1"  # Special: uses "_1" for backward compatibility
    return f"{name}_{expert_names[i]}"  # Descriptive suffix
```

**For v17 (3 experts):**

| Expert Index | expert_name in Config | Parameter Suffix | Example Parameter Path |
|--------------|----------------------|------------------|------------------------|
| 0 | `"prompt_expert"` | `_prompt_expert` | `PaliGemma/llm/layers/0/attn/qkv_einsum_prompt_expert/kernel` |
| 1 | `"state_expert"` | `_state_expert` | `PaliGemma/llm/layers/0/attn/qkv_einsum_state_expert/kernel` |
| 2 | `"action_expert"` | `_1` | `PaliGemma/llm/layers/0/attn/qkv_einsum_1/kernel` |

**Why action expert uses `_1`**: This preserves backward compatibility with existing checkpoints from v12 (2-expert model). When loading pretrained action expert weights, the parameter names must match exactly. Using `_1` allows seamless loading of v12 action expert into v17.

**Other components** (not expert-specific):
- Vision encoder: `PaliGemma/img/...` (SigLIP parameters)
- Embedder: `PaliGemma/llm/embedder/...` (shared across experts)
- Projection layers: `state_proj/kernel`, `action_in_proj/kernel`, etc.

### Default Configuration

**Recommended default for v17:**

```python
@dataclasses.dataclass(frozen=True)
class Pi0IncontextConfigv17(_model.BaseModelConfig):
    dtype: str = "bfloat16"

    # Default expert variants
    prompt_expert_variant: _gemma.Variant = "gemma_300m_v2"      # No LoRA → Fully trainable
    state_expert_variant: _gemma.Variant = "gemma_300m_lora"     # LoRA → Base frozen, LoRA trainable
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"    # LoRA → Base frozen, LoRA trainable

    # ... other config fields ...
```

**Expected freeze behavior with defaults:**
- ✅ Vision encoder (SigLIP): **Trainable**
- ✅ Prompt expert: **Fully trainable** (no LoRA)
- ❌ State expert base: **Frozen**
- ✅ State expert LoRA adapters: **Trainable**
- ❌ Action expert base: **Frozen**
- ✅ Action expert LoRA adapters: **Trainable**
- ✅ All projection layers: **Trainable** (state_proj, action_in_proj, etc.)

### Complete Implementation

```python
def get_freeze_filter(self) -> nnx.filterlib.Filter:
    """Returns the freeze filter for 3-expert architecture.

    Vision encoder is TRAINABLE by default (not frozen).

    Freeze Policy:
    - Vision encoder (img): trainable (no freeze)
    - Expert with LoRA: freeze base weights, train LoRA adapters
    - Expert without LoRA: fully trainable (no freeze)
    - Projection layers: always trainable (no freeze)

    With default config (prompt=no lora, state=lora, action=lora):
      - Freezes: state_expert base, action_expert base
      - Trains: vision, prompt_expert, state LoRA, action LoRA, projections
    """
    filters = []

    # Define regex filters for each component
    prompt_expert_filter = nnx_utils.PathRegex(".*llm.*_prompt_expert.*")
    state_expert_filter = nnx_utils.PathRegex(".*llm.*_state_expert.*")
    action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")
    lora_filter = nnx_utils.PathRegex(".*lora.*")

    # Check which experts use LoRA
    prompt_has_lora = "lora" in self.prompt_expert_variant
    state_has_lora = "lora" in self.state_expert_variant
    action_has_lora = "lora" in self.action_expert_variant

    # Case 1: No LoRA anywhere - train everything (including vision)
    if not (prompt_has_lora or state_has_lora or action_has_lora):
        return nnx.Nothing

    # Case 2+: At least one expert uses LoRA

    # Freeze base weights of LoRA experts
    if prompt_has_lora:
        filters.append(prompt_expert_filter)
    if state_has_lora:
        filters.append(state_expert_filter)
    if action_has_lora:
        filters.append(action_expert_filter)

    # Unfreeze LoRA adapters (they should be trainable)
    filters.append(nnx.Not(lora_filter))

    # Unfreeze non-LoRA experts (they should be fully trainable)
    if not prompt_has_lora:
        filters.append(nnx.Not(prompt_expert_filter))
    if not state_has_lora:
        filters.append(nnx.Not(state_expert_filter))
    if not action_has_lora:
        filters.append(nnx.Not(action_expert_filter))

    # Combine all filters with logical AND
    return nnx.All(*filters)
```

**How it works**: The freeze filter uses regex pattern matching on parameter paths. A parameter is frozen if:
1. It matches an expert base filter (e.g., `.*llm.*_state_expert.*`), AND
2. It does NOT match the LoRA filter (e.g., NOT `.*lora.*`), AND
3. It does NOT match a non-LoRA expert exclusion

### All Configuration Scenarios

The table below shows freeze behavior for all 8 combinations of LoRA usage (2^3):

| Prompt | State | Action | Vision | Prompt Expert | State Base | State LoRA | Action Base | Action LoRA |
|--------|-------|--------|--------|---------------|------------|------------|-------------|-------------|
| No LoRA | No LoRA | No LoRA | ✅ Train | ✅ Train | ✅ Train | N/A | ✅ Train | N/A |
| **No LoRA** | **LoRA** | **LoRA** | **✅ Train** | **✅ Train** | **❌ Freeze** | **✅ Train** | **❌ Freeze** | **✅ Train** |
| LoRA | No LoRA | No LoRA | ✅ Train | ❌ Freeze | ✅ Train | ✅ Train | ✅ Train | N/A |
| LoRA | LoRA | No LoRA | ✅ Train | ❌ Freeze | ❌ Freeze | ✅ Train | ✅ Train | N/A |
| LoRA | No LoRA | LoRA | ✅ Train | ❌ Freeze | ✅ Train | ✅ Train | ❌ Freeze | ✅ Train |
| No LoRA | LoRA | No LoRA | ✅ Train | ✅ Train | ❌ Freeze | ✅ Train | ✅ Train | N/A |
| No LoRA | No LoRA | LoRA | ✅ Train | ✅ Train | ✅ Train | N/A | ❌ Freeze | ✅ Train |
| LoRA | LoRA | LoRA | ✅ Train | ❌ Freeze | ❌ Freeze | ✅ Train | ❌ Freeze | ✅ Train |

**(Bold row = recommended default configuration)**

**Use cases:**
- **Row 1**: Full fine-tuning from scratch or with pretrained base model
- **Row 2** (default): Train prompt expert fully, use LoRA for state/action (efficient hierarchical learning)
- **Row 8**: Maximum efficiency - only train LoRA adapters across all experts

### Optional: Vision Encoder Freezing Control

If you want the ability to optionally freeze the vision encoder, add a config flag:

```python
@dataclasses.dataclass(frozen=True)
class Pi0IncontextConfigv17(_model.BaseModelConfig):
    # ... other fields ...

    freeze_vision_encoder: bool = False  # Set to True to freeze vision encoder

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter for 3-expert architecture."""
        filters = []

        # Define filters
        vision_filter = nnx_utils.PathRegex(".*img.*")
        prompt_expert_filter = nnx_utils.PathRegex(".*llm.*_prompt_expert.*")
        state_expert_filter = nnx_utils.PathRegex(".*llm.*_state_expert.*")
        action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        lora_filter = nnx_utils.PathRegex(".*lora.*")

        # Check which experts use LoRA
        prompt_has_lora = "lora" in self.prompt_expert_variant
        state_has_lora = "lora" in self.state_expert_variant
        action_has_lora = "lora" in self.action_expert_variant

        # Optionally freeze vision encoder
        if self.freeze_vision_encoder:
            filters.append(vision_filter)

        # No LoRA anywhere and vision not frozen - train everything
        if not (prompt_has_lora or state_has_lora or action_has_lora) and not self.freeze_vision_encoder:
            return nnx.Nothing

        # ... rest same as before (freeze LoRA expert bases, unfreeze adapters, etc.) ...
```

**Usage:**
```python
# Trainable vision (default)
config = Pi0IncontextConfigv17()

# Frozen vision (typical for transfer learning)
config = Pi0IncontextConfigv17(freeze_vision_encoder=True)
```

### Testing the Freeze Filter

Verify your freeze filter is working correctly:

```python
from openpi.models import pi0_incontextv17
import flax.nnx as nnx
import jax

# Create model with your config
config = pi0_incontextv17.Pi0IncontextConfigv17(
    prompt_expert_variant="gemma_300m_v2",      # No LoRA
    state_expert_variant="gemma_300m_lora",     # LoRA
    action_expert_variant="gemma_300m_lora",    # LoRA
)
model = config.create(jax.random.PRNGKey(0))

# Get filters
freeze_filter = config.get_freeze_filter()
trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))

# Extract parameter states
params = nnx.state(model)
frozen_params = params.filter(freeze_filter)
trainable_params = params.filter(trainable_filter)

# Print frozen parameters
print("=== FROZEN PARAMETERS ===")
frozen_count = 0
for path in sorted(frozen_params.flat_state().keys()):
    print(f"  ❌ {path}")
    frozen_count += 1

# Print trainable parameters (with categorization)
print("\n=== TRAINABLE PARAMETERS ===")
trainable_count = 0
for path in sorted(trainable_params.flat_state().keys()):
    if "img" in path:
        print(f"  ✅ {path}  <- VISION ENCODER")
    elif "lora" in path:
        print(f"  ✅ {path}  <- LoRA ADAPTER")
    elif "prompt_expert" in path:
        print(f"  ✅ {path}  <- PROMPT EXPERT (full)")
    elif "state_proj" in path or "action" in path:
        print(f"  ✅ {path}  <- PROJECTION LAYER")
    else:
        print(f"  ✅ {path}")
    trainable_count += 1

print(f"\nTotal frozen: {frozen_count}")
print(f"Total trainable: {trainable_count}")
print(f"Frozen ratio: {frozen_count / (frozen_count + trainable_count):.2%}")
```

**Expected output with default config:**
```
=== FROZEN PARAMETERS ===
  ❌ PaliGemma/llm/layers/0/attn/qkv_einsum_1/kernel
  ❌ PaliGemma/llm/layers/0/attn/qkv_einsum_state_expert/kernel
  ❌ PaliGemma/llm/layers/0/mlp_1/gating/kernel
  ❌ PaliGemma/llm/layers/0/mlp_state_expert/gating/kernel
  ... (state and action expert base weights)

=== TRAINABLE PARAMETERS ===
  ✅ PaliGemma/img/Transformer/encoderblock_0/...  <- VISION ENCODER
  ✅ PaliGemma/llm/layers/0/attn/qkv_einsum_1/lora_down/kernel  <- LoRA ADAPTER
  ✅ PaliGemma/llm/layers/0/attn/qkv_einsum_state_expert/lora_up/kernel  <- LoRA ADAPTER
  ✅ PaliGemma/llm/layers/0/attn/qkv_einsum_prompt_expert/kernel  <- PROMPT EXPERT (full)
  ✅ state_proj/kernel  <- PROJECTION LAYER
  ✅ action_in_proj/kernel  <- PROJECTION LAYER
  ... (prompt expert, LoRA adapters, projections)

Total frozen: 1234
Total trainable: 567
Frozen ratio: 68.52%
```

### Key Takeaways

1. **Vision encoder is trainable by default** - Different from typical transfer learning practices, but allows fine-tuning vision features for robotics tasks
2. **Action expert uses `_1` suffix** - Preserves backward compatibility for checkpoint loading
3. **LoRA reduces trainable parameters by ~90%** - Only adapter weights are trained, base model frozen
4. **Flexible configuration** - Independently control LoRA usage for each of 3 experts
5. **Projection layers always trainable** - Task-specific projections (state_proj, action_in_proj, etc.) are never frozen
6. **Testing is important** - Always verify freeze filter behavior before training to avoid surprises

---

## Preventing Action Tokens from Attending to State Tokens

By default in v17, **action tokens CAN attend to state expert tokens** during training because all experts share the same attention mechanism and tokens are concatenated: `[midfix | state_expert | action_expert]`.

The `make_attn_mask` function allows token i to attend to token j when `cumsum[j] <= cumsum[i]`. Since action tokens have a higher cumsum value (C_P + 3) than state tokens (C_P + 1), action tokens can attend to state tokens.

### Why You Might Want to Block This

1. **Prevent information leakage**: During training, state expert is learning to predict future states. If action tokens attend to state expert's intermediate representations, it creates a dependency that doesn't exist during inference.
2. **Training-testing mismatch**: At test time, state expert runs first, then action expert. Making them independent during training better matches inference.
3. **Cleaner hierarchical separation**: Enforces that action expert only depends on state expert's final outputs (via conditioning), not internal processing.

### Option 1: Custom Attention Mask Modification (Recommended)

Modify the attention mask after `make_attn_mask` to explicitly block action→state attention:

**In `compute_loss()` method**:

```python
@override
def compute_loss(
    self,
    rng: at.KeyArrayLike,
    observation: _model.ObservationIncontext,
    actions: _model.Actions,
    future_states: at.Float[at.Array, "b action_horizon state_dim"],
    *,
    train: bool = False,
) -> at.Float[at.Array, "*b"]:
    # ... existing preprocessing code ...

    # Embed all three experts
    midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)

    suffix_state_tokens, suffix_state_mask, suffix_state_ar_mask = self.embed_suffix_state(
        observation, x_t_state, time_state
    )

    suffix_action_tokens, suffix_action_mask, suffix_action_ar_mask = self.embed_suffix_action(
        observation, future_states, x_t_action, time_action
    )

    # Build base attention mask
    input_mask = jnp.concatenate([midfix_mask, suffix_state_mask, suffix_action_mask], axis=1)
    ar_mask = jnp.concatenate([midfix_ar_mask, suffix_state_ar_mask, suffix_action_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)

    # === NEW: Block action tokens from attending to state expert tokens ===
    midfix_len = midfix_mask.shape[1]
    state_len = suffix_state_mask.shape[1]

    state_start = midfix_len
    state_end = midfix_len + state_len
    action_start = midfix_len + state_len
    # action_end = midfix_len + state_len + action_len  # Not needed

    # Set attn_mask[action_positions, state_positions] = False
    # Shape: [B, T, S] - for all action tokens (dim 1), block state tokens (dim 2)
    attn_mask = attn_mask.at[:, action_start:, state_start:state_end].set(False)

    # Continue with forward pass
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (midfix_out, state_out, action_out), _ = self.PaliGemma.llm(
        [midfix_tokens, suffix_state_tokens, suffix_action_tokens],
        mask=attn_mask,
        positions=positions
    )

    # ... rest of loss computation ...
```

**In `sample_actions()` method** (action generation loop):

```python
def step_action(carry):
    x_t, time = carry
    suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_action(
        observation, future_states, x_t, jnp.broadcast_to(time, batch_size)
    )

    # Build attention mask
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    midfix_attn_mask_repeat = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([midfix_attn_mask_repeat, suffix_attn_mask], axis=-1)

    # NOTE: No need to block state tokens here because we skip state expert (None)
    # The attention mask only spans [midfix | action], so state tokens aren't present

    positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    (_, _, action_out), _ = self.PaliGemma.llm(
        [None, None, suffix_tokens],  # Only action expert
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache
    )

    v_t = self.action_out_proj(action_out[:, -self.action_horizon:])
    return x_t + dt * v_t, time + dt
```

**Implementation Notes**:
- The blocking only needs to happen in `compute_loss()` where all 3 experts run together
- During inference (`sample_actions()`), state and action experts run separately, so no blocking is needed
- This approach adds minimal overhead (just one `.at[].set()` operation)

### Training-Testing Mismatch Warning

⚠️ **Important**: Even with attention blocking, there's a fundamental mismatch between training and inference:

**Training**: Action expert is conditioned on **ground truth future states** (clean, accurate)
```python
suffix_action_tokens, _, _ = self.embed_suffix_action(
    observation,
    future_states,  # GT states from dataloader (pre-downsampled)
    x_t_action,
    time_action
)
```

**Inference**: Action expert is conditioned on **predicted future states** (noisy, potentially inaccurate)
```python
# future_states comes from state expert's diffusion loop
suffix_tokens, _, _ = self.embed_suffix_action(
    observation,
    future_states,  # Predicted by state expert
    x_t,
    time
)
```

This distribution shift can hurt performance because the action expert never learns to handle imperfect state predictions during training.

### Possible Solutions

#### Solution 1: Add Noise to Ground Truth States During Training

Add Gaussian noise to ground truth future states during training to simulate the prediction errors that will occur during inference. This helps the action expert learn to be robust to imperfect state predictions.

**Pros**: Simple, no architectural changes, minimal overhead
**Cons**: Requires tuning noise scale; may not match actual prediction error distribution

#### Solution 2: Two-Stage Training

First, train both experts with GT states (stage 1). Then, fine-tune the action expert using predicted states from the state expert (stage 2), with stop-gradient to prevent backprop through the state expert.

**Pros**: Action expert learns to handle real prediction errors
**Cons**: Two training runs required; more complex pipeline

#### Solution 3: Random State Conditioning Dropout

During training, randomly use predicted states (from state expert) or GT states with probability p (e.g., 30%). This exposes the action expert to both clean and noisy state conditioning in a single training run.

**Pros**: Single training run; action expert sees both GT and predicted states
**Cons**: Slower training (extra forward passes for state prediction); requires tuning dropout rate

### Recommendation

Start with **Solution 1 (add noise)** because it's simplest and has minimal overhead. If performance is still poor, try **Solution 3 (random dropout)** to expose the action expert to real prediction errors.

---

## Testing Checklist

- [ ] Config creation with 3 expert variants (prompt, state, action)
- [ ] Model initialization (verify 3 experts in llm.configs)
- [ ] Forward pass with dummy data (observation, actions, future_states)
- [ ] Loss computation (verify both state and action losses contribute)
- [ ] Inference on single observation (check both stages)
- [ ] Gradient flow (verify action loss doesn't affect state expert weights)
- [ ] Checkpoint saving/loading
- [ ] Integration with training config system
- [ ] Normalization of states vs actions (different statistics)
- [ ] State downsampling correctness (temporal alignment)
- [ ] Future state trajectory visualization
- [ ] **Attention mask verification** (if blocking action→state attention):
  - [ ] Verify `attn_mask[:, action_start:, state_start:state_end]` is all False during training
  - [ ] Check action expert outputs don't change when state expert tokens are modified
  - [ ] Confirm no gradient flows from action expert to state expert through attention
- [ ] **Training-testing mismatch mitigation** (if applicable):
  - [ ] Test with noisy GT states (Solution 1)
  - [ ] Verify two-stage training pipeline (Solution 2)
  - [ ] Validate random conditioning dropout (Solution 3)
  - [ ] Compare action expert performance with GT vs predicted state conditioning

---

## Future Extensions

1. **Learnable state interpolation**: Add a learned module to upsample/interpolate future states to full action_horizon resolution
2. **Multi-scale state hierarchy**: Extend to 4+ experts with multiple temporal resolutions (very coarse → coarse → fine states)
3. **Separate diffusion schedules**: Use different noise schedules for state prediction vs action generation
4. **Cross-expert attention**: Allow action expert to attend to state expert's internal representations (not just final outputs)
5. **Residual action prediction**: Action expert predicts actions as residuals on top of a dynamics model applied to predicted states
6. **State trajectory constraints**: Add physics-based constraints or priors to the state prediction expert
7. **Adaptive downsampling**: Learn the temporal downsampling factor instead of fixing it
8. **Multi-modal state prediction**: Predict multiple possible future state trajectories (mixture of distributions)

---

## References

- **[Gemma Multi-Expert Architecture Guide](gemma_multi_expert_architecture.md)** - Detailed explanation of how experts share parameters and attention
- v12 implementation: `src/openpi/models/pi0_incontextv12.py`
- Gemma multi-expert: `src/openpi/models/gemma.py`
- Flow matching paper: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Attention masks: `make_attn_mask()` in v12
