# Gemma Multi-Expert Architecture Guide

This document explains how the multi-expert system works in `_gemma.Module` (`src/openpi/models/gemma.py`). Understanding this is crucial for implementing models with multiple experts like Pi0 IncontextV12, V17, etc.

## Table of Contents
1. [Overview](#overview)
2. [Shared vs Separate Components](#shared-vs-separate-components)
3. [Attention Mechanism Deep Dive](#attention-mechanism-deep-dive)
4. [Parameter Naming Convention](#parameter-naming-convention)
5. [Information Flow Visualization](#information-flow-visualization)
6. [Constraints and Flexibility](#constraints-and-flexibility)
7. [Memory and Computation Analysis](#memory-and-computation-analysis)
8. [Practical Implications](#practical-implications)

---

## Overview

The `_gemma.Module` class (`src/openpi/models/gemma.py:407`) implements a transformer that can have **multiple experts** sharing the same layer structure but with mostly separate parameters.

**Key Design Principle**: Experts share the **attention computation** (after projecting to Q/K/V) but have **separate parameter matrices** for projections, norms, and feed-forward layers.

**Why this design?**
- Allows experts to "attend to each other's tokens" through shared attention
- Each expert can specialize (different dimensions, LoRA configs)
- More parameter-efficient than N completely separate models
- Enables cross-expert information flow controlled by attention masks

---

## Shared vs Separate Components

### 1. SHARED Components (All Experts Use Same Weights)

**Only ONE component is truly shared across experts:**

#### Embedder (Token → Embedding)
**Location**: `src/openpi/models/gemma.py:195-216, 422-425`

```python
self.embedder = Embedder(
    vocab_size=self.voc_size,
    embed_dim=self.configs[0].width,  # Only uses expert 0's width
    name="embedder",
)
```

- **Shared**: Single embedder for all experts
- **Limitation**: Only uses expert 0's embedding dimension
- **Note**: In Pi0 models, this embedder is rarely used because inputs come from vision encoder + MLP projections, not token IDs

#### Transformer Depth (Must Match)
**Location**: `src/openpi/models/gemma.py:419`

```python
# All experts must have the same depth
assert all(config.depth == self.configs[0].depth for config in self.configs)
```

- All experts go through the **same number of layers**
- However, within each layer, experts have **separate parameters**

### 2. SEPARATE Components (Each Expert Has Own Weights)

Within each transformer Block (`src/openpi/models/gemma.py:349-400`), every expert maintains:

#### A. RMSNorm Layers (3 per layer per expert)

**Pre-Attention Norm** (lines 369-373):
```python
pre_attn = []
for i, x in enumerate(xs):
    if x is not None:
        x = RMSNorm(name=_name("pre_attention_norm", i))(x)  # Separate!
    pre_attn.append(x)
```

**Pre-FFN Norm** (lines 383-391):
```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        x = RMSNorm(name=_name("pre_ffw_norm", i))(x)  # Separate!
```

**Final Norm** (line 448):
```python
self.final_norms = [
    RMSNorm(name=_name("final_norm", i))
    for i in range(len(self.configs))
]
```

**Result**: Each expert learns its own normalization statistics.

#### B. Attention Projections (Q, K, V, and Output)

**Q/K/V Projections** (lines 238-264):
```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is None:
        continue

    # Separate Q/K/V projection for each expert
    if config.num_kv_heads == config.num_heads:
        qkv_einsum = lora.Einsum(
            shape=(3, config.num_heads, config.width, config.head_dim),
            name=_name("qkv_einsum", i),  # <-- Separate per expert!
            ...
        )
        qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
    else:
        # Separate Q and KV einsums for MQA/GQA
        q_einsum = lora.Einsum(
            shape=(config.num_heads, config.width, config.head_dim),
            name=_name("q_einsum", i),  # <-- Separate!
            ...
        )
        kv_einsum = lora.Einsum(
            shape=(2, config.num_kv_heads, config.width, config.head_dim),
            name=_name("kv_einsum", i),  # <-- Separate!
            ...
        )
```

**Attention Output Projection** (lines 298-312):
```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        out_einsum = lora.Einsum(
            shape=(config.num_heads, config.head_dim, config.width),
            name=_name("attn_vec_einsum", i),  # <-- Separate per expert!
            ...
        )
        out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
```

**Result**: Each expert can have different input/output dimensions (config.width) for attention.

#### C. Feed-Forward Networks (MLP)

**Location**: Lines 383-392

```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        x = lora.FeedForward(
            features=config.width,      # Can differ per expert!
            hidden_dim=config.mlp_dim,  # Can differ per expert!
            name=_name("mlp", i),       # <-- Separate per expert!
            lora_config=config.lora_configs.get("ffn"),
        )(x)
    out.append(x)
```

**Result**: Each expert has completely independent feed-forward layers with potentially different dimensions.

---

## Attention Mechanism Deep Dive

This is the **most important** part to understand. Attention has both **shared** and **separate** components.

### Step-by-Step Attention Process

**Location**: `src/openpi/models/gemma.py:219-314` (Attention class)

#### Step 1: Compute Q, K, V (SEPARATE)

Each expert projects its tokens to Q, K, V using its own parameters:

```python
qkvs = []
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is None:
        continue
    # Each expert uses its own projection matrices
    qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
```

At this point:
- Expert 0: `q_0 [B, T_0, N, H]`, `k_0 [B, T_0, K, H]`, `v_0 [B, T_0, K, H]`
- Expert 1: `q_1 [B, T_1, N, H]`, `k_1 [B, T_1, K, H]`, `v_1 [B, T_1, K, H]`
- Expert 2: `q_2 [B, T_2, N, H]`, `k_2 [B, T_2, K, H]`, `v_2 [B, T_2, K, H]`

#### Step 2: Concatenate Q, K, V (SHARED)

**Line 266**:
```python
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

Now we have:
- `q [B, T_0+T_1+T_2, N, H]` - concatenated along sequence dimension
- `k [B, T_0+T_1+T_2, K, H]`
- `v [B, T_0+T_1+T_2, K, H]`

**Critical insight**: Tokens from all experts are now in a single sequence!

#### Step 3: Apply RoPE (SHARED)

**Lines 268-271**:
```python
q = _apply_rope(q, positions=positions)
q *= self.configs[0].head_dim ** -0.5
k = _apply_rope(k, positions=positions)
```

Positional encoding is applied to the concatenated sequence.

#### Step 4: Compute Attention (SHARED)

**Lines 281-295**:
```python
q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

# Apply attention mask
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

# Softmax and weighted sum
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
```

**Key point**:
- Attention is computed over **all tokens from all experts**
- The attention mask controls which tokens can attend to which
- This is where cross-expert information flow happens!

#### Step 5: Split and Project Output (SEPARATE)

**Lines 298-312**:
```python
out = []
start = 0
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        end = start + x.shape[1]
        # Extract this expert's portion of the encoded sequence
        out_einsum = lora.Einsum(
            shape=(config.num_heads, config.head_dim, config.width),
            name=_name("attn_vec_einsum", i),
            ...
        )
        # Project back to this expert's dimension
        out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
        start = end
    else:
        out.append(None)
```

Each expert:
1. Extracts its portion of the attended sequence
2. Projects back to its own dimension using separate output projection

### What This Means

**Expert 0's tokens can attend to Expert 1's tokens** (and vice versa), controlled by the attention mask!

Example with 2 experts (prompt + action):
```
Sequence: [img₁ img₂ text₁ text₂ | state action₁ action₂]
          |<--- prompt expert --->|<--- action expert --->|

Attention mask can allow:
- Action tokens attend to image tokens ✓
- Action tokens attend to text tokens ✓
- Action tokens attend to state token ✓
- Action tokens attend causally to previous action tokens ✓
```

---

## Parameter Naming Convention

**Function**: `_namev2` at line 512-519

```python
def _namev2(name, i, expert_names=["paligemma", "action_expert"]):
    """
    Creates parameter names for multi-expert models.

    Args:
        name: Base parameter name (e.g., "qkv_einsum")
        i: Expert index
        expert_names: List of expert names

    Returns:
        Formatted parameter name
    """
    assert expert_names[i] in ["paligemma", "action_expert", "prompt_expert"]

    if expert_names[i] == "paligemma":
        return name                      # No suffix for expert 0
    elif expert_names[i] == "action_expert":
        return f"{name}_{1}"             # Legacy: always use "_1"
    return f"{name}_{expert_names[i]}"   # Use expert name as suffix
```

### Naming Examples

For 3 experts: `["prompt_expert", "action_expert", "coarse_expert"]`

**Layer 0 parameters:**
```
# Expert 0 (prompt_expert)
pre_attention_norm
qkv_einsum
attn_vec_einsum
mlp/gating_einsum
mlp/linear

# Expert 1 (action_expert) - legacy format
pre_attention_norm_1
qkv_einsum_1
attn_vec_einsum_1
mlp_1/gating_einsum
mlp_1/linear

# Expert 2 (coarse_expert)
pre_attention_norm_coarse_expert
qkv_einsum_coarse_expert
attn_vec_einsum_coarse_expert
mlp_coarse_expert/gating_einsum
mlp_coarse_expert/linear
```

### Why This Matters

**Checkpoint compatibility**:
- Expert 0 ("paligemma") matches PaliGemma pre-trained weights
- Expert 1 ("action_expert") matches Pi0 pre-trained action expert
- New experts get unique names that won't conflict

**For v17**: You'll likely use:
```python
configs=[
    _gemma.get_config("gemma_300m_v2", "prompt_expert"),
    _gemma.get_config("gemma_300m", "action_expert"),      # Keep legacy name
    _gemma.get_config("gemma_300m", "coarse_expert"),      # New name
]
```

---

## Information Flow Visualization

### Forward Pass Through One Transformer Layer

```
INPUT:  xs = [tokens_0, tokens_1, tokens_2]
        shapes: [B, T₀, D₀], [B, T₁, D₁], [B, T₂, D₂]

┌─────────────────────────────────────────────────────────┐
│ 1. Pre-Attention Normalization (SEPARATE)              │
└─────────────────────────────────────────────────────────┘
        ↓            ↓            ↓
   [norm_0]      [norm_1]      [norm_2]      ← Separate RMSNorm per expert

┌─────────────────────────────────────────────────────────┐
│ 2. Q/K/V Projection (SEPARATE)                         │
└─────────────────────────────────────────────────────────┘
        ↓            ↓            ↓
   [Q₀,K₀,V₀]   [Q₁,K₁,V₁]   [Q₂,K₂,V₂]    ← Separate projections
   [B,T₀,N,H]   [B,T₁,N,H]   [B,T₂,N,H]

┌─────────────────────────────────────────────────────────┐
│ 3. Concatenate (SHARED)                                │
└─────────────────────────────────────────────────────────┘
        └────────────┴────────────┘
                     ↓
        Q = concat([Q₀, Q₁, Q₂], axis=1)  → [B, T₀+T₁+T₂, N, H]
        K = concat([K₀, K₁, K₂], axis=1)  → [B, T₀+T₁+T₂, K, H]
        V = concat([V₀, V₁, V₂], axis=1)  → [B, T₀+T₁+T₂, K, H]

┌─────────────────────────────────────────────────────────┐
│ 4. Attention Computation (SHARED)                      │
│    - RoPE positional encoding                          │
│    - Compute logits: Q @ K^T                           │
│    - Apply attention mask                              │
│    - Softmax                                           │
│    - Weighted sum: softmax @ V                         │
└─────────────────────────────────────────────────────────┘
                     ↓
        encoded [B, T₀+T₁+T₂, N, H]

        ↓ (all tokens can attend to each other, per mask)

┌─────────────────────────────────────────────────────────┐
│ 5. Split and Output Projection (SEPARATE)              │
└─────────────────────────────────────────────────────────┘
        ┌────────────┴────────────┐
        ↓            ↓            ↓
   encoded₀      encoded₁      encoded₂    ← Split by sequence length
   [:,:T₀,:]     [:,T₀:T₀+T₁,:] [:,T₀+T₁:,:]
        ↓            ↓            ↓
   [out_proj₀]  [out_proj₁]  [out_proj₂]   ← Separate output projections
   [B,T₀,D₀]    [B,T₁,D₁]    [B,T₂,D₂]

┌─────────────────────────────────────────────────────────┐
│ 6. Residual Connection                                 │
└─────────────────────────────────────────────────────────┘
        ↓            ↓            ↓
   tokens_0 +   tokens_1 +   tokens_2 +    ← Element-wise add
   out_0        out_1        out_2

┌─────────────────────────────────────────────────────────┐
│ 7. Pre-FFN Normalization (SEPARATE)                    │
└─────────────────────────────────────────────────────────┘
        ↓            ↓            ↓
   [norm_0]      [norm_1]      [norm_2]      ← Separate RMSNorm

┌─────────────────────────────────────────────────────────┐
│ 8. Feed-Forward (SEPARATE)                             │
└─────────────────────────────────────────────────────────┘
        ↓            ↓            ↓
   [mlp_0]       [mlp_1]       [mlp_2]       ← Separate FFN
   width=D₀      width=D₁      width=D₂      ← Can differ!
   mlp_dim=...   mlp_dim=...   mlp_dim=...

┌─────────────────────────────────────────────────────────┐
│ 9. Residual Connection                                 │
└─────────────────────────────────────────────────────────┘
        ↓            ↓            ↓
OUTPUT: [tokens_0', tokens_1', tokens_2']
```

### Key Observations

1. **Most computation is separate**: Only the attention softmax operates on concatenated tokens
2. **Cross-expert information flow**: Happens during attention computation
3. **Dimension flexibility**: Each expert can have different D (width) but must share N, K, H (heads)
4. **Masking is critical**: Controls which tokens attend to which, enabling causal/non-causal patterns

---

## Constraints and Flexibility

### MUST BE SAME Across All Experts

**Location**: Lines 227-229, 419

```python
# In Attention.__call__:
assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

# In Module.setup:
assert all(config.depth == self.configs[0].depth for config in self.configs)
```

**Why?**
- Q/K/V are concatenated along sequence dimension, not head dimension
- Attention computation assumes uniform head structure
- All experts must go through same number of layers

### CAN BE DIFFERENT Per Expert

| Parameter | Location | Notes |
|-----------|----------|-------|
| `width` | Config | Model dimension (D), can differ per expert |
| `mlp_dim` | Config | FFN hidden dimension, can differ per expert |
| `lora_configs` | Config | LoRA settings, can differ per expert |
| Input sequence length | Runtime | Expert 0 can have 100 tokens, Expert 1 can have 50 tokens |

**Example valid config**:
```python
configs = [
    Config(width=2048, depth=18, mlp_dim=8192,  num_heads=8, ...),  # Large prompt expert
    Config(width=1024, depth=18, mlp_dim=4096,  num_heads=8, ...),  # Smaller coarse expert
    Config(width=1024, depth=18, mlp_dim=4096,  num_heads=8, ...),  # Smaller fine expert
]
```

---

## Memory and Computation Analysis

### Parameter Count

For `gemma_300m` variant (width=1024, depth=18, mlp_dim=4096, num_heads=8):

**Per expert, per layer (~17M params/layer):**
- Q/K/V projections: `3 × (1024 × 8 × 256) ≈ 6.3M`
- Attention output: `(8 × 256 × 1024) ≈ 2.1M`
- FFN: `(1024 × 4096) + (4096 × 1024) ≈ 8.4M`
- RMSNorm (3×): `3 × 1024 ≈ 0.003M`

**Total per expert**: `18 layers × 17M ≈ 311M parameters`

**For 3 experts**: `3 × 311M ≈ 933M parameters`

**Plus shared**:
- Embedder: `257152 × 1024 ≈ 263M` (if used, typically frozen)

### Computation Per Forward Pass

**Assume**:
- Prompt expert: 800 tokens
- Coarse expert: 11 tokens
- Fine expert: 51 tokens
- **Total concatenated sequence**: 862 tokens

**Per layer computation**:

1. **Q/K/V projections** (separate): `O(T × D² × num_experts)`
   - `(800 × 1024²) + (11 × 1024²) + (51 × 1024²) ≈ 903M FLOPs`

2. **Attention** (shared): `O((T_total)² × D)`
   - `862² × 1024 ≈ 760M FLOPs`
   - **This is the expensive part!**

3. **Output projection** (separate): `O(T × D² × num_experts)`
   - Similar to Q/K/V: `≈ 903M FLOPs`

4. **FFN** (separate): `O(T × D × mlp_dim × num_experts)`
   - `(800 + 11 + 51) × 1024 × 4096 × 2 ≈ 7.1B FLOPs`

**Total per layer**: `≈ 9.7B FLOPs`
**Total model (18 layers)**: `≈ 175B FLOPs`

### Memory (Activation Storage)

**During forward pass** (batch_size=32):

- Concatenated Q/K/V: `3 × 32 × 862 × 8 × 256 × 2 bytes ≈ 34 MB`
- Attention logits: `32 × 8 × 862 × 862 × 4 bytes ≈ 770 MB` (largest!)
- Activations per expert: `32 × T × D × 2 bytes × 3 experts ≈ 53 MB`

**KV Cache** (for inference with num_steps=10):
- Per layer: `2 × 32 × 862 × 1 × 256 × 2 bytes ≈ 8.9 MB`
- All layers: `18 × 8.9 MB ≈ 160 MB`

### Comparison: Multi-Expert vs Separate Models

| Metric | 3 Separate Models | 3 Experts (Shared Attention) | Savings |
|--------|-------------------|------------------------------|---------|
| Parameters | ~933M | ~933M | 0% (but can share embedder) |
| FLOPs (training) | ~525B | ~175B | **67%** |
| Memory (attention) | ~2.3 GB | ~770 MB | **66%** |
| Cross-expert attention | ❌ No | ✅ Yes | - |

**Key insight**: The shared attention computation provides massive computational savings and enables cross-expert information flow, with minimal parameter overhead.

---

## Practical Implications

### 1. Training Considerations

**Gradient Flow**:
- Each expert's projection matrices receive gradients based on their own outputs
- Attention computation is differentiable w.r.t. all experts' Q/K/V
- Experts can learn to "cooperate" through attention

**Example**: In v12, the action expert learns to attend to specific image regions (computed by prompt expert) that are relevant for action prediction.

**Loss Design**:
- Can compute separate losses for different experts
- Backprop will flow through shared attention to all experts
- Can freeze specific experts by filtering gradients

### 2. Inference Optimization

**KV Cache Reuse**:
```python
# Compute prompt expert once
_, kv_cache = llm([prompt_tokens, None, None], ...)

# Reuse cache in diffusion loop (10 steps)
for step in range(10):
    (_, coarse_out, _), _ = llm([None, coarse_tokens, None], kv_cache=kv_cache, ...)
```

**Skipping Experts**:
- Pass `None` for experts you don't need
- Their Q/K/V projections and output projections are skipped
- Attention still computed over remaining experts

### 3. Checkpoint Loading

**Partial Loading**:
```python
# Load only prompt expert from PaliGemma
checkpoint_loader = weight_loaders.LocalCheckpointLoader(
    checkpoint_path="path/to/paligemma",
    filter=lambda path: "prompt_expert" in path or not any(f"_{i}" in path for i in range(10))
)

# Load only action expert from Pi0
checkpoint_loader = weight_loaders.LocalCheckpointLoader(
    checkpoint_path="path/to/pi0",
    filter=lambda path: "_1" in path  # action_expert uses "_1" suffix
)
```

### 4. LoRA Fine-Tuning

Each expert can have different LoRA configs:

```python
configs = [
    Config(..., lora_configs={}),  # Prompt expert: no LoRA (frozen)
    Config(..., lora_configs={     # Coarse expert: LoRA on attention only
        "attn": lora.LoRAConfig(rank=16, alpha=16.0)
    }),
    Config(..., lora_configs={     # Fine expert: LoRA on both
        "attn": lora.LoRAConfig(rank=16, alpha=16.0),
        "ffn": lora.LoRAConfig(rank=16, alpha=16.0),
    }),
]
```

### 5. Attention Mask Design

**For 3 experts with hierarchical structure**:

```python
# Token layout: [prompt_tokens | coarse_tokens | fine_tokens]
# Positions:     [0...P-1       | P...P+C-1     | P+C...P+C+F-1]

ar_mask = [
    # Prompt expert: bidirectional attention
    False, False, ..., False,  # P tokens

    # Coarse expert: can attend to prompt + causal within coarse
    True,                       # First coarse token starts new block
    False, False, ..., False,   # Remaining coarse tokens (C-1)

    # Fine expert: can attend to prompt + all coarse + causal within fine
    True,                       # First fine token starts new block
    False, False, ..., False,   # Remaining fine tokens (F-1)
]

# make_attn_mask converts this to 2D mask:
# - Coarse tokens CAN attend to all prompt tokens
# - Fine tokens CAN attend to all prompt + all coarse tokens
# - Each group has causal attention within itself
```

### 6. Debugging Tips

**Check parameter names**:
```python
# Print all parameter paths
for path, param in nnx.split(model, nnx.Param)[1].items():
    print(f"{path}: {param.shape}")

# Should see:
# PaliGemma.llm.layers_0.attn.qkv_einsum.kernel: (3, 8, 1024, 256)
# PaliGemma.llm.layers_0.attn.qkv_einsum_1.kernel: (3, 8, 1024, 256)
# PaliGemma.llm.layers_0.attn.qkv_einsum_coarse_expert.kernel: (3, 8, 1024, 256)
```

**Check attention mask shape**:
```python
# Should be [B, 1, T_total, T_total] where T_total = T0 + T1 + T2
print(f"Attention mask shape: {attn_mask.shape}")
print(f"Expected: [{batch_size}, 1, {T0 + T1 + T2}, {T0 + T1 + T2}]")
```

**Verify expert outputs**:
```python
(out0, out1, out2), _ = llm([tokens0, tokens1, tokens2], ...)
assert out0.shape == tokens0.shape  # Expert 0 output matches input
assert out1.shape == tokens1.shape  # Expert 1 output matches input
assert out2.shape == tokens2.shape  # Expert 2 output matches input
```

---

## Summary Table

| Component | Shared? | Parameters | Computation | Can Differ? |
|-----------|---------|------------|-------------|-------------|
| Embedder | ✅ Yes | Expert 0 only | Shared | No (uses expert 0's dim) |
| Depth | ✅ Must match | - | - | ❌ No |
| num_heads | ✅ Must match | - | Shared attention | ❌ No |
| head_dim | ✅ Must match | - | Shared attention | ❌ No |
| RMSNorm | ❌ No | Separate per expert | Separate | ✅ Yes (implicitly by width) |
| Q/K/V projections | ❌ No | Separate per expert | Separate | ✅ Yes (by width) |
| Attention computation | ✅ Yes | - | Shared (concat→attend→split) | - |
| Output projection | ❌ No | Separate per expert | Separate | ✅ Yes (by width) |
| FFN (MLP) | ❌ No | Separate per expert | Separate | ✅ Yes (width, mlp_dim) |
| width | - | - | - | ✅ Yes |
| mlp_dim | - | - | - | ✅ Yes |
| LoRA configs | - | - | - | ✅ Yes |

---

## Key Takeaways

1. **Experts are mostly independent** with separate parameter matrices for projections, norms, and FFNs

2. **Shared attention computation** enables cross-expert information flow while saving 67% of compute

3. **Token concatenation** means all experts' tokens exist in a single sequence during attention

4. **Attention mask controls interaction** between experts - this is how you design the information flow

5. **Dimension flexibility**: Experts can have different widths but must share head configuration

6. **Parameter naming** follows a convention to enable loading from pre-trained checkpoints

7. **Computational hotspot**: Attention over concatenated sequence is O(T²), where T = sum of all expert sequence lengths

8. **For v17**: You can create 3 experts with independent specializations (prompt processing, coarse actions, fine actions) while still allowing them to share information through attention

---

## Further Reading

- Original Gemma implementation: https://github.com/google-deepmind/gemma
- Flax Linen documentation: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/index.html
- Multi-Query Attention paper: https://arxiv.org/abs/1911.02150
- LoRA paper: https://arxiv.org/abs/2106.09685

For implementation examples, see:
- `src/openpi/models/pi0_incontextv12.py` - 2-expert model
- `implementationv17.md` - Guide for 3-expert hierarchical model
