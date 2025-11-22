# Config System Refactoring Analysis

**Document Date**: 2025-11-10
**Refactoring Date**: September 29, 2025
**Refactoring Commit**: `539a4df20f3004f0d16f15355852c94690c8129c`

---

## Executive Summary

The PR codebase (`q_former` branch) underwent a **major config system refactoring** on **September 29, 2025**, which happened **125 commits AFTER** the branch diverged from `v1.0`. This refactoring split the monolithic `config.py` (6,851 lines) into **domain-specific config files**:

- **Original System** (v1.0): Single `config.py` with all configurations
- **New System** (q_former): Modular config files with automatic discovery and loading

**Key Impact**: The original `v1.0` branch uses the **old monolithic config system**, while the PR branch uses the **new modular config system**. This creates **structural incompatibility** when porting configs between branches.

---

## Timeline

```
May 20, 2025
    │
    ├── 9018878 "fix incontext meta file load bug"
    │   (Branches diverge)
    │
    ├─────────────────────────────────────────────────────────────
    │                                                             │
    │ PR Branch (q_former)                         Original Branch (v1.0)
    │                                                             │
    │ +125 commits...                              No refactoring
    │                                                             │
    ├── Sep 29, 2025                                             │
    │   539a4df "segregate redundant config.py                   │
    │            into sub-config files"                           │
    │   ⭐ MAJOR REFACTORING ⭐                                   │
    │                                                             │
    │   - Split config.py (7725 lines) →                         │
    │     • config.py (672 lines, core only)                     │
    │     • config_aloha.py (628 lines)                          │
    │     • config_libero.py (4191 lines)                        │
    │     • config_template.py (75 lines)                        │
    │     • deprecated_config.py (8352 lines, backup)            │
    │                                                             │
    │ +40 more commits...                          +23 commits...│
    │                                                             │
    └── HEAD (q_former)                            HEAD (v1.0)   │
        Modular config system                      Monolithic    │
                                                   config system │
```

---

## Refactoring Details

### Commit Information

**Commit**: `539a4df20f3004f0d16f15355852c94690c8129c`
**Date**: Mon Sep 29 16:14:13 2025 +0300
**Author**: dingjiansw101 <jianding101@gmail.com>
**Message**: "segregate redundant config.py into sub-config files"

**Files Changed**:
```
src/openpi/training/config.py            | -7725 lines
src/openpi/training/config_aloha.py      | +628 lines
src/openpi/training/config_libero.py     | +4191 lines
src/openpi/training/config_template.py   | +75 lines
src/openpi/training/deprecated_config.py | +8352 lines (backup)

Total: +13,246 insertions, -7,725 deletions
```

---

## Config System Comparison

### Original System (v1.0 branch)

**File Structure**:
```
src/openpi/training/
└── config.py  (6,851 lines)
    ├── Core infrastructure
    ├── LiberoInputs/Outputs classes
    ├── DroidInputs/Outputs classes
    ├── AlohaInputs/Outputs classes
    ├── All DataConfig classes
    ├── All TrainConfig definitions
    └── _CONFIGS registry dict
```

**Registration Pattern**:
```python
# All configs defined in config.py
_CONFIGS = {
    "pi0_libero": lambda: get_pi0_libero_config(),
    "pi0_aloha_mobile": lambda: get_pi0_aloha_mobile_config(),
    ...
}

def get_config(config_name: str) -> TrainConfig:
    return _CONFIGS[config_name]()
```

**Pros**:
- ✅ Simple, all in one place
- ✅ Easy to grep for all configs
- ✅ No import complexity

**Cons**:
- ❌ 6,851 lines in single file (hard to navigate)
- ❌ Merge conflicts when multiple people edit
- ❌ Slow to load entire file

---

### New System (q_former branch)

**File Structure**:
```
src/openpi/training/
├── config.py  (672 lines, core infrastructure only)
│   ├── Core types (TrainConfig, DataConfig, etc.)
│   ├── Auto-discovery system
│   └── Config loader
├── config_aloha.py  (677 lines)
│   ├── LeRobotAlohaDataConfig
│   ├── LeRobotAlohaMobileDataConfig
│   ├── LeRobotAlohaMobileIncontextDataConfig
│   └── build(api) → list[TrainConfig]
├── config_libero.py  (7,173 lines)
│   ├── LiberoInputs/Outputs
│   ├── LeRobotLiberoDataConfig
│   └── All Libero TrainConfigs
├── config_sequence_debug.py  (3,452 lines)
└── deprecated_config.py  (old backup)
```

**Registration Pattern**:
```python
# In config.py (core):
def _discover_child_modules() -> list[str]:
    """Auto-discover config_*.py files"""
    pkg_dir = pathlib.Path(__file__).parent
    base_pkg = __name__.rsplit(".", 1)[0]
    out = []
    for p in pkg_dir.glob("config_*.py"):
        if p.name == "config.py":
            continue
        out.append(f"{base_pkg}.{p.stem}")
    return out

def _load_fragments(module_names: list[str]) -> list[TrainConfig]:
    """Load configs from each sub-file"""
    out: list[TrainConfig] = []
    api = sys.modules[__name__]  # Pass core module as API
    for m in module_names:
        mod = importlib.import_module(m)
        build = getattr(mod, "build", None)
        if callable(build):
            out.extend(build(api))  # Each file returns configs
    return out

_MODULES = _discover_child_modules()
_CONFIGS: list[TrainConfig] = _load_fragments(_MODULES)
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

def get_config(config_name: str) -> TrainConfig:
    """Same interface as before"""
    return _CONFIGS_DICT[config_name]
```

**In config_aloha.py**:
```python
def build(api) -> list["api.TrainConfig"]:
    """Called by core config.py to get all ALOHA configs"""
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    g["Group"] = getattr(api._transforms, "Group")
    g["TrainConfig"] = getattr(api, "TrainConfig")

    # Define DataConfig classes
    @dataclasses.dataclass(frozen=True)
    class LeRobotAlohaMobileIncontextDataConfig(api.DataConfigFactory):
        states_cache_path: str = "metadata/aloha_pen_uncap/episode_states_cache.json"
        actions_cache_path: str = "metadata/aloha_pen_uncap/episode_actions_first_cache.json"
        tracks_path: str = "metadata/aloha_pen_uncap/episode_tracks_combined.json"
        # ... other fields ...

        def create(self, assets_dirs, model_config):
            # Returns DataConfig
            pass

    # Return list of TrainConfig instances
    return [
        api.TrainConfig(
            name="pi0_aloha_pen_uncap_b5_low_mem_finetune",
            model=api.pi0.Pi0Config(...),
            data=LeRobotAlohaMobileIncontextDataConfig(...),
            ...
        ),
        # ... more configs ...
    ]
```

**Pros**:
- ✅ Modular, domain-specific files
- ✅ Easier to navigate (677 lines vs 6,851)
- ✅ Less merge conflicts
- ✅ Auto-discovery of new config files
- ✅ Cleaner separation of concerns

**Cons**:
- ⚠️ More complex loading mechanism
- ⚠️ Each config file imports from `api` parameter
- ⚠️ Harder to understand initially

---

## LeRobotAlohaMobileIncontextDataConfig Comparison

### Location

| Branch | File | Lines |
|--------|------|-------|
| **v1.0 (original)** | `src/openpi/training/config.py` | 650-730 |
| **q_former (PR)** | `src/openpi/training/config_aloha.py` | 135-220 |

### Key Differences

#### 1. Metadata Paths

**v1.0 (original)**:
```python
class LeRobotAlohaMobileIncontextDataConfig(DataConfigFactory):
    states_cache_path: str = "metadata/pen_incontext/episode_states_cache.json"
    actions_cache_path: str = "metadata/pen_incontext/episode_actions_first_cache.json"
    tracks_path: str = "metadata/pen_incontext/episode_tracks_combined.json"
    task_to_episode: str = "metadata/pen_incontext/task_to_episode.json"
    episode_to_indexes_file: str = "metadata/pen_incontext/episode_to_indexes.json"
```

**q_former (PR)**:
```python
class LeRobotAlohaMobileIncontextDataConfig(api.DataConfigFactory):
    states_cache_path: str = "metadata/aloha_pen_uncap/episode_states_cache.json"
    actions_cache_path: str = "metadata/aloha_pen_uncap/episode_actions_first_cache.json"
    tracks_path: str = "metadata/aloha_pen_uncap/episode_tracks_combined.json"
    # NOTE: task_to_episode and episode_to_indexes_file are hardcoded in create()
```

**Impact**:
- ⚠️ Different default metadata directories
- ⚠️ v1.0 has configurable `task_to_episode` field, PR hardcodes it
- ⚠️ Must update metadata paths when porting configs

#### 2. Repack Transforms

**v1.0 (original)**:
```python
repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
    default=_transforms.Group(
        inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",                    # ⭐ Included
                "episode_index": "episode_index",      # ⭐ Included
                "index": "index",                      # ⭐ Included
                "task_index": "task_index",            # ⭐ Included
            }
        )
    ]
),
)
```

**q_former (PR)**:
```python
repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
    default=api._transforms.Group(
        inputs=[
            api._transforms.RepackTransform(
                {
                    "images": {"cam_high": "observation.images.top"},  # ⚠️ Different key
                    "state": "observation.state",
                    "actions": "action",
                    # ⚠️ Missing: prompt, episode_index, index, task_index
                }
            )
        ]
    )
)
```

**Impact**:
- ⚠️ v1.0 explicitly repacks metadata fields (prompt, task_index, etc.)
- ⚠️ PR relies on implicit passing or later injection
- ⚠️ Image key changed: `observation.images.cam_high` → `observation.images.top`

#### 3. InjectDemoIndexes Transform

**v1.0 (original)**:
```python
data_transforms = _transforms.Group(
    inputs=[_transforms.InjectDemoIndexes(
        task_to_episode=self.task_to_episode,                   # ⭐ Configurable
        episode_to_indexes=self.episode_to_indexes_file,        # ⭐ Configurable
        sample_frames=model_config.sample_frames,
        random_select=model_config.random_select,
        sample_episodes=model_config.sample_episodes,
        train_episode_index_list=train_epi)],
    outputs=[],
)
```

**q_former (PR)**:
```python
data_transforms = api._transforms.Group(
    inputs=[api._transforms.InjectDemoIndexes(
        task_to_episode="metadata/aloha_pen_uncap/task_to_episode.json",  # ⚠️ Hardcoded
        episode_to_indexes="metadata/aloha_pen_uncap/episode_to_indexes.json",  # ⚠️ Hardcoded
        sample_frames=model_config.sample_frames,
        random_select=model_config.random_select,
        sample_episodes=model_config.sample_episodes,
        train_episode_index_list=train_epi)],
    outputs=[],
)
```

**Impact**:
- ⚠️ PR hardcodes metadata paths, less flexible
- ⚠️ Cannot easily switch between different metadata sets

#### 4. Multi-process Support

**v1.0 (original)**:
```python
class LeRobotAlohaMobileIncontextDataConfig(DataConfigFactory):
    multi_process: bool = False  # ⭐ Has multi-process flag
```

**q_former (PR)**:
```python
class LeRobotAlohaMobileIncontextDataConfig(api.DataConfigFactory):
    # ⚠️ No multi_process field
```

**Impact**:
- ⚠️ v1.0 supports multi-process data loading
- ⚠️ PR removed this feature (or moved elsewhere)

---

## Impact on Porting Between Branches

### Porting Configs from v1.0 → q_former

**Challenge**: v1.0 uses monolithic `config.py`, q_former uses modular system

**Steps**:

1. **Find the target config file** (e.g., `config_aloha.py` for ALOHA configs)

2. **Locate the `build(api)` function** in the target file

3. **Add DataConfig class inside `build()` if needed**:
   ```python
   @dataclasses.dataclass(frozen=True)
   class MyNewDataConfig(api.DataConfigFactory):
       # ... fields from v1.0 ...

       def create(self, assets_dirs, model_config):
           # ... copy logic from v1.0 ...
           pass
   ```

4. **Add TrainConfig to the return list**:
   ```python
   return [
       # ... existing configs ...
       api.TrainConfig(
           name="my_new_config",
           model=api.pi0.Pi0Config(...),
           data=MyNewDataConfig(...),
           ...
       ),
   ]
   ```

5. **Test config loading**:
   ```bash
   uv run scripts/train.py my_new_config --help
   ```

**Gotchas**:
- ⚠️ Must use `api.*` prefix for all types (e.g., `api.TrainConfig` not `TrainConfig`)
- ⚠️ Update metadata paths from `metadata/pen_incontext/` to `metadata/aloha_pen_uncap/`
- ⚠️ Check repack transforms match expected keys
- ⚠️ Multi-process flag may not be supported

---

### Porting Configs from q_former → v1.0

**Challenge**: q_former uses modular system, v1.0 uses monolithic `config.py`

**Steps**:

1. **Extract DataConfig class from `config_aloha.py`** (or other domain file)

2. **Add to main `config.py`** at the appropriate section:
   ```python
   # In config.py
   @dataclasses.dataclass(frozen=True)
   class MyNewDataConfig(DataConfigFactory):
       # Copy from q_former, but:
       # - Change api._transforms → _transforms
       # - Change api.DataConfig → DataConfig
       # - Add configurable paths instead of hardcoded

       task_to_episode: str = "metadata/pen_incontext/task_to_episode.json"
       episode_to_indexes_file: str = "metadata/pen_incontext/episode_to_indexes.json"

       def create(self, assets_dirs, model_config):
           # Use self.task_to_episode instead of hardcoded path
           pass
   ```

3. **Add TrainConfig function**:
   ```python
   def get_my_new_config() -> TrainConfig:
       return TrainConfig(
           name="my_new_config",
           model=pi0.Pi0Config(...),
           data=MyNewDataConfig(...),
           ...
       )
   ```

4. **Register in `_CONFIGS` dict**:
   ```python
   _CONFIGS = {
       # ... existing configs ...
       "my_new_config": lambda: get_my_new_config(),
   }
   ```

**Gotchas**:
- ⚠️ Remove all `api.` prefixes
- ⚠️ Make hardcoded paths configurable
- ⚠️ Update metadata directory from `aloha_pen_uncap` to `pen_incontext`
- ⚠️ Add explicit repack of metadata fields (prompt, task_index, etc.)

---

## Metadata Path Conventions

### v1.0 Branch
```
metadata/
├── pen_incontext/                    # Mobile ALOHA incontext
│   ├── task_to_episode.json
│   ├── episode_to_indexes.json
│   ├── episode_states_cache.json
│   ├── episode_actions_first_cache.json
│   └── episode_tracks_combined.json
└── (other datasets...)
```

### q_former Branch
```
metadata/
├── aloha_pen_uncap/                 # Mobile ALOHA incontext
│   ├── task_to_episode.json
│   ├── episode_to_indexes.json
│   ├── episode_states_cache.json
│   ├── episode_actions_first_cache.json
│   └── episode_tracks_combined.json
├── libero/
└── (other datasets...)
```

**Recommendation**: When porting deployment scripts (main_incontext.py, etc.), update metadata paths to match the target branch's convention.

---

## Recommendations

### For Config Porting

1. **Use Appropriate Config File**:
   - ALOHA configs → `config_aloha.py`
   - Libero configs → `config_libero.py`

2. **Metadata Paths**:
   - v1.0 → q_former: Change `metadata/pen_incontext/` to `metadata/aloha_pen_uncap/`
   - q_former → v1.0: Change `metadata/aloha_pen_uncap/` to `metadata/pen_incontext/`

3. **Repack Transforms**:
   - Ensure all required fields are repacked (prompt, task_index, episode_index, index)
   - Match image keys to dataset structure

4. **Testing**:
   ```bash
   # Test config loads
   uv run scripts/train.py <config_name> --help

   # Test dataset access
   uv run src/openpi/training/generate_task_to_index.py --config <config_name>
   ```

---

### For Deployment Script Porting

When porting `main_incontext.py`, `env_incontext.py` from v1.0 → q_former:

1. **Update metadata paths in code**:
   ```python
   # Change from:
   task_metadata_file = "metadata/pen_incontext/task_to_episode.json"

   # To:
   task_metadata_file = "metadata/aloha_pen_uncap/task_to_episode.json"
   ```

2. **Generate metadata for new path**:
   ```bash
   uv run src/openpi/training/generate_task_to_index.py \
     --config pi0_aloha_pen_uncap_b5_low_mem_finetune \
     --output_dir metadata/aloha_pen_uncap
   ```

3. **Test in-context loading**:
   ```bash
   # Verify env_incontext.py can load metadata
   python -c "
   from examples.aloha_mobile_real.env_incontext import AlohaRealIncontextEnvironment
   env = AlohaRealIncontextEnvironment('metadata/aloha_pen_uncap/task_to_episode.json')
   print('Success')
   "
   ```

---

## Summary

| Aspect | v1.0 (Original) | q_former (PR) |
|--------|-----------------|---------------|
| **Config Structure** | Monolithic config.py (6,851 lines) | Modular config_*.py (672 + 7,000+ lines) |
| **Registration** | Manual `_CONFIGS` dict | Auto-discovery + `build(api)` |
| **Metadata Path** | `metadata/pen_incontext/` | `metadata/aloha_pen_uncap/` |
| **Path Configuration** | ✅ Configurable fields | ⚠️ Some hardcoded paths |
| **Repack Metadata** | ✅ Explicit (prompt, task_index, etc.) | ⚠️ Minimal (may rely on implicit) |
| **Multi-process** | ✅ Supported | ⚠️ Not in DataConfig class |
| **Complexity** | ⭐ Simple, all in one place | ⭐⭐⭐ Modular, auto-loading |

**Key Takeaway**: The config system refactoring creates **structural differences** that require careful attention when porting configs or deployment scripts between branches. Always verify metadata paths and test config loading after porting.

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Analysis Date** | 2025-11-10 |
| **Refactoring Commit** | 539a4df20f3004f0d16f15355852c94690c8129c |
| **Refactoring Date** | 2025-09-29 16:14:13 +0300 |
| **Commits After Divergence** | 125 commits (May 20 → Sep 29, 2025) |
| **Lines Changed** | +14,446 insertions, -7,725 deletions |
| **v1.0 Config** | config.py (6,851 lines) |
| **q_former Config** | config.py (672 lines) + 6 domain files |

---

**End of Analysis**
