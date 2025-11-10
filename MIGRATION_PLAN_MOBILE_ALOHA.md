# Mobile ALOHA Real-World Integration: Comprehensive Migration Plan

**Document Version**: 1.0
**Date**: 2025-11-10
**Target**: Integrate mobile ALOHA real-world deployment code from v1.0 → q_former branch
**Estimated Total Time**: 2-3 working days

---

## Executive Summary

This document provides a **complete migration plan** to integrate mobile ALOHA real-world deployment infrastructure from the v1.0 branch into the q_former branch, enabling production deployment of latest Q-former models on real robots.

**Current State**:
- ✅ q_former branch has latest model architecture (v12-v18)
- ❌ q_former branch lacks real robot deployment infrastructure
- ✅ v1.0 branch has complete deployment tools
- ⚠️ Branches have significant structural differences

**Goal State**:
- ✅ q_former branch with Q-former v18 models
- ✅ Complete mobile ALOHA deployment infrastructure
- ✅ In-context learning deployment capability
- ✅ Trajectory recording and analysis tools
- ✅ Compatibility with existing training pipelines

---

## Table of Contents

1. [Analysis Summary](#1-analysis-summary)
2. [Critical Incompatibilities](#2-critical-incompatibilities)
3. [Migration Strategy](#3-migration-strategy)
4. [Phase 1: Core Deployment Files](#phase-1-core-deployment-files-day-1-morning)
5. [Phase 2: Metadata & Cache Files](#phase-2-metadata--cache-files-day-1-afternoon)
6. [Phase 3: Configuration Alignment](#phase-3-configuration-alignment-day-2-morning)
7. [Phase 4: Testing & Validation](#phase-4-testing--validation-day-2-afternoon)
8. [Phase 5: Optional Enhancements](#phase-5-optional-enhancements-day-3)
9. [Rollback Plan](#9-rollback-plan)
10. [Post-Migration Checklist](#10-post-migration-checklist)

---

## 1. Analysis Summary

### 1.1 Branch Divergence (from BRANCH_DIVERGENCE_ANALYSIS.md)

**Divergence Point**: May 20, 2025, commit `9018878`

| Metric | v1.0 (Original) | q_former (PR) |
|--------|-----------------|---------------|
| **Commits after divergence** | 23 | 90 |
| **Focus** | Production deployment | Research architecture |
| **Missing from PR** | - | ~900 lines deployment code |
| **Missing from v1.0** | - | Q-former v12-v18 models |

**Key Structural Changes**:
1. **Config system refactored** (Sep 29, 2025) - Monolithic → Modular
2. **Data loading redesigned** - Transform-based → Dataset-based (`CustomLeRobotDataset`)
3. **Metadata paths changed** - `pen_incontext/` → `aloha_pen_uncap/`

---

### 1.2 Missing Components (from MOBILE_ALOHA_MISSING_COMPONENTS.md)

#### Critical Files Missing in q_former:

| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `main_incontext.py` | 139 | In-context deployment with task prompts | 🔴 P1 |
| `env_incontext.py` | 100 | Task metadata environment wrapper | 🔴 P1 |
| `trajectory_recorder.py` | 150 | HDF5 trajectory recording | 🟠 P2 |
| `visualize_trajectory_videos.py` | 252 | Multi-camera video export | 🟠 P2 |
| `view_trajectory.py` | 96 | Quick trajectory inspection | 🟡 P3 |
| `main_icrt.py` | 77 | ICRT evaluation | 🟡 P3 |
| `test_trajectory_recording.py` | 90 | Unit tests | 🟢 P4 |

#### Metadata Infrastructure Missing:

```
metadata/
├── aloha_pen_uncap/              # Exists in PR (2 files, 265KB)
│   ├── task_to_episode.json      # ✓ Present
│   └── episode_to_indexes.json   # ✓ Present
│
├── pen_incontext/                # MISSING from PR (5 files, 39MB)
│   ├── task_to_episode.json      # ✗ Missing
│   ├── episode_to_indexes.json   # ✗ Missing
│   ├── episode_states_without_delta_cache.json  # ✗ Missing (19MB)
│   ├── episode_actions_without_delta_cache.json # ✗ Missing (20MB)
│   └── tasks.jsonl               # ✗ Missing
│
└── objects_all/                  # MISSING from PR (required for objects config)
    ├── task_to_episode.json      # ✗ Not generated
    ├── episode_to_indexes.json   # ✗ Not generated
    ├── episode_states_without_delta_cache.json  # ✗ Not generated
    └── episode_actions_without_delta_cache.json # ✗ Not generated
```

---

### 1.3 Config System Changes (from CONFIG_SYSTEM_CHANGES.md)

**Refactoring Date**: September 29, 2025 (125 commits after divergence)

| Aspect | v1.0 | q_former |
|--------|------|----------|
| **Structure** | `config.py` (6,851 lines) | `config.py` (672) + `config_aloha.py` (677) + 5 more |
| **Config Registration** | Manual `_CONFIGS` dict | Auto-discovery via `build(api)` |
| **ALOHA Configs** | In `config.py` | In `config_aloha.py` |

**Import Pattern Changes**:
```python
# v1.0 (old):
from openpi.training.config import get_config

# q_former (new):
from openpi.training.config import get_config  # Same API!
# But configs are defined in config_aloha.py using build(api) pattern
```

---

### 1.4 Data Processing Differences (from LeRobotAlohaMobileIncontextDataConfig analysis)

#### Critical Incompatibilities:

**1. Repack Transform Field Mismatch**

| Field | v1.0 | q_former | Impact |
|-------|------|----------|---------|
| `prompt` | ✓ Repacked | ✗ Not repacked | ❌ Will break in-context |
| `episode_index` | ✓ Repacked | ✗ Not repacked | ❌ Transform failures |
| `index` | ✓ Repacked | ✗ Not repacked | ❌ Transform failures |
| `task_index` | ✓ Repacked | ✗ Not repacked | ❌ Will break in-context |
| `cam_left_wrist` | ✓ Repacked | ✗ Not repacked | ⚠️ Missing camera |
| `cam_right_wrist` | ✓ Repacked | ✗ Not repacked | ⚠️ Missing camera |

**2. InjectDemoIndexes Configuration**

| Aspect | v1.0 | q_former | Impact |
|--------|------|----------|---------|
| `task_to_episode` | Configurable field | Hardcoded path | ⚠️ Less flexible |
| `episode_to_indexes` | Configurable field | Hardcoded path | ⚠️ Less flexible |
| `sample_frames` default | 16 | 2 | ❌ 8x difference! |
| Stage-wise prompting | Not implemented | Fully implemented | ⚠️ Feature gap |
| Inference caching | Not implemented | Fully implemented | ✅ Performance gain |

**3. Missing Config - Objects Dataset**

**Config Name**: `pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select`

**Status**: ❌ **DOES NOT EXIST** in q_former codebase

**Referenced in**:
- 4 SBATCH job scripts in `jobs/sbatch_scripts_real/`
- Expected but not defined

**Needs**:
- Config definition in `config_aloha.py`
- Dataset repo_id specification
- Metadata generation for `objects_all/`
- Cache file creation (39MB)

---

## 2. Critical Incompatibilities

### 2.1 Blocker Issues

These **MUST** be resolved before deployment will work:

#### ❌ Issue 1: Missing Metadata Fields in Repack Transform

**Problem**: q_former strips essential metadata fields that in-context transforms need.

**Location**: `config_aloha.py` line 153-165

**Current (broken)**:
```python
repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
    default=api._transforms.Group(
        inputs=[
            api._transforms.RepackTransform(
                {
                    "images": {"cam_high": "observation.images.top"},
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )
)
```

**Required (fixed)**:
```python
repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
    default=api._transforms.Group(
        inputs=[
            api._transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",                 # ← ADD
                    "episode_index": "episode_index",   # ← ADD
                    "index": "index",                   # ← ADD
                    "task_index": "task_index",         # ← ADD
                }
            )
        ]
    )
)
```

**Impact**: Without this, `main_incontext.py` will crash when trying to access metadata.

---

#### ❌ Issue 2: Missing Cache Files (39MB)

**Problem**: Configs reference cache files that don't exist in q_former.

**Referenced Files**:
```python
states_cache_path="metadata/aloha_pen_uncap/episode_states_cache.json"  # ✗ Missing
actions_cache_path="metadata/aloha_pen_uncap/episode_actions_cache.json"  # ✗ Missing
```

**Exists in v1.0**:
```
metadata/pen_incontext/
├── episode_states_without_delta_cache.json   (19MB)
└── episode_actions_without_delta_cache.json  (20MB)
```

**Solution Options**:

**Option A - Copy from v1.0** (Fast):
```bash
cp /home/dingj0b/code/openpi/metadata/pen_incontext/episode_states_without_delta_cache.json \
   metadata/aloha_pen_uncap/episode_states_cache.json

cp /home/dingj0b/code/openpi/metadata/pen_incontext/episode_actions_without_delta_cache.json \
   metadata/aloha_pen_uncap/episode_actions_cache.json
```

**Option B - Regenerate** (Slower but correct):
```bash
# Use the cache creation SBATCH script
sbatch jobs/sbatch_scripts_real/create_cache_pi0_aloha_pen_uncap_*.sh
```

**Impact**: Without cache files, in-context learning with action/state prompts will fail.

---

#### ❌ Issue 3: Hardcoded Metadata Paths in Deployment Scripts

**Problem**: `main_incontext.py` and `env_incontext.py` from v1.0 use old paths.

**v1.0 paths**:
```python
task_metadata = "metadata/pen_incontext/task_to_episode.json"
```

**q_former paths**:
```python
task_metadata = "metadata/aloha_pen_uncap/task_to_episode.json"
```

**Solution**: Update paths when copying files (covered in migration steps).

---

### 2.2 Warning Issues

These won't block deployment but will cause problems:

#### ⚠️ Issue 4: Sample Frames Mismatch (16 vs 2)

**Impact**: Models trained with different frame counts may not transfer.

**Recommendation**:
- Keep q_former default (2 frames) for new models
- Document difference for debugging
- Consider adding as configurable parameter

---

#### ⚠️ Issue 5: Missing Objects Dataset Config

**Impact**: SBATCH jobs referencing `pi0_aloha_objects_all_*` will fail.

**Solution**: Create the missing config (covered in Phase 3).

---

## 3. Migration Strategy

### 3.1 Approach: Selective File Porting

**Why not full merge?**
- 90 vs 23 commits - too much divergence
- Different architectural patterns (dataset vs transform)
- Config system refactoring creates conflicts

**Strategy**: **Port deployment files + align configurations**

**Principles**:
1. Keep q_former's advanced architecture (CustomLeRobotDataset, stage-wise prompting)
2. Port v1.0's deployment scripts (main_incontext.py, trajectory_recorder.py)
3. Align configurations for compatibility
4. Generate missing metadata
5. Test thoroughly before production use

---

### 3.2 Timeline Overview

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1**: Core deployment files | 3-4 hours | Copied and updated deployment scripts |
| **Phase 2**: Metadata & cache | 2-3 hours | Complete metadata directory |
| **Phase 3**: Config alignment | 2-3 hours | Fixed configs, new objects config |
| **Phase 4**: Testing | 3-4 hours | Validated on real robot |
| **Phase 5**: Optional enhancements | 4-8 hours | Enhanced features |
| **Total** | 14-22 hours | **2-3 working days** |

---

## Phase 1: Core Deployment Files (Day 1 Morning)

**Goal**: Copy essential deployment infrastructure
**Duration**: 3-4 hours
**Dependencies**: None

### Step 1.1: Copy Deployment Scripts (30 min)

**Priority 1 (Critical) Files**:

```bash
cd /home/dingj0b/code/openpi_pr/openpi

# Copy in-context deployment
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/main_incontext.py \
   examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/env_incontext.py \
   examples/aloha_mobile_real/
```

**Priority 2 (Important) Files**:

```bash
# Copy trajectory tools
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/trajectory_recorder.py \
   examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/visualize_trajectory_videos.py \
   examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/view_trajectory.py \
   examples/aloha_mobile_real/
```

**Priority 3 (Optional) Files**:

```bash
# Copy ICRT and tests
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/main_icrt.py \
   examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/test_trajectory_recording.py \
   examples/aloha_mobile_real/
```

**Verification**:
```bash
ls -lh examples/aloha_mobile_real/main_incontext.py  # Should show 139 lines
ls -lh examples/aloha_mobile_real/env_incontext.py   # Should show 100 lines
```

---

### Step 1.2: Update Metadata Paths in Copied Files (1 hour)

**Files to Update**:
1. `examples/aloha_mobile_real/main_incontext.py`
2. `examples/aloha_mobile_real/env_incontext.py`
3. `examples/aloha_mobile_real/main_icrt.py` (if copied)

**Search and Replace**:

```bash
# Find all hardcoded paths
grep -n "metadata/pen_incontext" examples/aloha_mobile_real/main_incontext.py
grep -n "metadata/pen_incontext" examples/aloha_mobile_real/env_incontext.py
```

**Expected Changes**:

**In `env_incontext.py`**:
```python
# Change from:
self.task_metadata_file = "metadata/pen_incontext/task_to_episode.json"

# To:
self.task_metadata_file = "metadata/aloha_pen_uncap/task_to_episode.json"
```

**In `main_incontext.py`**:
```python
# Change from:
metadata_file = "metadata/pen_incontext/task_to_episode.json"

# To:
metadata_file = "metadata/aloha_pen_uncap/task_to_episode.json"
```

**Verification**:
```bash
# Verify no old paths remain
grep -r "pen_incontext" examples/aloha_mobile_real/
# Should return NO matches
```

---

### Step 1.3: Test Import and Syntax (30 min)

```bash
# Test imports
python3 -c "from examples.aloha_mobile_real.main_incontext import *; print('✓ main_incontext imports OK')"
python3 -c "from examples.aloha_mobile_real.env_incontext import *; print('✓ env_incontext imports OK')"
python3 -c "from examples.aloha_mobile_real.trajectory_recorder import *; print('✓ trajectory_recorder imports OK')"

# Check for syntax errors
python3 -m py_compile examples/aloha_mobile_real/main_incontext.py
python3 -m py_compile examples/aloha_mobile_real/env_incontext.py
```

**Expected Output**: No errors

---

### Step 1.4: Update Enhanced main.py (1 hour)

**Goal**: Add recording and in-context features to standard `main.py`

**Current `main.py`** (54 lines, basic):
```python
# Simple inference loop, no recording
```

**Enhanced `main.py`** from v1.0 (81 lines):
- `--record` flag for trajectory recording
- `--action-horizon` parameter
- `--num-episodes`, `--max-steps-per-episode` limits
- Integration with `trajectory_recorder.py`

**Option A - Replace entirely**:
```bash
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/main.py \
   examples/aloha_mobile_real/main.py
```

**Option B - Merge manually** (safer):
1. Keep current main.py
2. Add missing CLI arguments from v1.0 version
3. Add trajectory recorder integration

**Recommendation**: Use Option B for safety.

**Verification**:
```bash
uv run examples/aloha_mobile_real/main.py --help
# Should show: --record, --action-horizon, --num-episodes flags
```

---

### Phase 1 Checklist

- [ ] Copied `main_incontext.py` (139 lines)
- [ ] Copied `env_incontext.py` (100 lines)
- [ ] Copied `trajectory_recorder.py` (150 lines)
- [ ] Copied `visualize_trajectory_videos.py` (252 lines)
- [ ] Copied `view_trajectory.py` (96 lines)
- [ ] Updated metadata paths (`pen_incontext` → `aloha_pen_uncap`)
- [ ] Verified imports work
- [ ] Enhanced `main.py` with recording flag
- [ ] Tested syntax with `py_compile`

---

## Phase 2: Metadata & Cache Files (Day 1 Afternoon)

**Goal**: Create complete metadata infrastructure
**Duration**: 2-3 hours
**Dependencies**: Phase 1 complete

### Step 2.1: Verify Existing Metadata (15 min)

```bash
cd /home/dingj0b/code/openpi_pr/openpi

# Check what exists
ls -lh metadata/aloha_pen_uncap/
# Expected output:
# task_to_episode.json (405B)
# episode_to_indexes.json (265KB)

# Check if cache files exist (they shouldn't)
ls -lh metadata/aloha_pen_uncap/episode_states*.json 2>/dev/null || echo "✗ Missing"
ls -lh metadata/aloha_pen_uncap/episode_actions*.json 2>/dev/null || echo "✗ Missing"
```

---

### Step 2.2: Decision Point - Copy vs Regenerate Cache Files

**Option A - Copy from v1.0** (FAST: 5 min):

**Pros**:
- Instant (just copy 39MB files)
- Known working data
- No computation needed

**Cons**:
- May not match q_former dataset exactly
- Different preprocessing possible

**Commands**:
```bash
# Copy state cache (19MB)
cp /home/dingj0b/code/openpi/metadata/pen_incontext/episode_states_without_delta_cache.json \
   metadata/aloha_pen_uncap/episode_states_cache.json

# Copy action cache (20MB)
cp /home/dingj0b/code/openpi/metadata/pen_incontext/episode_actions_without_delta_cache.json \
   metadata/aloha_pen_uncap/episode_actions_cache.json

# Verify
ls -lh metadata/aloha_pen_uncap/*.json
# Should show 4 files totaling ~39MB
```

---

**Option B - Regenerate** (SLOW: 1-2 hours):

**Pros**:
- Guaranteed correct for q_former dataset
- Matches current data processing
- Clean slate

**Cons**:
- Requires GPU/compute time
- More complex

**Commands**:
```bash
# Submit cache creation job
sbatch jobs/sbatch_scripts_real/create_cache_pi0_aloha_pen_uncap_*.sh

# Monitor job
squeue -u $USER

# Wait for completion (~1-2 hours)
# Check output
ls -lh metadata/aloha_pen_uncap/
```

**Recommendation**: **Use Option A** for speed, regenerate later if issues arise.

---

### Step 2.3: Copy Additional Metadata Files (15 min)

```bash
# Copy tasks.jsonl if it exists
if [ -f /home/dingj0b/code/openpi/metadata/pen_incontext/tasks.jsonl ]; then
    cp /home/dingj0b/code/openpi/metadata/pen_incontext/tasks.jsonl \
       metadata/aloha_pen_uncap/
    echo "✓ Copied tasks.jsonl"
fi

# Verify complete metadata structure
tree metadata/aloha_pen_uncap/
```

**Expected structure**:
```
metadata/aloha_pen_uncap/
├── task_to_episode.json                      (405B)
├── episode_to_indexes.json                   (265KB)
├── episode_states_cache.json                 (19MB)
├── episode_actions_cache.json                (20MB)
└── tasks.jsonl                               (optional)
```

---

### Step 2.4: Create Objects Dataset Metadata (30-60 min)

**For the missing config**: `pi0_aloha_objects_all_incontextv12_*`

**Step 2.4.1 - Determine Dataset Repo ID**:

```bash
# Search for potential repo_id in SBATCH files
grep -r "objects" jobs/sbatch_scripts_real/pi0_aloha_objects*.sh
```

**Expected**: Find dataset name like `vo2yager/objects_all` or similar

**Step 2.4.2 - Generate Metadata**:

```bash
# First, create the config (see Phase 3)
# Then generate metadata

uv run src/openpi/training/generate_task_to_index.py \
  --config pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select \
  --output_dir metadata/objects_all
```

**Output**:
```
Building lookup tables: 100%|████████| XXXX/XXXX [00:XX<00:00]
✅ Wrote task_to_episode → metadata/objects_all/task_to_episode.json
✅ Wrote episode_to_indexes → metadata/objects_all/episode_to_indexes.json
```

**Step 2.4.3 - Generate Cache Files**:

```bash
# Submit cache creation job for objects
sbatch jobs/sbatch_scripts_real/create_cache_pi0_aloha_objects_all_*.sh

# Or copy from v1.0 if available
if [ -d /home/dingj0b/code/openpi/metadata/objects_pickup_place ]; then
    cp /home/dingj0b/code/openpi/metadata/objects_pickup_place/*.json \
       metadata/objects_all/
fi
```

---

### Step 2.5: Verify Metadata Completeness (15 min)

```bash
# Create verification script
cat > verify_metadata.sh << 'EOF'
#!/bin/bash

check_metadata() {
    local dir=$1
    echo "Checking $dir..."

    if [ ! -f "$dir/task_to_episode.json" ]; then
        echo "  ✗ Missing task_to_episode.json"
    else
        echo "  ✓ task_to_episode.json ($(stat -f%z "$dir/task_to_episode.json" 2>/dev/null || stat -c%s "$dir/task_to_episode.json") bytes)"
    fi

    if [ ! -f "$dir/episode_to_indexes.json" ]; then
        echo "  ✗ Missing episode_to_indexes.json"
    else
        echo "  ✓ episode_to_indexes.json ($(stat -f%z "$dir/episode_to_indexes.json" 2>/dev/null || stat -c%s "$dir/episode_to_indexes.json") bytes)"
    fi

    if [ ! -f "$dir/episode_states_cache.json" ]; then
        echo "  ⚠ Missing episode_states_cache.json (optional)"
    else
        echo "  ✓ episode_states_cache.json ($(stat -f%z "$dir/episode_states_cache.json" 2>/dev/null || stat -c%s "$dir/episode_states_cache.json") bytes)"
    fi

    if [ ! -f "$dir/episode_actions_cache.json" ]; then
        echo "  ⚠ Missing episode_actions_cache.json (optional)"
    else
        echo "  ✓ episode_actions_cache.json ($(stat -f%z "$dir/episode_actions_cache.json" 2>/dev/null || stat -c%s "$dir/episode_actions_cache.json") bytes)"
    fi
}

check_metadata "metadata/aloha_pen_uncap"
check_metadata "metadata/objects_all"
EOF

chmod +x verify_metadata.sh
./verify_metadata.sh
```

---

### Phase 2 Checklist

- [ ] Verified existing metadata (task_to_episode, episode_to_indexes)
- [ ] Copied or regenerated state cache (19MB)
- [ ] Copied or regenerated action cache (20MB)
- [ ] Copied tasks.jsonl (if exists)
- [ ] Created objects_all metadata directory
- [ ] Generated objects_all task_to_episode.json
- [ ] Generated objects_all episode_to_indexes.json
- [ ] Generated objects_all cache files
- [ ] Verified all metadata with verification script

---

## Phase 3: Configuration Alignment (Day 2 Morning)

**Goal**: Fix config incompatibilities and create missing configs
**Duration**: 2-3 hours
**Dependencies**: Phase 1 & 2 complete

### Step 3.1: Fix Repack Transform (CRITICAL - 30 min)

**File**: `src/openpi/training/config_aloha.py`
**Location**: Lines 153-165 in `LeRobotAlohaMobileIncontextDataConfig`

**Current (broken)**:
```python
repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
    default=api._transforms.Group(
        inputs=[
            api._transforms.RepackTransform(
                {
                    "images": {"cam_high": "observation.images.top"},
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )
)
```

**Fixed version**:
```python
repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
    default=api._transforms.Group(
        inputs=[
            api._transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",       # Changed: top → cam_high
                        "cam_left_wrist": "observation.images.cam_left_wrist",  # Added
                        "cam_right_wrist": "observation.images.cam_right_wrist",  # Added
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",                 # Added
                    "episode_index": "episode_index",   # Added
                    "index": "index",                   # Added
                    "task_index": "task_index",         # Added
                }
            )
        ]
    )
)
```

**Impact**: **CRITICAL** - Without this, in-context deployment will fail.

**Verification**:
```bash
# Check the fix
grep -A 15 "repack_transforms.*Group" src/openpi/training/config_aloha.py | grep -c "prompt"
# Should output: 1 (meaning prompt is present)
```

---

### Step 3.2: Update Cache File Paths (20 min)

**File**: `src/openpi/training/config_aloha.py`
**Locations**: Multiple configs using `LeRobotAlohaMobileIncontextDataConfig`

**Find all configs referencing cache paths**:
```bash
grep -n "states_cache_path\|actions_cache_path" src/openpi/training/config_aloha.py
```

**Update example** (in `pi0_aloha_pen_uncap_incontextv12_*` config around line 437-438):

**Current**:
```python
states_cache_path="metadata/libero/episode_states_without_delta_cache.json",  # ← Wrong!
actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",  # ← Wrong!
```

**Fixed**:
```python
states_cache_path="metadata/aloha_pen_uncap/episode_states_cache.json",
actions_cache_path="metadata/aloha_pen_uncap/episode_actions_cache.json",
```

**Verification**:
```bash
# Check no configs reference non-existent files
grep -n "metadata/libero/episode_states" src/openpi/training/config_aloha.py
# Should return NO matches in ALOHA configs
```

---

### Step 3.3: Create Missing Objects Config (1 hour)

**File**: `src/openpi/training/config_aloha.py`
**Add after existing pen_uncap config** (around line 455)

**New config definition**:

```python
# Objects All In-Context Config
api.TrainConfig(
    name="pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
    model=api.pi0_incontextv12.Pi0IncontextConfigv12(
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m_lora",
        sample_frames=2,
        sample_actions=32,
        random_select=True,
    ),
    data=LeRobotAlohaMobileIncontextDataConfig(
        repo_id="vo2yager/objects_all",  # ← VERIFY THIS REPO ID!
        assets=api.AssetsConfig(
            assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen_mobile",
        ),
        default_prompt="pick and place objects",  # ← Customize as needed
        repack_transforms=api._transforms.Group(
            inputs=[
                api._transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "episode_index": "episode_index",
                        "index": "index",
                        "task_index": "task_index",
                    }
                )
            ]
        ),
        base_config=api.DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
        use_delta_joint_actions=False,
        states_cache_path="metadata/objects_all/episode_states_cache.json",
        actions_cache_path="metadata/objects_all/episode_actions_cache.json",
    ),
    weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=20_000,  # Default for base config
    freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m_lora",
        sample_frames=2,
        sample_actions=32,
        random_select=True,
    ).get_freeze_filter(),
    ema_decay=None,
    num_workers=1,
    batch_size=32,
),
```

**Variants to add** (for different training steps):

1. **40k variant**: Copy above, change:
   - `name`: append `_40k`
   - `num_train_steps`: 40_000

2. **80k variant**: Copy above, change:
   - `name`: append `_80k`
   - `num_train_steps`: 80_000

**Verification**:
```bash
# Test config loads
uv run scripts/train.py pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select --help

# Should show config parameters without errors
```

---

### Step 3.4: Document Config Changes (30 min)

Create a config migration guide:

```bash
cat > CONFIG_MIGRATION_NOTES.md << 'EOF'
# Config Migration Notes

## Changes Made

### LeRobotAlohaMobileIncontextDataConfig

1. **Repack Transform - CRITICAL FIX**
   - Added: prompt, episode_index, index, task_index
   - Added: cam_left_wrist, cam_right_wrist
   - Changed: observation.images.top → observation.images.cam_high

2. **Cache File Paths - UPDATED**
   - Changed: metadata/libero/* → metadata/aloha_pen_uncap/*
   - Matches actual file locations

3. **New Config Added**
   - pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select
   - With 20k, 40k, 80k variants

## Testing Required

- [ ] Test config loading
- [ ] Test dataset loading
- [ ] Test in-context demo injection
- [ ] Test training start (1 iteration)
- [ ] Test inference on real robot

## Rollback

If issues arise, revert `config_aloha.py` to commit: $(git rev-parse HEAD)
EOF
```

---

### Phase 3 Checklist

- [ ] Fixed repack transform (added 7 missing fields)
- [ ] Updated cache file paths
- [ ] Created objects_all config (20k variant)
- [ ] Created objects_all_40k config
- [ ] Created objects_all_80k config
- [ ] Verified config loading
- [ ] Tested dataset access
- [ ] Documented changes

---

## Phase 4: Testing & Validation (Day 2 Afternoon)

**Goal**: Validate migration on real robot
**Duration**: 3-4 hours
**Dependencies**: Phases 1, 2, 3 complete

### Step 4.1: Unit Tests (1 hour)

**Test 1: Config Loading**

```bash
# Test all modified configs load
uv run python3 << 'EOF'
from openpi.training.config import get_config

configs_to_test = [
    "pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
    "pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
]

for config_name in configs_to_test:
    try:
        cfg = get_config(config_name)
        print(f"✓ {config_name}: OK")
        print(f"  - Model: {cfg.model.__class__.__name__}")
        print(f"  - Data repo: {cfg.data.repo_id}")
    except Exception as e:
        print(f"✗ {config_name}: FAILED - {e}")
EOF
```

**Expected output**: All configs should load without errors.

---

**Test 2: Dataset Access**

```bash
# Test dataset loads with correct fields
uv run python3 << 'EOF'
from openpi.training.config import get_config
from openpi.training.data_loader import create_dataset

config = get_config("pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select")
data_config = config.data.create(config.assets_dirs, config.model)

dataset = create_dataset(data_config, config.model)

# Check first sample
sample = dataset[0]
required_fields = ['observation', 'action', 'episode_index', 'index', 'task_index', 'prompt']
missing = [f for f in required_fields if f not in sample]

if missing:
    print(f"✗ Missing fields: {missing}")
else:
    print("✓ All required fields present")
    print(f"  - Sample keys: {list(sample.keys())}")
EOF
```

**Expected**: No missing fields.

---

**Test 3: Metadata Loading**

```bash
# Test metadata files load correctly
uv run python3 << 'EOF'
import json

# Test pen_uncap metadata
with open("metadata/aloha_pen_uncap/task_to_episode.json") as f:
    task_to_ep = json.load(f)
    print(f"✓ Pen uncap: {len(task_to_ep)} tasks")

with open("metadata/aloha_pen_uncap/episode_to_indexes.json") as f:
    ep_to_idx = json.load(f)
    print(f"✓ Pen uncap: {len(ep_to_idx)} episodes")

# Test objects_all metadata (if created)
try:
    with open("metadata/objects_all/task_to_episode.json") as f:
        task_to_ep = json.load(f)
        print(f"✓ Objects all: {len(task_to_ep)} tasks")
except FileNotFoundError:
    print("⚠ Objects all metadata not yet created")
EOF
```

---

**Test 4: Environment Initialization**

```bash
# Test env_incontext can load metadata
uv run python3 << 'EOF'
import sys
sys.path.insert(0, "examples/aloha_mobile_real")

from env_incontext import AlohaRealIncontextEnvironment

try:
    env = AlohaRealIncontextEnvironment(
        task_metadata_file="metadata/aloha_pen_uncap/task_to_episode.json"
    )
    print("✓ Environment initialized successfully")
    print(f"  - Available tasks: {len(env.task_to_episode)}")
except Exception as e:
    print(f"✗ Environment initialization failed: {e}")
EOF
```

---

### Step 4.2: Integration Tests (1 hour)

**Test 5: Policy Server Start**

```bash
# Start policy server (non-blocking test)
timeout 30s uv run scripts/serve_policy_incontext.py \
  policy:checkpoint \
  --policy.config=pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_inference \
  --policy.dir=checkpoints/pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select/experiment/19999 &

SERVER_PID=$!

sleep 10

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Policy server started successfully"
    kill $SERVER_PID
else
    echo "✗ Policy server failed to start"
fi
```

---

**Test 6: Dry-Run Inference**

```bash
# Test main_incontext.py can parse arguments
uv run examples/aloha_mobile_real/main_incontext.py --help

# Should show:
# --prompt, --host, --port, --record, etc.
```

---

### Step 4.3: Real Robot Validation (1-2 hours)

**Prerequisites**:
- Policy server running on GPU machine
- Mobile ALOHA robot accessible
- Network connectivity between robot and server

**Test 7: Single Episode Deployment**

```bash
# On robot or control machine:
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick up the pen" \
  --host <GPU_SERVER_IP> \
  --port 8080 \
  --num-episodes 1 \
  --max-steps-per-episode 50 \
  --record

# Expected:
# - Connects to policy server
# - Loads task metadata
# - Executes 1 episode
# - Saves trajectory to HDF5
```

**Validation**:
```bash
# Check trajectory file created
ls -lh trajectory_*.hdf5

# Inspect trajectory
uv run examples/aloha_mobile_real/view_trajectory.py trajectory_ep001.hdf5

# Expected output:
# Episode: episode_001
# Duration: X.X seconds
# Steps: XX
# Actions: shape=(XX, 14)
```

---

**Test 8: Trajectory Visualization**

```bash
# Generate video from recorded trajectory
uv run examples/aloha_mobile_real/visualize_trajectory_videos.py \
  --trajectory trajectory_ep001.hdf5 \
  --output-format video

# Expected:
# Creates trajectory_ep001.mp4 with multi-camera view
```

**Verification**:
```bash
ls -lh trajectory_ep001.mp4
# Should show video file

# Play video to verify quality
# (Use local player or transfer to local machine)
```

---

**Test 9: Multi-Episode with Different Tasks**

```bash
# Test 3 episodes with different prompts
for prompt in "pick up the pen" "uncap the pen" "place the pen in the cup"; do
    uv run examples/aloha_mobile_real/main_incontext.py \
      --prompt "$prompt" \
      --host <GPU_SERVER_IP> \
      --port 8080 \
      --num-episodes 1 \
      --record
done

# Verify 3 trajectory files created
ls -lh trajectory_*.hdf5 | wc -l
# Should output: 3
```

---

### Step 4.4: Performance Validation (30 min)

**Test 10: Inference Latency**

```bash
# Measure end-to-end latency
uv run python3 << 'EOF'
import time
import numpy as np

# Simulate inference call timing
# (In production, measure actual robot → server → robot round-trip)

latencies = []
for i in range(10):
    start = time.time()
    # Placeholder: actual inference would go here
    time.sleep(0.05)  # Simulate 50ms inference
    latency = time.time() - start
    latencies.append(latency)

print(f"Average latency: {np.mean(latencies)*1000:.1f}ms")
print(f"Max latency: {np.max(latencies)*1000:.1f}ms")
print(f"Min latency: {np.min(latencies)*1000:.1f}ms")

# Target: < 100ms for real-time control
if np.mean(latencies) < 0.1:
    print("✓ Latency acceptable for real-time control")
else:
    print("⚠ Latency may be too high")
EOF
```

---

### Step 4.5: Error Handling Tests (30 min)

**Test 11: Missing Metadata Graceful Failure**

```bash
# Test with non-existent metadata file
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "nonexistent task" \
  --host <GPU_SERVER_IP> \
  --port 8080 2>&1 | grep -i "error\|exception"

# Should show clear error message, not crash
```

---

**Test 12: Network Failure Handling**

```bash
# Test with wrong server address
timeout 10s uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick up the pen" \
  --host 1.2.3.4 \
  --port 9999 2>&1 | grep -i "connection\|timeout"

# Should show connection error, not hang indefinitely
```

---

### Phase 4 Checklist

- [ ] Config loading tests pass
- [ ] Dataset access tests pass
- [ ] Metadata loading tests pass
- [ ] Environment initialization tests pass
- [ ] Policy server starts successfully
- [ ] main_incontext.py dry-run works
- [ ] Single episode deployment succeeds
- [ ] Trajectory file created correctly
- [ ] Trajectory visualization works
- [ ] Multi-episode deployment works
- [ ] Inference latency acceptable
- [ ] Error handling graceful

---

## Phase 5: Optional Enhancements (Day 3)

**Goal**: Add production-ready features
**Duration**: 4-8 hours
**Dependencies**: Phases 1-4 complete and validated

### Enhancement 1: Automated Metadata Generation (2 hours)

Create script to auto-generate all metadata for a new dataset:

```bash
cat > scripts/setup_aloha_dataset.sh << 'EOF'
#!/bin/bash
# Usage: ./setup_aloha_dataset.sh <dataset_name> <repo_id>

DATASET_NAME=$1
REPO_ID=$2

if [ -z "$DATASET_NAME" ] || [ -z "$REPO_ID" ]; then
    echo "Usage: $0 <dataset_name> <repo_id>"
    echo "Example: $0 objects_all vo2yager/objects_all"
    exit 1
fi

echo "Setting up dataset: $DATASET_NAME"
echo "Repo ID: $REPO_ID"

# Create metadata directory
mkdir -p metadata/$DATASET_NAME

# Generate task_to_episode and episode_to_indexes
echo "Generating metadata files..."
uv run src/openpi/training/generate_task_to_index.py \
  --config ${DATASET_NAME}_config \
  --output_dir metadata/$DATASET_NAME

# Generate cache files (submit SBATCH job)
echo "Submitting cache generation job..."
sbatch jobs/sbatch_scripts_real/create_cache_${DATASET_NAME}.sh

echo "✓ Metadata generation started"
echo "  Check job status with: squeue -u $USER"
EOF

chmod +x scripts/setup_aloha_dataset.sh
```

---

### Enhancement 2: Unified Deployment Script (2 hours)

Create single script for common deployment scenarios:

```bash
cat > scripts/deploy_aloha_incontext.sh << 'EOF'
#!/bin/bash
# Unified mobile ALOHA in-context deployment script

# Configuration
SERVER_HOST=${ALOHA_SERVER_HOST:-"localhost"}
SERVER_PORT=${ALOHA_SERVER_PORT:-8080}
RECORD=${ALOHA_RECORD:-true}
NUM_EPISODES=${ALOHA_NUM_EPISODES:-5}

# Parse arguments
TASK_PROMPT=$1
if [ -z "$TASK_PROMPT" ]; then
    echo "Usage: $0 \"<task prompt>\" [options]"
    echo "Example: $0 \"pick up the pen\" --episodes 3"
    exit 1
fi

shift  # Remove task prompt from args

# Override defaults with CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) SERVER_HOST=$2; shift 2 ;;
        --port) SERVER_PORT=$2; shift 2 ;;
        --episodes) NUM_EPISODES=$2; shift 2 ;;
        --no-record) RECORD=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build command
CMD="uv run examples/aloha_mobile_real/main_incontext.py"
CMD="$CMD --prompt \"$TASK_PROMPT\""
CMD="$CMD --host $SERVER_HOST"
CMD="$CMD --port $SERVER_PORT"
CMD="$CMD --num-episodes $NUM_EPISODES"
if [ "$RECORD" = true ]; then
    CMD="$CMD --record"
fi

echo "Deploying: $TASK_PROMPT"
echo "Server: $SERVER_HOST:$SERVER_PORT"
echo "Episodes: $NUM_EPISODES"
echo "Recording: $RECORD"
echo ""

eval $CMD
EOF

chmod +x scripts/deploy_aloha_incontext.sh
```

**Usage**:
```bash
# Simple usage
./scripts/deploy_aloha_incontext.sh "pick up the pen"

# With options
./scripts/deploy_aloha_incontext.sh "pick up the pen" --host 10.0.0.1 --episodes 3
```

---

### Enhancement 3: Performance Monitoring (2 hours)

Add telemetry to deployment scripts:

```python
# Add to main_incontext.py

import time
import json
from collections import defaultdict

class DeploymentMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, metric_name, value):
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": time.time()
        })

    def save(self, filename="deployment_metrics.json"):
        with open(filename, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)
        print(f"✓ Saved metrics to {filename}")

# Usage in main loop:
metrics = DeploymentMetrics()

for episode in range(num_episodes):
    start_time = time.time()

    # ... deployment code ...

    episode_duration = time.time() - start_time
    metrics.record("episode_duration", episode_duration)
    metrics.record("steps_per_episode", num_steps)
    metrics.record("avg_inference_latency", np.mean(inference_latencies))

metrics.save()
```

---

### Enhancement 4: Automated Testing Suite (2 hours)

Create comprehensive test suite:

```bash
cat > tests/test_aloha_migration.py << 'EOF'
import pytest
import json
from pathlib import Path

def test_metadata_exists():
    """Test all required metadata files exist"""
    assert Path("metadata/aloha_pen_uncap/task_to_episode.json").exists()
    assert Path("metadata/aloha_pen_uncap/episode_to_indexes.json").exists()

def test_metadata_valid():
    """Test metadata files are valid JSON"""
    with open("metadata/aloha_pen_uncap/task_to_episode.json") as f:
        data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

def test_config_loads():
    """Test critical configs load without errors"""
    from openpi.training.config import get_config

    configs = [
        "pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
    ]

    for config_name in configs:
        cfg = get_config(config_name)
        assert cfg is not None
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'data')

def test_repack_transform_has_required_fields():
    """Test repack transform includes all required fields"""
    from openpi.training.config import get_config

    cfg = get_config("pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select")

    # Check repack transform includes metadata fields
    repack = cfg.data.repack_transforms.inputs[0]
    mapping = repack.mapping

    required_fields = ["prompt", "episode_index", "index", "task_index"]
    for field in required_fields:
        assert field in mapping, f"Missing required field: {field}"

def test_cache_files_accessible():
    """Test cache files exist and are readable"""
    cache_files = [
        "metadata/aloha_pen_uncap/episode_states_cache.json",
        "metadata/aloha_pen_uncap/episode_actions_cache.json",
    ]

    for cache_file in cache_files:
        path = Path(cache_file)
        if path.exists():  # Optional files
            assert path.stat().st_size > 0, f"{cache_file} is empty"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF
```

**Run tests**:
```bash
uv run pytest tests/test_aloha_migration.py -v
```

---

### Phase 5 Checklist

- [ ] Created automated metadata generation script
- [ ] Created unified deployment script
- [ ] Added performance monitoring
- [ ] Created automated test suite
- [ ] All tests passing

---

## 9. Rollback Plan

If issues arise, use this rollback procedure:

### Step 1: Identify Current Commit

```bash
cd /home/dingj0b/code/openpi_pr/openpi
git log --oneline -1
# Note the commit hash (e.g., abc1234)
```

### Step 2: Create Rollback Branch

```bash
# Create safety branch before rollback
git checkout -b migration-backup-$(date +%Y%m%d)
git checkout q_former  # Return to main branch
```

### Step 3: Selective Rollback

**If only config changes failed**:
```bash
# Rollback just config_aloha.py
git checkout HEAD~1 src/openpi/training/config_aloha.py
```

**If deployment scripts have issues**:
```bash
# Remove copied files
rm examples/aloha_mobile_real/main_incontext.py
rm examples/aloha_mobile_real/env_incontext.py
rm examples/aloha_mobile_real/trajectory_recorder.py
# etc.
```

**Full rollback**:
```bash
# Rollback all changes
git reset --hard <commit_hash_before_migration>

# Remove metadata if needed
rm -rf metadata/aloha_pen_uncap/*cache.json
rm -rf metadata/objects_all/
```

### Step 4: Verify Rollback

```bash
# Test config loading still works
uv run python -c "from openpi.training.config import get_config; print('OK')"

# Test existing functionality
uv run scripts/train.py pi0_aloha_mobile --help
```

---

## 10. Post-Migration Checklist

### ✅ Code Integration

- [ ] All deployment files copied and updated
- [ ] Metadata paths corrected
- [ ] Config repack transforms fixed
- [ ] Cache files generated or copied
- [ ] Missing configs created

### ✅ Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Single episode deployment works
- [ ] Multi-episode deployment works
- [ ] Trajectory recording works
- [ ] Video visualization works

### ✅ Documentation

- [ ] Migration documented in this file
- [ ] Config changes documented
- [ ] Known issues documented
- [ ] Rollback procedure tested

### ✅ Production Readiness

- [ ] Performance validated (latency < 100ms)
- [ ] Error handling verified
- [ ] Monitoring in place
- [ ] Team trained on new workflow

---

## 11. Success Criteria

The migration is **successful** when:

1. ✅ **Deployment works**: Can run `main_incontext.py` on real robot
2. ✅ **In-context learning works**: Task prompts correctly select demonstrations
3. ✅ **Recording works**: Trajectories saved to HDF5 and can be visualized
4. ✅ **Performance acceptable**: Inference latency < 100ms
5. ✅ **No regressions**: Existing training workflows still work
6. ✅ **Documented**: Team understands new workflow

---

## 12. Known Limitations

### Architectural Differences

1. **v1.0 uses transform-based in-context**, **q_former uses dataset-based**
   - These are fundamentally different approaches
   - May see performance differences
   - Monitoring recommended

2. **Sample frames default changed** (16 → 2)
   - Models trained with different settings may not transfer
   - Document carefully

3. **Stage-wise prompting in PR only**
   - Advanced feature not in v1.0
   - Enable with `all_episode_stage` parameter

### Missing from v1.0

- Inference caching (PR has it, v1.0 doesn't)
- Stage-wise hierarchical prompting
- Episode validation with fallbacks

### Future Work

- Port PR's `CustomLeRobotDataset` back to v1.0 (if needed)
- Unify metadata path conventions
- Merge config systems (if branches merge)

---

## 13. Contact & Support

**Migration Owner**: [Your name]
**Date Completed**: [To be filled after migration]
**Issues**: File in repository issue tracker
**Documentation**: See related docs:
- `MOBILE_ALOHA_MISSING_COMPONENTS.md`
- `BRANCH_DIVERGENCE_ANALYSIS.md`
- `CONFIG_SYSTEM_CHANGES.md`
- `README_DIVERGENCE.md`

---

**End of Migration Plan**

**Version**: 1.0
**Last Updated**: 2025-11-10
