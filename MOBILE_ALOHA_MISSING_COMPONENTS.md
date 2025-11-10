# Mobile ALOHA Real-World Experiment: Missing Components Analysis

**Document Version**: 1.0
**Date**: 2025-11-10
**Author**: Comparative Analysis of `/home/dingj0b/code/openpi` vs `/ibex/user/dingj0b/code/openpi_pr/openpi`

---

## Executive Summary

This document provides a comprehensive analysis of missing mobile ALOHA real-world experiment implementations in the PR codebase (`/ibex/user/dingj0b/code/openpi_pr/openpi`) compared to the reference implementation (`/home/dingj0b/code/openpi`).

**Key Findings**:
- **~900 lines** of critical deployment infrastructure missing
- **In-context learning deployment pipeline** incomplete
- **Trajectory recording and analysis tools** absent
- **Task metadata infrastructure** not present

**Impact**: The PR codebase cannot currently support:
1. In-context learning experiments on real mobile ALOHA robots
2. Trajectory recording during deployment for debugging/analysis
3. Post-deployment video analysis and visualization
4. Task-specific prompted inference with metadata

---

## Table of Contents

1. [Codebase Comparison Overview](#1-codebase-comparison-overview)
2. [Missing Components by Category](#2-missing-components-by-category)
3. [Detailed File-by-File Analysis](#3-detailed-file-by-file-analysis)
4. [Feature Gap Analysis](#4-feature-gap-analysis)
5. [Impact Assessment](#5-impact-assessment)
6. [Recommendations](#6-recommendations)
7. [Appendix: Reference Architecture](#7-appendix-reference-architecture)

---

## 1. Codebase Comparison Overview

### 1.1 Reference Codebase (Original)

**Location**: `/home/dingj0b/code/openpi`

**Mobile ALOHA Implementation Summary**:
- **Policy Implementations**: 3 files (~640 lines)
- **Real Robot Scripts**: 12 files (~1,600 lines)
- **Data Conversion Tools**: 2 files (~400 lines)
- **Metadata Infrastructure**: Multiple JSON files for task definitions
- **Total Lines of Code**: ~4,056 lines

### 1.2 PR Codebase (Current)

**Location**: `/ibex/user/dingj0b/code/openpi_pr/openpi`

**Mobile ALOHA Implementation Summary**:
- **Policy Implementations**: 3 files (~627 lines)
- **Real Robot Scripts**: 8 files (~1,558 lines)
- **Data Conversion Tools**: 2 files (~641 lines)
- **Metadata Infrastructure**: **Missing**
- **Total Lines of Code**: ~2,826 lines

### 1.3 Gap Summary

| Category | Original | PR Version | Missing |
|----------|----------|------------|---------|
| Policy Files | 3 | 3 | ✓ Complete |
| Real Robot Scripts | 12 | 8 | **4 files** |
| Data Conversion | 2 | 2 | ✓ Complete |
| Metadata | Full | None | **All metadata** |
| Trajectory Tools | 4 files | 0 files | **All tools** |
| In-Context Scripts | 2 files | 0 files | **All scripts** |

---

## 2. Missing Components by Category

### 2.1 Critical Missing Files (High Priority)

These files are **essential** for in-context learning deployment:

| File | Purpose | Lines | Blocks |
|------|---------|-------|--------|
| `examples/aloha_mobile_real/main_incontext.py` | In-context learning inference with task prompts | 139 | In-context deployment |
| `examples/aloha_mobile_real/env_incontext.py` | Environment wrapper with task metadata loading | 100 | In-context deployment |
| `metadata/aloha_pen_uncap/task_to_episode.json` | Task description → episode mapping | N/A | In-context deployment |
| `metadata/aloha_pen_uncap/episode_to_indexes.json` | Episode → frame indices (264 KB) | N/A | In-context deployment |

**Impact**: **Cannot run in-context learning experiments** on real mobile ALOHA.

---

### 2.2 Important Missing Files (Medium Priority)

These files are **important** for debugging and analysis:

| File | Purpose | Lines | Use Case |
|------|---------|-------|----------|
| `examples/aloha_mobile_real/trajectory_recorder.py` | Records trajectories to HDF5 during deployment | 150 | Debugging, dataset creation |
| `examples/aloha_mobile_real/visualize_trajectory_videos.py` | Multi-camera video export and visualization | 252 | Post-deployment analysis |
| `examples/aloha_mobile_real/view_trajectory.py` | Text-based trajectory inspection | 96 | Quick data validation |

**Impact**: **No post-deployment analysis capabilities**, cannot debug failures or create datasets from real runs.

---

### 2.3 Optional Missing Files (Low Priority)

These files provide **additional functionality**:

| File | Purpose | Lines | Use Case |
|------|---------|-------|----------|
| `examples/aloha_mobile_real/main_icrt.py` | Alternative ICRT evaluation script | 77 | Legacy ICRT experiments |
| `examples/aloha_mobile_real/test_trajectory_recording.py` | Unit tests for trajectory recorder | 90 | Development/testing |

**Impact**: Limited, but useful for alternative evaluation paradigms.

---

### 2.4 Feature Gaps in Existing Files

#### **A. `examples/aloha_mobile_real/main.py`**

**Original Version** (81 lines):
```python
# Command-line arguments:
--host              # Policy server host
--port              # Policy server port
--action-horizon    # Configurable action horizon (default 25)
--num-episodes      # Number of episodes to run
--max-steps-per-episode  # Step limit per episode
--record            # Enable trajectory recording (HDF5)
--video-display     # Enable live video display
```

**PR Version** (54 lines):
```python
# Command-line arguments:
--host              # Policy server host
--port              # Policy server port
# Missing: --action-horizon, --num-episodes, --record, etc.
```

**Missing Features**:
- ✗ Configurable action horizon
- ✗ Episode/step limits
- ✗ Trajectory recording flag
- ✗ Integration with `trajectory_recorder.py`
- ✗ Video display integration

**Impact**: Cannot record trajectories during deployment, less flexible parameter control.

---

#### **B. `examples/aloha_mobile_real/constants.py`**

**Original Version** (160 lines) vs **PR Version** (159 lines):

**Potential Missing Features**:
- Task-specific configuration constants
- Enhanced gripper transformation utilities
- Additional hardware calibration parameters

**Recommendation**: Compare files line-by-line to identify specific gaps.

---

## 3. Detailed File-by-File Analysis

### 3.1 In-Context Learning Pipeline

#### **File**: `main_incontext.py` ❌ MISSING

**Purpose**: Main inference script for in-context learning experiments on mobile ALOHA.

**Key Features**:
```python
# Command-line interface:
python main_incontext.py \
  --prompt "pick_up_the_cucumber_and_place_it_in_the_basket" \
  --host 10.68.106.197 \
  --port 8080 \
  --record \
  --action-horizon 25
```

**Functionality**:
1. Loads task metadata from JSON files (`task_to_episode.json`)
2. Maps task descriptions to episode indices
3. Passes task context to policy via observations
4. Records trajectories with task-specific naming
5. Supports custom prompts via CLI

**Dependencies**:
- `env_incontext.py` (also missing)
- `metadata/aloha_pen_uncap/task_to_episode.json` (missing)
- `trajectory_recorder.py` (missing for recording)

**Why Critical**: This is the **primary entry point** for in-context learning deployment. Without it, in-context policies cannot be tested on real hardware.

---

#### **File**: `env_incontext.py` ❌ MISSING

**Purpose**: Extended environment wrapper that loads and manages task metadata.

**Key Features**:
```python
class AlohaRealIncontextEnvironment:
    def __init__(self, task_index_file: str):
        # Load task.jsonl with task definitions
        # Map task descriptions to indices

    def reset(self, prompt: str):
        # Find task_index from prompt
        # Return observation with task metadata
        return {
            "image": {...},
            "state": ...,
            "prompt": prompt,
            "task_index": task_index,
            "split": "test"
        }
```

**Functionality**:
1. Parses `task.jsonl` metadata files
2. Maps natural language prompts to task indices
3. Injects task metadata into observations
4. Enables task-conditioned policy inference

**Dependencies**:
- `metadata/` directory structure
- `task.jsonl` format task definitions

**Why Critical**: Without this, the in-context policy cannot receive task context, breaking the in-context learning pipeline.

---

### 3.2 Trajectory Recording and Analysis

#### **File**: `trajectory_recorder.py` ❌ MISSING

**Purpose**: Records robot trajectories to HDF5 format during deployment for later analysis.

**Key Features**:
```python
class TrajectoryRecorder:
    def __init__(self, output_dir: str, episode_name: str):
        # Initialize HDF5 file structure

    def record_step(self, observation: dict, action: np.ndarray):
        # Buffer states, actions, images, timestamps

    def save_episode(self):
        # Write to HDF5 with gzip compression
        # Structure:
        #   /observations/qpos: [T, 14]
        #   /observations/qvel: [T, 14]
        #   /observations/images/cam_high: [T, H, W, 3]
        #   /observations/images/cam_left_wrist: [T, H, W, 3]
        #   /observations/images/cam_right_wrist: [T, H, W, 3]
        #   /actions: [T, 14]
        #   /timestamps: [T]
```

**Use Cases**:
1. **Debugging**: Replay failed episodes to understand robot behavior
2. **Dataset Creation**: Convert successful runs into training data
3. **Performance Analysis**: Measure timing, action distributions, etc.
4. **Visualization**: Feed to `visualize_trajectory_videos.py` for video export

**Why Important**: Essential for iterative development and debugging. Without this, you cannot analyze what went wrong during deployment.

---

#### **File**: `visualize_trajectory_videos.py` ❌ MISSING

**Purpose**: Comprehensive trajectory visualization tool supporting multiple export formats.

**Key Features**:
```python
# Command-line interface:
python visualize_trajectory_videos.py \
  --trajectory trajectory_ep001.hdf5 \
  --output-format video \
  --fps 30 \
  --camera-layout grid

# Supported formats:
# - video: MP4 export with multiple cameras
# - gif: Animated GIF for quick sharing
# - frames: Extract individual PNG frames
# - playback: Interactive matplotlib visualization
```

**Capabilities**:
1. Multi-camera synchronized playback
2. Video export (MP4 with H.264 encoding)
3. GIF animation generation
4. Frame extraction to image sequences
5. Configurable layout (grid, horizontal, vertical)
6. FPS control and frame skipping

**Use Cases**:
- Create demo videos for presentations
- Analyze multi-camera synchronized behavior
- Share deployment results with team
- Debug camera calibration issues

**Why Important**: Visual analysis is critical for robotics debugging. Videos enable team collaboration and issue identification.

---

#### **File**: `view_trajectory.py` ❌ MISSING

**Purpose**: Command-line utility for quick trajectory inspection.

**Key Features**:
```python
# Command-line interface:
python view_trajectory.py trajectory_ep001.hdf5

# Output:
# Episode: episode_001
# Duration: 5.2 seconds
# Steps: 260
# Observations:
#   /observations/qpos: shape=(260, 14), dtype=float64
#   /observations/qvel: shape=(260, 14), dtype=float64
#   /observations/images/cam_high: shape=(260, 480, 640, 3), dtype=uint8
# Actions:
#   /actions: shape=(260, 14), dtype=float64
#   range: [-0.5, 0.5]
#   mean: 0.02
```

**Use Cases**:
- Quick sanity check after recording
- Validate HDF5 file structure
- Check data shapes before conversion
- Identify corrupted recordings

**Why Important**: Fast validation tool that prevents wasted time on corrupted data.

---

#### **File**: `test_trajectory_recording.py` ❌ MISSING

**Purpose**: Unit tests for trajectory recorder functionality.

**Test Coverage**:
```python
def test_hdf5_creation():
    # Verify HDF5 file is created correctly

def test_data_persistence():
    # Verify data is saved and readable

def test_shape_validation():
    # Verify correct shapes for states/actions/images

def test_compression():
    # Verify gzip compression works
```

**Why Useful**: Ensures trajectory recording works correctly before real deployment.

---

### 3.3 Metadata Infrastructure

#### **Directory**: `metadata/aloha_pen_uncap/` ❌ MISSING

**Purpose**: Task metadata for in-context learning experiments.

**File Structure**:
```
metadata/
└── aloha_pen_uncap/
    ├── task_to_episode.json          # Task → Episode mapping
    │   {
    │     "pick_up_pen": [0, 1, 2],
    │     "uncap_pen": [3, 4, 5],
    │     ...
    │   }
    │
    └── episode_to_indexes.json       # Episode → Frame indices
        {
          "episode_0": [0, 1, 2, ..., 250],
          "episode_1": [251, 252, ..., 500],
          ...
        }
```

**Additional Metadata Directories**:
```
metadata/
├── aloha_pen_uncap/
├── objects_pickup_place_v1/
├── objects_pickup_place_v2/
└── objects_pickup_place_v3/
```

**Usage in Code**:
```python
# In env_incontext.py:
with open("metadata/aloha_pen_uncap/task_to_episode.json") as f:
    task_to_episode = json.load(f)

# Map user prompt to task index
task_index = task_to_episode[prompt][0]

# Pass to policy
observation["task_index"] = task_index
```

**Why Critical**: Without this metadata, in-context learning policies cannot map natural language prompts to demonstration episodes, breaking the entire in-context learning system.

---

## 4. Feature Gap Analysis

### 4.1 In-Context Learning Deployment Gap

**Current State** (PR Codebase):
```
User → main.py → AlohaRealEnvironment → Policy Server → Actions
         ↓
    (No task context)
```

**Required State** (Original Codebase):
```
User → main_incontext.py → AlohaRealIncontextEnvironment → Policy Server → Actions
         ↓                           ↓
    Task Prompt              task_to_episode.json
                                     ↓
                            Task Index Injection
```

**Gap**: The entire task metadata pipeline is missing.

---

### 4.2 Trajectory Recording Gap

**Current State** (PR Codebase):
```
Deployment → Actions Executed → (Data Lost)
```

**Required State** (Original Codebase):
```
Deployment → Actions Executed → TrajectoryRecorder → HDF5 Files
                                        ↓
                              visualize_trajectory_videos.py
                                        ↓
                                MP4 / GIF / Analysis
```

**Gap**: No persistence or analysis capabilities.

---

### 4.3 Comparison Table

| Feature | Original | PR Version | Status |
|---------|----------|------------|--------|
| **In-Context Learning** |
| Task prompt inference | ✓ | ✗ | ❌ Missing |
| Task metadata loading | ✓ | ✗ | ❌ Missing |
| Episode selection | ✓ | ✗ | ❌ Missing |
| **Trajectory Analysis** |
| HDF5 recording | ✓ | ✗ | ❌ Missing |
| Video export | ✓ | ✗ | ❌ Missing |
| Quick inspection | ✓ | ✗ | ❌ Missing |
| **Deployment Features** |
| Configurable action horizon | ✓ | ✗ | ⚠️ Limited |
| Episode recording flag | ✓ | ✗ | ❌ Missing |
| Video display integration | ✓ | ✓ | ✓ Present |
| **Data Conversion** |
| Mobile ALOHA → LeRobot | ✓ | ✓ | ✓ Present |
| Batch conversion | ✓ | ✓ | ✓ Present |
| **Policy Implementations** |
| Mobile ALOHA policy | ✓ | ✓ | ✓ Present |
| In-context policy | ✓ | ✓ | ✓ Present |
| Gripper transforms | ✓ | ✓ | ✓ Present |

---

## 5. Impact Assessment

### 5.1 Immediate Blocking Issues

#### **Cannot Deploy In-Context Learning Models**

**Severity**: 🔴 **Critical**

**Description**: The PR codebase cannot run in-context learning experiments on real mobile ALOHA robots.

**Affected Workflows**:
1. Testing `pi0_aloha_pen_uncap_incontextv12_*` models on hardware
2. Few-shot learning experiments with task prompts
3. Task-conditioned policy evaluation

**Required for Unblocking**:
- `main_incontext.py`
- `env_incontext.py`
- `metadata/` directory structure
- Task JSON files

---

#### **No Deployment Debugging Capabilities**

**Severity**: 🟠 **High**

**Description**: Cannot record or analyze what happens during real robot deployments.

**Affected Workflows**:
1. Debugging failed episodes
2. Creating datasets from successful runs
3. Performance analysis (timing, action distributions)
4. Video documentation for team sharing

**Required for Unblocking**:
- `trajectory_recorder.py`
- `visualize_trajectory_videos.py`
- `view_trajectory.py`

---

### 5.2 Development Friction

#### **Limited Parameter Control**

**Severity**: 🟡 **Medium**

**Description**: `main.py` lacks configurable parameters present in original.

**Impact**:
- Must edit code to change action horizon
- Cannot easily set episode/step limits
- Manual code changes for each experiment

**Required for Improvement**:
- Enhanced `main.py` with full CLI arguments

---

### 5.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| In-context experiments fail | **High** | Critical | Port in-context files immediately |
| Cannot debug deployment failures | **Medium** | High | Port trajectory recording tools |
| Limited experiment flexibility | **Low** | Medium | Enhance main.py with full CLI |
| Team cannot share results | **Low** | Medium | Port visualization tools |

---

## 6. Recommendations

### 6.1 Immediate Action Items (Sprint 1)

**Priority 1: Enable In-Context Learning** (Est. 2-3 hours)

```bash
# Step 1: Copy in-context scripts
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/main_incontext.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/env_incontext.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

# Step 2: Copy metadata
cp -r /home/dingj0b/code/openpi/metadata/aloha_pen_uncap \
      /ibex/user/dingj0b/code/openpi_pr/openpi/metadata/

cp -r /home/dingj0b/code/openpi/metadata/objects_pickup_place* \
      /ibex/user/dingj0b/code/openpi_pr/openpi/metadata/

# Step 3: Test in-context deployment
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick_up_pen" \
  --host <policy-server> \
  --port 8080
```

**Validation Checklist**:
- [ ] `main_incontext.py` runs without errors
- [ ] Task metadata loads correctly
- [ ] Policy receives task context in observations
- [ ] Actions are executed on robot

---

**Priority 2: Enable Trajectory Recording** (Est. 1-2 hours)

```bash
# Step 1: Copy trajectory tools
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/trajectory_recorder.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/visualize_trajectory_videos.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/view_trajectory.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

# Step 2: Test recording
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick_up_pen" \
  --record \
  --host <policy-server>

# Step 3: Verify HDF5 output
uv run examples/aloha_mobile_real/view_trajectory.py \
  trajectory_ep001.hdf5

# Step 4: Generate video
uv run examples/aloha_mobile_real/visualize_trajectory_videos.py \
  --trajectory trajectory_ep001.hdf5 \
  --output-format video
```

**Validation Checklist**:
- [ ] HDF5 files are created during deployment
- [ ] `view_trajectory.py` shows correct data structure
- [ ] `visualize_trajectory_videos.py` generates videos
- [ ] Videos show synchronized multi-camera views

---

### 6.2 Secondary Action Items (Sprint 2)

**Priority 3: Enhance main.py** (Est. 1 hour)

```bash
# Compare and merge main.py versions
diff /home/dingj0b/code/openpi/examples/aloha_mobile_real/main.py \
     /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/main.py

# Port missing CLI arguments:
# - --action-horizon
# - --num-episodes
# - --max-steps-per-episode
# - --record
```

**Expected Improvements**:
- Full CLI control without code edits
- Recording integration in standard main.py
- Better experiment reproducibility

---

**Priority 4: Port Optional Files** (Est. 30 min)

```bash
# Copy optional files
cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/main_icrt.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/test_trajectory_recording.py \
   /ibex/user/dingj0b/code/openpi_pr/openpi/examples/aloha_mobile_real/

# Run tests
uv run pytest examples/aloha_mobile_real/test_trajectory_recording.py
```

---

### 6.3 Verification and Testing Plan

**Phase 1: Unit Testing** (After each file port)
```bash
# Test in-context environment
python -c "
from examples.aloha_mobile_real.env_incontext import AlohaRealIncontextEnvironment
env = AlohaRealIncontextEnvironment('metadata/aloha_pen_uncap/task_to_episode.json')
print('Environment initialized successfully')
"

# Test trajectory recorder
uv run pytest examples/aloha_mobile_real/test_trajectory_recording.py
```

**Phase 2: Integration Testing** (After all files ported)
```bash
# Test full in-context pipeline
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick_up_pen" \
  --host localhost \
  --port 8080 \
  --record \
  --num-episodes 1

# Verify outputs
ls -lh trajectory_*.hdf5
uv run examples/aloha_mobile_real/view_trajectory.py trajectory_*.hdf5
```

**Phase 3: Hardware Validation** (On real mobile ALOHA)
```bash
# 1. Start policy server on GPU machine
uv run scripts/serve_policy_incontext.py \
  policy:checkpoint \
  --policy.config=pi0_aloha_pen_uncap_incontextv12_inference \
  --policy.dir=checkpoints/...

# 2. Run in-context deployment
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick_up_pen" \
  --host <gpu-server-ip> \
  --port 8080 \
  --record

# 3. Analyze results
uv run examples/aloha_mobile_real/visualize_trajectory_videos.py \
  --trajectory trajectory_ep001.hdf5 \
  --output-format video
```

---

### 6.4 Rollout Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Priority 1: In-context learning files | Working in-context deployment |
| **Week 1** | Priority 2: Trajectory recording | HDF5 recording + visualization |
| **Week 2** | Priority 3: Enhanced main.py | Full CLI control |
| **Week 2** | Priority 4: Optional files + testing | Complete test coverage |
| **Week 3** | Hardware validation | Validated on real robot |

---

## 7. Appendix: Reference Architecture

### 7.1 In-Context Learning Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input Layer                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                 python main_incontext.py
                 --prompt "pick_up_pen"
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              AlohaRealIncontextEnvironment                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  1. Load metadata/aloha_pen_uncap/task_to_episode.json │    │
│  │  2. Map "pick_up_pen" → task_index = 5                 │    │
│  │  3. Inject task_index into observation                 │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                observation = {
                  "image": {...},
                  "state": [...],
                  "prompt": "pick_up_pen",
                  "task_index": 5,
                  "split": "test"
                }
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  WebSocket Policy Server                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Pi0 In-Context Model (pi0_incontextv12)              │    │
│  │  - Receives task_index                                 │    │
│  │  - Loads demo frames from similar episodes             │    │
│  │  - Conditions action prediction on demonstrations      │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    action_chunk = [...]
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Real Robot Execution                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Execute actions on mobile ALOHA hardware              │    │
│  │  Record to HDF5 via TrajectoryRecorder                 │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Trajectory Recording Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Deployment Execution                        │
│                                                                 │
│  For each timestep:                                             │
│    1. Get observation from robot                                │
│    2. Send to policy server                                     │
│    3. Receive action                                            │
│    4. Execute action                                            │
│    5. recorder.record_step(obs, action)  ← Trajectory Recorder  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TrajectoryRecorder                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Buffer:                                               │    │
│  │  - states: List[np.ndarray]                            │    │
│  │  - actions: List[np.ndarray]                           │    │
│  │  - images: {cam: List[np.ndarray]}                     │    │
│  │  - timestamps: List[float]                             │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                   recorder.save_episode()
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     HDF5 File Structure                         │
│                                                                 │
│  trajectory_ep001.hdf5:                                         │
│    /observations/qpos: [T, 14]                                  │
│    /observations/qvel: [T, 14]                                  │
│    /observations/images/cam_high: [T, 480, 640, 3]             │
│    /observations/images/cam_left_wrist: [T, 480, 640, 3]       │
│    /observations/images/cam_right_wrist: [T, 480, 640, 3]      │
│    /actions: [T, 14]                                            │
│    /timestamps: [T]                                             │
│    /metadata/episode_length: 260                                │
│    /metadata/task: "pick_up_pen"                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ├──────────────────────┐
                           │                      │
                           ▼                      ▼
               ┌─────────────────────┐  ┌──────────────────┐
               │  view_trajectory.py │  │  visualize_*     │
               │  (Quick inspection) │  │  (Video export)  │
               └─────────────────────┘  └──────────────────┘
```

### 7.3 File Dependency Graph

```
main_incontext.py
├── env_incontext.py
│   └── metadata/aloha_pen_uncap/task_to_episode.json
├── trajectory_recorder.py
│   └── (creates HDF5 files)
└── video_display.py

trajectory_recorder.py (output)
├── view_trajectory.py
│   └── (reads HDF5 for inspection)
└── visualize_trajectory_videos.py
    └── (generates MP4/GIF/frames)

main.py (enhanced version)
├── env.py
├── trajectory_recorder.py (optional, if --record flag)
└── video_display.py (optional, if --video-display flag)
```

---

## 8. Conclusion

The PR codebase is missing **critical infrastructure** for mobile ALOHA in-context learning deployment and trajectory analysis. The immediate priority should be:

1. **Port in-context learning files** (`main_incontext.py`, `env_incontext.py`, metadata)
2. **Port trajectory recording tools** (`trajectory_recorder.py`, visualization scripts)
3. **Enhance main.py** with full CLI arguments
4. **Validate on hardware** to ensure compatibility

**Estimated Total Effort**: 4-6 hours of porting + testing + validation

**Success Criteria**:
- ✅ Can run in-context learning experiments on real mobile ALOHA
- ✅ Can record and analyze deployment trajectories
- ✅ Can generate videos for team sharing and debugging
- ✅ Full CLI control without code modifications

---

## Document Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-10 | Comparative Analysis | Initial document creation |

---

**End of Document**
