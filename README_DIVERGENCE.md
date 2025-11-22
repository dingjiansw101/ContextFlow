# Branch Divergence Documentation

**Last Updated**: 2025-11-10

This directory contains comprehensive documentation analyzing the divergence between the PR codebase (`q_former` branch) and the original codebase (`v1.0` branch).

---

## Document Overview

### 1. [BRANCH_DIVERGENCE_ANALYSIS.md](BRANCH_DIVERGENCE_ANALYSIS.md)
**Main divergence analysis** - Start here!

**Contents**:
- When and where the branches diverged (May 20, 2025, commit `9018878`)
- Commit history comparison (90 commits vs 23 commits)
- What's missing in each branch
- Timeline of development
- Recommendations for merging/porting

**Key Findings**:
- Divergence point: May 20, 2025
- PR branch focused on: Q-former architecture (v12-v18), research
- Original branch focused on: Mobile ALOHA real-world deployment, production tools

---

### 2. [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md)
**Detailed file-by-file analysis** of missing mobile ALOHA components

**Contents**:
- Complete inventory of missing files (~900 lines of code)
- Critical vs optional missing components
- Detailed description of each missing file's purpose
- Step-by-step porting guide with commands
- Impact assessment and priorities

**Critical Missing Files**:
- `main_incontext.py` - In-context learning deployment (139 lines)
- `env_incontext.py` - Task metadata environment (100 lines)
- `trajectory_recorder.py` - HDF5 recording (150 lines)
- `visualize_trajectory_videos.py` - Video analysis (252 lines)
- `metadata/` directory - Task-episode mappings

**Priority Guide**:
- **Priority 1** (Critical): In-context deployment files (4-6 hours)
- **Priority 2** (Important): Trajectory tools (2-3 hours)
- **Priority 3** (Medium): Enhanced main.py (1 hour)
- **Priority 4** (Optional): Additional utilities (30 min)

---

### 3. [CONFIG_SYSTEM_CHANGES.md](CONFIG_SYSTEM_CHANGES.md)
**Config system refactoring analysis**

**Contents**:
- Detailed comparison of old vs new config systems
- When refactoring happened (Sep 29, 2025, 125 commits after divergence)
- How to port configs between branches
- Metadata path conventions
- `LeRobotAlohaMobileIncontextDataConfig` differences

**Key Changes**:
- Old system: Single `config.py` (6,851 lines)
- New system: Modular `config_*.py` files (7 files, 14,187 lines total)
- Auto-discovery and loading mechanism
- Different metadata path conventions

**Critical for**:
- Adding new training configurations
- Porting configs between branches
- Understanding metadata path differences

---

## Quick Reference

### Divergence Summary

| Aspect | v1.0 (Original) | q_former (PR) |
|--------|-----------------|---------------|
| **Divergence Date** | May 20, 2025 | May 20, 2025 |
| **Commits After Divergence** | 23 | 90 |
| **Focus** | Production deployment | Research architecture |
| **Config System** | Monolithic (6,851 lines) | Modular (7 files) |
| **Metadata Path** | `metadata/pen_incontext/` | `metadata/aloha_pen_uncap/` |
| **Mobile ALOHA Deployment** | ✅ Complete | ❌ Missing |
| **Model Architecture** | ⚠️ v12 (older) | ✅ Q-former v18 (latest) |
| **Trajectory Tools** | ✅ Complete | ❌ Missing |

---

## Common Tasks

### Task 1: Port Mobile ALOHA Deployment Files to PR Branch

**Goal**: Enable in-context learning experiments on real mobile ALOHA using q_former models

**Files to Copy** (from v1.0 → q_former):
```bash
# Essential files (~900 lines)
examples/aloha_mobile_real/main_incontext.py
examples/aloha_mobile_real/env_incontext.py
examples/aloha_mobile_real/trajectory_recorder.py
examples/aloha_mobile_real/visualize_trajectory_videos.py
examples/aloha_mobile_real/view_trajectory.py
metadata/aloha_pen_uncap/
```

**Steps**:
1. Read [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md), Section 6 "Recommendations"
2. Follow Priority 1 and Priority 2 porting guides
3. Update metadata paths from `pen_incontext` → `aloha_pen_uncap`
4. Test deployment on real robot

**Estimated Time**: 1 working day (6-8 hours)

**Reference**: See [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md) Section 6.1-6.2

---

### Task 2: Add New Training Config to q_former Branch

**Goal**: Create a new ALOHA training configuration in the modular config system

**Steps**:
1. Read [CONFIG_SYSTEM_CHANGES.md](CONFIG_SYSTEM_CHANGES.md), Section "Porting Configs from v1.0 → q_former"
2. Open `src/openpi/training/config_aloha.py`
3. Locate the `build(api)` function
4. Add your `DataConfig` class inside `build()`
5. Add `TrainConfig` to the return list
6. Test: `uv run scripts/train.py <config_name> --help`

**Example**:
```python
# In config_aloha.py, inside build(api):

@dataclasses.dataclass(frozen=True)
class MyCustomDataConfig(api.DataConfigFactory):
    repo_id: str = "my-org/my-dataset"
    default_prompt: str = "my task"
    # ... other fields ...

    def create(self, assets_dirs, model_config):
        # ... implementation ...
        pass

# In the return statement:
return [
    # ... existing configs ...
    api.TrainConfig(
        name="my_custom_config",
        model=api.pi0_incontextv18.Pi0Incontextv18Config(...),
        data=MyCustomDataConfig(...),
        ...
    ),
]
```

**Reference**: See [CONFIG_SYSTEM_CHANGES.md](CONFIG_SYSTEM_CHANGES.md), Section "Porting Configs from v1.0 → q_former"

---

### Task 3: Generate Metadata for In-Context Learning

**Goal**: Create `task_to_episode.json` and `episode_to_indexes.json` for a dataset

**Command**:
```bash
uv run src/openpi/training/generate_task_to_index.py \
  --config <config_name> \
  --output_dir metadata/<dataset_name>
```

**Example**:
```bash
# For mobile ALOHA pen uncap dataset
uv run src/openpi/training/generate_task_to_index.py \
  --config pi0_aloha_pen_uncap_b5_low_mem_finetune \
  --output_dir metadata/aloha_pen_uncap
```

**Output**:
```
metadata/aloha_pen_uncap/
├── task_to_episode.json        # Task index → episode indices
└── episode_to_indexes.json     # Episode index → frame indices
```

**Reference**: See documentation in `generate_task_to_index.py` docstring

---

### Task 4: Deploy In-Context Model on Real Robot

**Prerequisites**:
1. ✅ Deployment files ported (Task 1)
2. ✅ Metadata generated (Task 3)
3. ✅ Policy server running with q_former model

**Steps**:
```bash
# 1. Start policy server on GPU machine
uv run scripts/serve_policy_incontext.py \
  policy:checkpoint \
  --policy.config=<incontext_config_name> \
  --policy.dir=checkpoints/<path>

# 2. Run deployment on robot
uv run examples/aloha_mobile_real/main_incontext.py \
  --prompt "pick up the pen" \
  --host <gpu-server-ip> \
  --port 8080 \
  --record

# 3. Analyze recorded trajectory
uv run examples/aloha_mobile_real/visualize_trajectory_videos.py \
  --trajectory trajectory_ep001.hdf5 \
  --output-format video
```

**Reference**: See [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md) Appendix Section 7.1-7.2

---

## FAQ

### Q: Why are mobile ALOHA deployment files missing in the PR branch?

**A**: The branches diverged on May 20, 2025. After that, the v1.0 branch developed mobile ALOHA deployment infrastructure (23 commits), while the PR branch focused on Q-former architecture development (90 commits). The deployment files were added to v1.0 **after** divergence, so they never made it into the PR branch.

See: [BRANCH_DIVERGENCE_ANALYSIS.md](BRANCH_DIVERGENCE_ANALYSIS.md), Section "Why the Divergence Happened"

---

### Q: Can I just merge the v1.0 branch into q_former?

**A**: Not recommended. The branches have diverged significantly (90 vs 23 commits) and have different structural changes (especially the config system refactoring). **Selective file porting** is safer and faster.

**Better approach**: Follow the porting guide in [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md) to copy only the necessary deployment files.

---

### Q: Why do metadata paths differ between branches?

**A**: Different naming conventions evolved independently after divergence:
- v1.0 uses: `metadata/pen_incontext/`
- q_former uses: `metadata/aloha_pen_uncap/`

When porting files, you must update hardcoded paths to match the target branch's convention.

See: [CONFIG_SYSTEM_CHANGES.md](CONFIG_SYSTEM_CHANGES.md), Section "Metadata Path Conventions"

---

### Q: How do I know which config file to edit?

**A**: In the q_former branch (new system), configs are organized by domain:
- ALOHA configs → `config_aloha.py`
- Libero configs → `config_libero.py`

In the v1.0 branch (old system), all configs are in `config.py`.

See: [CONFIG_SYSTEM_CHANGES.md](CONFIG_SYSTEM_CHANGES.md), Section "Config System Comparison"

---

### Q: What's the fastest way to get q_former running on real mobile ALOHA?

**A**: Follow this fast-track (4-6 hours):

1. **Copy 4 critical files** (30 min):
   ```bash
   cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/main_incontext.py \
      examples/aloha_mobile_real/
   cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/env_incontext.py \
      examples/aloha_mobile_real/
   cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/trajectory_recorder.py \
      examples/aloha_mobile_real/
   cp /home/dingj0b/code/openpi/examples/aloha_mobile_real/visualize_trajectory_videos.py \
      examples/aloha_mobile_real/
   ```

2. **Copy metadata** (5 min):
   ```bash
   cp -r /home/dingj0b/code/openpi/metadata/aloha_pen_uncap metadata/
   ```

3. **Update metadata paths in copied files** (15 min):
   - Replace `metadata/pen_incontext/` → `metadata/aloha_pen_uncap/`

4. **Test on real robot** (3-5 hours):
   - Start policy server
   - Run main_incontext.py
   - Record and analyze trajectories

See: [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md), Section 6.1 "Immediate Action Items"

---

## Branch Strategy Recommendations

### For Production Deployment (Real Robots)

**Use v1.0 branch**:
- ✅ Complete deployment infrastructure
- ✅ Trajectory recording and analysis
- ✅ In-context learning deployment tested
- ⚠️ Older model architecture (v12)

### For Research & Development (New Models)

**Use q_former branch**:
- ✅ Latest Q-former architecture (v18)
- ✅ Modular config system
- ✅ Advanced in-context learning models
- ⚠️ Missing deployment infrastructure

### For Real Robot Experiments with Latest Models

**Hybrid approach**:
1. Develop models in q_former branch
2. Port deployment files from v1.0 (follow this guide)
3. Generate metadata for your dataset
4. Deploy on real robot

**Estimated setup time**: 1 working day

---

## Getting Help

### Step-by-Step Guides

1. **Porting mobile ALOHA files**: See [MOBILE_ALOHA_MISSING_COMPONENTS.md](MOBILE_ALOHA_MISSING_COMPONENTS.md), Section 6
2. **Adding new configs**: See [CONFIG_SYSTEM_CHANGES.md](CONFIG_SYSTEM_CHANGES.md), Section "Porting Configs"
3. **Understanding divergence**: See [BRANCH_DIVERGENCE_ANALYSIS.md](BRANCH_DIVERGENCE_ANALYSIS.md), Section "Divergence Point"

### Quick Command References

**Generate metadata**:
```bash
uv run src/openpi/training/generate_task_to_index.py --config <name> --output_dir metadata/<name>
```

**Test config loading**:
```bash
uv run scripts/train.py <config_name> --help
```

**Deploy in-context model**:
```bash
uv run examples/aloha_mobile_real/main_incontext.py --prompt "<task>" --host <ip> --port 8080
```

**Visualize trajectory**:
```bash
uv run examples/aloha_mobile_real/visualize_trajectory_videos.py --trajectory <file.hdf5>
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-10 | Initial analysis of branch divergence and missing components |
| 1.1 | 2025-11-10 | Added config system refactoring analysis |
| 1.2 | 2025-11-10 | Created this summary document |

---

## Contact

For questions or issues with this documentation, please:
1. Check the detailed analysis documents linked above
2. Review the relevant code sections mentioned in the guides
3. File an issue in the repository

---

**End of Summary**
