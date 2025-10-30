#!/usr/bin/env python3
"""
Quick diagnosis tool for Libero prompt coverage.

This script reproduces the filtering logic used by InjectDemoIndexes and
reports every task whose prompt episodes disappear after applying the
train/test split.  It helps explain errors like:

    ValueError: InjectDemoIndexes: no prompt episodes with frames for task X

Usage (from repo root):
    python openpi/scripts/check_libero_prompt_coverage.py

Optional arguments let you point to alternate metadata directories, override
the exclude list, or change the minimum number of frames required per episode.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import jsonlines

# The default test set used in openpi/training/config.py
DEFAULT_LIBERO_TEST_TASK = (
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "pick up the milk and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
)


@dataclass
class TaskCoverage:
    task_index: int
    task_text: str
    total_episodes: int
    kept_after_filter: int
    with_enough_frames: int
    sample_episode_ids: Sequence[int]

    @property
    def missing_prompts(self) -> bool:
        return self.with_enough_frames == 0


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def load_task_descriptions(tasks_jsonl: Path) -> Dict[int, str]:
    descriptions: Dict[int, str] = {}
    with jsonlines.open(tasks_jsonl) as reader:
        for entry in reader:
            idx = int(entry["task_index"])
            descriptions[idx] = entry.get("task", "")
    return descriptions


def load_episode_tasks(episodes_jsonl: Path) -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = {}
    with jsonlines.open(episodes_jsonl) as reader:
        for entry in reader:
            idx = int(entry["episode_index"])
            mapping[idx] = list(entry.get("tasks", []))
    return mapping


def build_exclude_set(explicit: Iterable[str], keep_default: bool) -> set[str]:
    exclude = set(explicit or [])
    if keep_default:
        exclude.update(DEFAULT_LIBERO_TEST_TASK)
    return exclude


def compute_coverage(
    task_to_episode: Dict[int, Sequence[int]],
    episode_to_indexes: Dict[int, Sequence[int]],
    episode_tasks: Dict[int, Sequence[str]],
    task_descriptions: Dict[int, str],
    exclude_task_text: Sequence[str],
    min_frames: int,
) -> List[TaskCoverage]:
    # Build the whitelist of episode indices that survive the train/test split.
    kept_episode_indices = {
        ep_idx
        for ep_idx, tasks in episode_tasks.items()
        if not any(task in exclude_task_text for task in tasks)
    }

    coverage: List[TaskCoverage] = []
    for task_idx, episode_ids in sorted(task_to_episode.items()):
        episode_ids = [int(ep) for ep in (episode_ids or [])]
        filtered_ids = [ep for ep in episode_ids if ep in kept_episode_indices]
        valid_ids = [
            ep for ep in filtered_ids if len(episode_to_indexes.get(ep, [])) >= min_frames
        ]

        coverage.append(
            TaskCoverage(
                task_index=task_idx,
                task_text=task_descriptions.get(task_idx, "<missing description>"),
                total_episodes=len(episode_ids),
                kept_after_filter=len(filtered_ids),
                with_enough_frames=len(valid_ids),
                sample_episode_ids=episode_ids[:5],
            )
        )
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Libero prompt coverage issues.")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("openpi/metadata/libero"),
        help="Directory containing task_to_episode.json and episode_to_indexes.json.",
    )
    parser.add_argument(
        "--episodes-json",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot/physical-intelligence/libero/meta/episodes.jsonl"),
        help="episodes.jsonl file describing each episode's tasks (default mirrors training config).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=2,
        help="Minimum number of frames required per prompt episode (matches sample_frames).",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=None,
        help="Additional task descriptions to exclude from prompts (can be repeated).",
    )
    parser.add_argument(
        "--no-default-exclude",
        action="store_true",
        help="Skip the built-in DEFAULT_LIBERO_TEST_TASK exclusion list.",
    )
    args = parser.parse_args()

    metadata_dir: Path = args.metadata_dir
    episodes_json = Path(os.path.expanduser(str(args.episodes_json)))

    task_to_episode_raw = load_json(metadata_dir / "task_to_episode.json")
    episode_to_indexes_raw = load_json(metadata_dir / "episode_to_indexes.json")

    task_to_episode: Dict[int, Sequence[int]] = {
        int(task): [int(ep) for ep in (episodes or [])]
        for task, episodes in task_to_episode_raw.items()
    }
    episode_to_indexes: Dict[int, Sequence[int]] = {
        int(ep): indexes for ep, indexes in episode_to_indexes_raw.items()
    }

    task_descriptions = load_task_descriptions(metadata_dir / "tasks.jsonl")
    episode_tasks = load_episode_tasks(episodes_json)

    exclude_task_text = build_exclude_set(
        args.exclude_task or [], keep_default=not args.no_default_exclude
    )

    coverage = compute_coverage(
        task_to_episode=task_to_episode,
        episode_to_indexes=episode_to_indexes,
        episode_tasks=episode_tasks,
        task_descriptions=task_descriptions,
        exclude_task_text=tuple(exclude_task_text),
        min_frames=args.min_frames,
    )

    flagged = [entry for entry in coverage if entry.missing_prompts]

    print(f"Checked {len(coverage)} tasks; {len(flagged)} lose all prompt episodes after filtering.")
    if not flagged:
        return

    print("\nTasks missing prompt coverage:")
    for entry in flagged:
        print(
            f"  - task {entry.task_index:>3}: {entry.task_text}\n"
            f"      total episodes: {entry.total_episodes}\n"
            f"      after split:    {entry.kept_after_filter}\n"
            f"      ≥{args.min_frames} frames: {entry.with_enough_frames}\n"
            f"      sample episode ids: {list(entry.sample_episode_ids)}"
        )


if __name__ == "__main__":
    main()
