"""Generate metadata/libero_90/task_to_episode.json from vo2yager/libero_90.

Output schema mirrors metadata/libero/task_to_episode.json: top-level dict,
keys are stringified task_index (0..89), values are lists of episode_index ints.
"""

import json
from pathlib import Path

LIBERO90_META = Path("~/.cache/huggingface/lerobot/vo2yager/libero_90/meta").expanduser()
OUT = Path(__file__).resolve().parents[1] / "metadata" / "libero_90" / "task_to_episode.json"


def main() -> None:
    task_str_to_idx: dict[str, int] = {}
    with (LIBERO90_META / "tasks.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            task_str_to_idx[rec["task"]] = int(rec["task_index"])

    task_to_episode: dict[str, list[int]] = {}
    with (LIBERO90_META / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            for task_str in rec["tasks"]:
                tidx = task_str_to_idx[task_str]
                task_to_episode.setdefault(str(tidx), []).append(int(rec["episode_index"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(task_to_episode, f, indent=2)
    n_tasks = len(task_to_episode)
    n_episodes = sum(len(v) for v in task_to_episode.values())
    print(f"Wrote {OUT} ({n_tasks} task indices, {n_episodes} episodes total)")


if __name__ == "__main__":
    main()
