"""Summarize completed release evaluations and identify missing suite results."""

import json
from pathlib import Path
import sys


def main(root: Path) -> None:
    names = (
        "ContextFlow_4gpu",
        "ContextFlow_726c829_2gpu_defaultprec_run1",
        "ContextFlow_defaultprec_2gpu_seed42_run1",
        "ContextFlow_refactor_merge_nw16",
        "ContextFlow_repro_nw16_seed3",
    )
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    rows = []
    for name in names:
        row = {"checkpoint": name, "suites": {}, "episodes": 0, "successes": 0, "missing": []}
        for suite in suites:
            path = root / name / f"{suite}.json"
            if not path.is_file():
                row["missing"].append(suite)
                continue
            result = json.loads(path.read_text())["summary"]
            row["suites"][suite] = result
            row["episodes"] += result["total_episodes"]
            row["successes"] += result["total_successes"]
        row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else None
        rows.append(row)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(f"{row['checkpoint']}: {row['successes']}/{row['episodes']}; missing={row['missing']}")
    if any(row["missing"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
