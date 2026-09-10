"""Verify a staged release checkpoint, then launch its evaluation exactly once."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(name: str, gpu: str, run_id: str, trials: str = "50", scope: str = "unseen") -> None:
    root = Path(__file__).resolve().parents[2]
    expected = json.loads(Path(__file__).with_name("release_checksums.json").read_text())[name]
    checkpoint_root = root / "checkpoints/ContextFlow" / name
    output = root / "logs/release_eval" / run_id / name
    if output.exists():
        raise FileExistsError(f"Refusing to duplicate or overwrite evaluation: {output}")
    deadline = time.monotonic() + 4 * 3600
    while time.monotonic() < deadline:
        complete = all(
            (checkpoint_root / path).is_file() and (checkpoint_root / path).stat().st_size == spec["size"]
            for path, spec in expected.items()
        )
        if complete:
            break
        print(f"Waiting for {name} checkpoint staging", flush=True)
        time.sleep(30)
    else:
        raise TimeoutError(f"Checkpoint staging timed out: {name}")
    for path, spec in expected.items():
        with (checkpoint_root / path).open("rb") as file:
            digest = hashlib.file_digest(file, "md5").hexdigest()
        if digest != spec["md5"]:
            raise ValueError(f"Release checksum mismatch: {name}/{path}")
    print(f"Verified all {len(expected)} files for {name}; launching on GPU {gpu}", flush=True)
    while time.monotonic() < deadline:
        memory, utilization = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(",")
        )
        if int(memory) > 60000 and int(utilization) < 5:
            break
        print(f"Waiting for GPU {gpu} to become free for {name}", flush=True)
        time.sleep(30)
    else:
        raise TimeoutError(f"GPU availability timed out: {name}")
    subprocess.run(
        [
            "bash",
            "jobs/local/eval_release_checkpoint.sh",
            str(checkpoint_root / "19999"),
            name,
            gpu,
            run_id,
            trials,
            scope,
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
