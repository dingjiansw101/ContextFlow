"""Wait for an ORIX job, verify transferred inference files, then evaluate locally."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def remote(command):
    return subprocess.check_output(["ssh", "-o", "ConnectTimeout=20", "orix", command], text=True, timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=int)
    parser.add_argument("root")
    parser.add_argument("experiment")
    parser.add_argument("--allow-transfer", action="store_true", help="User approved the exact rsync source and destination")
    args = parser.parse_args()
    if not args.allow_transfer:
        parser.error("Checkpoint transfer needs explicit approval before this watcher is launched")
    repo = Path(__file__).resolve().parents[2]
    destination = repo / "checkpoints/ContextFlow" / args.experiment / "19999"
    receipt_path = args.root + "/receipts/" + args.experiment + ".json"
    source = args.root + "/source/checkpoints/ContextFlow/" + args.experiment + "/19999/"
    logs = repo / "logs/reproduction_eval"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / (args.experiment + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 7 * 86400
        while time.monotonic() < deadline:
            try:
                output = remote(f"sacct -X -n -P -j {args.job_id} --format=JobID,State,ExitCode")
            except (subprocess.SubprocessError, OSError) as error:
                print(f"Status query failed; retrying the same job: {error}", flush=True)
                time.sleep(60)
                continue
            rows = [line.split("|") for line in output.splitlines() if line.strip()]
            row = next((r for r in rows if r[0] == str(args.job_id)), None)
            if row and row[1] == "COMPLETED" and row[2] == "0:0":
                break
            if row and row[1].split()[0].rstrip("+") in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "NODE_FAIL"}:
                raise RuntimeError(f"Training stopped: {row}; evaluation not launched")
            print(f"Waiting for ORIX job {args.job_id}: {row}", flush=True)
            time.sleep(60)
        else:
            raise TimeoutError("Training wait exceeded seven days")
        receipt = json.loads(remote("cat " + shlex.quote(receipt_path)))
        assert receipt["status"] == "completed" and receipt["slurm_job_id"] == str(args.job_id)
        assert receipt["checkpoint"].rstrip("/") == source.rstrip("/")
        subprocess.run(["git", "diff", "--exit-code", receipt["source_sha"], "--", "src", "scripts", "examples/libero", "packages"], cwd=repo, check=True)
        required = receipt["inference_bytes"] + 15 * 1024**3
        destination.parent.mkdir(parents=True, exist_ok=True)
        while shutil.disk_usage(destination.parent).free < required:
            if time.monotonic() > deadline:
                raise TimeoutError("Waiting for local storage")
            print(f"Waiting for {required / 1024**3:.1f} GiB local free space", flush=True)
            time.sleep(60)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint destination: {destination}")
        subprocess.run(["rsync", "-rt", "--partial", "--include=/params/***", "--include=/assets/***",
                        "--include=/_CHECKPOINT_METADATA", "--include=/EVAL_SHA256SUMS", "--exclude=*",
                        "orix:" + source, str(destination) + "/"], check=True)
        manifest = destination / "EVAL_SHA256SUMS"
        for line in manifest.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError("Invalid checksum path")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
        (destination / "training_receipt.json").write_text(json.dumps(receipt, indent=2))
        while time.monotonic() < deadline:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            available = [int(parts[0]) for line in output.splitlines() if len(parts := line.split(",")) == 3 and int(parts[1]) > 60000 and int(parts[2]) < 5]
            if available:
                gpu = str(min(available))
                print(f"Verified checkpoint; evaluating on local GPU {gpu}", flush=True)
                subprocess.run(["bash", "jobs/local/eval_reproduction.sh", str(destination), args.experiment, gpu], cwd=repo, check=True)
                return
            print("Waiting for a free local GPU", flush=True)
            time.sleep(60)
        raise TimeoutError("Waiting for local GPU")


if __name__ == "__main__":
    main()
