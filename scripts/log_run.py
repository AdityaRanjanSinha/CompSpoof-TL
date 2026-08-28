"""Append a run record to experiments/runs.csv. Results not in the log do not exist.

Usage:
    python scripts/log_run.py --run_id dualbranch_v1 --seed 42 --model dual_branch \
        --dataset compspoof-tl --hardware "1x A100 40GB" --wall_time 3h12m \
        --f1 0.71 --eer_env 0.33 --notes "first end-to-end run"
"""
import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "experiments" / "runs.csv"
FIELDS = ["timestamp", "run_id", "git_commit", "seed", "model", "dataset",
          "hardware", "wall_time", "f1", "eer_orig", "eer_sp", "eer_env", "notes"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


def main():
    p = argparse.ArgumentParser()
    for f in FIELDS:
        if f not in ("timestamp", "git_commit"):
            p.add_argument(f"--{f}", default="")
    a = vars(p.parse_args())
    a["timestamp"] = datetime.now().isoformat(timespec="seconds")
    a["git_commit"] = git_commit()

    new = not LOG.exists()
    with open(LOG, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: a.get(k, "") for k in FIELDS})
    print(f"logged {a['run_id']} @ {a['git_commit']}")


if __name__ == "__main__":
    main()
