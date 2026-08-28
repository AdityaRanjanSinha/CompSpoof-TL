# CompSpoof-TL

Component-level **temporal localization** of audio spoofing in speech + environment mixtures.
B.Tech Thesis Project (BTP) · Team B.

We build the first task, benchmark, metrics, and baseline for detecting **which component**
of an audio mixture is spoofed **and when**.

---

## Repository layout

```
configs/        Single source of truth for all parameters (default.yaml).
src/            Reusable code (seeds, paths, and later: toolchain, model, metrics).
scripts/        Runnable entry points (e.g. log_run.py).
experiments/    runs.csv — the tracked experiment log. "If it's not logged, it didn't happen."
notebooks/      Lab notebook: one markdown file per student per week.
data/           CompSpoofV2 + generated benchmark. GITIGNORED (lives on the server).
tests/          Unit tests (toolchain, metrics, seeds).
```

## Conventions (decided once, never changed)

- **Python 3.11** for this repo. (The baseline repo uses its own separate Python 3.10 env — see `environment.md`.)
- **Config:** plain YAML, single top-level file `configs/default.yaml`. No Hydra.
- **Seeds:** the only allowed seeds for headline numbers are **13, 42, 2027**.
- **Experiment log:** every run appends a row to `experiments/runs.csv` (via `scripts/log_run.py`)
  with git commit, seed, hardware, and final metrics. Results not in the log do not exist.
- **Lab notebook:** each student commits one markdown file per week under `notebooks/<name>/weekNN.md`,
  recording negative results with the same care as positive ones.
- **Git:** private now, public at submission. Every experiment has its config committed *before* launch.

## First-time setup

```bash
# 1. create and activate the project environment (Python 3.11)
conda create -n compspoof-tl python=3.11 -y
conda activate compspoof-tl

# 2. install dependencies
pip install -r requirements.txt

# 3. after everything installs, pin exact versions (the lockfile the plan requires)
pip freeze > requirements.lock

# 4. sanity check
pytest -q
```

## Data

The dataset is **not** stored in git. On the server, download CompSpoofV2 into `data/CompSpoofV2/`
and point `paths.compspoofv2_dir` in `configs/default.yaml` at it.
