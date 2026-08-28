# Environments

This project uses **two separate conda environments**. Do not mix them.

## 1. `compspoof-tl` (this repo) — Python 3.11
Our own code: toolchain, model, metrics, analysis. Created from `requirements.txt`.

## 2. `ESDD2-Baseline` (the official baseline repo) — Python 3.10
Used only to reproduce the official baseline. It depends on a pinned `fairseq` commit and
is built inside the cloned `ESDD2-Baseline` repo, on the server (compiled against server CUDA).
Never build this one on a CPU-only laptop — you'll just have to rebuild it on the server.
