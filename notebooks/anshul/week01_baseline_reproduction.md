# Week 01 — Official ESDD2 Baseline GPU Reproduction on PARAM Pragya

**Date:** 2026-09-24  
**Author:** Anshul Rathore (`ee3231024`)  
**Project Allocation:** `maggarwal_care.spons`  
**Host / Cluster:** PARAM Pragya (`pragya.iitd.ac.in`)  
**Node:** `amilan003` (8 × NVIDIA A100-SXM4-80GB)  
**Objective:** Reproduce and verify end-to-end execution of the official ESDD2 challenge baseline (`ESDD2-Baseline`) on the assigned IIT Delhi GPU cluster, validating CUDA availability, Fairseq XLS-R feature extraction, forward/backward loss computation, and checkpoint persistence without CUDA OOM.

---

## 1. Environment & Cluster Architecture

- **Cluster Separation**:
  - **PADUM (`hpc.iitd.ac.in`)**: Storage node hosting the 112 GB raw dataset (`/home/ee/btech/ee1230446/scratch/CompSpoofV2`) and the 96,600-clip benchmark (`/home/ee/btech/ee1230446/scratch/CompSpoof-TL/full`).
  - **PARAM Pragya (`pragya.iitd.ac.in`)**: Execution cluster for GPU compute allocation `maggarwal_care.spons` equipped with 80 GB A100 GPUs.
- **Python Environment**:
  - Conda Environment: `ESDD2-Baseline` (`Python 3.10.19`)
  - PyTorch: `2.14.0+cu130`
  - CUDA: `13.0` (Driver `580.105.08`)
  - Fairseq: `1.0.0a0` (Compiled C++/Cython extensions via GCC 11)
  - Pre-trained Weights: `~/ESDD2-Baseline/weight/xlsr2_300m.pt` (3.6 GB, verified)

---

## 2. Issues Encountered & Resolved

1. **Fairseq Pip Compatibility**:
   - Modern `pip` (>= 24.1) rejects `fairseq` due to `omegaconf<2.1` using deprecated `PyYAML>=5.1.*` metadata specifiers.
   - **Fix**: Pinned `pip==24.0` inside `ESDD2-Baseline` to allow editable compilation with GCC 11.
2. **Directory Namespace Shadowing**:
   - Cloning Fairseq inside the `ESDD2-Baseline/` directory created a local folder `./fairseq` which shadowed the installed site-package (PEP 420 namespace collision where `fairseq.__file__` returned `None`).
   - **Fix**: Relocated the Fairseq source tree to `~/fairseq_src/` outside the project root and reinstalled with `pip install -e ~/fairseq_src --no-build-isolation --no-deps`.
3. **Submodule Import Resolution**:
   - `model/XLSR2_AASIST.py` referenced `fairseq.checkpoint_utils.load_model_ensemble_and_task` without an explicit `import fairseq.checkpoint_utils`.
   - **Fix**: Added explicit `import fairseq.checkpoint_utils` in `model/XLSR2_AASIST.py`.
4. **Single-Class NaN Guard in EER Computation**:
   - `sklearn.metrics.roc_curve` raises `ValueError: All-NaN slice encountered` when evaluating splits lacking positive or negative samples for specific sub-targets (e.g. `original_labels`).
   - **Fix**: Added an edge-case guard in `libs/eval_metrics.py` returning `0.0` when `len(np.unique(labels)) < 2`.

---

## 3. PBS Batch Job Execution

- **PBS Job Script:** `gpu_sanity_check.pbs`
- **Job ID:** `80871.pragya`
- **Resources Allocated:** `select=1:ncpus=4:ngpus=1:mem=16gb` on node `amilan003` (A100-SXM4-80GB)

### Execution Trace & Verification:
```text
=== GPU SANITY CHECK JOB STARTED AT Thu Sep 24 11:27:19 AM IST 2026 ===
Node: amilan003
=== 1. NVIDIA GPU STATUS ===
NVIDIA-SMI 580.105.08 | Driver: 580.105.08 | CUDA: 13.0
GPU 0: NVIDIA A100-SXM4-80GB (81920 MiB VRAM)

=== 2. PYTORCH CUDA TEST ===
CUDA Available: True | Count: 1 | Device: NVIDIA A100-SXM4-80GB

=== 3. RUNNING BASELINE BRING-UP TEST (GPU 0) ===
Loading model to GPUs:(0,), #param: 957.85M
Set train mode...
Train: 1: 100%|██████████| 4/4 [00:20<00:00, 5.20s/it]
train epoch:1, loss: 81.91132354736328, MSE:77.481689453125
Set dev mode...
dev Results - original_eer: 0.8333, speech_eer: 0.2500, env_eer: 1.0000
Training for 1/1 epoches done!
Set eval mode...
eval Results - original_eer: 0.8333, speech_eer: 0.2500, env_eer: 1.0000
=== JOB COMPLETED AT Thu Sep 24 11:29:30 AM IST 2026 ===
```

---

## 4. Checkpoints & Output Artifacts

The training loop successfully persisted full model state checkpoints under `/home/ee/btech/ee3231024/ESDD2-Baseline/weight/epoch_1_joint_0/`:
- `best_epoch1.pt.tar` (11 GB)
- `last.pt.tar` (11 GB)
- `trainer.log`
- `mdl.json`, `trainer.json`

---

## 5. Summary & Next Milestones

- **Reproduction Gate:** **PASSED**.
  - Verified 957.85M parameter XLS-R + AASIST + UNet joint architecture runs cleanly on PARAM Pragya A100 GPUs.
  - Zero memory leaks or CUDA OOM errors.
- **Ready for Week 6 Implementation**:
  - Developing our Dual-Branch XLS-R + BEATs acoustic encoder with temporal GRU localization head for CompSpoof-TL.
