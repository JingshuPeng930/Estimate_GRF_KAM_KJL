# Estimate GRF, KFM, and KJL from IMU

This repository is organized into two independent code paths:

| Folder | Purpose |
| --- | --- |
| `grf_subject_independent/` | IMU-only vertical GRF estimation with subject-independent LOSO evaluation. |
| `kjl_subject_dependent_cascaded/` | Subject-dependent KJL estimation for AB03 Amy using a cascaded `IMU + predicted GRF + predicted KFM` input. |

Training datasets and model outputs are intentionally not committed. Place generated datasets and checkpoints in the paths described by each folder's README.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run subject-independent GRF LOSO:

```bash
cd grf_subject_independent
python TCN_Training_GRF_SubjectIndependent_LOSO.py --all
```

Run subject-dependent cascaded KJL:

```bash
cd kjl_subject_dependent_cascaded
python run_pipeline_GRFKFM_KJL.py
```

The cascaded KJL pipeline checks for existing GRF/KFM upstream models first. If they are missing, it trains them automatically; use `--retrain-grf` and/or `--retrain-kfm` to force a fresh upstream run.
