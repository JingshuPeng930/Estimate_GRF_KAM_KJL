# Estimate GRF, KFM, and KJL from IMU

This repository is organized into independent code paths:

| Folder | Purpose |
| --- | --- |
| `grf_subject_independent/` | IMU-only vertical GRF estimation with subject-independent LOSO evaluation. |
| `imu_processing/` | Standalone preprocessing utility for converting IMU_BI raw outputs into filtered 9-IMU CSV files. |
| `kjl_subject_dependent_cascaded/` | Subject-dependent KJL estimation for AB03 Amy using a cascaded `IMU + predicted GRF + predicted KFM` input. |
| `kjl_subject_dependent_noncascaded/` | Subject-dependent KJL non-cascaded baseline and IMU ablation code for AB03 Amy. |
| `kjl_subject_independent_cascaded/` | Subject-independent LOSO KJL estimation using the cascaded `IMU + predicted GRF + predicted KFM` input. |

Training outputs and expanded datasets are intentionally not committed. The subject-independent cascaded KJL folder includes a Git LFS data archive that can be unpacked into the expected training-data folders.
The subject-dependent non-cascaded KJL folder also includes a Git LFS AB03 data archive for baseline and ablation runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

If you want to run the subject-independent cascaded KJL model directly after cloning, install Git LFS and unpack the included generated-data archive first:

```bash
git lfs install
git lfs pull

python kjl_subject_independent_cascaded/unpack_generated_data.py
```

This creates:

```text
kjl_subject_independent_cascaded/data/grf
kjl_subject_independent_cascaded/data/kfm
kjl_subject_independent_cascaded/data/kjl
```

For the subject-dependent non-cascaded KJL ablation data, run:

```bash
python kjl_subject_dependent_noncascaded/unpack_generated_data.py
```

This creates:

```text
kjl_subject_dependent_noncascaded/data/kjl_ab03_dep
```

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

Run subject-dependent non-cascaded KJL baseline or ablations:

```bash
cd kjl_subject_dependent_noncascaded
python run_kjl_subject_dependent_ablation.py \
  --dataset-root data/kjl_ab03_dep \
  --split-json data/kjl_ab03_dep/splits/split_noexo_temporal_seed42.json \
  --ablation all
```

Run subject-independent cascaded KJL LOSO:

```bash
python kjl_subject_independent_cascaded/run_pipeline_GRFKFM_KJL_SI_LOSO.py \
  --window-size 150 \
  --seed 42 \
  --grf-epochs 30 \
  --kfm-epochs 30 \
  --kjl-epochs 50 \
  --kjl-lr 1e-5
```
