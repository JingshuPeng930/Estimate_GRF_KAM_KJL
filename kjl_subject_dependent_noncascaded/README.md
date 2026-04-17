# Subject-dependent KJL Non-cascaded Model

This folder contains the AB03 Amy subject-dependent KJL training code for the non-cascaded baseline and simple IMU ablations.

The model uses only IMU windows as input:

```text
pelvis IMU + right femur IMU + right tibia IMU + right calcaneus IMU -> KJL
```

It does not use predicted GRF or predicted KFM.

## Files

| Path | Role |
| --- | --- |
| `run_kjl_subject_dependent_ablation.py` | Recommended command-line entry point for baseline and ablation runs. |
| `TCN_Training_KJL_AB03_DEP.py` | Main TCN training code. Cascade inputs are disabled by the wrapper. |
| `kjl_ab03_tcn_dataset.py` | Windowed KJL dataloader. |
| `TCN_Header_Model.py` | TCN model definition. |
| `soft_delay_classifier.py` | Optional delay-conditioning helper imported by the trainer. |
| `generate_ab03_kjl_dep_dataset.py` | Dataset-generation script if prepared KJL data need to be rebuilt. |
| `generate_ab03_fixed_splits.py` | Utility for creating reproducible train/val/test split JSON files. |
| `requirements.txt` | Python package requirements. |

## Data

The generated AB03 KJL dataset is included as a Git LFS archive. After cloning, run:

```bash
git lfs install
git lfs pull
python kjl_subject_dependent_noncascaded/unpack_generated_data.py
```

This creates:

```text
kjl_subject_dependent_noncascaded/data/kjl_ab03_dep
```

Expected layout:

```text
data/kjl_ab03_dep/manifest.json
data/kjl_ab03_dep/splits/*.json
data/kjl_ab03_dep/AB03_Amy/LG/<condition>/trial_*/Input/imu.csv
data/kjl_ab03_dep/AB03_Amy/LG/<condition>/trial_*/Label/*.csv
```

If raw `IMU_Data_Process/` and `KJL_GT/` folders are available, the dataset can also be rebuilt with:

```bash
RAW_DATA_ROOT=/path/to/raw/root \
KJL_OUTPUT_ROOT=data/kjl_ab03_dep \
python generate_ab03_kjl_dep_dataset.py
```

`RAW_DATA_ROOT` should contain:

```text
IMU_Data_Process/AB03_Amy/...
KJL_GT/AB03_Amy/...
```

Then create fixed split files:

```bash
python generate_ab03_fixed_splits.py --dataset-root data/kjl_ab03_dep
```

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kjl_subject_dependent_noncascaded/requirements.txt
```

## Run Baseline

```bash
cd kjl_subject_dependent_noncascaded
python run_kjl_subject_dependent_ablation.py \
  --dataset-root data/kjl_ab03_dep \
  --split-json data/kjl_ab03_dep/splits/split_noexo_temporal_seed42.json \
  --ablation all \
  --epochs 50 \
  --window-size 150 \
  --seed 42
```

Outputs are written under:

```text
runs/kjl_ab03_dep_ablation/
```

## Run Ablations

Leave out one IMU:

```bash
python run_kjl_subject_dependent_ablation.py --ablation no_pelvis
python run_kjl_subject_dependent_ablation.py --ablation no_femur
python run_kjl_subject_dependent_ablation.py --ablation no_tibia
python run_kjl_subject_dependent_ablation.py --ablation no_calcn
```

Use one IMU only:

```bash
python run_kjl_subject_dependent_ablation.py --ablation pelvis_only
python run_kjl_subject_dependent_ablation.py --ablation femur_only
python run_kjl_subject_dependent_ablation.py --ablation tibia_only
python run_kjl_subject_dependent_ablation.py --ablation calcn_only
```

Use accelerometer-only or gyroscope-only input:

```bash
python run_kjl_subject_dependent_ablation.py --ablation acc_only
python run_kjl_subject_dependent_ablation.py --ablation gyr_only
```

You can also exclude exact columns manually:

```bash
python run_kjl_subject_dependent_ablation.py \
  --ablation all \
  --exclude-input-cols pelvis_imu_acc_y pelvis_imu_gyr_z
```

Each run writes `train_config.json`, model weights, prediction CSVs, per-trial metrics, and `ablation_result.json`.
