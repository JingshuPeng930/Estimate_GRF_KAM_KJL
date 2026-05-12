# Subject-independent KJL Cascaded Model

This folder contains the code used to train the leave-one-subject-out (LOSO) cascaded KJL model:

```text
IMU -> GRF model -> predicted GRF
IMU -> KFM model -> predicted KFM
IMU + predicted GRF + predicted KFM -> KJL model
```

The final KJL model uses 26 input channels: 24 IMU channels plus one predicted vertical GRF channel and one predicted KFM channel. The default LOSO subjects are `AB02_Rajiv`, `AB03_Amy`, and `AB05_Maria`.

## Fast Start With Included Data

This repository includes a compressed generated-data archive tracked with Git LFS:

```text
kjl_subject_independent_cascaded/data_archive/kjl_subject_independent_cascaded_generated_data.tar.gz
```

After cloning, run:

```bash
git lfs install
git lfs pull

python kjl_subject_independent_cascaded/unpack_generated_data.py
```

The unpack step creates:

```text
kjl_subject_independent_cascaded/data/grf
kjl_subject_independent_cascaded/data/kfm
kjl_subject_independent_cascaded/data/kjl
kjl_subject_independent_cascaded/data/grf_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kfm_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_double
```

This folder also includes a raw combined unilateral KJL archive:

```text
kjl_subject_independent_cascaded/data_archive/kjl_raw_unilateral_4imu_combined.tar.gz
```

Unpack it with:

```bash
python kjl_subject_independent_cascaded/unpack_generated_data.py \
  --archive kjl_subject_independent_cascaded/data_archive/kjl_raw_unilateral_4imu_combined.tar.gz
```

It creates:

```text
kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_raw_combined
```

Each trial folder contains one raw bilateral `Input/imu.csv` and one bilateral `Label/kjl_fy.csv`. The KJL dataloader expands each trial into right and left unilateral samples during training; left-side IMU channels are mirrored to the pseudo-right 24-channel convention only inside the dataloader.

Then train all LOSO folds:

```bash
python kjl_subject_independent_cascaded/run_pipeline_GRFKFM_KJL_SI_LOSO.py \
  --window-size 150 \
  --seed 42 \
  --grf-epochs 30 \
  --kfm-epochs 30 \
  --kjl-epochs 50 \
  --kjl-lr 1e-5
```

## Folder Layout

| Path | Role |
| --- | --- |
| `run_pipeline_GRFKFM_KJL_SI_LOSO.py` | End-to-end LOSO pipeline. For each held-out subject, it trains upstream GRF, trains upstream KFM, then trains the cascaded KJL model. |
| `TCN_Training_KJL_AB03_DEP.py` | KJL trainer with optional cascade inputs. |
| `kjl_ab03_tcn_dataset.py` | Windowed KJL dataloader and label filtering. |
| `generate_multisubject_kjl_dataset.py` | Builds the KJL dataset and LOSO splits from `KJL_GT` and `IMU_Data_Process`. |
| `generate_unilateral_grfkfmkjl_datasets.py` | Builds doubled pseudo-right unilateral GRF/KFM/KJL datasets using right-side samples and mirrored left-side samples. |
| `generate_raw_unilateral_kjl_dataset.py` | Builds raw combined unilateral KJL trials with raw left/right IMU columns and left/right KJL label columns. |
| `plot_loso_cascade_figures.py` | Generates time overlays, gait-cycle mean +/- SD plots, task metrics, and agreement plots. |
| `upstream_grf/` | GRF dataset generator, dataloader, model, and trainer used by the cascade. |
| `upstream_kfm/` | KFM dataset generator, dataloader, model, and trainer used by the cascade. |

Expanded generated data and model outputs are intentionally not committed. After unpacking the Git LFS archive, generated data are expected under:

```text
kjl_subject_independent_cascaded/data/grf
kjl_subject_independent_cascaded/data/kfm
kjl_subject_independent_cascaded/data/kjl
kjl_subject_independent_cascaded/data/grf_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kfm_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_raw_combined
kjl_subject_independent_cascaded/runs
```

The archive contains the latest generated training data used for the standard cascaded LOSO runs and the doubled unilateral runs: `Input/imu.csv`, `Label/*.csv`, `manifest.json`, and the LOSO split JSON files. It does not include raw `IMU_Data_Process`, `KJL_GT`, `ID_GT`, checkpoints, run outputs, or `aligned_debug.csv`.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kjl_subject_independent_cascaded/requirements.txt
```

## Prepare Data

If you want to regenerate the datasets from raw files instead of using the included archive, place the raw processing folders in the repository root, using the same names as the lab workspace:

```text
IMU_Data_Process/
KJL_GT/
ID_GT/
```

Then generate the three datasets:

```bash
python kjl_subject_independent_cascaded/upstream_grf/generate_multisubject_grf_dataset.py \
  --subjects AB02_Rajiv AB03_Amy AB05_Maria

python kjl_subject_independent_cascaded/upstream_kfm/generate_multisubject_kfm_dataset.py \
  --subjects AB02_Rajiv AB03_Amy AB05_Maria

python kjl_subject_independent_cascaded/generate_multisubject_kjl_dataset.py
```

Each generated dataset includes `split_subject_independent_loso_<subject>.json` files used by the pipeline.

### Optional: Doubled Unilateral Dataset

To train a unilateral pseudo-right model with doubled data, generate right-side and mirrored left-side samples:

```bash
python kjl_subject_independent_cascaded/generate_unilateral_grfkfmkjl_datasets.py \
  --subjects AB02_Rajiv AB03_Amy AB05_Maria \
  --height-overrides AB02_Rajiv=1.76
```

This creates:

```text
kjl_subject_independent_cascaded/data/grf_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kfm_unilateral_4imu_double
kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_double
```

Each source trial contributes:

```text
R sample: pelvis + femur_r + tibia_r + calcn_r -> right GRF/KFM/KJL
L sample: flipped pelvis + flipped femur_l/tibia_l/calcn_l renamed as pseudo-right -> left GRF/KFM/KJL
```

The flipped channels for left-side samples are `Acc_Y`, `Gyr_X`, and `Gyr_Z` for pelvis, femur, tibia, and calcaneus. The final input still has 24 channels.

### Optional: Raw Combined Unilateral KJL Dataset

To keep the data raw on disk and mirror left-side channels only during training, generate the raw combined KJL dataset:

```bash
python kjl_subject_independent_cascaded/generate_raw_unilateral_kjl_dataset.py \
  --subjects AB02_Rajiv AB03_Amy AB05_Maria
```

This creates:

```text
kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_raw_combined
```

Folder names no longer include `_R` or `_L`; each source condition has a single `trial_1` folder. `Input/imu.csv` keeps raw `*_l_*` and `*_r_*` IMU columns, and `Label/kjl_fy.csv` keeps both right and left KJL columns. When the KJL dataloader sees this format, it trains on both sides by reading the right side directly and mirroring the left side in memory.

Run the pipeline on this doubled unilateral dataset with:

```bash
python kjl_subject_independent_cascaded/run_pipeline_GRFKFM_KJL_SI_LOSO.py \
  --grf-data-root kjl_subject_independent_cascaded/data/grf_unilateral_4imu_double \
  --kfm-data-root kjl_subject_independent_cascaded/data/kfm_unilateral_4imu_double \
  --kjl-data-root kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_double \
  --output-tag unilateral4imu_double \
  --window-size 150 \
  --seed 42 \
  --grf-epochs 30 \
  --kfm-epochs 30 \
  --kjl-epochs 50 \
  --kjl-lr 1e-5
```

## Train LOSO Cascade

Run all three LOSO folds:

```bash
python kjl_subject_independent_cascaded/run_pipeline_GRFKFM_KJL_SI_LOSO.py \
  --window-size 150 \
  --seed 42 \
  --grf-epochs 30 \
  --kfm-epochs 30 \
  --kjl-epochs 50 \
  --kjl-lr 1e-5
```

Run a single held-out subject:

```bash
python kjl_subject_independent_cascaded/run_pipeline_GRFKFM_KJL_SI_LOSO.py \
  --subjects AB02_Rajiv \
  --window-size 150 \
  --seed 42 \
  --grf-epochs 30 \
  --kfm-epochs 30 \
  --kjl-epochs 50 \
  --kjl-lr 1e-5
```

Outputs are written to:

```text
kjl_subject_independent_cascaded/runs/
```

The pipeline writes both:

```text
summary_loso.json
summary_all_tasks_loso.csv
```

## Plot Results

After the LOSO runs finish:

```bash
python kjl_subject_independent_cascaded/plot_loso_cascade_figures.py
```

Figures are saved under:

```text
kjl_subject_independent_cascaded/runs/figures
```

For the doubled unilateral run, pass the matching run root and dataset roots:

```bash
python kjl_subject_independent_cascaded/plot_loso_cascade_figures.py \
  --run-root kjl_subject_independent_cascaded/runs \
  --summary-csv kjl_subject_independent_cascaded/runs/summary_all_tasks_loso_unilateral4imu_double.csv \
  --out-dir kjl_subject_independent_cascaded/runs/figures_unilateral4imu_double \
  --grf-data-root kjl_subject_independent_cascaded/data/grf_unilateral_4imu_double \
  --kfm-data-root kjl_subject_independent_cascaded/data/kfm_unilateral_4imu_double \
  --kjl-data-root kjl_subject_independent_cascaded/data/kjl_unilateral_4imu_double \
  --overlay-condition 20p200ms_R
```

## Notes

- The default upstream targets are `FPR_fz_up_norm_bw` for GRF and `kfm_bwbh` for KFM.
- The default KJL target is `knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw`.
- GRF and KFM dataloaders apply a 15 Hz Butterworth low-pass filter to labels during training/evaluation by default.
- KJL labels are also low-pass filtered in the dataloader.
- The dataset generators include a 15 Hz IMU low-pass step. If the source IMU files have already been filtered upstream, disable or adjust `IMU_FILTER_CUTOFF_HZ` in the generators to avoid double filtering.
