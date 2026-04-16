# Subject-independent KJL Cascaded Model

This folder contains the code used to train the leave-one-subject-out (LOSO) cascaded KJL model:

```text
IMU -> GRF model -> predicted GRF
IMU -> KFM model -> predicted KFM
IMU + predicted GRF + predicted KFM -> KJL model
```

The final KJL model uses 26 input channels: 24 IMU channels plus one predicted vertical GRF channel and one predicted KFM channel. The default LOSO subjects are `AB02_Rajiv`, `AB03_Amy`, and `AB05_Maria`.

## Folder Layout

| Path | Role |
| --- | --- |
| `run_pipeline_GRFKFM_KJL_SI_LOSO.py` | End-to-end LOSO pipeline. For each held-out subject, it trains upstream GRF, trains upstream KFM, then trains the cascaded KJL model. |
| `TCN_Training_KJL_AB03_DEP.py` | KJL trainer with optional cascade inputs. |
| `kjl_ab03_tcn_dataset.py` | Windowed KJL dataloader and label filtering. |
| `generate_multisubject_kjl_dataset.py` | Builds the KJL dataset and LOSO splits from `KJL_GT` and `IMU_Data_Process`. |
| `plot_loso_cascade_figures.py` | Generates time overlays, gait-cycle mean +/- SD plots, task metrics, and agreement plots. |
| `upstream_grf/` | GRF dataset generator, dataloader, model, and trainer used by the cascade. |
| `upstream_kfm/` | KFM dataset generator, dataloader, model, and trainer used by the cascade. |

Generated data and model outputs are intentionally not committed. They are expected under:

```text
kjl_subject_independent_cascaded/data/grf
kjl_subject_independent_cascaded/data/kfm
kjl_subject_independent_cascaded/data/kjl
kjl_subject_independent_cascaded/runs
```

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kjl_subject_independent_cascaded/requirements.txt
```

## Prepare Data

Place the raw processing folders in the repository root, using the same names as the lab workspace:

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

## Notes

- The default upstream targets are `FPR_fz_up_norm_bw` for GRF and `kfm_bwbh` for KFM.
- The default KJL target is `knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw`.
- GRF and KFM dataloaders apply a 15 Hz Butterworth low-pass filter to labels during training/evaluation by default.
- KJL labels are also low-pass filtered in the dataloader.
- The dataset generators include a 15 Hz IMU low-pass step. If the source IMU files have already been filtered upstream, disable or adjust `IMU_FILTER_CUTOFF_HZ` in the generators to avoid double filtering.
