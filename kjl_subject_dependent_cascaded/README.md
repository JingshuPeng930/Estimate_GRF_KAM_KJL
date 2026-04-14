# Subject-dependent KJL Cascaded Model

This folder contains the code needed to train an AB03 Amy subject-dependent KJL model with cascaded upstream predictions:

```text
IMU -> GRF model -> predicted GRF
IMU -> KFM model -> predicted KFM
IMU + predicted GRF + predicted KFM -> KJL model
```

The default setup uses `window_size=150`, GRF target `FPR_fz_up_norm_bw`, KFM target `kfm_bwbh`, and KJL target `knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw`.

## Folder Layout

| Path | Role |
| --- | --- |
| `TCN_Training_KJL_AB03_DEP.py` | Main KJL training code with optional cascade inputs. |
| `TCN_Training_KJL_AB03_DEP_CASCADE_Amy03.py` | Direct wrapper for `GRF + KFM -> KJL` when upstream runs already exist. |
| `run_pipeline_GRFKFM_KJL.py` | Staged pipeline that checks or trains GRF/KFM, then trains cascaded KJL. |
| `kjl_ab03_tcn_dataset.py` | KJL subject-dependent dataloader. |
| `soft_delay_classifier.py` | Optional delay-conditioning helper used by KJL training. |
| `upstream_grf/` | Subject-dependent GRF training and dataset code used as an upstream model. |
| `upstream_kfm/` | Subject-dependent KFM training and dataset code used as an upstream model. |

## Data

Data are not committed. The pipeline expects prepared datasets at:

```text
kjl_subject_dependent_cascaded/data/grf_ab03_imu
kjl_subject_dependent_cascaded/data/kfm_ab03_id
kjl_subject_dependent_cascaded/data/kjl_ab03_dep
```

The required trial layout is the same as the original training folders:

```text
<dataset_root>/AB03_Amy/LG/<condition>/trial_1/Input/imu.csv
<dataset_root>/AB03_Amy/LG/<condition>/trial_1/Label/*.csv
```

For KFM, `generate_ab03_kfm_dataset.py` extracts `knee_angle_r_moment` from OpenSim ID `.sto` files and creates `kfm_bwbh` using trial-specific mass:

```text
Exo/NoAssi: KFM_Nm / (57.3 * 9.81 * 1.67)
NoExo:      KFM_Nm / (53.0 * 9.81 * 1.67)
```

## Run

Use existing upstream checkpoints if present; otherwise train missing upstream models and then train KJL:

```bash
python run_pipeline_GRFKFM_KJL.py
```

Force GRF and KFM to retrain first, then train KJL:

```bash
python run_pipeline_GRFKFM_KJL.py --retrain-grf --retrain-kfm
```

Only prepare/check upstream models:

```bash
python run_pipeline_GRFKFM_KJL.py --skip-kjl
```

Quick KJL smoke test after upstream models exist:

```bash
python run_pipeline_GRFKFM_KJL.py --kjl-epochs 1
```

Outputs are written under:

```text
kjl_subject_dependent_cascaded/runs/
```
