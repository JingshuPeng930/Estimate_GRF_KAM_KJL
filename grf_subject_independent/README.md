# IMU-only GRF estimation — subject-independent (LOSO)

Temporal convolutional network (TCN) training code for estimating vertical GRF (normalized by body weight) from IMU, evaluated with leave-one-subject-out (LOSO) splits.

Training data are **not** included. Prepare your dataset locally or generate it from a project tree that contains `IMU_Data_Process/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Build the multi-subject dataset (optional)

Requires a directory layout with `IMU_Data_Process/<subject>/...` (see `generate_multisubject_grf_dataset.py` for subject names and paths). Output defaults to `./data_grf_all_subjects_imu/`.

```bash
python generate_multisubject_grf_dataset.py --project-root /path/to/parent-of-IMU_Data_Process
```

If `IMU_Data_Process` sits next to these scripts, you can omit `--project-root`.

## 2) Train LOSO

From this folder (so `data_grf_all_subjects_imu/` is visible next to the scripts):

```bash
# All folds (hold out each subject once)
python TCN_Training_GRF_SubjectIndependent_LOSO.py --all

# Single fold
python TCN_Training_GRF_SubjectIndependent_LOSO.py --held-out-subject AB08_Adrian
```

Checkpoints and logs are written under `runs_grf_si_loso/`.

## Files

| File | Role |
|------|------|
| `TCN_Training_GRF_SubjectIndependent_LOSO.py` | LOSO entry point |
| `TCN_Training_GRF_AB03.py` | Training loop and hyperparameters |
| `grf_ab03_tcn_dataset.py` | Windowed trials and dataloaders |
| `TCN_Header_Model.py` | TCN architecture |
| `generate_multisubject_grf_dataset.py` | Build aligned IMU/GRF trials from raw processed data |

## License

Add your license here if you publish this repository.
