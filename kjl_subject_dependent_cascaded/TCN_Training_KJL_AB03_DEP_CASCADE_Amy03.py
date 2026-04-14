import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from TCN_Training_KJL_AB03_DEP import train


THIS_DIR = Path(__file__).resolve().parent

GRF_RUN_DIR = str(
    THIS_DIR
    / "runs"
    / "upstream_grf"
    / "GRF_AB03_Amy_TCN_IMU_seed42_FPR_fz_upNBW_w150_bs256_do0p15_huber_b10.0_lpf15p0_ch32x32x32x32"
)

KFM_RUN_DIR = str(
    THIS_DIR
    / "runs"
    / "upstream_kfm"
    / "KFM_AB03_Amy_TCN_IMU_seed42_kfm_bwbh_w150_bs32_do0p15_huber_b10.0_lpf15p0_ch32x32x32x32"
)


CASCADE_CFG = {
    "run_name": "KJL_AB03_Amy_TCN_DEP_CASCADE_GRFKFM",
    "seed": 42,
    "seeds": [42],
    "dataset_root": str(THIS_DIR / "data" / "kjl_ab03_dep"),
    "output_dir": str(THIS_DIR / "runs" / "kjl_cascade"),
    "window_size": 150,
    "batch_size": 32,
    "epochs": 50,
    "use_cascade_inputs": True,
    "cascade_prediction_mode": "normalized",
    "cascade_allow_window_adapter": False,
    "cascade_sources": [
        {
            "name": "grf",
            "enabled": True,
            "run_dir": GRF_RUN_DIR,
            "checkpoint_path": "",
            "output_indices": [0],
        },
        {
            "name": "kfm",
            "enabled": True,
            "run_dir": KFM_RUN_DIR,
            "checkpoint_path": "",
            "output_indices": [0],
        },
    ],
}


if __name__ == "__main__":
    train(CASCADE_CFG)
