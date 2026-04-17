#!/usr/bin/env python3
"""Run AB03 subject-dependent non-cascaded KJL training and simple IMU ablations."""

import argparse
import json
from pathlib import Path

from TCN_Training_KJL_AB03_DEP import train


IMU_COLS = [
    "pelvis_imu_acc_x",
    "pelvis_imu_acc_y",
    "pelvis_imu_acc_z",
    "tibia_r_imu_acc_x",
    "tibia_r_imu_acc_y",
    "tibia_r_imu_acc_z",
    "femur_r_imu_acc_x",
    "femur_r_imu_acc_y",
    "femur_r_imu_acc_z",
    "calcn_r_imu_acc_x",
    "calcn_r_imu_acc_y",
    "calcn_r_imu_acc_z",
    "pelvis_imu_gyr_x",
    "pelvis_imu_gyr_y",
    "pelvis_imu_gyr_z",
    "tibia_r_imu_gyr_x",
    "tibia_r_imu_gyr_y",
    "tibia_r_imu_gyr_z",
    "femur_r_imu_gyr_x",
    "femur_r_imu_gyr_y",
    "femur_r_imu_gyr_z",
    "calcn_r_imu_gyr_x",
    "calcn_r_imu_gyr_y",
    "calcn_r_imu_gyr_z",
]

SENSOR_PREFIX = {
    "pelvis": "pelvis_imu",
    "femur": "femur_r_imu",
    "tibia": "tibia_r_imu",
    "calcn": "calcn_r_imu",
}


def _sensor_cols(sensor: str) -> list[str]:
    prefix = SENSOR_PREFIX[sensor]
    return [col for col in IMU_COLS if col.startswith(prefix)]


def _preset_exclusions(preset: str) -> list[str]:
    if preset == "all":
        return []
    if preset.startswith("no_"):
        sensor = preset.removeprefix("no_")
        return _sensor_cols(sensor)
    if preset.endswith("_only"):
        sensor = preset.removesuffix("_only")
        keep = set(_sensor_cols(sensor))
        return [col for col in IMU_COLS if col not in keep]
    if preset == "acc_only":
        return [col for col in IMU_COLS if "_gyr_" in col]
    if preset == "gyr_only":
        return [col for col in IMU_COLS if "_acc_" in col]
    raise ValueError(f"Unknown ablation preset: {preset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AB03 subject-dependent non-cascaded KJL model."
    )
    parser.add_argument("--dataset-root", default="data/kjl_ab03_dep")
    parser.add_argument("--split-json", default=None)
    parser.add_argument(
        "--target-col",
        default="knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="runs/kjl_ab03_dep_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--ablation",
        default="all",
        choices=[
            "all",
            "no_pelvis",
            "no_femur",
            "no_tibia",
            "no_calcn",
            "pelvis_only",
            "femur_only",
            "tibia_only",
            "calcn_only",
            "acc_only",
            "gyr_only",
        ],
        help="Convenience preset implemented by excluding input columns.",
    )
    parser.add_argument(
        "--exclude-input-cols",
        nargs="*",
        default=[],
        help="Additional exact input column names to exclude.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = sorted(set(_preset_exclusions(args.ablation) + args.exclude_input_cols))
    run_name = args.run_name or f"KJL_AB03_Amy_TCN_DEP_NONCASCADE_{args.ablation}_seed{args.seed}"

    cfg = {
        "run_name": run_name,
        "seed": args.seed,
        "seeds": [args.seed],
        "dataset_root": args.dataset_root,
        "split_json": args.split_json,
        "target_col": args.target_col,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "output_dir": args.output_dir,
        "exclude_input_cols": excluded,
        "use_cascade_inputs": False,
        "cascade_sources": [],
    }

    result = train(cfg)
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
