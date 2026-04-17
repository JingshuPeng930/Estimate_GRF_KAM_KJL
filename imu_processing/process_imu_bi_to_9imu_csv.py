#!/usr/bin/env python3
"""Convert IMU_BI raw outputs into filtered 9-IMU CSV files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as spsignal


SENSORS_9 = [
    "pelvis_imu",
    "tibia_r_imu",
    "femur_r_imu",
    "tibia_l_imu",
    "femur_l_imu",
    "calcn_r_imu",
    "calcn_l_imu",
    "thigh_r_imu",
    "thigh_l_imu",
]

ACC_COLS = [f"{sensor}_acc_{axis}" for sensor in SENSORS_9 for axis in ("x", "y", "z")]
GYR_COLS = [f"{sensor}_gyr_{axis}" for sensor in SENSORS_9 for axis in ("x", "y", "z")]
IMU_COLS_54 = [*ACC_COLS, *GYR_COLS]


def read_opensim_sto(path: Path) -> pd.DataFrame:
    """Read an OpenSim .sto table into a DataFrame."""
    skip_rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().lower() == "endheader":
                skip_rows = i + 1
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=skip_rows, engine="python")


def _split_vec3_series(series: pd.Series) -> pd.DataFrame:
    cleaned = (
        series.astype(str)
        .str.replace(r"[()\[\]]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    parts = cleaned.str.split(",", expand=True)
    if parts.shape[1] != 3:
        raise ValueError(f"Expected Vec3 values, got {parts.shape[1]} columns.")
    return parts.astype(float)


def _safe_sensor_name(name: str) -> str:
    return re.sub(r"\W+", "_", str(name)).strip("_")


def merge_sto_pair(accel_path: Path, gyro_path: Path) -> pd.DataFrame:
    """Merge OpenSim linear acceleration and angular velocity .sto files."""
    acc_df = read_opensim_sto(accel_path)
    gyr_df = read_opensim_sto(gyro_path)

    if "time" not in acc_df.columns or "time" not in gyr_df.columns:
        raise ValueError(f"Missing time column in {accel_path} or {gyro_path}")

    shared = [c for c in acc_df.columns if c != "time" and c in gyr_df.columns]
    if not shared:
        raise ValueError(f"No shared IMU vector columns in {accel_path} and {gyro_path}")

    acc_data: dict[str, pd.Series] = {"time": acc_df["time"]}
    for col in shared:
        vec = _split_vec3_series(acc_df[col])
        safe = _safe_sensor_name(col)
        acc_data[f"{safe}_acc_x"] = vec[0]
        acc_data[f"{safe}_acc_y"] = vec[1]
        acc_data[f"{safe}_acc_z"] = vec[2]

    gyr_data: dict[str, pd.Series] = {"time": gyr_df["time"]}
    for col in shared:
        vec = _split_vec3_series(gyr_df[col])
        safe = _safe_sensor_name(col)
        gyr_data[f"{safe}_gyr_x"] = vec[0]
        gyr_data[f"{safe}_gyr_y"] = vec[1]
        gyr_data[f"{safe}_gyr_z"] = vec[2]

    return pd.merge(pd.DataFrame(acc_data), pd.DataFrame(gyr_data), on="time")


def infer_fs_hz(time: np.ndarray, default_fs_hz: float = 100.0) -> float:
    if len(time) < 3:
        return default_fs_hz
    dt = np.diff(np.asarray(time, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return default_fs_hz
    return float(1.0 / np.median(dt))


def standardize_acc_units(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    meta = {"acc_unit_scale": 1.0, "acc_unit_mode": mode}
    if mode == "none":
        return out, meta
    if mode != "auto":
        scale = float(mode)
        out.loc[:, ACC_COLS] = out[ACC_COLS] * scale
        meta["acc_unit_scale"] = scale
        return out, meta

    acc = out[ACC_COLS].to_numpy(dtype=float)
    median_abs = float(np.nanmedian(np.abs(acc)))
    scale = 1000.0 if np.isfinite(median_abs) and median_abs < 0.1 else 1.0
    out.loc[:, ACC_COLS] = out[ACC_COLS] * scale
    meta.update(
        {
            "acc_unit_scale": scale,
            "acc_median_abs_before": median_abs,
            "acc_median_abs_after": float(np.nanmedian(np.abs(out[ACC_COLS].to_numpy(dtype=float)))),
        }
    )
    return out, meta


def lowpass_filter_imu(
    df: pd.DataFrame,
    cutoff_hz: float | None,
    fs_hz: float,
    order: int,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    meta = {
        "lowpass_applied": False,
        "lowpass_cutoff_hz": cutoff_hz,
        "lowpass_order": int(order),
        "lowpass_fs_hz": float(fs_hz),
    }
    if cutoff_hz is None:
        return out, meta
    if len(out) < max(16, order * 4):
        meta["lowpass_reason"] = "too_few_samples"
        return out, meta

    wn = float(cutoff_hz) / (0.5 * float(fs_hz))
    if wn <= 0 or wn >= 1:
        meta["lowpass_reason"] = "invalid_cutoff"
        return out, meta

    b, a = spsignal.butter(int(order), wn, btype="low")
    try:
        out.loc[:, IMU_COLS_54] = spsignal.filtfilt(
            b,
            a,
            out[IMU_COLS_54].to_numpy(dtype=float),
            axis=0,
        )
        meta["lowpass_applied"] = True
    except ValueError as exc:
        meta["lowpass_reason"] = str(exc)
    return out, meta


def normalize_9imu_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["time", *IMU_COLS_54] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required 9-IMU columns: {missing}")
    out = df[["time", *IMU_COLS_54]].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def process_dataframe(
    df: pd.DataFrame,
    cutoff_hz: float | None,
    fs_hz: float | None,
    order: int,
    acc_unit_scale: str,
) -> tuple[pd.DataFrame, dict]:
    out = normalize_9imu_columns(df)
    inferred_fs = infer_fs_hz(out["time"].to_numpy(dtype=float))
    filter_fs = float(fs_hz) if fs_hz is not None else inferred_fs
    out, unit_meta = standardize_acc_units(out, acc_unit_scale)
    out, filter_meta = lowpass_filter_imu(out, cutoff_hz=cutoff_hz, fs_hz=filter_fs, order=order)
    meta = {
        "n_rows": int(len(out)),
        "n_imu_channels": len(IMU_COLS_54),
        "inferred_fs_hz": inferred_fs,
        **unit_meta,
        **filter_meta,
    }
    return out, meta


def _find_sto_trials(input_root: Path) -> list[tuple[str, Path, Path]]:
    trials = []
    for trial_dir in sorted(p for p in input_root.rglob("*") if p.is_dir()):
        acc_files = sorted(trial_dir.glob("*linear_accelerations.sto"))
        gyr_files = sorted(trial_dir.glob("*angular_velocity.sto"))
        if len(acc_files) == 1 and len(gyr_files) == 1:
            trials.append((trial_dir.name, acc_files[0], gyr_files[0]))
    return trials


def _process_csv_files(args: argparse.Namespace) -> list[dict]:
    records = []
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    csv_files = [input_root] if input_root.is_file() else sorted(input_root.rglob("*.csv"))
    if args.trial_regex:
        pattern = re.compile(args.trial_regex)
        csv_files = [p for p in csv_files if pattern.search(p.stem)]

    for csv_path in csv_files:
        rel = Path(csv_path.name) if input_root.is_file() else csv_path.relative_to(input_root)
        out_path = output_root / rel
        if out_path.exists() and not args.overwrite:
            records.append({"input": str(csv_path), "output": str(out_path), "status": "skipped_exists"})
            continue
        df = pd.read_csv(csv_path)
        out, meta = process_dataframe(
            df,
            cutoff_hz=args.cutoff_hz,
            fs_hz=args.fs_hz,
            order=args.order,
            acc_unit_scale=args.acc_unit_scale,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        records.append({"input": str(csv_path), "output": str(out_path), "status": "ok", **meta})
    return records


def _process_sto_files(args: argparse.Namespace) -> list[dict]:
    records = []
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    trials = _find_sto_trials(input_root)
    if args.trial_regex:
        pattern = re.compile(args.trial_regex)
        trials = [item for item in trials if pattern.search(item[0])]

    for trial_name, accel_path, gyro_path in trials:
        out_path = output_root / f"{trial_name}.csv"
        if out_path.exists() and not args.overwrite:
            records.append({"trial": trial_name, "output": str(out_path), "status": "skipped_exists"})
            continue
        df = merge_sto_pair(accel_path, gyro_path)
        out, meta = process_dataframe(
            df,
            cutoff_hz=args.cutoff_hz,
            fs_hz=args.fs_hz,
            order=args.order,
            acc_unit_scale=args.acc_unit_scale,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        records.append(
            {
                "trial": trial_name,
                "accel_input": str(accel_path),
                "gyro_input": str(gyro_path),
                "output": str(out_path),
                "status": "ok",
                **meta,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input IMU_BI_CSV file/directory or IMU_BI_Data directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for filtered 9-IMU CSV files.")
    parser.add_argument("--source", choices=["auto", "csv", "sto"], default="auto")
    parser.add_argument("--cutoff-hz", type=float, default=15.0, help="Low-pass cutoff in Hz. Use <=0 to disable.")
    parser.add_argument("--fs-hz", type=float, default=None, help="Sampling frequency. Default infers from time.")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order.")
    parser.add_argument(
        "--acc-unit-scale",
        default="none",
        help="Acceleration unit scaling: none, auto, or numeric scale factor.",
    )
    parser.add_argument("--trial-regex", default=None, help="Optional regex filter applied to trial/file stems.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff_hz = None if args.cutoff_hz is not None and args.cutoff_hz <= 0 else args.cutoff_hz
    args.cutoff_hz = cutoff_hz

    if args.source == "auto":
        source = "csv" if any(args.input.rglob("*.csv")) else "sto"
    else:
        source = args.source

    if source == "csv":
        records = _process_csv_files(args)
    else:
        records = _process_sto_files(args)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": source,
        "input_root": str(args.input.resolve()),
        "output_root": str(args.output.resolve()),
        "sensors": SENSORS_9,
        "imu_columns": IMU_COLS_54,
        "cutoff_hz": args.cutoff_hz,
        "fs_hz": args.fs_hz,
        "order": args.order,
        "acc_unit_scale": args.acc_unit_scale,
        "trial_regex": args.trial_regex,
        "records": records,
    }
    manifest_path = args.output / "manifest_imu_bi_preprocessing.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok = sum(1 for r in records if r.get("status") == "ok")
    skipped = sum(1 for r in records if r.get("status") != "ok")
    print(f"Processed {ok} file(s); skipped {skipped}.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
