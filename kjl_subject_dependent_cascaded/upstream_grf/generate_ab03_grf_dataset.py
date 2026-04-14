import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMU_EXO_DIR = ROOT / "IMU_Data_Process" / "AB03_Amy" / "LG_Exo" / "IMU_CSV"
SCALE_EXO_MODEL_PATH = (
    ROOT
    / "IMU_Data_Process"
    / "AB03_Amy"
    / "LG_Exo"
    / "SCALE"
    / "AB03_Amy_Scaled_unilateral.osim"
)
IMU_NOEXO_DIR = ROOT / "IMU_Data_Process" / "AB03_Amy" / "LG_NoExo" / "IMU_CSV"
SCALE_NOEXO_MODEL_PATH = (
    ROOT
    / "IMU_Data_Process"
    / "AB03_Amy"
    / "LG_NoExo"
    / "SCALE"
    / "AB03_Amy_Scaled_unilateral.osim"
)
ANALOG_DIR = ROOT / "IMU_Data_Process" / "AB03_Amy" / "1101 Amy CSV"
OUT_ROOT = ROOT / "training_code_IMUonly_GRF" / "data_grf_ab03_imu"

SUBJECT = "AB03_Amy"
GRAVITY = 9.81

IMU_COLS = [
    "pelvis_imu_acc_x", "pelvis_imu_acc_y", "pelvis_imu_acc_z",
    "tibia_r_imu_acc_x", "tibia_r_imu_acc_y", "tibia_r_imu_acc_z",
    "femur_r_imu_acc_x", "femur_r_imu_acc_y", "femur_r_imu_acc_z",
    "calcn_r_imu_acc_x", "calcn_r_imu_acc_y", "calcn_r_imu_acc_z",
    "pelvis_imu_gyr_x", "pelvis_imu_gyr_y", "pelvis_imu_gyr_z",
    "tibia_r_imu_gyr_x", "tibia_r_imu_gyr_y", "tibia_r_imu_gyr_z",
    "femur_r_imu_gyr_x", "femur_r_imu_gyr_y", "femur_r_imu_gyr_z",
    "calcn_r_imu_gyr_x", "calcn_r_imu_gyr_y", "calcn_r_imu_gyr_z",
]

FORCE_COLS = ["FPR_fx", "FPR_fy", "FPR_fz"]
DEFAULT_TARGET_COL = "FPR_fz_up_norm_bw"


def load_total_model_mass_kg(osim_path: Path) -> float:
    root = ET.parse(osim_path).getroot()
    masses = []
    for body in root.findall(".//Body"):
        m = body.find("mass")
        if m is None or m.text is None:
            continue
        try:
            masses.append(float(m.text))
        except ValueError:
            continue
    if not masses:
        raise ValueError(f"No body masses found in {osim_path}")
    return float(sum(masses))


def read_right_force_and_trigger(analog_csv_path: Path) -> pd.DataFrame:
    """Read first (Devices) block from 1101 CSV: Right Force (Fx,Fy,Fz) + trigger."""
    frames = []
    subframes = []
    fx = []
    fy = []
    fz = []
    trigger = []

    with open(analog_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for _ in range(5):
            next(reader, None)

        for row in reader:
            if not row or len(row) < 21:
                break
            try:
                fr = float(row[0])
                sf = float(row[1])
                f_x = float(row[2])
                f_y = float(row[3])
                f_z = float(row[4])
                tr = float(row[20])
            except Exception:
                break
            frames.append(fr)
            subframes.append(sf)
            fx.append(f_x)
            fy.append(f_y)
            fz.append(f_z)
            trigger.append(tr)

    if len(frames) < 1000:
        raise ValueError(f"Insufficient parsed analog rows in {analog_csv_path}")

    fr = np.asarray(frames, dtype=float)
    sf = np.asarray(subframes, dtype=float)
    t = ((fr - 1.0) * 10.0 + sf) / 1000.0

    return pd.DataFrame(
        {
            "time_force": t,
            "FPR_fx": np.asarray(fx, dtype=float),
            "FPR_fy": np.asarray(fy, dtype=float),
            "FPR_fz": np.asarray(fz, dtype=float),
            "trigger": np.asarray(trigger, dtype=float),
        }
    )


def detect_trigger_crop_window(trigger_t: np.ndarray, trigger_v: np.ndarray) -> tuple[tuple[float, float] | None, dict]:
    base_n = min(len(trigger_v), 3000)
    baseline = trigger_v[:base_n]
    base_med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - base_med)))
    robust_std = 1.4826 * mad + 1e-8
    threshold = max(base_med + 8.0 * robust_std, 0.5)

    active_idx = np.where(trigger_v > threshold)[0]
    if active_idx.size == 0:
        return None, {
            "trigger_threshold": float(threshold),
            "trigger_segments": [],
            "trigger_detected": False,
            "trigger_reason": "no_active_samples_above_threshold",
        }

    split_idx = np.where(np.diff(active_idx) > 1)[0]
    starts = np.r_[active_idx[0], active_idx[split_idx + 1]]
    ends = np.r_[active_idx[split_idx], active_idx[-1]]

    segments = []
    for s, e in zip(starts, ends):
        if int(e - s + 1) < 50:
            continue
        segments.append((int(s), int(e)))

    seg_info = [
        {
            "start_time": float(trigger_t[s]),
            "end_time": float(trigger_t[e]),
            "n_samples": int(e - s + 1),
            "peak": float(np.max(trigger_v[s:e + 1])),
        }
        for s, e in segments
    ]

    if len(segments) < 2:
        return None, {
            "trigger_threshold": float(threshold),
            "trigger_segments": seg_info,
            "trigger_detected": False,
            "trigger_reason": "less_than_two_trigger_segments",
        }

    first_end = segments[0][1]
    second_start = segments[1][0]
    if second_start <= first_end:
        return None, {
            "trigger_threshold": float(threshold),
            "trigger_segments": seg_info,
            "trigger_detected": False,
            "trigger_reason": "invalid_trigger_order",
        }

    return (float(trigger_t[first_end]), float(trigger_t[second_start])), {
        "trigger_threshold": float(threshold),
        "trigger_segments": seg_info,
        "trigger_detected": True,
        "trigger_reason": "ok",
    }


def resample_force_to_imu_time(force_df: pd.DataFrame, imu_time: np.ndarray, value_cols: list[str]) -> pd.DataFrame:
    tf = force_df["time_force"].to_numpy(dtype=float)
    tf_unique, unique_idx = np.unique(tf, return_index=True)
    force_unique = force_df.iloc[unique_idx].reset_index(drop=True)

    out = {"time_force": imu_time.copy()}
    for col in value_cols:
        out[col] = np.interp(
            imu_time,
            tf_unique,
            force_unique[col].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
    return pd.DataFrame(out)


def _default_split(trial_paths: list[str], seed: int = 42) -> dict:
    if len(trial_paths) < 3:
        return {
            "train_trials": trial_paths,
            "val_trials": [],
            "test_trials": trial_paths,
        }

    rng = np.random.default_rng(seed)
    idx = np.arange(len(trial_paths))
    rng.shuffle(idx)
    shuffled = [trial_paths[i] for i in idx]

    n = len(shuffled)
    n_train = max(1, int(round(n * 0.7)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    return {
        "train_trials": shuffled[:n_train],
        "val_trials": shuffled[n_train : n_train + n_val],
        "test_trials": shuffled[n_train + n_val :],
    }


def build_trial(
    imu_path: Path,
    analog_path: Path,
    force_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    crop_time_window: tuple[float, float] | None = None,
    crop_meta: dict | None = None,
) -> dict:
    imu_df = pd.read_csv(imu_path)

    missing_imu = [c for c in IMU_COLS if c not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"Missing IMU columns in {imu_path.name}: {missing_imu}")

    imu_feat = imu_df[["time", *IMU_COLS]].copy()
    imu_time = imu_feat["time"].to_numpy(dtype=float)

    value_cols = [*FORCE_COLS, "trigger"]
    force_on_imu = resample_force_to_imu_time(force_df, imu_time, value_cols=value_cols)

    finite_mask = np.isfinite(force_on_imu[value_cols].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite_mask).sum())
    if not finite_mask.any():
        raise ValueError(f"No overlapping timestamps between IMU and analog force for {imu_path.name}")

    imu_aligned = imu_feat.loc[finite_mask].rename(columns={"time": "time_imu"}).reset_index(drop=True)
    force_aligned = force_on_imu.loc[finite_mask].reset_index(drop=True)
    merged = pd.concat([imu_aligned, force_aligned], axis=1)

    bw_newton = total_model_mass_kg * GRAVITY
    for col in FORCE_COLS:
        merged[f"{col}_norm_bw"] = merged[col] / bw_newton
    merged["FPR_fz_up_norm_bw"] = -merged["FPR_fz_norm_bw"]

    rows_before_crop = int(len(merged))
    crop_applied = False
    crop_start = None
    crop_end = None
    if crop_time_window is not None:
        crop_start, crop_end = float(crop_time_window[0]), float(crop_time_window[1])
        crop_mask = (merged["time_imu"] >= crop_start) & (merged["time_imu"] <= crop_end)
        if crop_mask.any():
            merged = merged.loc[crop_mask].reset_index(drop=True)
            crop_applied = True

    merged.insert(0, "sample_idx", np.arange(len(merged), dtype=int))

    input_dir = out_trial_dir / "Input"
    label_dir = out_trial_dir / "Label"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    merged[["sample_idx", "time_imu", *IMU_COLS]].to_csv(input_dir / "imu.csv", index=False)

    label_cols = [
        "sample_idx",
        "time_force",
        *FORCE_COLS,
        "trigger",
        *[f"{c}_norm_bw" for c in FORCE_COLS],
        "FPR_fz_up_norm_bw",
    ]
    merged[label_cols].to_csv(label_dir / "grf.csv", index=False)
    merged.to_csv(out_trial_dir / "aligned_debug.csv", index=False)

    return {
        "imu_file": str(imu_path.relative_to(ROOT)),
        "analog_file": str(analog_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "imu_rows_original": int(len(imu_df)),
        "analog_rows_original": int(len(force_df)),
        "rows_after_alignment": int(len(merged)),
        "rows_dropped_outside_force_time_range": dropped,
        "rows_before_trigger_crop": rows_before_crop,
        "rows_after_trigger_crop": int(len(merged)),
        "imu_time_start_used": float(merged["time_imu"].iloc[0]),
        "imu_time_end_used": float(merged["time_imu"].iloc[-1]),
        "force_time_start": float(force_df["time_force"].iloc[0]),
        "force_time_end": float(force_df["time_force"].iloc[-1]),
        "trigger_crop_applied": bool(crop_applied),
        "trigger_crop_start_time": crop_start,
        "trigger_crop_end_time": crop_end,
        "alignment": "right_force_fz_interpolated_to_imu_time",
        "total_model_mass_kg": float(total_model_mass_kg),
        "normalization": "force_N_divided_by_bodyweight_N",
        "default_target_col": DEFAULT_TARGET_COL,
        "trigger_meta": crop_meta or {},
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    exo_mass_kg = load_total_model_mass_kg(SCALE_EXO_MODEL_PATH)
    noexo_mass_kg = load_total_model_mass_kg(SCALE_NOEXO_MODEL_PATH)

    manifest = {
        "subject": SUBJECT,
        "imu_feature_count": len(IMU_COLS),
        "imu_features": IMU_COLS,
        "label_source": "1101 Amy CSV -> Right - Force Fz",
        "force_columns": FORCE_COLS,
        "sampling_rate_hz": 100,
        "label_resampling": "linear_interpolation_from_analog_1000Hz_to_IMU_100Hz",
        "alignment_assumption": "IMU stays on native timeline; right-force channels are interpolated onto IMU timestamps.",
        "normalization": {
            "type": "bodyweight",
            "formula": "force_norm_bw = force_N / (total_model_mass_kg * 9.81)",
            "exo_total_model_mass_kg": exo_mass_kg,
            "noexo_total_model_mass_kg": noexo_mass_kg,
        },
        "default_target_col": DEFAULT_TARGET_COL,
        "trials": [],
        "skipped_static": [],
        "skipped_no_analog": [],
        "skipped_bad_analog": [],
    }

    exo_files = sorted(IMU_EXO_DIR.glob(f"{SUBJECT}_LG_*_1.csv"))
    noexo_files = sorted(IMU_NOEXO_DIR.glob(f"{SUBJECT}_LG_NoExo_1.csv")) if IMU_NOEXO_DIR.exists() else []

    file_specs = []
    file_specs.extend((p, exo_mass_kg, "LG_Exo") for p in exo_files)
    file_specs.extend((p, noexo_mass_kg, "LG_NoExo") for p in noexo_files)

    for imu_path, mass_kg, source_group in file_specs:
        stem = imu_path.stem
        if "Static" in stem:
            manifest["skipped_static"].append(stem)
            continue

        analog_path = ANALOG_DIR / f"{stem}.csv"
        if not analog_path.exists():
            manifest["skipped_no_analog"].append(stem)
            continue

        try:
            force_df = read_right_force_and_trigger(analog_path)
        except Exception:
            manifest["skipped_bad_analog"].append(stem)
            continue

        trial_base = stem[:-2] if stem.endswith("_1") else stem
        cond = trial_base.split("_LG_")[1]
        out_trial_dir = OUT_ROOT / SUBJECT / "LG" / cond / "trial_1"

        crop_window = None
        trigger_meta = {}
        if cond in {"NoAssi", "NoExo"}:
            trigger_meta = {
                "trigger_detected": False,
                "trigger_reason": "condition_excluded_from_trigger_crop",
            }
        else:
            crop_window, trigger_meta = detect_trigger_crop_window(
                force_df["time_force"].to_numpy(dtype=float),
                force_df["trigger"].to_numpy(dtype=float),
            )

        info = build_trial(
            imu_path=imu_path,
            analog_path=analog_path,
            force_df=force_df,
            out_trial_dir=out_trial_dir,
            total_model_mass_kg=mass_kg,
            crop_time_window=crop_window,
            crop_meta=trigger_meta,
        )
        info["source_group"] = source_group
        manifest["trials"].append(info)
        print(f"[OK] {stem} -> {out_trial_dir}")

    with open(OUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    trial_rel_paths = [t["output_trial_dir"] for t in manifest["trials"]]
    split = _default_split(trial_rel_paths)
    with open(OUT_ROOT / "split_subject_dependent.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=True, indent=2)

    print(f"Generated {len(manifest['trials'])} trials")
    print(f"Skipped static: {len(manifest['skipped_static'])}")
    print(f"Skipped (no analog file): {len(manifest['skipped_no_analog'])}")
    print(f"Skipped (bad analog parse): {len(manifest['skipped_bad_analog'])}")
    print(f"Manifest: {OUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
