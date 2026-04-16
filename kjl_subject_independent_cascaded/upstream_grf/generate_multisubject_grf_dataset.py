import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import scipy.signal as spsignal


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT_DEFAULT = PACKAGE_DIR / "data" / "grf"
GRAVITY = 9.81
DEFAULT_SUBJECTS = ["AB02_Rajiv", "AB03_Amy", "AB05_Maria"]

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

FORCE_COLS = ["FPR_fx", "FPR_fy", "FPR_fz"]
DEFAULT_TARGET_COL = "FPR_fz_up_norm_bw"
ACC_COLS = [c for c in IMU_COLS if "_acc_" in c]
IMU_FILTER_CUTOFF_HZ: float | None = 15.0
IMU_FILTER_ORDER = 4
IMU_FILTER_FS_HZ = 100.0


def lowpass_filter_imu(
    imu_df: pd.DataFrame,
    cutoff_hz: float | None = IMU_FILTER_CUTOFF_HZ,
    order: int = IMU_FILTER_ORDER,
    fs_hz: float = IMU_FILTER_FS_HZ,
) -> tuple[pd.DataFrame, dict]:
    out = imu_df.copy()
    meta = {
        "imu_lowpass_filter_applied": False,
        "imu_lowpass_cutoff_hz": cutoff_hz,
        "imu_lowpass_order": int(order),
        "imu_lowpass_fs_hz": float(fs_hz),
    }
    if cutoff_hz is None or len(out) < 16:
        return out, meta

    nyq = 0.5 * float(fs_hz)
    wn = float(cutoff_hz) / nyq
    if wn <= 0 or wn >= 1:
        return out, meta

    b, a = spsignal.butter(int(order), wn, btype="low")
    try:
        out.loc[:, IMU_COLS] = spsignal.filtfilt(
            b,
            a,
            out[IMU_COLS].to_numpy(dtype=float),
            axis=0,
        )
        meta["imu_lowpass_filter_applied"] = True
    except ValueError as exc:
        meta["imu_lowpass_filter_error"] = str(exc)
    return out, meta


def standardize_imu_acc_units(imu_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = imu_df.copy()
    acc = out[ACC_COLS].to_numpy(dtype=float)
    median_abs = float(np.nanmedian(np.abs(acc)))
    scale = 1.0
    reason = "already_mps2"
    if np.isfinite(median_abs) and median_abs < 0.1:
        scale = 1000.0
        out.loc[:, ACC_COLS] = out[ACC_COLS] * scale
        reason = "acc_pipeline_units_to_mps2"
    return out, {
        "imu_acc_unit_scale_applied": scale,
        "imu_acc_unit_reason": reason,
        "imu_acc_median_abs_before": median_abs,
        "imu_acc_median_abs_after": float(np.nanmedian(np.abs(out[ACC_COLS].to_numpy(dtype=float)))),
    }

SUBJECT_CONFIGS = {
    "AB03_Amy": {"analog_dir": "1101 Amy CSV"},
    "AB05_Maria": {"analog_dir": "1025 Maria CSV"},
    "AB08_Adrian": {"analog_dir": "1109 Adrian CSV"},
    "AB02_Rajiv": {
        "analog_dir": "1122 Rajiv CSV",
        "exo_mass_kg": 62.7,
        "noexo_mass_kg": 58.4,
        "force_source": "fp_mot",
    },
}


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


def read_storage_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            end_idx = i
            break
    if end_idx is None:
        raise ValueError(f"No endheader found in {path}")
    if end_idx + 1 >= len(lines):
        raise ValueError(f"No header line after endheader in {path}")

    header = lines[end_idx + 1].strip().split()
    data_lines = lines[end_idx + 2 :]
    if not data_lines:
        raise ValueError(f"No data rows in {path}")

    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data[None, :]
    return pd.DataFrame(data, columns=header)


def read_right_force_from_fp_mot(fp_mot_path: Path) -> pd.DataFrame:
    fp_df = read_storage_table(fp_mot_path)
    required = ["time", "FPR_vx", "FPR_vy", "FPR_vz"]
    missing = [c for c in required if c not in fp_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {fp_mot_path}: {missing}")

    # The FP.mot exported by OpenSim stores vertical right GRF in FPR_vy
    # as positive-up. The legacy GRF target uses -FPR_fz/bodyweight.
    return pd.DataFrame(
        {
            "time_force": fp_df["time"].to_numpy(dtype=float),
            "FPR_fx": fp_df["FPR_vx"].to_numpy(dtype=float),
            "FPR_fy": fp_df["FPR_vz"].to_numpy(dtype=float),
            "FPR_fz": -fp_df["FPR_vy"].to_numpy(dtype=float),
            "trigger": np.zeros(len(fp_df), dtype=float),
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
            "peak": float(np.max(trigger_v[s : e + 1])),
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


def build_trial(
    imu_path: Path,
    force_source_path: Path,
    force_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    force_source: str,
    crop_time_window: tuple[float, float] | None = None,
    crop_meta: dict | None = None,
) -> dict:
    imu_df = pd.read_csv(imu_path)

    missing_imu = [c for c in IMU_COLS if c not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"Missing IMU columns in {imu_path.name}: {missing_imu}")
    imu_df, imu_unit_meta = standardize_imu_acc_units(imu_df)
    imu_df, imu_filter_meta = lowpass_filter_imu(imu_df)

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
        "force_source_file": str(force_source_path.relative_to(ROOT)),
        "force_source": force_source,
        "analog_file": str(force_source_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "imu_rows_original": int(len(imu_df)),
        **imu_unit_meta,
        **imu_filter_meta,
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


def _extract_subject_from_trial_rel(trial_rel: str) -> str:
    p = Path(trial_rel)
    if len(p.parts) < 4:
        raise ValueError(f"Unexpected trial path format: {trial_rel}")
    return p.parts[0]


def _build_loso_split(
    trial_rel_paths: list[str],
    held_out_subject: str,
    seed: int = 42,
    val_ratio: float = 0.15,
) -> dict:
    test_trials = sorted([t for t in trial_rel_paths if _extract_subject_from_trial_rel(t) == held_out_subject])
    train_pool = sorted([t for t in trial_rel_paths if _extract_subject_from_trial_rel(t) != held_out_subject])

    if not test_trials:
        raise ValueError(f"No test trials found for held-out subject {held_out_subject}")
    if len(train_pool) < 2:
        raise ValueError(
            f"Not enough training trials for held-out subject {held_out_subject}. "
            f"Need >=2, got {len(train_pool)}."
        )

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_pool))
    rng.shuffle(idx)
    shuffled = [train_pool[i] for i in idx]

    n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
    n_val = min(n_val, len(shuffled) - 1)
    val_trials = sorted(shuffled[:n_val])
    train_trials = sorted(shuffled[n_val:])

    return {
        "split_type": "subject_independent_loso",
        "held_out_subject": held_out_subject,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "test_trials": test_trials,
    }


def _pick_model_path(subject: str, source_group: str) -> Path | None:
    scale_dir = ROOT / "IMU_Data_Process" / subject / source_group / "SCALE"
    preferred_names = [
        f"{subject}_Scaled_unilateral.osim",
        f"{subject}_Scaled_bilateral_unilateral.osim",
        f"{subject}_Scaled_bilateral.osim",
        f"{subject}_Scaled.osim",
        "subject01_Scaled_knee2dof_test.osim",
    ]
    for name in preferred_names:
        path = scale_dir / name
        if path.exists():
            return path
    candidates = sorted(scale_dir.glob("*.osim"))
    return candidates[0] if candidates else None


def _load_mass_for_subject_condition(subject: str, condition_group: str, config: dict) -> float:
    override_key = f"{condition_group}_mass_kg"
    if override_key in config:
        return float(config[override_key])

    source_group = "LG_Exo" if condition_group == "exo" else "LG_NoExo"
    model_path = _pick_model_path(subject, source_group)
    if model_path is None:
        raise FileNotFoundError(f"No SCALE model found for {subject} {source_group}")
    return load_total_model_mass_kg(model_path)


def _find_force_source(
    subject: str,
    source_group: str,
    stem: str,
    analog_dir: Path | None,
) -> tuple[Path | None, str | None]:
    if analog_dir is not None:
        analog_path = analog_dir / f"{stem}.csv"
        if analog_path.exists():
            return analog_path, "analog_csv"

    mot_path = ROOT / "IMU_Data_Process" / subject / source_group / "MOT" / f"{stem}_FP.mot"
    if mot_path.exists():
        return mot_path, "fp_mot"

    return None, None


def _pick_imu_dir(subject: str, source_group: str) -> Path:
    root = ROOT / "IMU_Data_Process" / subject / source_group
    bi_dir = root / "IMU_BI_CSV"
    if bi_dir.exists():
        return bi_dir
    subject_bi_dir = ROOT / "IMU_Data_Process" / subject / "IMU_BI_CSV"
    if subject_bi_dir.exists():
        return subject_bi_dir
    return root / "IMU_CSV"


def generate_dataset(
    subjects: list[str],
    out_root: Path,
    seed: int = 42,
    val_ratio: float = 0.15,
    held_out_subject: str | None = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "subjects": subjects,
        "imu_feature_count": len(IMU_COLS),
        "imu_features": IMU_COLS,
        "label_source": "per-subject analog CSV or OpenSim FP.mot -> right GRF",
        "force_columns": FORCE_COLS,
        "sampling_rate_hz": 100,
        "imu_preprocessing": {
            "acc_unit_standardization": "multiply accelerometer channels by 1000 when median abs < 0.1",
            "lowpass_filter": {
                "type": "zero_phase_butterworth",
                "cutoff_hz": IMU_FILTER_CUTOFF_HZ,
                "order": IMU_FILTER_ORDER,
                "fs_hz": IMU_FILTER_FS_HZ,
                "channels": IMU_COLS,
            },
        },
        "label_resampling": "linear_interpolation_from_force_time_to_IMU_100Hz",
        "alignment_assumption": (
            "IMU stays on native timeline; right-force channels are interpolated onto IMU timestamps."
        ),
        "default_target_col": DEFAULT_TARGET_COL,
        "trials": [],
        "skipped_static": [],
        "skipped_no_analog": [],
        "skipped_no_force_source": [],
        "skipped_bad_analog": [],
        "skipped_bad_force_source": [],
        "subject_model_mass_kg": {},
    }

    for subject in subjects:
        if subject not in SUBJECT_CONFIGS:
            raise ValueError(f"Unsupported subject `{subject}`. Available: {sorted(SUBJECT_CONFIGS)}")

        config = SUBJECT_CONFIGS[subject]
        analog_dir = None
        if config.get("analog_dir"):
            analog_dir = ROOT / "IMU_Data_Process" / subject / str(config["analog_dir"])
        imu_exo_dir = _pick_imu_dir(subject, "LG_Exo")
        imu_noexo_dir = _pick_imu_dir(subject, "LG_NoExo")

        exo_mass_kg = _load_mass_for_subject_condition(subject, "exo", config)
        noexo_mass_kg = _load_mass_for_subject_condition(subject, "noexo", config)
        manifest["subject_model_mass_kg"][subject] = {
            "exo_total_model_mass_kg": exo_mass_kg,
            "noexo_total_model_mass_kg": noexo_mass_kg,
        }

        exo_files = sorted(
            p for p in imu_exo_dir.glob(f"{subject}_LG_*_1.csv") if p.name != f"{subject}_LG_NoExo_1.csv"
        )
        noexo_files = sorted(imu_noexo_dir.glob(f"{subject}_LG_NoExo_1.csv")) if imu_noexo_dir.exists() else []

        file_specs = []
        file_specs.extend((p, exo_mass_kg, "LG_Exo") for p in exo_files)
        file_specs.extend((p, noexo_mass_kg, "LG_NoExo") for p in noexo_files)

        for imu_path, mass_kg, source_group in file_specs:
            stem = imu_path.stem
            if "Static" in stem or "Standing" in stem:
                manifest["skipped_static"].append(stem)
                continue

            force_source_path, force_source = _find_force_source(subject, source_group, stem, analog_dir)
            if force_source_path is None:
                manifest["skipped_no_force_source"].append(stem)
                manifest["skipped_no_analog"].append(stem)
                continue

            try:
                if force_source == "analog_csv":
                    force_df = read_right_force_and_trigger(force_source_path)
                elif force_source == "fp_mot":
                    force_df = read_right_force_from_fp_mot(force_source_path)
                else:
                    raise ValueError(f"Unsupported force source: {force_source}")
            except Exception:
                manifest["skipped_bad_force_source"].append(stem)
                manifest["skipped_bad_analog"].append(stem)
                continue

            trial_base = stem[:-2] if stem.endswith("_1") else stem
            cond = trial_base.split("_LG_")[1]
            out_trial_dir = out_root / subject / "LG" / cond / "trial_1"

            crop_window = None
            trigger_meta = {}
            if force_source == "fp_mot":
                trigger_meta = {
                    "trigger_detected": False,
                    "trigger_reason": "fp_mot_source_has_no_trigger_channel",
                }
            elif cond in {"NoAssi", "NoExo"}:
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
                force_source_path=force_source_path,
                force_df=force_df,
                out_trial_dir=out_trial_dir,
                total_model_mass_kg=mass_kg,
                force_source=force_source,
                crop_time_window=crop_window,
                crop_meta=trigger_meta,
            )
            info["subject"] = subject
            info["condition"] = cond
            info["source_group"] = source_group
            info["output_trial_dir_dataset_rel"] = str(out_trial_dir.relative_to(out_root))
            manifest["trials"].append(info)
            print(f"[OK] {subject}: {stem} -> {out_trial_dir}")

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    trial_rel_paths = sorted([t["output_trial_dir_dataset_rel"] for t in manifest["trials"]])
    split_subjects = [held_out_subject] if held_out_subject else subjects
    for held_out_subject in split_subjects:
        split = _build_loso_split(
            trial_rel_paths=trial_rel_paths,
            held_out_subject=held_out_subject,
            seed=seed,
            val_ratio=val_ratio,
        )
        split["seed"] = int(seed)
        split["val_ratio"] = float(val_ratio)
        out_split = out_root / f"split_subject_independent_loso_{held_out_subject}.json"
        with open(out_split, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=True, indent=2)
        print(
            f"[SPLIT] {held_out_subject}: "
            f"train/val/test={len(split['train_trials'])}/{len(split['val_trials'])}/{len(split['test_trials'])}"
        )

    if held_out_subject is None:
        all_splits = {
            "split_type": "subject_independent_loso",
            "subjects": subjects,
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "folds": [],
        }
        for subj in subjects:
            split = json.loads((out_root / f"split_subject_independent_loso_{subj}.json").read_text())
            all_splits["folds"].append(split)
        with open(out_root / "split_subject_independent_loso_all.json", "w", encoding="utf-8") as f:
            json.dump(all_splits, f, ensure_ascii=True, indent=2)

    print(f"Generated trials: {len(manifest['trials'])}")
    print(f"Skipped static: {len(manifest['skipped_static'])}")
    print(f"Skipped (no analog file): {len(manifest['skipped_no_analog'])}")
    print(f"Skipped (bad analog parse): {len(manifest['skipped_bad_analog'])}")
    print(f"Manifest: {manifest_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-subject GRF+IMU dataset and LOSO splits.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=DEFAULT_SUBJECTS,
        help=f"Subjects to include. Default: {DEFAULT_SUBJECTS}",
    )
    parser.add_argument(
        "--out-root",
        default=str(OUT_ROOT_DEFAULT),
        help=f"Output dataset root. Default: {OUT_ROOT_DEFAULT}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for LOSO train/val split.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio within training subjects.")
    parser.add_argument(
        "--held-out-subject",
        default=None,
        help="If set, only generate one LOSO split for this held-out subject.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    subjects = [str(s) for s in args.subjects]
    held_out_subject = str(args.held_out_subject) if args.held_out_subject else None
    if held_out_subject is not None and held_out_subject not in subjects:
        raise ValueError(f"--held-out-subject `{held_out_subject}` is not in --subjects {subjects}")
    out_root = Path(args.out_root).resolve()
    generate_dataset(
        subjects=subjects,
        out_root=out_root,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        held_out_subject=held_out_subject,
    )


if __name__ == "__main__":
    main()
