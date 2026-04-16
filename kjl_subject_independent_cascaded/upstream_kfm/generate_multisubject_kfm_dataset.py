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
OUT_ROOT_DEFAULT = PACKAGE_DIR / "data" / "kfm"
GRAVITY = 9.81
KFM_COL = "knee_angle_r_moment"
DEFAULT_TARGET_COL = "kfm_bwbh"

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
    "AB03_Amy": {
        "analog_dir": "1101 Amy CSV",
        "height_m": 1.67,
        "exo_mass_kg": 57.3,
        "noexo_mass_kg": 53.0,
    },
    "AB05_Maria": {
        "analog_dir": "1025 Maria CSV",
        "height_m": 1.71,
    },
    "AB02_Rajiv": {
        "analog_dir": "1122 Rajiv CSV",
        "exo_mass_kg": 62.7,
        "noexo_mass_kg": 58.4,
        "force_source": "fp_mot",
    },
}


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


def read_trigger_from_analog(analog_csv_path: Path) -> pd.DataFrame:
    frames = []
    subframes = []
    trigger = []
    with open(analog_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for _ in range(5):
            next(reader, None)
        for row in reader:
            if not row or len(row) < 21:
                break
            try:
                frames.append(float(row[0]))
                subframes.append(float(row[1]))
                trigger.append(float(row[20]))
            except Exception:
                break
    if len(frames) < 1000:
        raise ValueError(f"Insufficient parsed analog rows in {analog_csv_path}")

    fr = np.asarray(frames, dtype=float)
    sf = np.asarray(subframes, dtype=float)
    t = ((fr - 1.0) * 10.0 + sf) / 1000.0
    return pd.DataFrame({"time_force": t, "trigger": np.asarray(trigger, dtype=float)})


def read_trigger_from_fp_mot(fp_mot_path: Path) -> pd.DataFrame:
    fp_df = read_storage_table(fp_mot_path)
    if "time" not in fp_df.columns:
        raise ValueError(f"Missing time column in {fp_mot_path}")
    return pd.DataFrame(
        {
            "time_force": fp_df["time"].to_numpy(dtype=float),
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
    segments = [(int(s), int(e)) for s, e in zip(starts, ends) if int(e - s + 1) >= 50]
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


def resample_to_imu_time(src_time: np.ndarray, src_values: np.ndarray, imu_time: np.ndarray) -> np.ndarray:
    t_unique, unique_idx = np.unique(src_time, return_index=True)
    v_unique = src_values[unique_idx]
    return np.interp(imu_time, t_unique, v_unique, left=np.nan, right=np.nan)


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


def _load_mass(subject: str, condition_group: str, config: dict) -> float:
    override_key = f"{condition_group}_mass_kg"
    if override_key in config:
        return float(config[override_key])

    source_group = "LG_Exo" if condition_group == "exo" else "LG_NoExo"
    model_path = _pick_model_path(subject, source_group)
    if model_path is None:
        raise FileNotFoundError(f"No SCALE model found for {subject} {source_group}")
    return load_total_model_mass_kg(model_path)


def _find_id_file(subject: str, source_group: str, stem: str) -> Path | None:
    search_roots = [
        ROOT / "IMU_Data_Process" / subject / source_group / "ID",
        ROOT / "ID_GT" / subject,
    ]
    for root in search_roots:
        direct = root / f"{stem}_ID.sto"
        if direct.exists():
            return direct
        matches = sorted(root.rglob(f"{stem}_ID.sto")) if root.exists() else []
        if matches:
            return matches[0]
    return None


def _find_trigger_source(
    subject: str,
    source_group: str,
    stem: str,
    analog_dir: Path | None,
) -> tuple[Path | None, str | None]:
    if analog_dir is not None:
        analog_path = analog_dir / f"{stem}.csv"
        if analog_path.exists():
            return analog_path, "analog_csv"
    fp_path = ROOT / "IMU_Data_Process" / subject / source_group / "MOT" / f"{stem}_FP.mot"
    if fp_path.exists():
        return fp_path, "fp_mot"
    return None, None


def _condition_from_stem(stem: str) -> str:
    trial_base = stem[:-2] if stem.endswith("_1") else stem
    return trial_base.split("_LG_")[1]


def _pick_imu_dir(subject: str, source_group: str) -> Path:
    root = ROOT / "IMU_Data_Process" / subject / source_group
    bi_dir = root / "IMU_BI_CSV"
    if bi_dir.exists():
        return bi_dir
    subject_bi_dir = ROOT / "IMU_Data_Process" / subject / "IMU_BI_CSV"
    if subject_bi_dir.exists():
        return subject_bi_dir
    return root / "IMU_CSV"


def build_trial(
    imu_path: Path,
    id_path: Path,
    trigger_source_path: Path,
    trigger_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    body_mass_kg: float,
    height_m: float,
    trigger_source: str,
    crop_time_window: tuple[float, float] | None = None,
    crop_meta: dict | None = None,
) -> dict:
    imu_df = pd.read_csv(imu_path)
    missing_imu = [c for c in IMU_COLS if c not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"Missing IMU columns in {imu_path.name}: {missing_imu}")
    imu_df, imu_unit_meta = standardize_imu_acc_units(imu_df)
    imu_df, imu_filter_meta = lowpass_filter_imu(imu_df)

    id_df = read_storage_table(id_path)
    if "time" not in id_df.columns:
        raise ValueError(f"Missing time column in {id_path}")
    if KFM_COL not in id_df.columns:
        raise ValueError(f"Missing KFM column `{KFM_COL}` in {id_path}")

    imu_feat = imu_df[["time", *IMU_COLS]].copy()
    imu_time = imu_feat["time"].to_numpy(dtype=float)

    kfm_nm = resample_to_imu_time(
        id_df["time"].to_numpy(dtype=float),
        id_df[KFM_COL].to_numpy(dtype=float),
        imu_time,
    )
    trigger = resample_to_imu_time(
        trigger_df["time_force"].to_numpy(dtype=float),
        trigger_df["trigger"].to_numpy(dtype=float),
        imu_time,
    )

    aligned = imu_feat.rename(columns={"time": "time_imu"}).copy()
    aligned["time_id"] = imu_time
    aligned[KFM_COL] = kfm_nm
    aligned["trigger"] = trigger

    finite_mask = np.isfinite(aligned[[KFM_COL, "trigger"]].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite_mask).sum())
    if not finite_mask.any():
        raise ValueError(f"No overlapping timestamps between IMU and ID for {imu_path.name}")
    aligned = aligned.loc[finite_mask].reset_index(drop=True)

    bw_newton = total_model_mass_kg * GRAVITY
    bw_bh_nm = body_mass_kg * GRAVITY * height_m
    aligned["knee_angle_r_moment_norm_kg"] = aligned[KFM_COL] / total_model_mass_kg
    aligned["knee_angle_r_moment_norm_bw"] = aligned[KFM_COL] / bw_newton
    aligned["knee_angle_r_moment_norm_bw_bh"] = aligned[KFM_COL] / bw_bh_nm
    aligned["kfm_bwbh"] = aligned["knee_angle_r_moment_norm_bw_bh"]

    rows_before_crop = int(len(aligned))
    crop_applied = False
    crop_start = None
    crop_end = None
    if crop_time_window is not None:
        crop_start, crop_end = float(crop_time_window[0]), float(crop_time_window[1])
        crop_mask = (aligned["time_imu"] >= crop_start) & (aligned["time_imu"] <= crop_end)
        if crop_mask.any():
            aligned = aligned.loc[crop_mask].reset_index(drop=True)
            crop_applied = True

    aligned.insert(0, "sample_idx", np.arange(len(aligned), dtype=int))

    input_dir = out_trial_dir / "Input"
    label_dir = out_trial_dir / "Label"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    aligned[["sample_idx", "time_imu", *IMU_COLS]].to_csv(input_dir / "imu.csv", index=False)
    label_cols = [
        "sample_idx",
        "time_id",
        KFM_COL,
        "knee_angle_r_moment_norm_kg",
        "knee_angle_r_moment_norm_bw",
        "knee_angle_r_moment_norm_bw_bh",
        "kfm_bwbh",
        "trigger",
    ]
    aligned[label_cols].to_csv(label_dir / "kfm.csv", index=False)
    aligned.to_csv(out_trial_dir / "aligned_debug.csv", index=False)

    return {
        "imu_file": str(imu_path.relative_to(ROOT)),
        "id_file": str(id_path.relative_to(ROOT)),
        "trigger_source_file": str(trigger_source_path.relative_to(ROOT)),
        "trigger_source": trigger_source,
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "imu_rows_original": int(len(imu_df)),
        **imu_unit_meta,
        **imu_filter_meta,
        "id_rows_original": int(len(id_df)),
        "trigger_rows_original": int(len(trigger_df)),
        "rows_after_alignment": int(len(aligned)),
        "rows_dropped_outside_overlap": dropped,
        "rows_before_trigger_crop": rows_before_crop,
        "rows_after_trigger_crop": int(len(aligned)),
        "trigger_crop_applied": bool(crop_applied),
        "trigger_crop_start_time": crop_start,
        "trigger_crop_end_time": crop_end,
        "total_model_mass_kg": float(total_model_mass_kg),
        "body_mass_kg": float(body_mass_kg),
        "height_m": float(height_m),
        "bw_bh_Nm": float(bw_bh_nm),
        "default_target_col": DEFAULT_TARGET_COL,
        "label_source_col": KFM_COL,
        "trigger_meta": crop_meta or {},
    }


def _default_split(trial_paths: list[str], seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(trial_paths))
    rng.shuffle(idx)
    shuffled = [trial_paths[i] for i in idx]
    n = len(shuffled)
    if n < 3:
        return {"train_trials": shuffled, "val_trials": [], "test_trials": shuffled, "seed": seed}

    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train_trials": shuffled[:n_train],
        "val_trials": shuffled[n_train:n_train + n_val],
        "test_trials": shuffled[n_train + n_val:],
        "split_type": "trial_random_subject_dependent",
        "seed": seed,
    }


def _loso_split(trial_infos: list[dict], held_out_subject: str, seed: int = 42) -> dict:
    train_pool = [t["output_trial_dir"] for t in trial_infos if t["subject"] != held_out_subject]
    test_trials = [t["output_trial_dir"] for t in trial_infos if t["subject"] == held_out_subject]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_pool))
    rng.shuffle(idx)
    shuffled = [train_pool[i] for i in idx]
    n_val = max(1, int(round(len(shuffled) * 0.15))) if shuffled else 0
    return {
        "train_trials": shuffled[n_val:],
        "val_trials": shuffled[:n_val],
        "test_trials": test_trials,
        "split_type": "subject_independent_loso",
        "held_out_subject": held_out_subject,
        "seed": seed,
    }


def _parse_height_overrides(values: list[str]) -> dict[str, float]:
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected SUBJECT=HEIGHT_M, got `{item}`")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def generate_dataset(subjects: list[str], out_root: Path, height_overrides: dict[str, float], seed: int = 42) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "subjects_requested": subjects,
        "imu_feature_count": len(IMU_COLS),
        "imu_features": IMU_COLS,
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
        "label_source": "OpenSim inverse dynamics .sto -> knee_angle_r_moment",
        "sampling_rate_hz": 100,
        "default_target_col": DEFAULT_TARGET_COL,
        "normalization": {
            "type": "mass_bodyweight_and_bwbh",
            "formula_norm_kg": "KFM_Nm / total_model_mass_kg",
            "formula_norm_bw": "KFM_Nm / (total_model_mass_kg * 9.81)",
            "formula_bwbh": "kfm_bwbh = KFM_Nm / (body_mass_kg * 9.81 * height_m)",
        },
        "subject_metadata": {},
        "trials": [],
        "skipped_static": [],
        "skipped_no_height": [],
        "skipped_no_trigger_source": [],
        "skipped_no_id": [],
        "skipped_bad_trigger_source": [],
        "skipped_bad_trial": [],
    }

    for subject in subjects:
        if subject not in SUBJECT_CONFIGS:
            raise ValueError(f"Unsupported subject `{subject}`. Available: {sorted(SUBJECT_CONFIGS)}")

        config = dict(SUBJECT_CONFIGS[subject])
        if subject in height_overrides:
            config["height_m"] = height_overrides[subject]
        height_m = config.get("height_m")

        exo_mass_kg = _load_mass(subject, "exo", config)
        noexo_mass_kg = _load_mass(subject, "noexo", config)
        manifest["subject_metadata"][subject] = {
            "height_m": height_m,
            "exo_mass_kg": exo_mass_kg,
            "noexo_mass_kg": noexo_mass_kg,
            "height_source": "override_or_config" if height_m is not None else None,
        }

        analog_dir = None
        if config.get("analog_dir"):
            analog_dir = ROOT / "IMU_Data_Process" / subject / str(config["analog_dir"])

        exo_files = sorted(
            p
            for p in _pick_imu_dir(subject, "LG_Exo").glob(f"{subject}_LG_*_1.csv")
            if p.name != f"{subject}_LG_NoExo_1.csv"
        )
        noexo_dir = _pick_imu_dir(subject, "LG_NoExo")
        noexo_files = sorted(noexo_dir.glob(f"{subject}_LG_NoExo_1.csv")) if noexo_dir.exists() else []

        file_specs = []
        file_specs.extend((p, exo_mass_kg, "LG_Exo") for p in exo_files)
        file_specs.extend((p, noexo_mass_kg, "LG_NoExo") for p in noexo_files)

        for imu_path, mass_kg, source_group in file_specs:
            stem = imu_path.stem
            if "Static" in stem or "Standing" in stem:
                manifest["skipped_static"].append(stem)
                continue
            if height_m is None:
                manifest["skipped_no_height"].append({"subject": subject, "trial": stem})
                continue

            id_path = _find_id_file(subject, source_group, stem)
            if id_path is None:
                manifest["skipped_no_id"].append(stem)
                continue

            trigger_source_path, trigger_source = _find_trigger_source(subject, source_group, stem, analog_dir)
            if trigger_source_path is None:
                manifest["skipped_no_trigger_source"].append(stem)
                continue

            try:
                if trigger_source == "analog_csv":
                    trigger_df = read_trigger_from_analog(trigger_source_path)
                elif trigger_source == "fp_mot":
                    trigger_df = read_trigger_from_fp_mot(trigger_source_path)
                else:
                    raise ValueError(f"Unsupported trigger source {trigger_source}")
            except Exception as exc:
                manifest["skipped_bad_trigger_source"].append({"trial": stem, "reason": str(exc)})
                continue

            cond = _condition_from_stem(stem)
            out_trial_dir = out_root / subject / "LG" / cond / "trial_1"
            crop_window = None
            trigger_meta = {}
            if trigger_source == "fp_mot":
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
                    trigger_df["time_force"].to_numpy(dtype=float),
                    trigger_df["trigger"].to_numpy(dtype=float),
                )

            try:
                info = build_trial(
                    imu_path=imu_path,
                    id_path=id_path,
                    trigger_source_path=trigger_source_path,
                    trigger_df=trigger_df,
                    out_trial_dir=out_trial_dir,
                    total_model_mass_kg=mass_kg,
                    body_mass_kg=mass_kg,
                    height_m=float(height_m),
                    trigger_source=str(trigger_source),
                    crop_time_window=crop_window,
                    crop_meta=trigger_meta,
                )
            except Exception as exc:
                manifest["skipped_bad_trial"].append({"trial": stem, "reason": str(exc)})
                continue

            info["subject"] = subject
            info["condition"] = cond
            info["source_group"] = source_group
            info["output_trial_dir"] = str(out_trial_dir.relative_to(ROOT))
            manifest["trials"].append(info)
            print(f"[OK] {subject} {cond} -> {out_trial_dir}")

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    trial_rel_paths = [t["output_trial_dir"] for t in manifest["trials"]]
    with open(out_root / "split_subject_dependent.json", "w", encoding="utf-8") as f:
        json.dump(_default_split(trial_rel_paths, seed=seed), f, ensure_ascii=True, indent=2)

    generated_subjects = sorted({t["subject"] for t in manifest["trials"]})
    for subject in generated_subjects:
        split = _loso_split(manifest["trials"], held_out_subject=subject, seed=seed)
        with open(out_root / f"split_subject_independent_loso_{subject}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=True, indent=2)

    print(f"Generated {len(manifest['trials'])} trials.")
    print(f"Skipped (no height): {len(manifest['skipped_no_height'])}")
    print(f"Skipped (no ID): {len(manifest['skipped_no_id'])}")
    print(f"Skipped (no trigger source): {len(manifest['skipped_no_trigger_source'])}")
    print(f"Skipped (bad trial): {len(manifest['skipped_bad_trial'])}")
    print(f"Manifest: {out_root / 'manifest.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-subject KFM+IMU dataset.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=sorted(SUBJECT_CONFIGS.keys()),
        help=f"Subjects to include. Default: {sorted(SUBJECT_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--height-overrides",
        nargs="*",
        default=[],
        help="Subject height overrides in meters, e.g. AB02_Rajiv=1.72",
    )
    parser.add_argument(
        "--out-root",
        default=str(OUT_ROOT_DEFAULT),
        help=f"Output dataset root. Default: {OUT_ROOT_DEFAULT}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_dataset(
        subjects=[str(s) for s in args.subjects],
        out_root=Path(args.out_root).resolve(),
        height_overrides=_parse_height_overrides([str(v) for v in args.height_overrides]),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
