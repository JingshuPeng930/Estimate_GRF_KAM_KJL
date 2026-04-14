import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "AB03_Amy"
GRAVITY = 9.81
SUBJECT_HEIGHT_M = 1.67

IMU_EXO_DIR = ROOT / "IMU_Data_Process" / SUBJECT / "LG_Exo" / "IMU_CSV"
IMU_NOEXO_DIR = ROOT / "IMU_Data_Process" / SUBJECT / "LG_NoExo" / "IMU_CSV"
SCALE_EXO_DIR = ROOT / "IMU_Data_Process" / SUBJECT / "LG_Exo" / "SCALE"
SCALE_NOEXO_DIR = ROOT / "IMU_Data_Process" / SUBJECT / "LG_NoExo" / "SCALE"
ANALOG_DIR = ROOT / "IMU_Data_Process" / SUBJECT / "1101 Amy CSV"
ID_DIR = ROOT / "ID_GT" / SUBJECT

OUT_ROOT = ROOT / "training_code_IMUonly_KFM" / "data_kfm_ab03_id"

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


def _pick_model_path(scale_dir: Path) -> Path:
    preferred = scale_dir / "subject01_Scaled_knee2dof_test.osim"
    if preferred.exists():
        return preferred
    candidates = sorted(
        p for p in scale_dir.glob("*_Scaled.osim")
        if "_unilateral" not in p.stem.lower() and "knee2dof" not in p.stem.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"No suitable *_Scaled.osim found in {scale_dir}")
    return candidates[0]


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
    data_lines = lines[end_idx + 2:]
    if not data_lines:
        raise ValueError(f"No data rows in {path}")

    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data[None, :]
    return pd.DataFrame(data, columns=header)


def read_right_force_and_trigger(analog_csv_path: Path) -> pd.DataFrame:
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
                fr = float(row[0])
                sf = float(row[1])
                tr = float(row[20])
            except Exception:
                break
            frames.append(fr)
            subframes.append(sf)
            trigger.append(tr)

    if len(frames) < 1000:
        raise ValueError(f"Insufficient parsed analog rows in {analog_csv_path}")

    fr = np.asarray(frames, dtype=float)
    sf = np.asarray(subframes, dtype=float)
    t = ((fr - 1.0) * 10.0 + sf) / 1000.0
    return pd.DataFrame({"time_force": t, "trigger": np.asarray(trigger, dtype=float)})


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
        if int(e - s + 1) >= 50:
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


def resample_to_imu_time(src_time: np.ndarray, src_values: np.ndarray, imu_time: np.ndarray) -> np.ndarray:
    t_unique, unique_idx = np.unique(src_time, return_index=True)
    v_unique = src_values[unique_idx]
    return np.interp(imu_time, t_unique, v_unique, left=np.nan, right=np.nan)


def _find_id_file(stem: str) -> Path | None:
    direct = ID_DIR / f"{stem}_ID.sto"
    if direct.exists():
        return direct
    matches = sorted(ID_DIR.rglob(f"{stem}_ID.sto"))
    return matches[0] if matches else None


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
        "val_trials": shuffled[n_train:n_train + n_val],
        "test_trials": shuffled[n_train + n_val:],
    }


def build_trial(
    imu_path: Path,
    id_path: Path,
    analog_path: Path,
    trigger_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    body_mass_kg: float,
    height_m: float,
    crop_time_window: tuple[float, float] | None = None,
    crop_meta: dict | None = None,
) -> dict:
    imu_df = pd.read_csv(imu_path)
    missing_imu = [c for c in IMU_COLS if c not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"Missing IMU columns in {imu_path.name}: {missing_imu}")

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
    subject_bw_newton = body_mass_kg * GRAVITY
    bw_bh_nm = subject_bw_newton * height_m
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
        "analog_file": str(analog_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "imu_rows_original": int(len(imu_df)),
        "id_rows_original": int(len(id_df)),
        "analog_rows_original": int(len(trigger_df)),
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
        "normalization": {
            "norm_kg": "knee_angle_r_moment_norm_kg = KFM_Nm / total_model_mass_kg",
            "norm_bw": "knee_angle_r_moment_norm_bw = KFM_Nm / (total_model_mass_kg * 9.81)",
            "bwbh": "kfm_bwbh = KFM_Nm / (body_mass_kg * 9.81 * height_m)",
        },
        "trigger_meta": crop_meta or {},
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    exo_model_path = _pick_model_path(SCALE_EXO_DIR)
    noexo_model_path = _pick_model_path(SCALE_NOEXO_DIR)
    exo_mass_kg = load_total_model_mass_kg(exo_model_path)
    noexo_mass_kg = load_total_model_mass_kg(noexo_model_path)

    manifest = {
        "subject": SUBJECT,
        "imu_feature_count": len(IMU_COLS),
        "imu_features": IMU_COLS,
        "label_source": "OpenSim inverse dynamics .sto -> knee_angle_r_moment",
        "sampling_rate_hz": 100,
        "default_target_col": DEFAULT_TARGET_COL,
        "normalization": {
            "type": "mass_bodyweight_and_bwbh",
            "formula_norm_kg": "KFM_Nm / total_model_mass_kg",
            "formula_norm_bw": "KFM_Nm / (total_model_mass_kg * 9.81)",
            "formula_bwbh": "KFM_Nm / (body_mass_kg * 9.81 * height_m)",
            "body_mass_rule": "Use trial/model mass: Exo/NoAssi use exo_total_model_mass_kg, NoExo uses noexo_total_model_mass_kg",
            "height_m": SUBJECT_HEIGHT_M,
            "exo_bw_bh_Nm": exo_mass_kg * GRAVITY * SUBJECT_HEIGHT_M,
            "noexo_bw_bh_Nm": noexo_mass_kg * GRAVITY * SUBJECT_HEIGHT_M,
            "exo_total_model_mass_kg": exo_mass_kg,
            "noexo_total_model_mass_kg": noexo_mass_kg,
            "exo_model_path": str(exo_model_path.relative_to(ROOT)),
            "noexo_model_path": str(noexo_model_path.relative_to(ROOT)),
        },
        "trials": [],
        "skipped_static": [],
        "skipped_no_analog": [],
        "skipped_no_id": [],
        "skipped_bad_analog": [],
        "skipped_bad_id": [],
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

        id_path = _find_id_file(stem)
        if id_path is None:
            manifest["skipped_no_id"].append(stem)
            continue

        try:
            trigger_df = read_right_force_and_trigger(analog_path)
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
                trigger_df["time_force"].to_numpy(dtype=float),
                trigger_df["trigger"].to_numpy(dtype=float),
            )

        try:
            info = build_trial(
                imu_path=imu_path,
                id_path=id_path,
                analog_path=analog_path,
                trigger_df=trigger_df,
                out_trial_dir=out_trial_dir,
                total_model_mass_kg=mass_kg,
                body_mass_kg=mass_kg,
                height_m=SUBJECT_HEIGHT_M,
                crop_time_window=crop_window,
                crop_meta=trigger_meta,
            )
        except ValueError as e:
            manifest["skipped_bad_id"].append({"trial": stem, "reason": str(e)})
            continue

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
    print(f"Skipped (no ID file): {len(manifest['skipped_no_id'])}")
    print(f"Skipped (bad analog parse): {len(manifest['skipped_bad_analog'])}")
    print(f"Skipped (bad ID parse): {len(manifest['skipped_bad_id'])}")
    print(f"Manifest: {OUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
