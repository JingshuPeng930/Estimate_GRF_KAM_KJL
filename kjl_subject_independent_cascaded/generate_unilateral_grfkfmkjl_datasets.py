#!/usr/bin/env python3
"""Generate pseudo-right unilateral GRF/KFM/KJL datasets.

Each source trial contributes two samples:
  - side=R: pelvis + right femur/tibia/calcn IMUs, right label.
  - side=L: pelvis + left femur/tibia/calcn IMUs mirrored into right-side
    convention, left label.

The output input columns stay compatible with the existing 4-IMU models:
pelvis, tibia_r, femur_r, calcn_r x acc/gyr xyz = 24 channels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import scipy.signal as spsignal


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
for candidate in (PACKAGE_DIR / "upstream_kfm", ROOT / "training_code_IMUonly_KFM"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from generate_multisubject_kfm_dataset import (  # noqa: E402
    SUBJECT_CONFIGS,
    detect_trigger_crop_window,
    read_storage_table,
    read_trigger_from_analog,
    read_trigger_from_fp_mot,
    resample_to_imu_time,
    _condition_from_stem,
    _find_id_file,
    _find_trigger_source,
    _load_mass,
    _parse_height_overrides,
    _pick_imu_dir,
)


GRAVITY = 9.81
KFM_R_COL = "knee_angle_r_moment"
KFM_L_COL = "knee_angle_l_moment"
KFM_TARGET_COL = "kfm_bwbh"
KJL_R_COL = "knee_r_on_tibia_r_in_tibia_r_fy"
KJL_L_COL = "knee_l_on_tibia_l_in_tibia_l_fy"
KJL_TARGET_COL = "knee_r_on_tibia_r_in_tibia_r_fy"
KJL_TARGET_COL_NORM = "knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw"
GRF_TARGET_COL = "FPR_fz_up_norm_bw"

OUT_GRF_DEFAULT = PACKAGE_DIR / "data" / "grf_unilateral_4imu_double"
OUT_KFM_DEFAULT = PACKAGE_DIR / "data" / "kfm_unilateral_4imu_double"
OUT_KJL_DEFAULT = PACKAGE_DIR / "data" / "kjl_unilateral_4imu_double"

PSEUDO_IMU_COLS = [
    "pelvis_imu_acc_x", "pelvis_imu_acc_y", "pelvis_imu_acc_z",
    "tibia_r_imu_acc_x", "tibia_r_imu_acc_y", "tibia_r_imu_acc_z",
    "femur_r_imu_acc_x", "femur_r_imu_acc_y", "femur_r_imu_acc_z",
    "calcn_r_imu_acc_x", "calcn_r_imu_acc_y", "calcn_r_imu_acc_z",
    "pelvis_imu_gyr_x", "pelvis_imu_gyr_y", "pelvis_imu_gyr_z",
    "tibia_r_imu_gyr_x", "tibia_r_imu_gyr_y", "tibia_r_imu_gyr_z",
    "femur_r_imu_gyr_x", "femur_r_imu_gyr_y", "femur_r_imu_gyr_z",
    "calcn_r_imu_gyr_x", "calcn_r_imu_gyr_y", "calcn_r_imu_gyr_z",
]

SOURCE_IMU_COLS = sorted({
    c.replace("_r_", "_l_") if any(seg in c for seg in ["tibia_r", "femur_r", "calcn_r"]) else c
    for c in PSEUDO_IMU_COLS
} | set(PSEUDO_IMU_COLS))
ACC_COLS = [c for c in SOURCE_IMU_COLS if "_acc_" in c]
IMU_FILTER_CUTOFF_HZ: float | None = 15.0
IMU_FILTER_ORDER = 4
IMU_FILTER_FS_HZ = 100.0

FLIP_COMPONENTS = {"acc_y", "gyr_x", "gyr_z"}
LEFT_LABEL_SIGN = {
    "grf": 1.0,
    "kfm": 1.0,
    "kjl": 1.0,
}


def _component(col: str) -> str:
    parts = col.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def _standardize_imu_acc_units(imu_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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


def _lowpass_filter_imu(imu_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = imu_df.copy()
    meta = {
        "imu_lowpass_filter_applied": False,
        "imu_lowpass_cutoff_hz": IMU_FILTER_CUTOFF_HZ,
        "imu_lowpass_order": IMU_FILTER_ORDER,
        "imu_lowpass_fs_hz": IMU_FILTER_FS_HZ,
    }
    if IMU_FILTER_CUTOFF_HZ is None or len(out) < 16:
        return out, meta
    wn = float(IMU_FILTER_CUTOFF_HZ) / (0.5 * IMU_FILTER_FS_HZ)
    b, a = spsignal.butter(IMU_FILTER_ORDER, wn, btype="low")
    try:
        out.loc[:, SOURCE_IMU_COLS] = spsignal.filtfilt(
            b,
            a,
            out[SOURCE_IMU_COLS].to_numpy(dtype=float),
            axis=0,
        )
        meta["imu_lowpass_filter_applied"] = True
    except ValueError as exc:
        meta["imu_lowpass_filter_error"] = str(exc)
    return out, meta


def _pseudo_right_imu(imu_df: pd.DataFrame, side: str) -> pd.DataFrame:
    out = pd.DataFrame({"time": imu_df["time"].to_numpy(dtype=float)})
    for col in PSEUDO_IMU_COLS:
        if col.startswith("pelvis_imu_"):
            src = col
        elif side == "R":
            src = col
        else:
            src = col.replace("_r_", "_l_")
        values = imu_df[src].to_numpy(dtype=float).copy()
        if side == "L" and _component(col) in FLIP_COMPONENTS:
            values *= -1.0
        out[col] = values
    return out


def _prepare_imu(imu_path: Path, side: str) -> tuple[pd.DataFrame, dict]:
    imu_df = pd.read_csv(imu_path)
    missing = [c for c in ["time", *SOURCE_IMU_COLS] if c not in imu_df.columns]
    if missing:
        raise ValueError(f"Missing required IMU columns in {imu_path}: {missing}")
    imu_df, unit_meta = _standardize_imu_acc_units(imu_df)
    imu_df, filter_meta = _lowpass_filter_imu(imu_df)
    return _pseudo_right_imu(imu_df, side), {**unit_meta, **filter_meta}


def _read_force_and_trigger(analog_csv_path: Path, side: str) -> pd.DataFrame:
    frames, subframes, fx, fy, fz, trigger = [], [], [], [], [], []
    offset = 2 if side == "R" else 11
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
                fx.append(float(row[offset]))
                fy.append(float(row[offset + 1]))
                fz.append(float(row[offset + 2]))
                trigger.append(float(row[20]))
            except Exception:
                break
    if len(frames) < 1000:
        raise ValueError(f"Insufficient parsed analog rows in {analog_csv_path}")
    fr = np.asarray(frames)
    sf = np.asarray(subframes)
    return pd.DataFrame(
        {
            "time_force": ((fr - 1.0) * 10.0 + sf) / 1000.0,
            "FPR_fx": np.asarray(fx, dtype=float),
            "FPR_fy": np.asarray(fy, dtype=float),
            "FPR_fz": np.asarray(fz, dtype=float),
            "trigger": np.asarray(trigger, dtype=float),
        }
    )


def _read_force_from_fp_mot(fp_mot_path: Path, side: str) -> pd.DataFrame:
    fp_df = read_storage_table(fp_mot_path)
    prefix = "FPR" if side == "R" else "FPL"
    required = ["time", f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_vz"]
    missing = [c for c in required if c not in fp_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {fp_mot_path}: {missing}")
    return pd.DataFrame(
        {
            "time_force": fp_df["time"].to_numpy(dtype=float),
            "FPR_fx": fp_df[f"{prefix}_vx"].to_numpy(dtype=float),
            "FPR_fy": fp_df[f"{prefix}_vz"].to_numpy(dtype=float),
            "FPR_fz": -fp_df[f"{prefix}_vy"].to_numpy(dtype=float),
            "trigger": np.zeros(len(fp_df), dtype=float),
        }
    )


def _resample_table_to_imu_time(src_df: pd.DataFrame, imu_time: np.ndarray, value_cols: list[str], time_col: str) -> pd.DataFrame:
    t = src_df[time_col].to_numpy(dtype=float)
    t_unique, unique_idx = np.unique(t, return_index=True)
    src_unique = src_df.iloc[unique_idx].reset_index(drop=True)
    out = {time_col: imu_time.copy()}
    for col in value_cols:
        out[col] = np.interp(
            imu_time,
            t_unique,
            src_unique[col].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
    return pd.DataFrame(out)


def _apply_crop(df: pd.DataFrame, time_col: str, crop_time_window: tuple[float, float] | None) -> tuple[pd.DataFrame, dict]:
    rows_before = int(len(df))
    if crop_time_window is None:
        return df, {
            "rows_before_trigger_crop": rows_before,
            "rows_after_trigger_crop": rows_before,
            "trigger_crop_applied": False,
            "trigger_crop_start_time": None,
            "trigger_crop_end_time": None,
        }
    start, end = float(crop_time_window[0]), float(crop_time_window[1])
    mask = (df[time_col] >= start) & (df[time_col] <= end)
    if not mask.any():
        return df, {
            "rows_before_trigger_crop": rows_before,
            "rows_after_trigger_crop": rows_before,
            "trigger_crop_applied": False,
            "trigger_crop_start_time": start,
            "trigger_crop_end_time": end,
        }
    out = df.loc[mask].reset_index(drop=True)
    return out, {
        "rows_before_trigger_crop": rows_before,
        "rows_after_trigger_crop": int(len(out)),
        "trigger_crop_applied": True,
        "trigger_crop_start_time": start,
        "trigger_crop_end_time": end,
    }


def _write_input(df: pd.DataFrame, out_trial_dir: Path) -> None:
    input_dir = out_trial_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    df[["sample_idx", "time_imu", *PSEUDO_IMU_COLS]].to_csv(input_dir / "imu.csv", index=False)


def build_grf_trial(
    imu_path: Path,
    force_source_path: Path,
    force_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    side: str,
    crop_time_window: tuple[float, float] | None,
    trigger_meta: dict,
) -> dict:
    imu_feat, imu_meta = _prepare_imu(imu_path, side)
    imu_time = imu_feat["time"].to_numpy(dtype=float)
    force_on_imu = _resample_table_to_imu_time(force_df, imu_time, ["FPR_fx", "FPR_fy", "FPR_fz", "trigger"], "time_force")
    finite = np.isfinite(force_on_imu[["FPR_fx", "FPR_fy", "FPR_fz", "trigger"]].to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        raise ValueError("No overlapping timestamps between IMU and force")
    merged = pd.concat(
        [imu_feat.loc[finite].rename(columns={"time": "time_imu"}).reset_index(drop=True), force_on_imu.loc[finite].reset_index(drop=True)],
        axis=1,
    )
    bw = total_model_mass_kg * GRAVITY
    for col in ["FPR_fx", "FPR_fy", "FPR_fz"]:
        merged[f"{col}_norm_bw"] = merged[col] / bw
    merged[GRF_TARGET_COL] = -merged["FPR_fz_norm_bw"] * (LEFT_LABEL_SIGN["grf"] if side == "L" else 1.0)
    merged, crop_info = _apply_crop(merged, "time_imu", crop_time_window)
    merged.insert(0, "sample_idx", np.arange(len(merged), dtype=int))
    (out_trial_dir / "Label").mkdir(parents=True, exist_ok=True)
    _write_input(merged, out_trial_dir)
    label_cols = ["sample_idx", "time_force", "FPR_fx", "FPR_fy", "FPR_fz", "trigger", "FPR_fx_norm_bw", "FPR_fy_norm_bw", "FPR_fz_norm_bw", GRF_TARGET_COL]
    merged[label_cols].to_csv(out_trial_dir / "Label" / "grf.csv", index=False)
    return {
        "force_source_file": str(force_source_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "side": side,
        "rows_after_alignment": int(len(merged)),
        "rows_dropped_outside_overlap": int((~finite).sum()),
        **imu_meta,
        **crop_info,
        "trigger_meta": trigger_meta,
        "label_source_side": side,
        "label_target_col": GRF_TARGET_COL,
    }


def build_kfm_trial(
    imu_path: Path,
    id_path: Path,
    trigger_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    body_mass_kg: float,
    height_m: float,
    side: str,
    crop_time_window: tuple[float, float] | None,
    trigger_meta: dict,
) -> dict:
    imu_feat, imu_meta = _prepare_imu(imu_path, side)
    id_df = read_storage_table(id_path)
    src_col = KFM_R_COL if side == "R" else KFM_L_COL
    if src_col not in id_df.columns:
        raise ValueError(f"Missing KFM column {src_col} in {id_path}")
    imu_time = imu_feat["time"].to_numpy(dtype=float)
    kfm_nm = resample_to_imu_time(id_df["time"].to_numpy(dtype=float), id_df[src_col].to_numpy(dtype=float), imu_time)
    if side == "L":
        kfm_nm = kfm_nm * LEFT_LABEL_SIGN["kfm"]
    trigger = resample_to_imu_time(trigger_df["time_force"].to_numpy(dtype=float), trigger_df["trigger"].to_numpy(dtype=float), imu_time)
    aligned = imu_feat.rename(columns={"time": "time_imu"}).copy()
    aligned["time_id"] = imu_time
    aligned[KFM_R_COL] = kfm_nm
    aligned["trigger"] = trigger
    finite = np.isfinite(aligned[[KFM_R_COL, "trigger"]].to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        raise ValueError("No overlapping timestamps between IMU and ID")
    aligned = aligned.loc[finite].reset_index(drop=True)
    bw = total_model_mass_kg * GRAVITY
    bw_bh = body_mass_kg * GRAVITY * height_m
    aligned["knee_angle_r_moment_norm_kg"] = aligned[KFM_R_COL] / total_model_mass_kg
    aligned["knee_angle_r_moment_norm_bw"] = aligned[KFM_R_COL] / bw
    aligned["knee_angle_r_moment_norm_bw_bh"] = aligned[KFM_R_COL] / bw_bh
    aligned[KFM_TARGET_COL] = aligned["knee_angle_r_moment_norm_bw_bh"]
    aligned, crop_info = _apply_crop(aligned, "time_imu", crop_time_window)
    aligned.insert(0, "sample_idx", np.arange(len(aligned), dtype=int))
    (out_trial_dir / "Label").mkdir(parents=True, exist_ok=True)
    _write_input(aligned, out_trial_dir)
    label_cols = ["sample_idx", "time_id", KFM_R_COL, "knee_angle_r_moment_norm_kg", "knee_angle_r_moment_norm_bw", "knee_angle_r_moment_norm_bw_bh", KFM_TARGET_COL, "trigger"]
    aligned[label_cols].to_csv(out_trial_dir / "Label" / "kfm.csv", index=False)
    return {
        "id_file": str(id_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "side": side,
        "rows_after_alignment": int(len(aligned)),
        "rows_dropped_outside_overlap": int((~finite).sum()),
        **imu_meta,
        **crop_info,
        "trigger_meta": trigger_meta,
        "label_source_col": src_col,
        "label_target_col": KFM_TARGET_COL,
    }


def build_kjl_trial(
    imu_path: Path,
    jr_path: Path,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    side: str,
) -> dict:
    imu_feat, imu_meta = _prepare_imu(imu_path, side)
    jr_df = pd.read_csv(jr_path)
    src_col = KJL_R_COL if side == "R" else KJL_L_COL
    if "time" not in jr_df.columns or src_col not in jr_df.columns:
        raise ValueError(f"Missing time/{src_col} in {jr_path}")
    imu_time = imu_feat["time"].to_numpy(dtype=float)
    kjl = resample_to_imu_time(jr_df["time"].to_numpy(dtype=float), jr_df[src_col].to_numpy(dtype=float), imu_time)
    if side == "L":
        kjl = kjl * LEFT_LABEL_SIGN["kjl"]
    merged = imu_feat.rename(columns={"time": "time_imu"}).copy()
    merged["time_jr"] = imu_time
    merged[KJL_TARGET_COL] = kjl
    finite = np.isfinite(merged[[KJL_TARGET_COL]].to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        raise ValueError("No overlapping timestamps between IMU and JR")
    merged = merged.loc[finite].reset_index(drop=True)
    merged[KJL_TARGET_COL_NORM] = merged[KJL_TARGET_COL] / (total_model_mass_kg * GRAVITY)
    merged.insert(0, "sample_idx", np.arange(len(merged), dtype=int))
    (out_trial_dir / "Label").mkdir(parents=True, exist_ok=True)
    _write_input(merged, out_trial_dir)
    merged[["sample_idx", "time_jr", KJL_TARGET_COL, KJL_TARGET_COL_NORM]].to_csv(out_trial_dir / "Label" / "kjl_fy.csv", index=False)
    return {
        "jr_file": str(jr_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "side": side,
        "rows_after_alignment": int(len(merged)),
        "rows_dropped_outside_overlap": int((~finite).sum()),
        **imu_meta,
        "label_source_col": src_col,
        "label_target_col": KJL_TARGET_COL_NORM,
    }


def _find_jr_file(subject: str, source_group: str, stem: str, cond: str) -> Path | None:
    roots = [
        ROOT / "KJL_GT" / subject,
        ROOT / "IMU_Data_Process" / subject / source_group,
    ]
    names = [
        f"{stem[:-2] if stem.endswith('_1') else stem}_JointReaction_ReactionLoads.csv",
        f"{stem}_JointReaction_ReactionLoads.csv",
    ]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / ("NoExo" if cond == "NoExo" else "Exo") / "LG" / "Analyze" / cond / name
            if direct.exists():
                return direct
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    return None


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


def _write_manifest_and_splits(out_root: Path, manifest: dict, seed: int) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for subject in sorted({t["subject"] for t in manifest["trials"]}):
        split = _loso_split(manifest["trials"], held_out_subject=subject, seed=seed)
        (out_root / f"split_subject_independent_loso_{subject}.json").write_text(json.dumps(split, indent=2), encoding="utf-8")


def _manifest(task: str, subjects: list[str], target_col: str) -> dict:
    return {
        "task": task,
        "dataset_type": "pseudo_right_unilateral_double",
        "subjects_requested": subjects,
        "imu_feature_count": len(PSEUDO_IMU_COLS),
        "imu_features": PSEUDO_IMU_COLS,
        "imu_mapping": {
            "right_sample": "pelvis + femur_r/tibia_r/calcn_r",
            "left_sample": "pelvis flipped + femur_l/tibia_l/calcn_l flipped and renamed to right-side feature names",
            "flipped_channels": sorted(FLIP_COMPONENTS),
            "left_label_sign": LEFT_LABEL_SIGN[task],
        },
        "target_col": target_col,
        "trials": [],
        "skipped": [],
    }


def generate(subjects: list[str], out_grf: Path, out_kfm: Path, out_kjl: Path, height_overrides: dict[str, float], seed: int) -> None:
    manifests = {
        "grf": _manifest("grf", subjects, GRF_TARGET_COL),
        "kfm": _manifest("kfm", subjects, KFM_TARGET_COL),
        "kjl": _manifest("kjl", subjects, KJL_TARGET_COL_NORM),
    }
    out_roots = {"grf": out_grf, "kfm": out_kfm, "kjl": out_kjl}

    for subject in subjects:
        config = dict(SUBJECT_CONFIGS[subject])
        if subject in height_overrides:
            config["height_m"] = height_overrides[subject]
        height_m = config.get("height_m")
        exo_mass = _load_mass(subject, "exo", config)
        noexo_mass = _load_mass(subject, "noexo", config)
        analog_dir = ROOT / "IMU_Data_Process" / subject / str(config["analog_dir"]) if config.get("analog_dir") else None

        exo_files = sorted(p for p in _pick_imu_dir(subject, "LG_Exo").glob(f"{subject}_LG_*_1.csv") if p.name != f"{subject}_LG_NoExo_1.csv")
        noexo_dir = _pick_imu_dir(subject, "LG_NoExo")
        noexo_files = sorted(noexo_dir.glob(f"{subject}_LG_NoExo_1.csv")) if noexo_dir.exists() else []
        file_specs = [(p, exo_mass, "LG_Exo") for p in exo_files] + [(p, noexo_mass, "LG_NoExo") for p in noexo_files]

        for imu_path, mass_kg, source_group in file_specs:
            stem = imu_path.stem
            if "Static" in stem or "Standing" in stem:
                continue
            cond = _condition_from_stem(stem)
            id_path = _find_id_file(subject, source_group, stem)
            jr_path = _find_jr_file(subject, source_group, stem, cond)
            trigger_source_path, trigger_source = _find_trigger_source(subject, source_group, stem, analog_dir)

            trigger_df = None
            crop_window = None
            trigger_meta = {}
            if trigger_source_path is not None:
                try:
                    if trigger_source == "analog_csv":
                        trigger_df = read_trigger_from_analog(trigger_source_path)
                    else:
                        trigger_df = read_trigger_from_fp_mot(trigger_source_path)
                    if trigger_source == "analog_csv" and cond not in {"NoAssi", "NoExo"}:
                        crop_window, trigger_meta = detect_trigger_crop_window(
                            trigger_df["time_force"].to_numpy(dtype=float),
                            trigger_df["trigger"].to_numpy(dtype=float),
                        )
                    elif cond in {"NoAssi", "NoExo"}:
                        trigger_meta = {"trigger_detected": False, "trigger_reason": "condition_excluded_from_trigger_crop"}
                    else:
                        trigger_meta = {"trigger_detected": False, "trigger_reason": "fp_mot_source_has_no_trigger_channel"}
                except Exception as exc:
                    for m in manifests.values():
                        m["skipped"].append({"subject": subject, "trial": stem, "reason": f"bad trigger source: {exc}"})
                    continue

            for side in ("R", "L"):
                side_cond = f"{cond}_{side}"
                if trigger_source_path is not None:
                    try:
                        force_df = _read_force_and_trigger(trigger_source_path, side) if trigger_source == "analog_csv" else _read_force_from_fp_mot(trigger_source_path, side)
                        out_trial = out_grf / subject / "LG" / side_cond / "trial_1"
                        info = build_grf_trial(imu_path, trigger_source_path, force_df, out_trial, mass_kg, side, crop_window, trigger_meta)
                        info.update({"subject": subject, "condition": side_cond, "source_condition": cond})
                        manifests["grf"]["trials"].append(info)
                    except Exception as exc:
                        manifests["grf"]["skipped"].append({"subject": subject, "trial": stem, "side": side, "reason": str(exc)})

                if id_path is not None and trigger_df is not None and height_m is not None:
                    try:
                        out_trial = out_kfm / subject / "LG" / side_cond / "trial_1"
                        info = build_kfm_trial(imu_path, id_path, trigger_df, out_trial, mass_kg, mass_kg, float(height_m), side, crop_window, trigger_meta)
                        info.update({"subject": subject, "condition": side_cond, "source_condition": cond})
                        manifests["kfm"]["trials"].append(info)
                    except Exception as exc:
                        manifests["kfm"]["skipped"].append({"subject": subject, "trial": stem, "side": side, "reason": str(exc)})

                if jr_path is not None:
                    try:
                        out_trial = out_kjl / subject / "LG" / side_cond / "trial_1"
                        info = build_kjl_trial(imu_path, jr_path, out_trial, mass_kg, side)
                        info.update({"subject": subject, "condition": side_cond, "source_condition": cond})
                        manifests["kjl"]["trials"].append(info)
                    except Exception as exc:
                        manifests["kjl"]["skipped"].append({"subject": subject, "trial": stem, "side": side, "reason": str(exc)})

    for task, manifest in manifests.items():
        _write_manifest_and_splits(out_roots[task], manifest, seed=seed)
        print(f"[{task.upper()}] trials={len(manifest['trials'])}, skipped={len(manifest['skipped'])}, root={out_roots[task]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=["AB02_Rajiv", "AB03_Amy", "AB05_Maria"])
    parser.add_argument("--height-overrides", nargs="*", default=[], help="Subject height overrides, e.g. AB02_Rajiv=1.76")
    parser.add_argument("--out-grf", type=Path, default=OUT_GRF_DEFAULT)
    parser.add_argument("--out-kfm", type=Path, default=OUT_KFM_DEFAULT)
    parser.add_argument("--out-kjl", type=Path, default=OUT_KJL_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        subjects=[str(s) for s in args.subjects],
        out_grf=args.out_grf.resolve(),
        out_kfm=args.out_kfm.resolve(),
        out_kjl=args.out_kjl.resolve(),
        height_overrides=_parse_height_overrides([str(v) for v in args.height_overrides]),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
