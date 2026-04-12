import argparse
import csv
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT_DEFAULT = _SCRIPT_DIR / "data_grf_all_subjects_imu"
# Set from CLI in main(): directory that contains IMU_Data_Process/
ROOT = _SCRIPT_DIR


def _safe_rel(path: Path, anchor: Path) -> str:
    try:
        return str(path.resolve().relative_to(anchor.resolve()))
    except ValueError:
        return str(path.resolve())
GRAVITY = 9.81

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

SUBJECT_CONFIGS = {
    "AB03_Amy": {"analog_dir": "1101 Amy CSV"},
    "AB05_Maria": {"analog_dir": "1025 Maria CSV"},
    "AB08_Adrian": {"analog_dir": "1109 Adrian CSV"},
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
    analog_path: Path,
    force_df: pd.DataFrame,
    out_trial_dir: Path,
    total_model_mass_kg: float,
    dataset_out_root: Path,
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
        "imu_file": _safe_rel(imu_path, ROOT),
        "analog_file": _safe_rel(analog_path, ROOT),
        "output_trial_dir": _safe_rel(out_trial_dir, dataset_out_root),
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
        "label_source": "per-subject analog CSV -> Right Force (Fx, Fy, Fz) + trigger",
        "force_columns": FORCE_COLS,
        "sampling_rate_hz": 100,
        "label_resampling": "linear_interpolation_from_analog_1000Hz_to_IMU_100Hz",
        "alignment_assumption": (
            "IMU stays on native timeline; right-force channels are interpolated onto IMU timestamps."
        ),
        "default_target_col": DEFAULT_TARGET_COL,
        "trials": [],
        "skipped_static": [],
        "skipped_no_analog": [],
        "skipped_bad_analog": [],
        "subject_model_mass_kg": {},
    }

    for subject in subjects:
        if subject not in SUBJECT_CONFIGS:
            raise ValueError(f"Unsupported subject `{subject}`. Available: {sorted(SUBJECT_CONFIGS)}")

        analog_dir = ROOT / "IMU_Data_Process" / subject / SUBJECT_CONFIGS[subject]["analog_dir"]
        imu_exo_dir = ROOT / "IMU_Data_Process" / subject / "LG_Exo" / "IMU_CSV"
        imu_noexo_dir = ROOT / "IMU_Data_Process" / subject / "LG_NoExo" / "IMU_CSV"
        scale_exo_model_path = ROOT / "IMU_Data_Process" / subject / "LG_Exo" / "SCALE" / f"{subject}_Scaled_unilateral.osim"
        scale_noexo_model_path = ROOT / "IMU_Data_Process" / subject / "LG_NoExo" / "SCALE" / f"{subject}_Scaled_unilateral.osim"

        exo_mass_kg = load_total_model_mass_kg(scale_exo_model_path)
        noexo_mass_kg = load_total_model_mass_kg(scale_noexo_model_path)
        manifest["subject_model_mass_kg"][subject] = {
            "exo_total_model_mass_kg": exo_mass_kg,
            "noexo_total_model_mass_kg": noexo_mass_kg,
        }

        exo_files = sorted(imu_exo_dir.glob(f"{subject}_LG_*_1.csv"))
        noexo_files = sorted(imu_noexo_dir.glob(f"{subject}_LG_NoExo_1.csv")) if imu_noexo_dir.exists() else []

        file_specs = []
        file_specs.extend((p, exo_mass_kg, "LG_Exo") for p in exo_files)
        file_specs.extend((p, noexo_mass_kg, "LG_NoExo") for p in noexo_files)

        for imu_path, mass_kg, source_group in file_specs:
            stem = imu_path.stem
            if "Static" in stem or "Standing" in stem:
                manifest["skipped_static"].append(stem)
                continue

            analog_path = analog_dir / f"{stem}.csv"
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
            out_trial_dir = out_root / subject / "LG" / cond / "trial_1"

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
                dataset_out_root=out_root,
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
        "--project-root",
        type=str,
        default=None,
        help=(
            "Parent directory of IMU_Data_Process/ (your processed lab data tree). "
            "If omitted, uses this script's directory only if IMU_Data_Process exists there."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=sorted(SUBJECT_CONFIGS.keys()),
        help=f"Subjects to include. Default: {sorted(SUBJECT_CONFIGS.keys())}",
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
    global ROOT
    args = _parse_args()
    if args.project_root:
        ROOT = Path(args.project_root).resolve()
    else:
        ROOT = _SCRIPT_DIR
    imu_proc = ROOT / "IMU_Data_Process"
    if not imu_proc.is_dir():
        print(
            "ERROR: IMU_Data_Process not found.\n"
            f"  Looked under: {ROOT}\n"
            "  Pass --project-root /path/to/project_that_contains_IMU_Data_Process",
            file=sys.stderr,
        )
        raise SystemExit(2)

    subjects = [str(s) for s in args.subjects]
    held_out_subject = str(args.held_out_subject) if args.held_out_subject else None
    if held_out_subject is not None and held_out_subject not in subjects:
        raise ValueError(f"--held-out-subject `{held_out_subject}` is not in --subjects {subjects}")
    out_root = Path(args.out_root)
    generate_dataset(
        subjects=subjects,
        out_root=out_root,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        held_out_subject=held_out_subject,
    )


if __name__ == "__main__":
    main()
