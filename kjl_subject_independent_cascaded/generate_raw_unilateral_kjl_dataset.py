#!/usr/bin/env python3
"""Generate raw bilateral KJL trials for unilateral training.

Each source condition is stored once:

    <subject>/LG/<condition>/trial_1/Input/imu.csv
    <subject>/LG/<condition>/trial_1/Label/kjl_fy.csv

The input CSV keeps raw synced left/right IMU columns. The dataloader mirrors
the left-side channels at training time when it expands each trial into R/L
unilateral samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_multisubject_kjl_dataset import (
    ROOT,
    SUBJECTS,
    _condition_from_jr,
    _find_imu_path,
    _jr_stem,
    _loso_split,
    _subject_masses,
)
from kjl_ab03_tcn_dataset import RAW_IMU_COLS, TARGET_COL, LEFT_TARGET_COL


PACKAGE_DIR = Path(__file__).resolve().parent
GRAVITY = 9.81
KJL_R_COL = "knee_r_on_tibia_r_in_tibia_r_fy"
KJL_L_COL = "knee_l_on_tibia_l_in_tibia_l_fy"
KJL_R_NORM_COL = TARGET_COL
KJL_L_NORM_COL = LEFT_TARGET_COL
OUT_ROOT = PACKAGE_DIR / "data" / "kjl_unilateral_4imu_raw_combined"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _interp_to_imu(jr_df: pd.DataFrame, imu_time: np.ndarray, col: str) -> np.ndarray:
    t = jr_df["time"].to_numpy(dtype=float)
    y = jr_df[col].to_numpy(dtype=float)
    t_unique, unique_idx = np.unique(t, return_index=True)
    return np.interp(imu_time, t_unique, y[unique_idx], left=np.nan, right=np.nan)


def build_raw_trial(
    imu_path: Path,
    jr_path: Path,
    out_trial_dir: Path,
    total_model_mass_kg: float,
) -> dict:
    imu_df = pd.read_csv(imu_path)
    jr_df = pd.read_csv(jr_path)

    missing_imu = [c for c in ["time", *RAW_IMU_COLS] if c not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"Missing raw IMU columns in {imu_path.name}: {missing_imu}")
    missing_label = [c for c in ["time", KJL_R_COL, KJL_L_COL] if c not in jr_df.columns]
    if missing_label:
        raise ValueError(f"Missing KJL columns in {jr_path.name}: {missing_label}")

    imu_time = imu_df["time"].to_numpy(dtype=float)
    labels = pd.DataFrame(
        {
            "time_jr": imu_time.copy(),
            KJL_R_COL: _interp_to_imu(jr_df, imu_time, KJL_R_COL),
            KJL_L_COL: _interp_to_imu(jr_df, imu_time, KJL_L_COL),
        }
    )
    finite = np.isfinite(labels[[KJL_R_COL, KJL_L_COL]].to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        raise ValueError(f"No overlapping timestamps between IMU and JR for {imu_path.name}")

    imu_out = imu_df.loc[finite, ["time", *RAW_IMU_COLS]].rename(columns={"time": "time_imu"}).reset_index(drop=True)
    label_out = labels.loc[finite].reset_index(drop=True)
    sample_idx = np.arange(len(imu_out), dtype=int)
    imu_out.insert(0, "sample_idx", sample_idx)
    label_out.insert(0, "sample_idx", sample_idx)

    bw = float(total_model_mass_kg) * GRAVITY
    label_out[KJL_R_NORM_COL] = label_out[KJL_R_COL] / bw
    label_out[KJL_L_NORM_COL] = label_out[KJL_L_COL] / bw

    input_dir = out_trial_dir / "Input"
    label_dir = out_trial_dir / "Label"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    imu_out.to_csv(input_dir / "imu.csv", index=False)
    label_out.to_csv(label_dir / "kjl_fy.csv", index=False)

    return {
        "imu_file": str(imu_path.relative_to(ROOT)),
        "jr_file": str(jr_path.relative_to(ROOT)),
        "output_trial_dir": _display_path(out_trial_dir),
        "imu_rows_original": int(len(imu_df)),
        "jr_rows_original": int(len(jr_df)),
        "rows_after_timestamp_alignment": int(len(imu_out)),
        "rows_dropped_outside_jr_overlap": int((~finite).sum()),
        "imu_time_start_used": float(imu_out["time_imu"].iloc[0]),
        "imu_time_end_used": float(imu_out["time_imu"].iloc[-1]),
        "total_model_mass_kg": float(total_model_mass_kg),
    }


def generate(subjects: list[str], out_root: Path, seed: int, overwrite: bool) -> None:
    out_root = out_root.resolve()
    if out_root.exists() and overwrite:
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_type": "raw_bilateral_combined_unilateral",
        "description": "Raw synced bilateral IMU inputs with right/left KJL labels; left side is mirrored inside the dataloader.",
        "subjects_requested": subjects,
        "imu_feature_count": len(RAW_IMU_COLS),
        "imu_features": RAW_IMU_COLS,
        "input_preprocessing": "none; source IMU columns are copied without unit scaling, low-pass filtering, or mirroring",
        "label_resampling": "linear_interpolation_from_JR_time_to_IMU_time",
        "label_columns": [KJL_R_COL, KJL_L_COL, KJL_R_NORM_COL, KJL_L_NORM_COL],
        "training_loader_behavior": "each raw trial is expanded into R and L samples; L uses left IMU columns mirrored to pseudo-right channel order",
        "trials": [],
        "skipped": [],
    }

    for subject in subjects:
        exo_mass, noexo_mass, mass_meta = _subject_masses(subject)
        jr_files = sorted((ROOT / "KJL_GT" / subject).rglob("*_JointReaction_ReactionLoads.csv"))
        for jr_path in jr_files:
            cond = _condition_from_jr(jr_path)
            stem = _jr_stem(jr_path)
            imu_path, source_group = _find_imu_path(subject, cond, stem)
            mass_kg = noexo_mass if cond == "NoExo" else exo_mass
            if mass_kg is None or not imu_path.exists():
                manifest["skipped"].append(
                    {
                        "subject": subject,
                        "condition": cond,
                        "jr_file": str(jr_path.relative_to(ROOT)),
                        "expected_imu_file": str(imu_path.relative_to(ROOT)),
                        "reason": "missing mass or IMU file",
                    }
                )
                continue

            out_trial_dir = out_root / subject / "LG" / cond / "trial_1"
            try:
                info = build_raw_trial(imu_path, jr_path, out_trial_dir, float(mass_kg))
            except Exception as exc:
                manifest["skipped"].append(
                    {
                        "subject": subject,
                        "condition": cond,
                        "jr_file": str(jr_path.relative_to(ROOT)),
                        "imu_file": str(imu_path.relative_to(ROOT)),
                        "reason": str(exc),
                    }
                )
                continue
            info.update(
                {
                    "subject": subject,
                    "condition": cond,
                    "source_group": source_group,
                    "subject_mass_meta": mass_meta,
                }
            )
            manifest["trials"].append(info)
            print(f"[OK] {subject} {cond} -> {out_trial_dir}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    for subject in sorted({t["subject"] for t in manifest["trials"]}):
        split = _loso_split(manifest["trials"], held_out_subject=subject, seed=seed)
        (out_root / f"split_subject_independent_loso_{subject}.json").write_text(
            json.dumps(split, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    print(f"[DONE] Generated {len(manifest['trials'])} raw combined trials at {out_root}")
    print(f"[DONE] Skipped {len(manifest['skipped'])} trials")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=SUBJECTS)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate([str(s) for s in args.subjects], args.out_root, int(args.seed), bool(args.overwrite))


if __name__ == "__main__":
    main()
