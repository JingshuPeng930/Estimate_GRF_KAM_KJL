import json
from pathlib import Path

import numpy as np

from generate_ab03_kjl_dep_dataset import (
    IMU_COLS,
    TARGET_COL,
    TARGET_COL_NORM_TOTAL_BW,
    build_trial,
    load_total_model_mass_kg,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
SUBJECTS = ["AB02_Rajiv", "AB03_Amy", "AB05_Maria"]
OUT_ROOT = PACKAGE_DIR / "data" / "kjl"
MASS_OVERRIDES_KG = {
    # Rajiv's OpenSim SCALE folders are not present in the current workspace.
    # Use provided total model masses so labels can be normalized once IMU data
    # are available.
    "AB02_Rajiv": {
        "exo": 62.7,
        "noexo": 58.4,
    },
}


def _pick_model_path(scale_dir: Path) -> Path | None:
    preferred_names = [
        "subject01_Scaled_knee2dof_test.osim",
        f"{scale_dir.parents[1].name}_Scaled_unilateral.osim",
        f"{scale_dir.parents[1].name}_Scaled.osim",
    ]
    for name in preferred_names:
        path = scale_dir / name
        if path.exists():
            return path
    candidates = sorted(scale_dir.glob("*.osim"))
    return candidates[0] if candidates else None


def _subject_masses(subject: str) -> tuple[float | None, float | None, dict]:
    meta = {}
    exo_scale = ROOT / "IMU_Data_Process" / subject / "LG_Exo" / "SCALE"
    noexo_scale = ROOT / "IMU_Data_Process" / subject / "LG_NoExo" / "SCALE"
    exo_model = _pick_model_path(exo_scale)
    noexo_model = _pick_model_path(noexo_scale)

    exo_mass = None
    noexo_mass = None
    if exo_model is not None:
        exo_mass = load_total_model_mass_kg(exo_model)
        meta["exo_model_path"] = str(exo_model.relative_to(ROOT))
    if noexo_model is not None:
        noexo_mass = load_total_model_mass_kg(noexo_model)
        meta["noexo_model_path"] = str(noexo_model.relative_to(ROOT))

    override = MASS_OVERRIDES_KG.get(subject, {})
    if exo_mass is None and "exo" in override:
        exo_mass = float(override["exo"])
        meta["exo_mass_source"] = "manual_override"
    if noexo_mass is None and "noexo" in override:
        noexo_mass = float(override["noexo"])
        meta["noexo_mass_source"] = "manual_override"

    meta["exo_total_model_mass_kg"] = exo_mass
    meta["noexo_total_model_mass_kg"] = noexo_mass
    return exo_mass, noexo_mass, meta


def _jr_stem(jr_path: Path) -> str:
    suffix = "_JointReaction_ReactionLoads"
    if not jr_path.stem.endswith(suffix):
        raise ValueError(f"Unexpected JR filename: {jr_path.name}")
    return jr_path.stem[: -len(suffix)]


def _condition_from_jr(jr_path: Path) -> str:
    return jr_path.parent.name


def _find_imu_path(subject: str, cond: str, jr_stem: str) -> tuple[Path, str]:
    group = "LG_NoExo" if cond == "NoExo" else "LG_Exo"
    root = ROOT / "IMU_Data_Process" / subject / group
    bi_dir = root / "IMU_BI_CSV"
    subject_bi_dir = ROOT / "IMU_Data_Process" / subject / "IMU_BI_CSV"
    if bi_dir.exists():
        imu_dir = bi_dir
    elif subject_bi_dir.exists():
        imu_dir = subject_bi_dir
    else:
        imu_dir = root / "IMU_CSV"
    return imu_dir / f"{jr_stem}_1.csv", group


def _default_split(trial_paths: list[str], seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(trial_paths))
    rng.shuffle(idx)
    shuffled = [trial_paths[i] for i in idx]

    n = len(shuffled)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train_trials": shuffled[:n_train],
        "val_trials": shuffled[n_train : n_train + n_val],
        "test_trials": shuffled[n_train + n_val :],
        "split_type": "trial_random_subject_dependent",
        "seed": seed,
    }


def _loso_split(trial_infos: list[dict], held_out_subject: str, seed: int = 42) -> dict:
    train_pool = [
        t["output_trial_dir"]
        for t in trial_infos
        if t["subject"] != held_out_subject
    ]
    test_trials = [
        t["output_trial_dir"]
        for t in trial_infos
        if t["subject"] == held_out_subject
    ]

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


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "subjects_requested": SUBJECTS,
        "target": TARGET_COL,
        "normalized_target_col": TARGET_COL_NORM_TOTAL_BW,
        "imu_feature_count": len(IMU_COLS),
        "imu_features": IMU_COLS,
        "imu_preprocessing": {
            "acc_unit_standardization": "multiply accelerometer channels by 1000 when median abs < 0.1",
            "lowpass_filter": {
                "type": "zero_phase_butterworth",
                "cutoff_hz": 15.0,
                "order": 4,
                "fs_hz": 100.0,
                "channels": IMU_COLS,
            },
        },
        "sampling_rate_hz": 100,
        "label_resampling": "linear_interpolation_from_JR_time_to_IMU_time",
        "normalization": {
            "type": "total_model_weight",
            "formula": f"{TARGET_COL} / (total_model_mass_kg * 9.81)",
        },
        "subject_masses": {},
        "trials": [],
        "skipped_no_imu": [],
        "skipped_no_mass": [],
        "skipped_bad_trial": [],
    }

    for subject in SUBJECTS:
        exo_mass, noexo_mass, mass_meta = _subject_masses(subject)
        manifest["subject_masses"][subject] = mass_meta

        jr_files = sorted((ROOT / "KJL_GT" / subject).rglob("*_JointReaction_ReactionLoads.csv"))
        for jr_path in jr_files:
            cond = _condition_from_jr(jr_path)
            stem = _jr_stem(jr_path)
            imu_path, source_group = _find_imu_path(subject, cond, stem)
            if not imu_path.exists():
                manifest["skipped_no_imu"].append(
                    {
                        "subject": subject,
                        "condition": cond,
                        "jr_file": str(jr_path.relative_to(ROOT)),
                        "expected_imu_file": str(imu_path.relative_to(ROOT)),
                    }
                )
                continue

            mass_kg = noexo_mass if cond == "NoExo" else exo_mass
            if mass_kg is None:
                manifest["skipped_no_mass"].append(
                    {
                        "subject": subject,
                        "condition": cond,
                        "jr_file": str(jr_path.relative_to(ROOT)),
                    }
                )
                continue

            out_trial_dir = OUT_ROOT / subject / "LG" / cond / "trial_1"
            try:
                info = build_trial(
                    imu_path=imu_path,
                    jr_path=jr_path,
                    out_trial_dir=out_trial_dir,
                    total_model_mass_kg=mass_kg,
                )
            except Exception as exc:
                manifest["skipped_bad_trial"].append(
                    {
                        "subject": subject,
                        "condition": cond,
                        "jr_file": str(jr_path.relative_to(ROOT)),
                        "imu_file": str(imu_path.relative_to(ROOT)),
                        "reason": str(exc),
                    }
                )
                continue

            info["subject"] = subject
            info["condition"] = cond
            info["source_group"] = source_group
            info["output_trial_dir"] = str(out_trial_dir.relative_to(ROOT))
            manifest["trials"].append(info)
            print(f"[OK] {subject} {cond} -> {out_trial_dir}")

    with open(OUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    trial_rel_paths = [t["output_trial_dir"] for t in manifest["trials"]]
    with open(OUT_ROOT / "split_subject_dependent.json", "w", encoding="utf-8") as f:
        json.dump(_default_split(trial_rel_paths), f, ensure_ascii=True, indent=2)

    generated_subjects = sorted({t["subject"] for t in manifest["trials"]})
    for subject in generated_subjects:
        split = _loso_split(manifest["trials"], held_out_subject=subject)
        with open(OUT_ROOT / f"split_subject_independent_loso_{subject}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=True, indent=2)

    print(f"Generated {len(manifest['trials'])} trials.")
    print(f"Skipped (no IMU): {len(manifest['skipped_no_imu'])}")
    print(f"Skipped (no mass): {len(manifest['skipped_no_mass'])}")
    print(f"Skipped (bad trial): {len(manifest['skipped_bad_trial'])}")
    print(f"Manifest: {OUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
