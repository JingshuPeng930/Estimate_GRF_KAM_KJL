import argparse
import csv
import json
from pathlib import Path

import numpy as np

from TCN_Training_GRF_AB03 import CONFIG as BASE_CONFIG, train as train_single_run


CONFIG = BASE_CONFIG.copy()
CONFIG.update(
    {
        "run_name": "GRF_SI_LOSO_TCN_IMU",
        "dataset_root": "data_grf_all_subjects_imu",
        "output_dir": "runs_grf_si_loso",
        "split_json": None,
        "seed": 42,
        "seeds": [42],
        "subjects": ["AB03_Amy", "AB05_Maria", "AB08_Adrian"],
        "held_out_subject": "AB08_Adrian",
        "val_ratio": 0.15,
    }
)


def _resolve_dataset_root_path(dataset_root: str) -> Path:
    p = Path(dataset_root)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / p,
        Path(__file__).resolve().parent / p,
        repo_root / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _collect_trial_rel_paths(dataset_root: Path, subjects: list[str]) -> list[str]:
    wanted = set(subjects)
    out = []
    for p in sorted(dataset_root.glob("AB*/LG/*/trial_1")):
        rel = p.relative_to(dataset_root)
        if not rel.parts:
            continue
        if rel.parts[0] not in wanted:
            continue
        out.append(str(rel))
    return out


def _subject_from_trial_rel(trial_rel: str) -> str:
    p = Path(trial_rel)
    if len(p.parts) < 4:
        raise ValueError(f"Unexpected trial path format: {trial_rel}")
    return p.parts[0]


def _build_single_loso_split(
    trial_rel_paths: list[str],
    held_out_subject: str,
    seed: int = 42,
    val_ratio: float = 0.15,
) -> dict:
    test_trials = sorted([t for t in trial_rel_paths if _subject_from_trial_rel(t) == held_out_subject])
    train_pool = sorted([t for t in trial_rel_paths if _subject_from_trial_rel(t) != held_out_subject])
    if not test_trials:
        raise ValueError(f"No test trials found for held-out subject {held_out_subject}")
    if len(train_pool) < 2:
        raise ValueError(f"Need >=2 train-pool trials, got {len(train_pool)}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_pool))
    rng.shuffle(idx)
    shuffled = [train_pool[i] for i in idx]

    n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
    n_val = min(n_val, len(shuffled) - 1)
    val_trials = sorted(shuffled[:n_val])
    train_trials = sorted(shuffled[n_val:])

    return {
        "split_type": "subject_independent_loso_single",
        "held_out_subject": held_out_subject,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "train_trials": train_trials,
        "val_trials": val_trials,
        "test_trials": test_trials,
    }


def _train_one_fold(cfg: dict, held_out_subject: str) -> dict:
    dataset_root = _resolve_dataset_root_path(str(cfg["dataset_root"]))
    subjects = [str(s) for s in cfg.get("subjects", [])]
    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.15))

    if held_out_subject not in subjects:
        raise ValueError(f"held_out_subject `{held_out_subject}` must be in subjects={subjects}")

    trial_rel_paths = _collect_trial_rel_paths(dataset_root, subjects=subjects)
    if not trial_rel_paths:
        raise FileNotFoundError(f"No trials found under {dataset_root} for subjects={subjects}")

    split = _build_single_loso_split(
        trial_rel_paths=trial_rel_paths,
        held_out_subject=held_out_subject,
        seed=seed,
        val_ratio=val_ratio,
    )
    split_path = dataset_root / f"split_subject_independent_loso_{held_out_subject}_single.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=True, indent=2)

    print(
        "Prepared LOSO split: "
        f"held_out={held_out_subject}, "
        f"train/val/test={len(split['train_trials'])}/{len(split['val_trials'])}/{len(split['test_trials'])}"
    )

    run_name = f"GRF_SI_LOSO_{held_out_subject}"
    cfg_for_base = {
        "run_name": run_name,
        "dataset_root": str(dataset_root),
        "split_json": str(split_path),
        "seed": seed,
        "output_dir": cfg["output_dir"],
    }

    for k in BASE_CONFIG.keys():
        if k in cfg and k not in cfg_for_base:
            cfg_for_base[k] = cfg[k]

    result = train_single_run(cfg_for_base)
    result["held_out_subject"] = held_out_subject
    result["split_json"] = str(split_path)
    result["subjects"] = json.dumps(subjects, ensure_ascii=True)
    return result


def _append_summary_row(summary_csv: Path, result: dict) -> None:
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def run_one_loso_model(cfg_override=None):
    cfg = CONFIG.copy()
    if cfg_override:
        cfg.update(cfg_override)

    held_out_subject = str(cfg["held_out_subject"])
    result = _train_one_fold(cfg=cfg, held_out_subject=held_out_subject)

    root_out = Path(cfg["output_dir"])
    root_out.mkdir(parents=True, exist_ok=True)
    summary_csv = root_out / "summary_single_loso.csv"
    _append_summary_row(summary_csv=summary_csv, result=result)

    print(f"Saved LOSO summary row -> {summary_csv}")
    return result


def run_all_loso_models(cfg_override=None):
    cfg = CONFIG.copy()
    if cfg_override:
        cfg.update(cfg_override)

    subjects = [str(s) for s in cfg.get("subjects", [])]
    if len(subjects) < 2:
        raise ValueError(f"Need at least 2 subjects for LOSO, got {subjects}")

    root_out = Path(cfg["output_dir"])
    root_out.mkdir(parents=True, exist_ok=True)
    summary_csv = root_out / "summary_all_loso.csv"

    results = []
    for held_out_subject in subjects:
        print("\n" + "=" * 90)
        print(f"Starting LOSO fold: hold out {held_out_subject}")
        print("=" * 90)
        result = _train_one_fold(cfg=cfg, held_out_subject=held_out_subject)
        _append_summary_row(summary_csv=summary_csv, result=result)
        results.append(result)

    if results:
        rmse = np.array([r["final_test_rmse"] for r in results], dtype=float)
        r2 = np.array([r["final_test_r2"] for r in results], dtype=float)
        pr = np.array([r["final_test_pearson_r"] for r in results], dtype=float)
        nrmse = np.array([r["final_test_nrmse_pct"] for r in results], dtype=float)
        print("\n" + "=" * 90)
        print("All-LOSO summary")
        print(f"Subjects (held-out each fold): {subjects}")
        print(f"RMSE mean+-std: {np.nanmean(rmse):.4f} +- {np.nanstd(rmse):.4f}")
        print(f"R2   mean+-std: {np.nanmean(r2):.4f} +- {np.nanstd(r2):.4f}")
        print(f"r    mean+-std: {np.nanmean(pr):.4f} +- {np.nanstd(pr):.4f}")
        print(f"nRMSE mean+-std: {np.nanmean(nrmse):.4f}% +- {np.nanstd(nrmse):.4f}%")
        print(f"Summary CSV: {summary_csv}")
        print("=" * 90)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train subject-independent GRF TCN with LOSO splits.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all LOSO folds (hold out each subject once).",
    )
    parser.add_argument(
        "--held-out-subject",
        default=None,
        help="Run one LOSO fold with this held-out subject (e.g., AB08_Adrian).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.all:
        run_all_loso_models()
    elif args.held_out_subject:
        run_one_loso_model({"held_out_subject": str(args.held_out_subject)})
    else:
        # Keep backward compatibility: default to config-specified single fold.
        run_one_loso_model()
