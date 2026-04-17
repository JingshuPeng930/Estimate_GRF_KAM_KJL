import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

GRF_MODULE_DIR = PACKAGE_DIR / "upstream_grf"
KFM_MODULE_DIR = PACKAGE_DIR / "upstream_kfm"
KJL_MODULE_DIR = PACKAGE_DIR

GRF_DATA_ROOT = PACKAGE_DIR / "data" / "grf"
KFM_DATA_ROOT = PACKAGE_DIR / "data" / "kfm"
KJL_DATA_ROOT = PACKAGE_DIR / "data" / "kjl"
RUN_ROOT = PACKAGE_DIR / "runs"


def _data_root(args: argparse.Namespace, task: str) -> Path:
    if task == "grf" and args.grf_data_root:
        return Path(args.grf_data_root).resolve()
    if task == "kfm" and args.kfm_data_root:
        return Path(args.kfm_data_root).resolve()
    if task == "kjl" and args.kjl_data_root:
        return Path(args.kjl_data_root).resolve()
    return {"grf": GRF_DATA_ROOT, "kfm": KFM_DATA_ROOT, "kjl": KJL_DATA_ROOT}[task]


def _run_suffix(args: argparse.Namespace) -> str:
    tag = str(args.output_tag or "").strip()
    return f"_{tag}" if tag else ""


def _run_train(module_dir: Path, module_name: str, cfg_override: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="grfkfm_kjl_si_loso_") as td:
        tmp_dir = Path(td)
        cfg_path = tmp_dir / "cfg.json"
        result_path = tmp_dir / "result.json"
        cfg_path.write_text(json.dumps(cfg_override, indent=2), encoding="utf-8")

        code = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(module_dir)!r})\n"
            f"from {module_name} import train\n"
            "cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))\n"
            "result = train(cfg)\n"
            "json.dump(result, open(sys.argv[2], 'w', encoding='utf-8'), indent=2)\n"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(cfg_path), str(result_path)],
            cwd=REPO_ROOT,
            check=True,
        )
        return json.loads(result_path.read_text(encoding="utf-8"))


def _split_path(data_root: Path, held_out_subject: str) -> Path:
    path = data_root / f"split_subject_independent_loso_{held_out_subject}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing LOSO split: {path}")
    return path


def _train_grf(args: argparse.Namespace, held_out_subject: str) -> Path:
    data_root = _data_root(args, "grf")
    result = _run_train(
        GRF_MODULE_DIR,
        "TCN_Training_GRF_AB03",
        {
            "run_name": f"GRF_SI_LOSO_{held_out_subject}_TCN_IMU{_run_suffix(args)}",
            "seed": args.seed,
            "seeds": [args.seed],
            "dataset_root": str(data_root),
            "split_json": str(_split_path(data_root, held_out_subject)),
            "target_col": "FPR_fz_up_norm_bw",
            "window_size": args.window_size,
            "batch_size": args.grf_batch_size,
            "epochs": args.grf_epochs,
            "output_dir": str(RUN_ROOT / f"upstream_grf{_run_suffix(args)}" / held_out_subject),
        },
    )
    return result


def _train_kfm(args: argparse.Namespace, held_out_subject: str) -> Path:
    data_root = _data_root(args, "kfm")
    result = _run_train(
        KFM_MODULE_DIR,
        "TCN_Training_KFM_AB03",
        {
            "run_name": f"KFM_SI_LOSO_{held_out_subject}_TCN_IMU{_run_suffix(args)}",
            "seed": args.seed,
            "seeds": [args.seed],
            "dataset_root": str(data_root),
            "split_json": str(_split_path(data_root, held_out_subject)),
            "target_col": "kfm_bwbh",
            "window_size": args.window_size,
            "batch_size": args.kfm_batch_size,
            "epochs": args.kfm_epochs,
            "output_dir": str(RUN_ROOT / f"upstream_kfm{_run_suffix(args)}" / held_out_subject),
        },
    )
    return result


def _train_kjl(args: argparse.Namespace, held_out_subject: str, grf_run_dir: Path, kfm_run_dir: Path) -> dict:
    data_root = _data_root(args, "kjl")
    return _run_train(
        KJL_MODULE_DIR,
        "TCN_Training_KJL_AB03_DEP",
        {
            "run_name": f"KJL_SI_LOSO_{held_out_subject}_TCN_DEP_CASCADE_GRFKFM{_run_suffix(args)}",
            "seed": args.seed,
            "seeds": [args.seed],
            "dataset_root": str(data_root),
            "split_json": str(_split_path(data_root, held_out_subject)),
            "target_col": "knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw",
            "output_dir": str(RUN_ROOT / f"kjl_cascade{_run_suffix(args)}" / held_out_subject),
            "window_size": args.window_size,
            "batch_size": args.kjl_batch_size,
            "epochs": args.kjl_epochs,
            "lr": args.kjl_lr,
            "use_cascade_inputs": True,
            "cascade_prediction_mode": args.cascade_prediction_mode,
            "cascade_allow_window_adapter": False,
            "cascade_sources": [
                {
                    "name": "grf",
                    "enabled": True,
                    "run_dir": str(grf_run_dir),
                    "checkpoint_path": "",
                    "output_indices": [0],
                },
                {
                    "name": "kfm",
                    "enabled": True,
                    "run_dir": str(kfm_run_dir),
                    "checkpoint_path": "",
                    "output_indices": [0],
                },
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train subject-independent LOSO GRF + KFM -> KJL cascaded models."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=["AB02_Rajiv", "AB03_Amy", "AB05_Maria"],
        help="Held-out subjects/folds to run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=150)
    parser.add_argument("--grf-epochs", type=int, default=30)
    parser.add_argument("--kfm-epochs", type=int, default=30)
    parser.add_argument("--kjl-epochs", type=int, default=50)
    parser.add_argument("--grf-batch-size", type=int, default=32)
    parser.add_argument("--kfm-batch-size", type=int, default=32)
    parser.add_argument("--kjl-batch-size", type=int, default=32)
    parser.add_argument("--kjl-lr", type=float, default=1e-5)
    parser.add_argument("--grf-data-root", default="", help="Override GRF dataset root.")
    parser.add_argument("--kfm-data-root", default="", help="Override KFM dataset root.")
    parser.add_argument("--kjl-data-root", default="", help="Override KJL dataset root.")
    parser.add_argument("--output-tag", default="", help="Suffix for run names and output folders, e.g. unilateral4imu.")
    parser.add_argument(
        "--cascade-prediction-mode",
        choices=["normalized", "denormalized"],
        default="normalized",
    )
    return parser.parse_args()


def _summary_row(task: str, held_out_subject: str, result: dict) -> dict:
    return {
        "task": task,
        "held_out_subject": held_out_subject,
        "run_name": result.get("run_name", ""),
        "seed": result.get("seed", ""),
        "target_col": result.get("target_col", ""),
        "window_size": result.get("window_size", ""),
        "batch_size": result.get("batch_size", ""),
        "rmse": result.get("final_test_rmse", ""),
        "r2": result.get("final_test_r2", ""),
        "pearson_r": result.get("final_test_pearson_r", ""),
        "nrmse_pct": result.get("final_test_nrmse_pct", ""),
        "run_dir": result.get("out_dir", ""),
    }


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_results = []
    summary_rows = []
    for held_out_subject in args.subjects:
        print("\n" + "=" * 80)
        print(f"[LOSO] Held-out subject: {held_out_subject}")
        print("=" * 80)

        print("[LOSO] Training upstream GRF...")
        grf_result = _train_grf(args, held_out_subject)
        grf_run_dir = Path(grf_result["out_dir"])
        grf_result["held_out_subject"] = held_out_subject
        summary_rows.append(_summary_row("GRF", held_out_subject, grf_result))
        print(f"[LOSO] GRF run: {grf_run_dir}")

        print("[LOSO] Training upstream KFM...")
        kfm_result = _train_kfm(args, held_out_subject)
        kfm_run_dir = Path(kfm_result["out_dir"])
        kfm_result["held_out_subject"] = held_out_subject
        summary_rows.append(_summary_row("KFM", held_out_subject, kfm_result))
        print(f"[LOSO] KFM run: {kfm_run_dir}")

        print("[LOSO] Training KJL cascade...")
        kjl_result = _train_kjl(args, held_out_subject, grf_run_dir, kfm_run_dir)
        kjl_result["held_out_subject"] = held_out_subject
        kjl_result["grf_run_dir"] = str(grf_run_dir)
        kjl_result["kfm_run_dir"] = str(kfm_run_dir)
        summary_rows.append(_summary_row("KJL", held_out_subject, kjl_result))
        all_results.append(kjl_result)
        print(f"[LOSO] KJL run: {kjl_result['out_dir']}")

    summary_path = RUN_ROOT / f"summary_loso{_run_suffix(args)}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    summary_csv_path = RUN_ROOT / f"summary_all_tasks_loso{_run_suffix(args)}.csv"
    _write_summary_csv(summary_rows, summary_csv_path)
    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
