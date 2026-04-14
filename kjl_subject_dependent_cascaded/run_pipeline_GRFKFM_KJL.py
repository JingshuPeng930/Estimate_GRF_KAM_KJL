import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_GRF_RUN_DIR = (
    THIS_DIR
    / "runs"
    / "upstream_grf"
    / "GRF_AB03_Amy_TCN_IMU_seed42_FPR_fz_upNBW_w150_bs256_do0p15_huber_b10.0_lpf15p0_ch32x32x32x32"
)
DEFAULT_KFM_RUN_DIR = (
    THIS_DIR
    / "runs"
    / "upstream_kfm"
    / "KFM_AB03_Amy_TCN_IMU_seed42_kfm_bwbh_w150_bs32_do0p15_huber_b10.0_lpf15p0_ch32x32x32x32"
)

GRF_MODULE_DIR = THIS_DIR / "upstream_grf"
KFM_MODULE_DIR = THIS_DIR / "upstream_kfm"
KJL_MODULE_DIR = THIS_DIR

GRF_DATASET_ROOT = THIS_DIR / "data" / "grf_ab03_imu"
KFM_DATASET_ROOT = THIS_DIR / "data" / "kfm_ab03_id"
KJL_DATASET_ROOT = THIS_DIR / "data" / "kjl_ab03_dep"

REQUIRED_RUN_FILES = [
    "train_config.json",
    "input_mean.npy",
    "input_std.npy",
    "label_mean.npy",
    "label_std.npy",
]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_checkpoint(run_dir: Path) -> Path | None:
    cfg_path = run_dir / "train_config.json"
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        run_name = str(cfg.get("run_name", ""))
        if run_name:
            preferred = run_dir / f"{run_name}.pt"
            if preferred.exists():
                return preferred

    non_epoch = sorted(p for p in run_dir.glob("*.pt") if "_epoch_" not in p.name)
    if non_epoch:
        return non_epoch[0]

    checkpoints = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def _is_complete_run(run_dir: Path, window_size: int | None = None) -> bool:
    if not run_dir.exists():
        return False
    if any(not (run_dir / fname).exists() for fname in REQUIRED_RUN_FILES):
        return False
    if _run_checkpoint(run_dir) is None:
        return False
    if window_size is not None:
        cfg = _load_json(run_dir / "train_config.json")
        if int(cfg.get("window_size", -1)) != int(window_size):
            return False
    return True


def _run_train(module_dir: Path, module_name: str, cfg_override: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="grfkfm_kjl_pipeline_") as td:
        tmp_dir = Path(td)
        cfg_path = tmp_dir / "cfg.json"
        result_path = tmp_dir / "result.json"
        cfg_path.write_text(json.dumps(cfg_override, indent=2), encoding="utf-8")

        code = (
            "import json, sys\n"
            "from pathlib import Path\n"
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
        return _load_json(result_path)


def _prepare_grf(args) -> Path:
    if not args.retrain_grf and _is_complete_run(DEFAULT_GRF_RUN_DIR, window_size=args.window_size):
        print(f"[Pipeline] Using existing GRF run: {DEFAULT_GRF_RUN_DIR}")
        return DEFAULT_GRF_RUN_DIR

    print("[Pipeline] Training GRF upstream model...")
    result = _run_train(
        GRF_MODULE_DIR,
        "TCN_Training_GRF_AB03",
        {
            "seed": args.seed,
            "seeds": [args.seed],
            "window_size": args.window_size,
            "batch_size": args.grf_batch_size,
            "epochs": args.grf_epochs,
            "dataset_root": str(GRF_DATASET_ROOT),
            "output_dir": str(THIS_DIR / "runs" / "upstream_grf"),
        },
    )
    run_dir = Path(result["out_dir"])
    if not _is_complete_run(run_dir, window_size=args.window_size):
        raise RuntimeError(f"GRF training finished but run is incomplete: {run_dir}")
    return run_dir


def _prepare_kfm(args) -> Path:
    if not args.retrain_kfm and _is_complete_run(DEFAULT_KFM_RUN_DIR, window_size=args.window_size):
        print(f"[Pipeline] Using existing KFM run: {DEFAULT_KFM_RUN_DIR}")
        return DEFAULT_KFM_RUN_DIR

    print("[Pipeline] Training KFM upstream model...")
    result = _run_train(
        KFM_MODULE_DIR,
        "TCN_Training_KFM_AB03",
        {
            "seed": args.seed,
            "seeds": [args.seed],
            "window_size": args.window_size,
            "batch_size": args.kfm_batch_size,
            "epochs": args.kfm_epochs,
            "target_col": "kfm_bwbh",
            "dataset_root": str(KFM_DATASET_ROOT),
            "output_dir": str(THIS_DIR / "runs" / "upstream_kfm"),
        },
    )
    run_dir = Path(result["out_dir"])
    if not _is_complete_run(run_dir, window_size=args.window_size):
        raise RuntimeError(f"KFM training finished but run is incomplete: {run_dir}")
    return run_dir


def _run_kjl(args, grf_run_dir: Path, kfm_run_dir: Path) -> dict:
    print("[Pipeline] Training KJL cascaded model...")
    return _run_train(
        KJL_MODULE_DIR,
        "TCN_Training_KJL_AB03_DEP",
        {
            "run_name": "KJL_AB03_Amy_TCN_DEP_CASCADE_GRFKFM",
            "seed": args.seed,
            "seeds": [args.seed],
            "dataset_root": str(KJL_DATASET_ROOT),
            "output_dir": str(THIS_DIR / "runs" / "kjl_cascade"),
            "window_size": args.window_size,
            "batch_size": args.kjl_batch_size,
            "epochs": args.kjl_epochs,
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
        description="Run the staged GRF + KFM -> KJL cascaded pipeline for AB03 Amy."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=150)
    parser.add_argument("--retrain-grf", action="store_true")
    parser.add_argument("--retrain-kfm", action="store_true")
    parser.add_argument("--skip-kjl", action="store_true", help="Only prepare/check upstream GRF and KFM.")
    parser.add_argument("--grf-epochs", type=int, default=30)
    parser.add_argument("--kfm-epochs", type=int, default=30)
    parser.add_argument("--kjl-epochs", type=int, default=50)
    parser.add_argument("--grf-batch-size", type=int, default=32)
    parser.add_argument("--kfm-batch-size", type=int, default=32)
    parser.add_argument("--kjl-batch-size", type=int, default=32)
    parser.add_argument(
        "--cascade-prediction-mode",
        choices=["normalized", "denormalized"],
        default="normalized",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grf_run_dir = _prepare_grf(args)
    kfm_run_dir = _prepare_kfm(args)

    print("[Pipeline] Upstream sources ready:")
    print(f"  GRF: {grf_run_dir}")
    print(f"  KFM: {kfm_run_dir}")

    if args.skip_kjl:
        print("[Pipeline] --skip-kjl set; stopping before KJL training.")
        return

    result = _run_kjl(args, grf_run_dir, kfm_run_dir)
    print("[Pipeline] KJL cascade finished.")
    print(f"[Pipeline] Output: {result['out_dir']}")


if __name__ == "__main__":
    main()
