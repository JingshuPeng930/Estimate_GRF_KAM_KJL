import json
from pathlib import Path
import argparse
import numpy as np


SUBJECT_TRIAL_GLOB = "AB03_Amy/LG/*/trial_1"


def _rel(p: Path) -> str:
    return str(p.as_posix())


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    print(f"[WRITE] {path}")


def _split_random(trials, seed, n_train, n_val, n_test):
    assert n_train + n_val + n_test == len(trials)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(trials))
    rng.shuffle(idx)
    s = [trials[i] for i in idx]
    return s[:n_train], s[n_train:n_train + n_val], s[n_train + n_val:]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed AB03 subject-dependent train/val/test splits.")
    parser.add_argument("--dataset-root", default="data/kjl_ab03_dep")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    trial_dirs = sorted(dataset_root.glob(SUBJECT_TRIAL_GLOB))
    if not trial_dirs:
        raise SystemExit(f"No trials found under {dataset_root}")

    noexo = [p for p in trial_dirs if "/NoExo/" in _rel(p)]
    exo_like = [p for p in trial_dirs if "/NoExo/" not in _rel(p)]

    print(f"Total trials: {len(trial_dirs)}")
    print(f"Exo/NoAssi trials: {len(exo_like)}")
    print(f"NoExo trials: {len(noexo)}")

    out_dir = dataset_root / "splits"

    # 1) Exo + NoAssi only (legacy-compatible baseline)
    if len(exo_like) >= 26:
        tr, va, te = _split_random(exo_like, seed=42, n_train=18, n_val=4, n_test=len(exo_like) - 22)
        _write_json(
            out_dir / "split_exo_noassi_only_seed42.json",
            {
                "description": "AB03 LG Exo+NoAssi only (NoExo excluded), random fixed split seed42.",
                "train_trials": [_rel(p) for p in tr],
                "val_trials": [_rel(p) for p in va],
                "test_trials": [_rel(p) for p in te],
            },
        )

    # If NoExo exists, generate the two comparison splits.
    if len(noexo) == 1 and len(exo_like) >= 26:
        noexo_trial = noexo[0]

        # 2) Mixed 27-trial split with NoExo forced into train.
        tr_exo, va, te = _split_random(exo_like, seed=42, n_train=18, n_val=4, n_test=len(exo_like) - 22)
        tr = [noexo_trial] + tr_exo
        _write_json(
            out_dir / "split_mixed27_noexo_in_train_seed42.json",
            {
                "description": "27-trial mixed split with NoExo forced into train; exo/noassi split fixed by seed42.",
                "train_trials": [_rel(p) for p in tr],
                "val_trials": [_rel(p) for p in va],
                "test_trials": [_rel(p) for p in te],
            },
        )

        # 3) Leave-NoExo-out test split.
        tr, va, _ = _split_random(exo_like, seed=42, n_train=22, n_val=4, n_test=len(exo_like) - 26)
        _write_json(
            out_dir / "split_leave_noexo_out_seed42.json",
            {
                "description": "Train/val on Exo+NoAssi only; test on NoExo only.",
                "train_trials": [_rel(p) for p in tr],
                "val_trials": [_rel(p) for p in va],
                "test_trials": [_rel(noexo_trial)],
            },
        )

        # 4) NoExo temporal split: first 70% in train, last 30% in test.
        # Trials that appear in both train_trials and test_trials are automatically
        # temporally split at runtime by build_kjl_ab03_dataloaders.
        tr_exo, va, te = _split_random(exo_like, seed=42, n_train=18, n_val=4, n_test=len(exo_like) - 22)
        _write_json(
            out_dir / "split_noexo_temporal_seed42.json",
            {
                "description": (
                    "NoExo temporal split: first 70% of NoExo trial in train, "
                    "last 30% in test. Exo+NoAssi split fixed by seed42 (18/4/4). "
                    "Trials listed in both train_trials and test_trials are "
                    "temporally split at runtime by build_kjl_ab03_dataloaders."
                ),
                "train_trials": [_rel(noexo_trial)] + [_rel(p) for p in tr_exo],
                "val_trials": [_rel(p) for p in va],
                "test_trials": [_rel(noexo_trial)] + [_rel(p) for p in te],
                "noexo_split_ratio": 0.7,
            },
        )
        # 5) Same as split #4, but swap 20p250ms into train and 20p200ms into test.
        #    Motivation: 20p250ms sits at the biomechanical boundary between the
        #    "standard loading" and "high-assistance shifted loading" regimes; training
        #    on it lets the model learn this boundary, while 20p200ms (neighbouring
        #    condition) becomes the held-out test case for that torque level.
        base_path = out_dir / "split_noexo_temporal_seed42.json"
        if base_path.exists():
            base = json.loads(base_path.read_text())
            cond_20p250 = next((p for p in base["test_trials"] if "20p250ms" in p), None)
            cond_20p200 = next((p for p in base["train_trials"] if "20p200ms" in p), None)
            if cond_20p250 and cond_20p200:
                new_train = [p for p in base["train_trials"] if "20p200ms" not in p] + [cond_20p250]
                new_test  = [p for p in base["test_trials"]  if "20p250ms" not in p] + [cond_20p200]
                _write_json(
                    out_dir / "split_noexo_temporal_20p250ms_in_train.json",
                    {
                        "description": (
                            "Same as split_noexo_temporal_seed42 but 20p250ms moved to train "
                            "and 20p200ms moved to test. NoExo temporal split ratio 0.7."
                        ),
                        "train_trials": new_train,
                        "val_trials": base["val_trials"],
                        "test_trials": new_test,
                        "noexo_split_ratio": base.get("noexo_split_ratio", 0.7),
                    },
                )
    else:
        print("NoExo-specific splits not created (need exactly one NoExo trial and 26 exo-like trials).")


if __name__ == "__main__":
    main()
