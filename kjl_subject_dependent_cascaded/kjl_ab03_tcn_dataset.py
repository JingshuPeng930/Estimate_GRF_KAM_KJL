import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.signal as spsignal
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


TARGET_COL = "knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw"

# Low-pass filter applied to the KJL label before training/evaluation.
# OpenSim inverse dynamics can produce numerical spikes (max sample-to-sample
# jump observed: 1.477 BW/10ms ≈ 147 BW/s, far exceeding physiological rates).
# A 15 Hz Butterworth removes these artefacts while preserving all gait-cycle
# harmonics (walking fundamental ~1 Hz; meaningful KJL content < 10 Hz).
# Set to None to disable filtering.
LABEL_FILTER_CUTOFF_HZ: float | None = 15.0
LABEL_FILTER_ORDER: int = 4
LABEL_FS_HZ: float = 100.0
_COND_RE = re.compile(r"^(?P<torque>\d+)p(?P<delay>\d+)ms$")


def _filter_label(y: np.ndarray) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter to the label column."""
    if LABEL_FILTER_CUTOFF_HZ is None or len(y) < 15:
        return y
    nyq = 0.5 * LABEL_FS_HZ
    b, a = spsignal.butter(LABEL_FILTER_ORDER, LABEL_FILTER_CUTOFF_HZ / nyq, btype="low")
    return spsignal.filtfilt(b, a, y.ravel()).reshape(-1, 1).astype(np.float32)


def _load_trial_arrays(
    trial_dir: Path,
    target_col: str = TARGET_COL,
    row_start: int = 0,
    row_end: int = None,
    exclude_feature_cols: Sequence[str] | None = None,
):
    input_path = trial_dir / "Input" / "imu.csv"
    label_path = trial_dir / "Label" / "kjl_fy.csv"

    input_df = pd.read_csv(input_path)
    label_df = pd.read_csv(label_path)

    exclude_set = {str(c) for c in (exclude_feature_cols or [])}
    feature_cols = [
        c
        for c in input_df.columns
        if c not in ("sample_idx", "time_imu") and c not in exclude_set
    ]
    if not feature_cols:
        raise ValueError(f"No input features left after exclusion for {input_path}")
    if target_col not in label_df.columns:
        raise ValueError(f"Missing target column `{target_col}` in {label_path}")

    x = input_df[feature_cols].to_numpy(dtype=np.float32)
    y = label_df[[target_col]].to_numpy(dtype=np.float32)

    # Filter BEFORE slicing so edge effects from filtfilt don't contaminate
    # the temporal split boundary.
    y = _filter_label(y)

    n = min(len(x), len(y))
    row_end_actual = n if row_end is None else min(row_end, n)
    return x[row_start:row_end_actual], y[row_start:row_end_actual], feature_cols


class WindowedTrialDataset(Dataset):
    def __init__(
        self,
        trial_dirs: Sequence[Path],
        window_size: int,
        input_mean=None,
        input_std=None,
        label_mean=None,
        label_std=None,
        fit_norm: bool = False,
        target_col: str = TARGET_COL,
        row_ranges: dict = None,
        exclude_feature_cols: Sequence[str] | None = None,
    ):
        self.trial_dirs = [Path(p) for p in trial_dirs]
        self.window_size = int(window_size)
        self.trials_x: List[np.ndarray] = []
        self.trials_y: List[np.ndarray] = []
        self.kept_trial_dirs: List[Path] = []
        self.index_map = []  # (trial_idx, start_idx)
        self.feature_cols = None
        self.target_col = target_col
        self.row_ranges = row_ranges or {}  # {str(trial_dir): (row_start, row_end)}
        self.exclude_feature_cols = tuple(exclude_feature_cols or [])
        self.trial_delay_ms: List[int] = []

        for tdir in self.trial_dirs:
            rng = self.row_ranges.get(str(tdir), (0, None))
            x, y, cols = _load_trial_arrays(
                tdir,
                target_col=self.target_col,
                row_start=rng[0],
                row_end=rng[1],
                exclude_feature_cols=self.exclude_feature_cols,
            )
            if len(x) < self.window_size:
                continue
            if self.feature_cols is None:
                self.feature_cols = cols
            elif cols != self.feature_cols:
                raise ValueError(
                    f"Inconsistent feature columns across trials. "
                    f"Expected {self.feature_cols}, got {cols} in {tdir}"
                )
            self.trials_x.append(x)
            self.trials_y.append(y)
            self.kept_trial_dirs.append(tdir)
            cond = tdir.parts[-2]
            m = _COND_RE.match(cond)
            self.trial_delay_ms.append(int(m.group("delay")) if m else -1)
            self.index_map.extend((len(self.trials_x) - 1, i) for i in range(len(x) - self.window_size + 1))

        if not self.trials_x:
            raise ValueError("No valid trials found for dataset.")

        if fit_norm:
            all_x = np.concatenate(self.trials_x, axis=0)
            all_y = np.concatenate(self.trials_y, axis=0)
            self.input_mean = all_x.mean(axis=0).astype(np.float32)
            self.input_std = (all_x.std(axis=0) + 1e-8).astype(np.float32)
            self.label_mean = all_y.mean(axis=0).astype(np.float32)
            self.label_std = (all_y.std(axis=0) + 1e-8).astype(np.float32)
        else:
            self.input_mean = np.asarray(input_mean, dtype=np.float32)
            self.input_std = np.asarray(input_std, dtype=np.float32)
            self.label_mean = np.asarray(label_mean, dtype=np.float32)
            self.label_std = np.asarray(label_std, dtype=np.float32)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        trial_idx, start = self.index_map[idx]
        x = self.trials_x[trial_idx][start:start + self.window_size]
        y = self.trials_y[trial_idx][start + self.window_size - 1]

        x = (x - self.input_mean) / self.input_std
        y = (y - self.label_mean) / self.label_std

        x_tensor = torch.from_numpy(x.T.copy()).float()  # (C, T)
        y_tensor = torch.from_numpy(y.copy()).float()     # (1,)
        trial_idx_t = torch.tensor(trial_idx, dtype=torch.long)
        start_t = torch.tensor(start, dtype=torch.long)
        delay_ms_t = torch.tensor(self.trial_delay_ms[trial_idx], dtype=torch.long)
        return x_tensor, y_tensor, trial_idx_t, start_t, delay_ms_t

    def get_sample_weights(self) -> np.ndarray:
        """Per-window weights so each trial contributes equally during sampling.

        Weight of window i = 1 / (number of windows in trial i).
        This counteracts imbalance between large exo trials and the small NoExo segment.
        """
        trial_window_counts = np.zeros(len(self.trials_x), dtype=float)
        for trial_idx, _ in self.index_map:
            trial_window_counts[trial_idx] += 1.0
        return np.array(
            [1.0 / trial_window_counts[trial_idx] for trial_idx, _ in self.index_map],
            dtype=np.float32,
        )

    def get_sample_weights_with_condition_bias(
        self,
        delay_weight_map: dict | None = None,
        noexo_weight: float = 1.0,
        noassi_weight: float = 1.0,
        default_weight: float = 1.0,
    ) -> np.ndarray:
        """Per-window weights with optional condition-aware multipliers.

        This keeps trial-balanced sampling as the base, then multiplies windows from
        selected conditions (e.g. high-delay trials) to bias training toward harder
        regimes without changing deployment inputs.
        """
        base = self.get_sample_weights().astype(np.float32)
        if not delay_weight_map and noexo_weight == 1.0 and noassi_weight == 1.0 and default_weight == 1.0:
            return base

        # Normalize keys once so JSON-loaded config {"200": 2.0} works.
        norm_delay_map = {}
        for k, v in (delay_weight_map or {}).items():
            try:
                norm_delay_map[int(k)] = float(v)
            except Exception:
                continue

        trial_multipliers = np.ones(len(self.trials_x), dtype=np.float32) * float(default_weight)
        for trial_idx, tdir in enumerate(self.kept_trial_dirs):
            cond = tdir.parts[-2]  # .../LG/<cond>/trial_1
            if cond == "NoExo":
                trial_multipliers[trial_idx] = float(noexo_weight)
                continue
            if cond == "NoAssi":
                trial_multipliers[trial_idx] = float(noassi_weight)
                continue
            m = _COND_RE.match(cond)
            if m:
                delay_ms = int(m.group("delay"))
                trial_multipliers[trial_idx] = float(norm_delay_map.get(delay_ms, default_weight))

        out = np.array(
            [base[i] * trial_multipliers[trial_idx] for i, (trial_idx, _) in enumerate(self.index_map)],
            dtype=np.float32,
        )
        # Keep average weight ~1 for sampler stability.
        mean_w = float(np.mean(out))
        if mean_w > 0:
            out = out / mean_w
        return out


@dataclass
class KJLDataBundle:
    train_dataset: WindowedTrialDataset
    val_dataset: Optional[WindowedTrialDataset]
    test_dataset: WindowedTrialDataset
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: DataLoader
    train_trials: List[str]
    val_trials: List[str]
    test_trials: List[str]

    @property
    def input_mean(self):
        return self.train_dataset.input_mean

    @property
    def input_std(self):
        return self.train_dataset.input_std

    @property
    def label_mean(self):
        return self.train_dataset.label_mean

    @property
    def label_std(self):
        return self.train_dataset.label_std

    @property
    def input_size(self):
        return len(self.train_dataset.feature_cols)


def _split_trials(
    trial_dirs: List[Path],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    if len(trial_dirs) < 3:
        raise ValueError("Need at least 3 trials to create train/val/test split.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(trial_dirs))
    rng.shuffle(idx)
    shuffled = [trial_dirs[i] for i in idx]

    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_trials = shuffled[:n_train]
    val_trials = shuffled[n_train:n_train + n_val]
    test_trials = shuffled[n_train + n_val:]
    return train_trials, val_trials, test_trials


def build_kjl_ab03_dataloaders(
    dataset_root: str,
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    split_json: str = None,
    target_col: str = TARGET_COL,
    noexo_split_ratio: float = 0.7,
    trial_balanced_sampling: bool = True,
    delay_weight_map: dict | None = None,
    noexo_train_weight: float = 1.0,
    noassi_train_weight: float = 1.0,
    default_train_weight: float = 1.0,
    sequential_train_batches: bool = False,
    exclude_feature_cols: Sequence[str] | None = None,
):
    root = Path(dataset_root)
    if split_json:
        split_path = Path(split_json)
        split = json.loads(split_path.read_text())
        train_trials = [Path(p) for p in split["train_trials"]]
        val_trials = [Path(p) for p in split["val_trials"]]
        test_trials = [Path(p) for p in split["test_trials"]]
        noexo_split_ratio = split.get("noexo_split_ratio", noexo_split_ratio)
    else:
        noexo_dirs = sorted(root.glob("AB03_Amy/LG/NoExo/trial_1"))
        exo_dirs = [p for p in sorted(root.glob("AB03_Amy/LG/*/trial_1")) if "NoExo" not in p.parts]
        train_trials, val_trials, test_trials = _split_trials(exo_dirs, seed=seed)
        for p in noexo_dirs:
            train_trials.append(p)
            test_trials.append(p)

    # Trials appearing in both train and test get a temporal split:
    # first noexo_split_ratio fraction → train, remainder → test.
    train_strs = {str(p) for p in train_trials}
    test_strs = {str(p) for p in test_trials}
    overlap_strs = train_strs & test_strs
    train_row_ranges: dict = {}
    test_row_ranges: dict = {}
    for p_str in overlap_strs:
        x_tmp, _, _ = _load_trial_arrays(Path(p_str), target_col=target_col)
        split_row = int(len(x_tmp) * noexo_split_ratio)
        train_row_ranges[p_str] = (0, split_row)
        test_row_ranges[p_str] = (split_row, None)

    train_ds = WindowedTrialDataset(
        train_trials, window_size=window_size, fit_norm=True,
        target_col=target_col, row_ranges=train_row_ranges,
        exclude_feature_cols=exclude_feature_cols,
    )
    if val_trials:
        val_ds = WindowedTrialDataset(
            val_trials,
            window_size=window_size,
            input_mean=train_ds.input_mean,
            input_std=train_ds.input_std,
            label_mean=train_ds.label_mean,
            label_std=train_ds.label_std,
            target_col=target_col,
            exclude_feature_cols=exclude_feature_cols,
        )
    else:
        val_ds = None
    test_ds = WindowedTrialDataset(
        test_trials,
        window_size=window_size,
        input_mean=train_ds.input_mean,
        input_std=train_ds.input_std,
        label_mean=train_ds.label_mean,
        label_std=train_ds.label_std,
        target_col=target_col,
        row_ranges=test_row_ranges,
        exclude_feature_cols=exclude_feature_cols,
    )

    if sequential_train_batches:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif trial_balanced_sampling:
        sample_weights = torch.from_numpy(
            train_ds.get_sample_weights_with_condition_bias(
                delay_weight_map=delay_weight_map,
                noexo_weight=noexo_train_weight,
                noassi_weight=noassi_train_weight,
                default_weight=default_train_weight,
            )
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return KJLDataBundle(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_trials=[str(p) for p in train_trials],
        val_trials=[str(p) for p in val_trials],
        test_trials=[str(p) for p in test_trials],
    )
