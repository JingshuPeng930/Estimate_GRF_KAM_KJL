#!/usr/bin/env python3
"""Create presentation figures for the SI LOSO KJL cascade results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
KJL_DATA_ROOT = PACKAGE_DIR / "data" / "kjl"
KFM_DATA_ROOT = PACKAGE_DIR / "data" / "kfm"
GRF_DATA_ROOT = PACKAGE_DIR / "data" / "grf"
DEFAULT_RUN_ROOT = PACKAGE_DIR / "runs"
SUBJECTS = ["AB02_Rajiv", "AB03_Amy", "AB05_Maria"]
SUBJECT_LABELS = {
    "AB02_Rajiv": "Rajiv",
    "AB03_Amy": "Amy",
    "AB05_Maria": "Maria",
}
SUBJECT_COLORS = {
    "AB02_Rajiv": "#1f77b4",
    "AB03_Amy": "#d62728",
    "AB05_Maria": "#2ca02c",
}
GT_COLOR = "#222222"
PRED_COLOR = "#0072B2"
TASK_CONFIG = {
    "GRF": {
        "data_root": GRF_DATA_ROOT,
        "label_file": "grf.csv",
        "time_col": "time_force",
        "ylabel": "Vertical GRF / BW",
        "title": "GRF",
        "plot_sign": 1.0,
    },
    "KFM": {
        "data_root": KFM_DATA_ROOT,
        "label_file": "kfm.csv",
        "time_col": "time_id",
        "ylabel": "KFM / (BW*BH)",
        "title": "KFM",
        "plot_sign": 1.0,
    },
    "KJL": {
        "data_root": KJL_DATA_ROOT,
        "label_file": "kjl_fy.csv",
        "time_col": "time_jr",
        "ylabel": "-KJL / BW",
        "title": "KJL",
        "plot_sign": -1.0,
    },
}


def _condition_sort_key(condition: str) -> tuple[int, int, str]:
    if condition == "NoExo":
        return (-2, -2, condition)
    if condition == "NoAssi":
        return (-1, -1, condition)
    if "p" in condition and "ms" in condition:
        left, right = condition.replace("ms", "").split("p", 1)
        return (int(left), int(right), condition)
    return (999, 999, condition)


def _load_summary(run_root: Path) -> pd.DataFrame:
    csv_path = run_root / "summary_all_tasks_loso.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    summary = pd.read_csv(csv_path)
    if "run_dir" in summary:
        def _resolve_run_dir(path_value: object) -> str | float:
            if pd.isna(path_value):
                return np.nan
            path = str(path_value)
            return str((REPO_ROOT / path).resolve()) if not path.startswith("/") else path

        summary["run_dir"] = summary["run_dir"].map(_resolve_run_dir)
    return summary


def _task_run_dir(summary: pd.DataFrame, task: str, subject: str) -> Path:
    rows = summary[(summary["task"] == task) & (summary["held_out_subject"] == subject)]
    if rows.empty:
        raise FileNotFoundError(f"No {task} run found in summary for {subject}")
    return Path(rows.iloc[0]["run_dir"])


def _kjl_run_dir(summary: pd.DataFrame, subject: str) -> Path:
    return _task_run_dir(summary, "KJL", subject)


def _per_trial_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "per_trial_test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing per-trial metrics: {metrics_path}")
    return pd.DataFrame(json.loads(metrics_path.read_text()))


def _load_prediction(run_dir: Path, condition: str) -> tuple[np.ndarray, np.ndarray]:
    pred_path = run_dir / f"preds_{condition}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    data = np.load(pred_path)
    return np.asarray(data["y_true"]).reshape(-1), np.asarray(data["y_pred"]).reshape(-1)


def _load_prediction_time(task: str, subject: str, condition: str, pred_len: int) -> np.ndarray:
    config = TASK_CONFIG[task]
    label_path = config["data_root"] / subject / "LG" / condition / "trial_1" / "Label" / config["label_file"]
    if not label_path.exists():
        return np.arange(pred_len) * 0.01
    label = pd.read_csv(label_path)
    time_col = config["time_col"] if config["time_col"] in label else label.columns[1]
    times = label[time_col].to_numpy(dtype=float)
    offset = max(0, len(times) - pred_len)
    aligned = times[offset : offset + pred_len]
    if len(aligned) == pred_len:
        return aligned
    return np.arange(pred_len) * 0.01


def _load_grf(subject: str, condition: str) -> tuple[np.ndarray, np.ndarray] | None:
    grf_path = GRF_DATA_ROOT / subject / "LG" / condition / "trial_1" / "Label" / "grf.csv"
    if not grf_path.exists():
        return None
    grf = pd.read_csv(grf_path)
    if "time_force" not in grf or "FPR_fz_up_norm_bw" not in grf:
        return None
    time = grf["time_force"].to_numpy(dtype=float)
    fz = grf["FPR_fz_up_norm_bw"].to_numpy(dtype=float)
    return time, fz


def _detect_contacts(time: np.ndarray, fz: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    above = np.asarray(fz) > threshold
    rising = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    if len(rising) == 0:
        return rising

    contacts = [int(rising[0])]
    for idx in rising[1:]:
        dt = time[idx] - time[contacts[-1]]
        if dt >= 0.55:
            contacts.append(int(idx))
    return np.asarray(contacts, dtype=int)


def _cycles_from_contacts(signal: np.ndarray, contacts: np.ndarray, n_points: int = 101) -> np.ndarray:
    cycles = []
    x_out = np.linspace(0.0, 1.0, n_points)
    for start, stop in zip(contacts[:-1], contacts[1:]):
        length = stop - start
        if length < 45 or length > 220:
            continue
        segment = signal[start:stop]
        if not np.isfinite(segment).all():
            continue
        x_in = np.linspace(0.0, 1.0, len(segment))
        cycles.append(np.interp(x_out, x_in, segment))
    if not cycles:
        return np.empty((0, n_points))
    return np.vstack(cycles)


def _annotate_metrics(ax: plt.Axes, row: pd.Series) -> None:
    text = (
        f"R2={row['r2']:.2f}\n"
        f"r={row['pearson_r']:.2f}\n"
        f"nRMSE={row['nrmse_pct']:.1f}%"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
    )


def plot_time_overlay(summary: pd.DataFrame, out_dir: Path, condition: str) -> Path:
    fig, axes = plt.subplots(len(SUBJECTS), 1, figsize=(10, 7.5), sharex=False, constrained_layout=True)

    for ax, subject in zip(axes, SUBJECTS):
        run_dir = _kjl_run_dir(summary, subject)
        metrics = _per_trial_metrics(run_dir)
        if condition not in set(metrics["condition"]):
            condition_use = metrics.sort_values("nrmse_pct").iloc[len(metrics) // 2]["condition"]
        else:
            condition_use = condition

        y_true, y_pred = _load_prediction(run_dir, condition_use)
        y_true = y_true * TASK_CONFIG["KJL"]["plot_sign"]
        y_pred = y_pred * TASK_CONFIG["KJL"]["plot_sign"]
        time = _load_prediction_time("KJL", subject, condition_use, len(y_true))
        time_rel = time - time[0]

        mask = time_rel <= min(8.0, time_rel[-1])
        ax.plot(time_rel[mask], y_true[mask], color=GT_COLOR, lw=1.8, label="GT")
        ax.plot(time_rel[mask], y_pred[mask], color=PRED_COLOR, lw=1.6, label="Pred")
        row = metrics[metrics["condition"] == condition_use].iloc[0]
        ax.set_title(f"{SUBJECT_LABELS[subject]} held out - {condition_use}")
        ax.set_ylabel(TASK_CONFIG["KJL"]["ylabel"])
        ax.grid(alpha=0.25)
        _annotate_metrics(ax, row)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    out_path = out_dir / f"kjl_time_overlay_{condition}.png"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def plot_gait_cycle_mean_sd(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(13.5, 4.4), sharey=True, constrained_layout=True)
    x = np.linspace(0, 100, 101)

    for ax, subject in zip(axes, SUBJECTS):
        run_dir = _kjl_run_dir(summary, subject)
        metrics = _per_trial_metrics(run_dir)
        true_cycles = []
        pred_cycles = []

        for condition in sorted(metrics["condition"].unique(), key=_condition_sort_key):
            try:
                y_true, y_pred = _load_prediction(run_dir, condition)
            except FileNotFoundError:
                continue
            y_true = y_true * TASK_CONFIG["KJL"]["plot_sign"]
            y_pred = y_pred * TASK_CONFIG["KJL"]["plot_sign"]
            pred_time = _load_prediction_time("KJL", subject, condition, len(y_true))
            grf = _load_grf(subject, condition)
            if grf is None:
                continue
            grf_time, grf_fz = grf
            fz_aligned = np.interp(pred_time, grf_time, grf_fz)
            contacts = _detect_contacts(pred_time, fz_aligned)
            true_cycle = _cycles_from_contacts(y_true, contacts)
            pred_cycle = _cycles_from_contacts(y_pred, contacts)
            if len(true_cycle) and len(pred_cycle):
                true_cycles.append(true_cycle)
                pred_cycles.append(pred_cycle)

        if true_cycles:
            true_all = np.vstack(true_cycles)
            pred_all = np.vstack(pred_cycles)
            true_mean, true_std = true_all.mean(axis=0), true_all.std(axis=0)
            pred_mean, pred_std = pred_all.mean(axis=0), pred_all.std(axis=0)
            ax.fill_between(x, true_mean - true_std, true_mean + true_std, color=GT_COLOR, alpha=0.13, linewidth=0)
            ax.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, color=PRED_COLOR, alpha=0.18, linewidth=0)
            ax.plot(x, true_mean, color=GT_COLOR, lw=2.0, label="GT mean +/- SD")
            ax.plot(x, pred_mean, color=PRED_COLOR, lw=2.0, label="Pred mean +/- SD")
            ax.set_title(f"{SUBJECT_LABELS[subject]} (n={len(true_all)} cycles)")
        else:
            ax.text(0.5, 0.5, "No cycles detected", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(SUBJECT_LABELS[subject])

        ax.set_xlabel("Right gait cycle (%)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(TASK_CONFIG["KJL"]["ylabel"])
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    out_path = out_dir / "kjl_gait_cycle_mean_sd_by_subject.png"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def plot_task_time_overlay(summary: pd.DataFrame, out_dir: Path, task: str, condition: str) -> Path:
    config = TASK_CONFIG[task]
    fig, axes = plt.subplots(len(SUBJECTS), 1, figsize=(10, 7.5), sharex=False, constrained_layout=True)

    for ax, subject in zip(axes, SUBJECTS):
        run_dir = _task_run_dir(summary, task, subject)
        metrics = _per_trial_metrics(run_dir)
        if condition not in set(metrics["condition"]):
            condition_use = metrics.sort_values("nrmse_pct").iloc[len(metrics) // 2]["condition"]
        else:
            condition_use = condition

        y_true, y_pred = _load_prediction(run_dir, condition_use)
        y_true = y_true * config["plot_sign"]
        y_pred = y_pred * config["plot_sign"]
        time = _load_prediction_time(task, subject, condition_use, len(y_true))
        time_rel = time - time[0]

        mask = time_rel <= min(8.0, time_rel[-1])
        ax.plot(time_rel[mask], y_true[mask], color=GT_COLOR, lw=1.8, label="GT")
        ax.plot(time_rel[mask], y_pred[mask], color=PRED_COLOR, lw=1.6, label="Pred")
        row = metrics[metrics["condition"] == condition_use].iloc[0]
        ax.set_title(f"{SUBJECT_LABELS[subject]} held out - {condition_use}")
        ax.set_ylabel(config["ylabel"])
        ax.grid(alpha=0.25)
        _annotate_metrics(ax, row)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    out_path = out_dir / f"{task.lower()}_time_overlay_{condition}.png"
    fig.suptitle(f"{config['title']} prediction vs GT", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_task_gait_cycle_mean_sd(summary: pd.DataFrame, out_dir: Path, task: str) -> Path:
    config = TASK_CONFIG[task]
    fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(13.5, 4.4), sharey=True, constrained_layout=True)
    x = np.linspace(0, 100, 101)

    for ax, subject in zip(axes, SUBJECTS):
        run_dir = _task_run_dir(summary, task, subject)
        metrics = _per_trial_metrics(run_dir)
        true_cycles = []
        pred_cycles = []

        for condition in sorted(metrics["condition"].unique(), key=_condition_sort_key):
            try:
                y_true, y_pred = _load_prediction(run_dir, condition)
            except FileNotFoundError:
                continue
            y_true = y_true * config["plot_sign"]
            y_pred = y_pred * config["plot_sign"]
            pred_time = _load_prediction_time(task, subject, condition, len(y_true))
            grf = _load_grf(subject, condition)
            if grf is None:
                continue
            grf_time, grf_fz = grf
            fz_aligned = np.interp(pred_time, grf_time, grf_fz)
            contacts = _detect_contacts(pred_time, fz_aligned)
            true_cycle = _cycles_from_contacts(y_true, contacts)
            pred_cycle = _cycles_from_contacts(y_pred, contacts)
            if len(true_cycle) and len(pred_cycle):
                true_cycles.append(true_cycle)
                pred_cycles.append(pred_cycle)

        if true_cycles:
            true_all = np.vstack(true_cycles)
            pred_all = np.vstack(pred_cycles)
            true_mean, true_std = true_all.mean(axis=0), true_all.std(axis=0)
            pred_mean, pred_std = pred_all.mean(axis=0), pred_all.std(axis=0)
            ax.fill_between(x, true_mean - true_std, true_mean + true_std, color=GT_COLOR, alpha=0.13, linewidth=0)
            ax.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, color=PRED_COLOR, alpha=0.18, linewidth=0)
            ax.plot(x, true_mean, color=GT_COLOR, lw=2.0, label="GT mean +/- SD")
            ax.plot(x, pred_mean, color=PRED_COLOR, lw=2.0, label="Pred mean +/- SD")
            ax.set_title(f"{SUBJECT_LABELS[subject]} (n={len(true_all)} cycles)")
        else:
            ax.text(0.5, 0.5, "No cycles detected", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(SUBJECT_LABELS[subject])

        ax.set_xlabel("Right gait cycle (%)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(config["ylabel"])
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    out_path = out_dir / f"{task.lower()}_gait_cycle_mean_sd_by_subject.png"
    fig.suptitle(f"{config['title']} gait-cycle mean +/- SD", y=1.04)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics_summary(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    tasks = ["GRF", "KFM", "KJL"]
    x = np.arange(len(tasks))
    width = 0.24

    for i, subject in enumerate(SUBJECTS):
        rows = summary[summary["held_out_subject"] == subject].set_index("task").reindex(tasks)
        offset = (i - 1) * width
        label = SUBJECT_LABELS[subject]
        color = SUBJECT_COLORS[subject]
        axes[0].bar(x + offset, rows["nrmse_pct"], width=width, label=label, color=color, alpha=0.85)
        axes[1].bar(x + offset, rows["r2"], width=width, label=label, color=color, alpha=0.85)

    axes[0].set_ylabel("nRMSE (%)")
    axes[1].set_ylabel("R2")
    for ax in axes:
        ax.set_xticks(x, tasks)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    out_path = out_dir / "loso_task_metrics_by_subject.png"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def plot_parity_and_bland(summary: pd.DataFrame, out_dir: Path) -> Path:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    all_true = []
    all_pred = []

    for subject in SUBJECTS:
        run_dir = _kjl_run_dir(summary, subject)
        metrics = _per_trial_metrics(run_dir)
        y_true_subject = []
        y_pred_subject = []
        for condition in metrics["condition"]:
            y_true, y_pred = _load_prediction(run_dir, condition)
            y_true_subject.append(y_true * TASK_CONFIG["KJL"]["plot_sign"])
            y_pred_subject.append(y_pred * TASK_CONFIG["KJL"]["plot_sign"])
        y_true_all = np.concatenate(y_true_subject)
        y_pred_all = np.concatenate(y_pred_subject)
        n = min(5000, len(y_true_all))
        idx = rng.choice(len(y_true_all), size=n, replace=False)
        color = SUBJECT_COLORS[subject]
        axes[0].scatter(y_true_all[idx], y_pred_all[idx], s=4, alpha=0.18, color=color, label=SUBJECT_LABELS[subject])
        mean = (y_true_all[idx] + y_pred_all[idx]) / 2.0
        diff = y_pred_all[idx] - y_true_all[idx]
        axes[1].scatter(mean, diff, s=4, alpha=0.18, color=color, label=SUBJECT_LABELS[subject])
        all_true.append(y_true_all)
        all_pred.append(y_pred_all)

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    lo = math.floor(min(true_all.min(), pred_all.min()))
    hi = math.ceil(max(true_all.max(), pred_all.max()))
    axes[0].plot([lo, hi], [lo, hi], color="#666666", lw=1.2, ls="--")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(f"GT {TASK_CONFIG['KJL']['ylabel']}")
    axes[0].set_ylabel(f"Predicted {TASK_CONFIG['KJL']['ylabel']}")
    axes[0].set_title("Point-wise agreement")
    axes[0].grid(alpha=0.25)

    diff_all = pred_all - true_all
    mean_all = (pred_all + true_all) / 2.0
    bias = float(np.mean(diff_all))
    loa = 1.96 * float(np.std(diff_all))
    axes[1].axhline(bias, color="#111111", lw=1.3, label=f"Bias={bias:.2f}")
    axes[1].axhline(bias + loa, color="#666666", lw=1.1, ls="--", label="95% LoA")
    axes[1].axhline(bias - loa, color="#666666", lw=1.1, ls="--")
    axes[1].set_xlim(np.percentile(mean_all, 0.5), np.percentile(mean_all, 99.5))
    axes[1].set_xlabel(f"Mean of GT and prediction ({TASK_CONFIG['KJL']['ylabel']})")
    axes[1].set_ylabel(f"Prediction - GT ({TASK_CONFIG['KJL']['ylabel']})")
    axes[1].set_title("Bland-Altman style residuals")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, markerscale=3)

    out_path = out_dir / "kjl_parity_and_bland_altman.png"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def plot_condition_heatmap(summary: pd.DataFrame, out_dir: Path) -> Path:
    rows = []
    for subject in SUBJECTS:
        run_dir = _kjl_run_dir(summary, subject)
        metrics = _per_trial_metrics(run_dir)
        metrics = metrics[["condition", "nrmse_pct", "r2"]].copy()
        metrics["subject"] = SUBJECT_LABELS[subject]
        rows.append(metrics)
    metrics_all = pd.concat(rows, ignore_index=True)
    conditions = sorted(metrics_all["condition"].unique(), key=_condition_sort_key)
    pivot = metrics_all.pivot(index="condition", columns="subject", values="nrmse_pct").loc[conditions]

    fig, ax = plt.subplots(figsize=(6.2, 9.2), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu_r")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_title("KJL LOSO nRMSE by held-out subject and condition")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            value = pivot.iloc[r, c]
            ax.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("nRMSE (%)")
    out_path = out_dir / "kjl_condition_nrmse_heatmap.png"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overlay-condition", default="20p200ms")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(run_root)
    paths = [
        plot_time_overlay(summary, out_dir, args.overlay_condition),
        plot_gait_cycle_mean_sd(summary, out_dir),
        plot_task_time_overlay(summary, out_dir, "GRF", args.overlay_condition),
        plot_task_time_overlay(summary, out_dir, "KFM", args.overlay_condition),
        plot_task_gait_cycle_mean_sd(summary, out_dir, "GRF"),
        plot_task_gait_cycle_mean_sd(summary, out_dir, "KFM"),
        plot_metrics_summary(summary, out_dir),
        plot_parity_and_bland(summary, out_dir),
        plot_condition_heatmap(summary, out_dir),
    ]

    print("Saved figures:")
    for path in paths:
        print(f"  {path}")
        print(f"  {path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
