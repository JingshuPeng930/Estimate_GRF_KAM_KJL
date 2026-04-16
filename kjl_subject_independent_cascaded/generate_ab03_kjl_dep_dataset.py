import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import scipy.signal as spsignal


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
IMU_DIR = ROOT / "IMU_Data_Process" / "AB03_Amy" / "LG_Exo" / "IMU_CSV"
IMU_NOEXO_DIR = ROOT / "IMU_Data_Process" / "AB03_Amy" / "LG_NoExo" / "IMU_CSV"
KJL_DIR = ROOT / "KJL_GT" / "AB03_Amy"
OUTPUT_DATASET_NAME = "data_kjl_ab03_dep_IMU"  # e.g. "data_kjl_ab03_dep_noxcorr" for A/B tests
OUT_ROOT = PACKAGE_DIR / "data" / "kjl_ab03_dep"

SUBJECT = "AB03_Amy"
TARGET_COL = "knee_r_on_tibia_r_in_tibia_r_fy"
TARGET_COL_NORM_TOTAL_BW = "knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw"
IMU_COLS = [
    "pelvis_imu_acc_x", "pelvis_imu_acc_y", "pelvis_imu_acc_z",
    "tibia_r_imu_acc_x", "tibia_r_imu_acc_y", "tibia_r_imu_acc_z",
    "femur_r_imu_acc_x", "femur_r_imu_acc_y", "femur_r_imu_acc_z",
    "calcn_r_imu_acc_x", "calcn_r_imu_acc_y", "calcn_r_imu_acc_z",
    "pelvis_imu_gyr_x", "pelvis_imu_gyr_y", "pelvis_imu_gyr_z",
    "tibia_r_imu_gyr_x", "tibia_r_imu_gyr_y", "tibia_r_imu_gyr_z",
    "femur_r_imu_gyr_x", "femur_r_imu_gyr_y", "femur_r_imu_gyr_z",
    "calcn_r_imu_gyr_x", "calcn_r_imu_gyr_y", "calcn_r_imu_gyr_z",
]
ACC_COLS = [c for c in IMU_COLS if "_acc_" in c]
IMU_FILTER_CUTOFF_HZ: float | None = 15.0
IMU_FILTER_ORDER = 4
IMU_FILTER_FS_HZ = 100.0
XCORR_PROXY_SPECS = [
    {"col": "femur_r_imu_gyr_y", "transform": "abs"},
]
# Keep xcorr fine-tuning local. Larger ranges (e.g., +/-0.5 s) can lock onto the wrong
# gait-cycle peak because the signals are strongly periodic.
USE_XCORR_FINE_TUNE = False
# How to apply the selected xcorr lag to JR timestamps during interpolation.
# +1: use jr_time + lag_sec (current behavior)
# -1: use jr_time - lag_sec (flip direction; often useful when model prediction is lagging GT)
XCORR_APPLY_SIGN = +1
XCORR_MAX_LAG_SAMPLES = 10  # +/- 0.10 s at 100 Hz (local correction only)
XCORR_MAX_ABS_APPLY_SAMPLES = 10  # hard sanity gate for applied lag
XCORR_SMOOTH_WIN = 11
# Restore the original "full-open" local xcorr behavior:
# single proxy, no consensus gating, no high-corr skip, no min-gain filtering.
XCORR_PROXY_MAX_SPREAD_SAMPLES = 999
XCORR_IGNORE_EDGE_LAG = False
XCORR_CONSENSUS_MIN_VALID_PROXIES = 1
XCORR_SKIP_IF_CORR_BEFORE_GE = 2.0
XCORR_LOW_CORR_BEFORE_THRESHOLD = 0.70
XCORR_MIN_IMPROVEMENT_LOW = -1.0
XCORR_MIN_IMPROVEMENT_MID = -1.0
GRAVITY = 9.81


def lowpass_filter_imu(
    imu_df: pd.DataFrame,
    cutoff_hz: float | None = IMU_FILTER_CUTOFF_HZ,
    order: int = IMU_FILTER_ORDER,
    fs_hz: float = IMU_FILTER_FS_HZ,
) -> tuple[pd.DataFrame, dict]:
    out = imu_df.copy()
    meta = {
        "imu_lowpass_filter_applied": False,
        "imu_lowpass_cutoff_hz": cutoff_hz,
        "imu_lowpass_order": int(order),
        "imu_lowpass_fs_hz": float(fs_hz),
    }
    if cutoff_hz is None or len(out) < 16:
        return out, meta

    nyq = 0.5 * float(fs_hz)
    wn = float(cutoff_hz) / nyq
    if wn <= 0 or wn >= 1:
        return out, meta

    b, a = spsignal.butter(int(order), wn, btype="low")
    try:
        out.loc[:, IMU_COLS] = spsignal.filtfilt(
            b,
            a,
            out[IMU_COLS].to_numpy(dtype=float),
            axis=0,
        )
        meta["imu_lowpass_filter_applied"] = True
    except ValueError as exc:
        meta["imu_lowpass_filter_error"] = str(exc)
    return out, meta


def standardize_imu_acc_units(imu_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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


def resample_label_to_target_time(
    jr_df: pd.DataFrame,
    target_time: np.ndarray,
    output_time: np.ndarray | None = None,
) -> pd.DataFrame:
    t = jr_df["time"].to_numpy(dtype=float)
    y = jr_df[TARGET_COL].to_numpy(dtype=float)
    if output_time is None:
        output_time = target_time

    # `np.interp` requires increasing x. Use stable deduplication for repeated timestamps.
    t_unique, unique_idx = np.unique(t, return_index=True)
    y_unique = y[unique_idx]
    y_interp = np.interp(target_time, t_unique, y_unique, left=np.nan, right=np.nan)

    return pd.DataFrame(
        {
            "time_jr": np.asarray(output_time, dtype=float).copy(),
            "time_jr_source": np.asarray(target_time, dtype=float).copy(),
            TARGET_COL: y_interp,
        }
    )


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


def interpolate_imu_to_jr_time(imu_df: pd.DataFrame, jr_time: np.ndarray) -> pd.DataFrame:
    imu_t = imu_df["time"].to_numpy(dtype=float)
    imu_t_unique, unique_idx = np.unique(imu_t, return_index=True)
    imu_unique = imu_df.iloc[unique_idx].reset_index(drop=True)

    out = {"time_imu": jr_time.copy()}
    for col in IMU_COLS:
        v = imu_unique[col].to_numpy(dtype=float)
        out[col] = np.interp(jr_time, imu_t_unique, v, left=np.nan, right=np.nan)
    return pd.DataFrame(out)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s + 1e-8)


def _transform_proxy_signal(x: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if transform == "signed":
        return x
    if transform == "abs":
        return np.abs(x)
    raise ValueError(f"Unsupported proxy transform: {transform}")


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 200:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.nanstd(aa) < 1e-8 or np.nanstd(bb) < 1e-8:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _best_lag_for_proxy(
    imu_t_unique: np.ndarray,
    imu_proxy: np.ndarray,
    jr_time: np.ndarray,
    kjl_sig: np.ndarray,
) -> dict:
    best = {"lag_samples": 0, "corr": float("nan")}
    corr_lag0 = float("nan")
    corr_map = {}
    for lag in range(-XCORR_MAX_LAG_SAMPLES, XCORR_MAX_LAG_SAMPLES + 1):
        lag_sec = lag / 100.0
        imu_interp = np.interp(jr_time + lag_sec, imu_t_unique, imu_proxy, left=np.nan, right=np.nan)
        imu_sig = _zscore(_moving_average(imu_interp, XCORR_SMOOTH_WIN))
        corr = _corrcoef_safe(imu_sig, kjl_sig)
        corr_map[int(lag)] = corr
        if lag == 0:
            corr_lag0 = corr
        if np.isfinite(corr) and (not np.isfinite(best["corr"]) or corr > best["corr"]):
            best = {"lag_samples": int(lag), "corr": float(corr)}
    return {"best": best, "corr_lag0": corr_lag0, "corr_map": corr_map}


def find_best_lag_samples(imu_df: pd.DataFrame, jr_100: pd.DataFrame) -> dict:
    proxy_cols = [spec["col"] for spec in XCORR_PROXY_SPECS]
    proxy_desc = [f'{spec["col"]}:{spec["transform"]}' for spec in XCORR_PROXY_SPECS]
    if not USE_XCORR_FINE_TUNE:
        return {
            "lag_samples_raw": 0,
            "lag_sec_raw": 0.0,
            "lag_samples_applied": 0,
            "lag_sec_applied": 0.0,
            "corr_before": None,
            "corr_after": None,
            "proxy_col": proxy_cols[0] if proxy_cols else None,
            "proxy_cols": proxy_cols,
            "proxy_specs": proxy_desc,
            "proxy_best_lags": {},
            "proxy_valid_lags": {},
            "proxy_edge_lag_ignored": {},
            "proxy_n_valid": 0,
            "proxy_spread_samples": 0,
            "max_lag_samples": 0,
            "max_abs_apply_samples": 0,
            "apply_sign": int(XCORR_APPLY_SIGN),
            "xcorr_fallback_to_lag0": True,
            "xcorr_fallback_reason": "xcorr_disabled",
        }

    imu_t = imu_df["time"].to_numpy(dtype=float)
    imu_t_unique, unique_idx = np.unique(imu_t, return_index=True)
    imu_unique = imu_df.iloc[unique_idx].reset_index(drop=True)

    jr_time = jr_100["time_jr"].to_numpy(dtype=float)
    kjl = -jr_100[TARGET_COL].to_numpy(dtype=float)

    kjl_sig = _zscore(_moving_average(kjl, XCORR_SMOOTH_WIN))

    proxy_lags = []
    proxy_valid_lags = {}
    proxy_edge_lag_ignored = {}
    corr_lag0_list = []
    proxy_corr_at_consensus = []
    proxy_best_lags = {}
    proxy_corr_maps = {}

    for spec in XCORR_PROXY_SPECS:
        proxy_col = spec["col"]
        imu_proxy = _transform_proxy_signal(
            imu_unique[proxy_col].to_numpy(dtype=float),
            spec["transform"],
        )
        proxy_out = _best_lag_for_proxy(imu_t_unique, imu_proxy, jr_time, kjl_sig)
        best = proxy_out["best"]
        corr_lag0 = proxy_out["corr_lag0"]
        proxy_corr_maps[proxy_col] = proxy_out["corr_map"]
        best_lag = int(best["lag_samples"])
        proxy_best_lags[proxy_col] = best_lag
        is_edge = abs(best_lag) >= XCORR_MAX_LAG_SAMPLES
        proxy_edge_lag_ignored[proxy_col] = bool(XCORR_IGNORE_EDGE_LAG and is_edge)
        if np.isfinite(best["corr"]) and not (XCORR_IGNORE_EDGE_LAG and is_edge):
            proxy_lags.append(best_lag)
            proxy_valid_lags[proxy_col] = best_lag
        if np.isfinite(corr_lag0):
            corr_lag0_list.append(float(corr_lag0))

    if proxy_lags:
        consensus_lag = int(np.median(np.asarray(proxy_lags, dtype=int)))
        proxy_spread = int(max(proxy_lags) - min(proxy_lags))
    else:
        consensus_lag = 0
        proxy_spread = 0

    for proxy_col in proxy_cols:
        corr_at_lag = proxy_corr_maps.get(proxy_col, {}).get(int(consensus_lag), float("nan"))
        if np.isfinite(corr_at_lag):
            proxy_corr_at_consensus.append(float(corr_at_lag))

    best = {
        "lag_samples": int(consensus_lag),
        "corr": (float(np.mean(proxy_corr_at_consensus)) if proxy_corr_at_consensus else float("nan")),
    }
    corr_lag0 = float(np.mean(corr_lag0_list)) if corr_lag0_list else float("nan")

    xcorr_fallback = False
    xcorr_fallback_reason = None
    corr_before = None if not np.isfinite(corr_lag0) else float(corr_lag0)
    corr_after = None if not np.isfinite(best["corr"]) else float(best["corr"])

    # Conservative layered rules:
    # 1) skip xcorr entirely if lag=0 proxy alignment is already high
    # 2) enforce a strict max applied lag
    # 3) require a gain threshold that depends on corr_before
    if len(proxy_lags) < XCORR_CONSENSUS_MIN_VALID_PROXIES:
        best = {"lag_samples": 0, "corr": corr_lag0}
        xcorr_fallback = True
        xcorr_fallback_reason = "insufficient_valid_proxies"
    elif proxy_lags and proxy_spread > XCORR_PROXY_MAX_SPREAD_SAMPLES:
        best = {"lag_samples": 0, "corr": corr_lag0}
        xcorr_fallback = True
        xcorr_fallback_reason = "proxy_disagreement_too_large"
    elif corr_before is not None and corr_before >= XCORR_SKIP_IF_CORR_BEFORE_GE:
        best = {"lag_samples": 0, "corr": corr_lag0}
        xcorr_fallback = True
        xcorr_fallback_reason = "high_corr_before_skip"
    elif abs(int(best["lag_samples"])) > XCORR_MAX_ABS_APPLY_SAMPLES:
        best = {"lag_samples": 0, "corr": corr_lag0}
        xcorr_fallback = True
        xcorr_fallback_reason = "lag_too_large"
    elif corr_before is not None and corr_after is not None:
        gain = corr_after - corr_before
        min_gain = (
            XCORR_MIN_IMPROVEMENT_LOW
            if corr_before < XCORR_LOW_CORR_BEFORE_THRESHOLD
            else XCORR_MIN_IMPROVEMENT_MID
        )
        if gain < min_gain:
            best = {"lag_samples": 0, "corr": corr_lag0}
            xcorr_fallback = True
            xcorr_fallback_reason = "min_improvement_not_met"

    lag_samples_raw = int(best["lag_samples"])
    lag_samples_applied = int(XCORR_APPLY_SIGN * lag_samples_raw)
    return {
        "lag_samples_raw": lag_samples_raw,
        "lag_sec_raw": float(lag_samples_raw / 100.0),
        "lag_samples_applied": lag_samples_applied,
        "lag_sec_applied": float(lag_samples_applied / 100.0),
        "corr_before": corr_before,
        "corr_after": None if not np.isfinite(best["corr"]) else float(best["corr"]),
        "proxy_col": proxy_cols[0] if proxy_cols else None,
        "proxy_cols": proxy_cols,
        "proxy_specs": proxy_desc,
        "proxy_best_lags": proxy_best_lags,
        "proxy_valid_lags": proxy_valid_lags,
        "proxy_edge_lag_ignored": proxy_edge_lag_ignored,
        "proxy_n_valid": int(len(proxy_lags)),
        "proxy_spread_samples": int(proxy_spread),
        "max_lag_samples": XCORR_MAX_LAG_SAMPLES,
        "max_abs_apply_samples": XCORR_MAX_ABS_APPLY_SAMPLES,
        "apply_sign": int(XCORR_APPLY_SIGN),
        "xcorr_fallback_to_lag0": xcorr_fallback,
        "xcorr_fallback_reason": xcorr_fallback_reason,
    }


def build_trial(imu_path: Path, jr_path: Path, out_trial_dir: Path, total_model_mass_kg: float) -> dict:
    imu_df = pd.read_csv(imu_path)
    jr_df = pd.read_csv(jr_path)

    missing = [c for c in IMU_COLS if c not in imu_df.columns]
    if missing:
        raise ValueError(f"Missing IMU columns in {imu_path.name}: {missing}")
    if TARGET_COL not in jr_df.columns:
        raise ValueError(f"Missing target column in {jr_path.name}: {TARGET_COL}")
    imu_df, imu_unit_meta = standardize_imu_acc_units(imu_df)
    imu_df, imu_filter_meta = lowpass_filter_imu(imu_df)

    imu_feat = imu_df[["time", *IMU_COLS]].copy()
    imu_time = imu_feat["time"].to_numpy(dtype=float)
    jr_on_imu = resample_label_to_target_time(jr_df, imu_time, output_time=imu_time)
    lag_info = find_best_lag_samples(imu_feat, jr_on_imu)
    lag_sec = lag_info["lag_sec_applied"]

    # Keep IMU on its native timeline. Shift only the label sampling timeline if xcorr is enabled.
    jr_on_imu = resample_label_to_target_time(
        jr_df,
        imu_time + lag_sec,
        output_time=imu_time,
    )
    imu_aligned = imu_feat.rename(columns={"time": "time_imu"}).reset_index(drop=True)

    # Keep only timestamps that fall within JR time range after interpolation.
    finite_mask = np.isfinite(jr_on_imu[[TARGET_COL]].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite_mask).sum())
    if not finite_mask.any():
        raise ValueError(f"No overlapping timestamps between IMU and JR for {imu_path.name}")
    imu_aligned = imu_aligned.loc[finite_mask].reset_index(drop=True)
    jr_on_imu = jr_on_imu.loc[finite_mask].reset_index(drop=True)

    merged = pd.concat([imu_aligned, jr_on_imu], axis=1)
    merged.insert(0, "sample_idx", np.arange(len(merged), dtype=int))
    merged[TARGET_COL_NORM_TOTAL_BW] = merged[TARGET_COL] / (total_model_mass_kg * GRAVITY)

    input_dir = out_trial_dir / "Input"
    label_dir = out_trial_dir / "Label"
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    merged[["sample_idx", "time_imu", *IMU_COLS]].to_csv(input_dir / "imu.csv", index=False)
    merged[["sample_idx", "time_jr", TARGET_COL, TARGET_COL_NORM_TOTAL_BW]].to_csv(
        label_dir / "kjl_fy.csv", index=False
    )
    merged.to_csv(out_trial_dir / "aligned_debug.csv", index=False)

    return {
        "imu_file": str(imu_path.relative_to(ROOT)),
        "jr_file": str(jr_path.relative_to(ROOT)),
        "output_trial_dir": str(out_trial_dir.relative_to(ROOT)),
        "imu_rows_original": int(len(imu_df)),
        **imu_unit_meta,
        **imu_filter_meta,
        "jr_rows_original": int(len(jr_df)),
        "label_rows_resampled_100hz": int(len(jr_on_imu)),
        "rows_after_timestamp_alignment": int(len(imu_aligned)),
        "rows_dropped_outside_imu_time_range": dropped,
        "imu_time_start_used": float(imu_aligned["time_imu"].iloc[0]),
        "imu_time_end_used": float(imu_aligned["time_imu"].iloc[-1]),
        "jr_time_start": float(jr_on_imu["time_jr"].iloc[0]),
        "jr_time_end": float(jr_on_imu["time_jr"].iloc[-1]),
        "alignment": "label_interpolation_to_imu_time",
        "total_model_mass_kg": float(total_model_mass_kg),
        "target_normalization": "divided_by_total_model_weight_(mass_kg*9.81)",
        "normalized_target_col": TARGET_COL_NORM_TOTAL_BW,
        "xcorr_lag_enabled": bool(USE_XCORR_FINE_TUNE),
        "xcorr_proxy_col": lag_info["proxy_col"],
        "xcorr_proxy_cols": lag_info["proxy_cols"],
        "xcorr_proxy_specs": lag_info["proxy_specs"],
        "xcorr_proxy_best_lags": lag_info["proxy_best_lags"],
        "xcorr_proxy_valid_lags": lag_info["proxy_valid_lags"],
        "xcorr_proxy_edge_lag_ignored": lag_info["proxy_edge_lag_ignored"],
        "xcorr_proxy_n_valid": int(lag_info["proxy_n_valid"]),
        "xcorr_proxy_spread_samples": int(lag_info["proxy_spread_samples"]),
        "xcorr_max_lag_samples": int(lag_info["max_lag_samples"]),
        "xcorr_max_abs_apply_samples": int(lag_info["max_abs_apply_samples"]),
        "xcorr_apply_sign": int(lag_info["apply_sign"]),
        "lag_samples_raw": int(lag_info["lag_samples_raw"]),
        "lag_sec_raw": float(lag_info["lag_sec_raw"]),
        "lag_samples": int(lag_info["lag_samples_applied"]),
        "lag_sec": float(lag_info["lag_sec_applied"]),
        "xcorr_corr_before": lag_info["corr_before"],
        "xcorr_corr_after": lag_info["corr_after"],
        "xcorr_fallback_to_lag0": bool(lag_info["xcorr_fallback_to_lag0"]),
        "xcorr_fallback_reason": lag_info["xcorr_fallback_reason"],
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    exo_model_path = ROOT / "IMU_Data_Process" / SUBJECT / "LG_Exo" / "SCALE" / f"{SUBJECT}_Scaled_unilateral.osim"
    noexo_model_path = ROOT / "IMU_Data_Process" / SUBJECT / "LG_NoExo" / "SCALE" / f"{SUBJECT}_Scaled_unilateral.osim"
    exo_total_mass_kg = load_total_model_mass_kg(exo_model_path)
    noexo_total_mass_kg = load_total_model_mass_kg(noexo_model_path)

    manifest = {
        "subject": SUBJECT,
        "target": TARGET_COL,
        "normalized_target_col": TARGET_COL_NORM_TOTAL_BW,
        "imu_feature_count": len(IMU_COLS),
        "imu_features": IMU_COLS,
        "imu_preprocessing": {
            "acc_unit_standardization": "multiply accelerometer channels by 1000 when median abs < 0.1",
            "lowpass_filter": {
                "type": "zero_phase_butterworth",
                "cutoff_hz": IMU_FILTER_CUTOFF_HZ,
                "order": IMU_FILTER_ORDER,
                "fs_hz": IMU_FILTER_FS_HZ,
                "channels": IMU_COLS,
            },
        },
        "sampling_rate_hz": 100,
        "label_resampling": "linear_interpolation_from_JR_time_to_IMU_time",
        "alignment_assumption": (
            "IMU stays on its native 100Hz timeline and JR is linearly interpolated "
            "onto the IMU timestamps; out-of-range timestamps are dropped."
        ),
        "xcorr_lag_finetune": {
            "enabled": bool(USE_XCORR_FINE_TUNE),
            "proxy_col": XCORR_PROXY_SPECS[0]["col"] if XCORR_PROXY_SPECS else None,
            "proxy_cols": [spec["col"] for spec in XCORR_PROXY_SPECS],
            "proxy_specs": [f'{spec["col"]}:{spec["transform"]}' for spec in XCORR_PROXY_SPECS],
            "max_lag_samples": (XCORR_MAX_LAG_SAMPLES if USE_XCORR_FINE_TUNE else 0),
            "max_abs_apply_samples": (XCORR_MAX_ABS_APPLY_SAMPLES if USE_XCORR_FINE_TUNE else 0),
            "proxy_max_spread_samples": XCORR_PROXY_MAX_SPREAD_SAMPLES,
            "ignore_edge_lag": bool(XCORR_IGNORE_EDGE_LAG),
            "consensus_min_valid_proxies": int(XCORR_CONSENSUS_MIN_VALID_PROXIES),
            "apply_sign": int(XCORR_APPLY_SIGN),
            "skip_if_corr_before_ge": XCORR_SKIP_IF_CORR_BEFORE_GE,
            "low_corr_before_threshold": XCORR_LOW_CORR_BEFORE_THRESHOLD,
            "min_improvement_low_corr": XCORR_MIN_IMPROVEMENT_LOW,
            "min_improvement_mid_corr": XCORR_MIN_IMPROVEMENT_MID,
            "smooth_window_samples": XCORR_SMOOTH_WIN,
        },
        "normalization": {
            "type": "total_model_weight",
            "formula": f"{TARGET_COL} / (total_model_mass_kg * {GRAVITY})",
            "exo_total_model_mass_kg": exo_total_mass_kg,
            "noexo_total_model_mass_kg": noexo_total_mass_kg,
        },
        "trials": [],
        "skipped_trials_no_jr": [],
    }

    exo_imu_files = sorted(IMU_DIR.glob("AB03_Amy_LG_*_1.csv"))
    noexo_imu_files = sorted(IMU_NOEXO_DIR.glob("AB03_Amy_LG_NoExo_1.csv")) if IMU_NOEXO_DIR.exists() else []

    for imu_path in [*exo_imu_files, *noexo_imu_files]:
        stem = imu_path.stem  # AB03_Amy_LG_10p100ms_1
        is_noexo = "NoExo" in stem and "LG_NoExo" in str(imu_path)
        if is_noexo:
            cond = "NoExo"
            trial_base = stem[:-2] if stem.endswith("_1") else stem  # AB03_Amy_LG_NoExo
            jr_stem = trial_base  # GT uses no _1 suffix
        else:
            trial_base = stem[:-2] if stem.endswith("_1") else stem
            cond = trial_base.split("_LG_")[1]
            jr_stem = trial_base

        jr_path = KJL_DIR / cond / f"{jr_stem}_JointReaction_ReactionLoads.csv"

        if not jr_path.exists():
            manifest["skipped_trials_no_jr"].append(stem)
            continue

        out_trial_dir = OUT_ROOT / SUBJECT / "LG" / cond / "trial_1"
        mass_kg = noexo_total_mass_kg if is_noexo else exo_total_mass_kg
        info = build_trial(imu_path, jr_path, out_trial_dir, total_model_mass_kg=mass_kg)
        info["source_group"] = "LG_NoExo" if is_noexo else "LG_Exo"
        manifest["trials"].append(info)
        print(f"[OK] {stem} -> {out_trial_dir}")

    with open(OUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    # Simple split file placeholder (training script can also auto-split with seed).
    trial_rel_paths = [t["output_trial_dir"] for t in manifest["trials"]]
    split = {
        "train_trials": trial_rel_paths[:1],
        "val_trials": trial_rel_paths[1:],
        "test_trials": trial_rel_paths[1:],
    }
    with open(OUT_ROOT / "split_subject_dependent.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=True, indent=2)

    print(f"Generated {len(manifest['trials'])} trials.")
    print(f"Skipped (no JR): {len(manifest['skipped_trials_no_jr'])} trials.")
    print(f"Manifest: {OUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
