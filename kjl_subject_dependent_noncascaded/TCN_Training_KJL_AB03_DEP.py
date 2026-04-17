import copy
import json
import os
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from TCN_Header_Model import TCNModel
from kjl_ab03_tcn_dataset import build_kjl_ab03_dataloaders, WindowedTrialDataset
from soft_delay_classifier import (
    DelayClassifier,
    build_delay_classes,
    predict_delay_probs,
    train_delay_classifier,
)


CONFIG = {
    "run_name": "KJL_AB03_Amy_TCN_DEP",
    "seeds": [42, 123, 2026, 7, 99],  # e.g. [42, 123, 2026]
    "dataset_root": "data/kjl_ab03_dep",
    # Optional input feature exclusion for ablation studies.
    # Example: ["pelvis_imu_acc_y"]
    "exclude_input_cols": [],
    "target_col": "knee_r_on_tibia_r_in_tibia_r_fy_norm_totalmodel_bw",
    # split_json=None → each seed randomly assigns trials to train/val/test via _split_trials,
    # so val distribution is not systematically biased (fixes the 300ms-in-val problem).
    "split_json": None,
    "noexo_split_ratio": 0.7,  # fallback if split_json does not encode it
    "trial_balanced_sampling": True,  # upsample small trials (e.g. NoExo) to equal per-trial weight
    # Optional condition-aware training-time weighting (no deployment input change).
    # Example to emphasize harder high-delay regimes: {200: 1.3, 250: 1.8, 300: 2.0}
    "delay_weight_map": {},
    "noexo_train_weight": 1.0,
    "noassi_train_weight": 1.0,
    "default_train_weight": 1.0,
    "seed": 42,
    "window_size": 150,
    "batch_size": 32,
    "num_workers": 0,
    "epochs": 50,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "dropout": 0.15,
    "loss_type": "huber",  # "mse" or "huber"
    "huber_beta": 10.0,
    # Optional shape-aware auxiliary loss: total = huber + lambda * (1 - PearsonCorr)
    "use_corr_loss": False,
    "corr_loss_weight": 0.05,
    "corr_loss_eps": 1e-8,
    # Optional local slope-matching loss on adjacent windows from the same trial.
    "use_deriv_loss": False,
    "deriv_loss_weight": 0.05,
    "deriv_loss_beta": 1.0,
    "deriv_loss_min_delay_ms": 200,
    # Optional waveform-shape loss on consecutive chunks from the same trial.
    # total += chunk_shape_weight * (chunk_corr_weight * corr_loss + chunk_fft_weight * fft_mag_loss)
    "use_chunk_shape_loss": False,
    "chunk_shape_weight": 0.10,
    "chunk_size": 32,
    "chunk_stride": 16,
    "chunk_min_delay_ms": 200,
    "chunk_corr_weight": 1.0,
    "chunk_fft_weight": 0.5,
    "chunk_fft_bins": 8,
    "chunk_fft_beta": 1.0,
    "chunk_shape_eps": 1e-8,
    # Oracle upper-bound switch: append ground-truth delay one-hot per window.
    # For analysis only (not realistic deployment).
    "use_oracle_delay_input": False,
    # Two-stage option: train a separate delay classifier first, then append
    # its predicted delay probabilities as soft conditioning inputs.
    "use_soft_delay_input": False,
    "delay_clf_hidden_dim": 128,
    "delay_clf_dropout": 0.1,
    "delay_clf_epochs": 12,
    "delay_clf_lr": 1e-3,
    "delay_clf_weight_decay": 1e-4,
    "delay_softmax_temperature": 1.0,
    # Cascaded option: append frozen upstream model predictions as extra
    # per-window input channels for KJL. Each prediction is repeated across
    # the KJL window time axis before concatenation.
    #
    # Fill run_dir with a trained run folder containing:
    #   train_config.json, input_mean.npy, input_std.npy, label_mean.npy,
    #   label_std.npy, and <run_name>.pt
    # Alternatively, set checkpoint_path directly; run_dir is still used for
    # the config/stat files when provided.
    "use_cascade_inputs": False,
    "cascade_prediction_mode": "normalized",  # "normalized" or "denormalized"
    "cascade_allow_window_adapter": True,
    "cascade_sources": [
        {
            "name": "grf",
            "enabled": True,
            "run_dir": "",
            "checkpoint_path": "",
            "output_indices": [0],
        },
        {
            "name": "kam",
            "enabled": True,
            "run_dir": "",
            "checkpoint_path": "",
            "output_indices": [0],
        },
    ],
    "number_of_layers": 2,
    "num_channels": [32, 32, 32, 32],
    "kernel_size": 5,
    "dilations": [1, 2, 4, 8, 16],
    "patience": 10,
    "use_last_epoch": False,  # val-based early stopping (val is randomly selected)
    "output_dir": "runs/kjl_ab03_dep",
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1).astype(float)
    y_pred = y_pred.reshape(-1).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"r2": float("nan"), "pearson_r": float("nan"), "nrmse_pct": float("nan")}

    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if sst <= 1e-12 else 1.0 - sse / sst

    yt_std = float(np.std(y_true))
    yp_std = float(np.std(y_pred))
    if yt_std <= 1e-12 or yp_std <= 1e-12:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    y_range = float(np.max(y_true) - np.min(y_true))
    nrmse_pct = float("nan") if y_range <= 1e-12 else (rmse / y_range) * 100.0
    return {"r2": r2, "pearson_r": pearson_r, "nrmse_pct": nrmse_pct}


def _batch_pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Batch-wise Pearson correlation on flattened predictions/targets.
    p = pred.reshape(-1)
    t = target.reshape(-1)
    p = p - p.mean()
    t = t - t.mean()
    denom = torch.sqrt((p.pow(2).sum()) * (t.pow(2).sum())).clamp_min(eps)
    corr = (p * t).sum() / denom
    corr = torch.clamp(corr, -1.0, 1.0)
    return 1.0 - corr


def _batch_derivative_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    trial_idx: torch.Tensor,
    start_idx: torch.Tensor,
    delay_ms: torch.Tensor | None = None,
    min_delay_ms: int = 200,
    beta: float = 1.0,
) -> torch.Tensor:
    p = pred.reshape(-1)
    t = target.reshape(-1)
    ti = trial_idx.reshape(-1)
    si = start_idx.reshape(-1)

    order = torch.argsort(ti * 1_000_000 + si)
    p = p[order]
    t = t[order]
    ti = ti[order]
    si = si[order]

    same_trial = ti[1:] == ti[:-1]
    consecutive = (si[1:] - si[:-1]) == 1
    valid = same_trial & consecutive
    if delay_ms is not None:
        di = delay_ms.reshape(-1)[order]
        valid = valid & (di[1:] >= int(min_delay_ms)) & (di[:-1] >= int(min_delay_ms))
    if not torch.any(valid):
        return pred.new_tensor(0.0)

    pred_diff = p[1:] - p[:-1]
    true_diff = t[1:] - t[:-1]
    return torch.nn.functional.smooth_l1_loss(pred_diff[valid], true_diff[valid], beta=beta)


def _batch_chunk_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    trial_idx: torch.Tensor,
    start_idx: torch.Tensor,
    delay_ms: torch.Tensor | None = None,
    chunk_size: int = 32,
    chunk_stride: int = 16,
    min_delay_ms: int = 200,
    corr_weight: float = 1.0,
    fft_weight: float = 0.5,
    fft_bins: int = 8,
    fft_beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Waveform-shape loss on consecutive chunks from the same trial.

    This is designed for single-step targets by reconstructing local sequences
    from consecutive windows in each batch (same trial, start index +1).
    """
    if chunk_size <= 1 or (corr_weight <= 0 and fft_weight <= 0):
        return pred.new_tensor(0.0)

    p = pred.reshape(-1)
    t = target.reshape(-1)
    ti = trial_idx.reshape(-1)
    si = start_idx.reshape(-1)

    order = torch.argsort(ti * 1_000_000 + si)
    p = p[order]
    t = t[order]
    ti = ti[order]
    si = si[order]
    di = delay_ms.reshape(-1)[order] if delay_ms is not None else None

    ti_cpu = ti.detach().cpu()
    si_cpu = si.detach().cpu()
    point_valid = torch.ones_like(ti_cpu, dtype=torch.bool)
    if di is not None:
        point_valid = (di >= int(min_delay_ms)).detach().cpu()

    n = int(p.numel())
    losses = []
    i = 0
    stride = max(1, int(chunk_stride))
    ksize = int(chunk_size)
    n_fft_bins = max(1, int(fft_bins))
    fft_beta = float(fft_beta)
    eps = float(eps)

    def _chunk_loss(pp: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        l = pp.new_tensor(0.0)
        if corr_weight > 0:
            pz = pp - pp.mean()
            tz = tt - tt.mean()
            denom = torch.sqrt((pz.pow(2).sum()) * (tz.pow(2).sum())).clamp_min(eps)
            corr = torch.clamp((pz * tz).sum() / denom, -1.0, 1.0)
            l = l + float(corr_weight) * (1.0 - corr)
        if fft_weight > 0:
            pz = pp - pp.mean()
            tz = tt - tt.mean()
            pm = torch.abs(torch.fft.rfft(pz))[1:]  # drop DC
            tm = torch.abs(torch.fft.rfft(tz))[1:]
            if pm.numel() > 0 and tm.numel() > 0:
                k = min(n_fft_bins, int(pm.numel()), int(tm.numel()))
                pm = pm[:k]
                tm = tm[:k]
                pm = pm / pm.sum().clamp_min(eps)
                tm = tm / tm.sum().clamp_min(eps)
                l = l + float(fft_weight) * torch.nn.functional.smooth_l1_loss(pm, tm, beta=fft_beta)
        return l

    while i < n:
        if not bool(point_valid[i].item()):
            i += 1
            continue
        j = i + 1
        while (
            j < n
            and bool(point_valid[j].item())
            and int(ti_cpu[j].item()) == int(ti_cpu[j - 1].item())
            and int(si_cpu[j].item()) - int(si_cpu[j - 1].item()) == 1
        ):
            j += 1

        run_len = j - i
        if run_len >= ksize:
            last_start = j - ksize
            starts = list(range(i, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            for s in starts:
                losses.append(_chunk_loss(p[s:s + ksize], t[s:s + ksize]))

        i = j

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _augment_with_soft_delay_probs(
    x: torch.Tensor,
    delay_classifier: DelayClassifier | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if delay_classifier is None:
        return x
    with torch.no_grad():
        probs = predict_delay_probs(delay_classifier, x, temperature=temperature)
    probs_seq = probs.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, K, T)
    return torch.cat([x, probs_seq], dim=1)


def _collect_delay_class_values(data_bundle) -> list[int]:
    delay_values = []
    for ds in (data_bundle.train_dataset, data_bundle.val_dataset, data_bundle.test_dataset):
        if ds is None:
            continue
        delay_values.extend(int(v) for v in getattr(ds, "trial_delay_ms", []))
    return build_delay_classes(delay_values)


def _augment_with_oracle_delay_onehot(
    x: torch.Tensor,
    delay_ms: torch.Tensor | None,
    class_values: list[int] | None,
) -> torch.Tensor:
    if delay_ms is None or not class_values:
        return x
    cls = torch.tensor(class_values, dtype=delay_ms.dtype, device=delay_ms.device)
    onehot = delay_ms.reshape(-1, 1) == cls.reshape(1, -1)  # (B, K)
    no_match = ~torch.any(onehot, dim=1)
    if torch.any(no_match):
        onehot[no_match, 0] = True
    onehot = onehot.float()
    onehot_seq = onehot.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, K, T)
    return torch.cat([x, onehot_seq], dim=1)


@dataclass
class CascadeSource:
    name: str
    model: TCNModel
    run_dir: Path
    checkpoint_path: Path
    input_mean: torch.Tensor
    input_std: torch.Tensor
    label_mean: torch.Tensor
    label_std: torch.Tensor
    output_indices: list[int]
    input_size: int
    output_size: int
    window_size: int

    @property
    def feature_dim(self) -> int:
        return len(self.output_indices)


def _resolve_path(path_value: str | os.PathLike | None) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(str(path_value)).expanduser()
    if path.exists() or path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / path
    if repo_path.exists():
        return repo_path
    return path


def _format_cfg_path(path_value: str | os.PathLike | None, cfg: dict) -> str:
    if path_value is None:
        return ""
    text = str(path_value)
    if not text:
        return text
    return text.format(seed=cfg.get("seed", ""), run_name=cfg.get("run_name", ""))


def _load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_checkpoint(run_dir: Path, train_cfg: dict, checkpoint_path: Path | None = None) -> Path:
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cascade checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    run_name = str(train_cfg.get("run_name", ""))
    if run_name:
        preferred = run_dir / f"{run_name}.pt"
        if preferred.exists():
            return preferred

    non_epoch = sorted(p for p in run_dir.glob("*.pt") if "_epoch_" not in p.name)
    if non_epoch:
        return non_epoch[0]

    candidates = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .pt checkpoint found in cascade run_dir: {run_dir}")


def _model_cfg_from_train_cfg(train_cfg: dict, run_dir: Path) -> dict:
    label_mean_path = run_dir / "label_mean.npy"
    output_size = int(train_cfg.get("output_size", 0))
    if output_size <= 0 and label_mean_path.exists():
        output_size = int(np.asarray(np.load(label_mean_path)).reshape(-1).shape[0])
    if output_size <= 0:
        output_size = 1

    return {
        "input_size": int(train_cfg["input_size"]),
        "output_size": output_size,
        "dropout": float(train_cfg.get("dropout", 0.0)),
        "number_of_layers": int(train_cfg["number_of_layers"]),
        "num_channels": [int(v) for v in train_cfg["num_channels"]],
        "kernel_size": int(train_cfg["kernel_size"]),
        "dilations": [int(v) for v in train_cfg["dilations"]],
        "window_size": int(train_cfg["window_size"]),
    }


def _load_cascade_sources(cfg: dict, device: str) -> list[CascadeSource]:
    if not bool(cfg.get("use_cascade_inputs", False)):
        return []

    sources = []
    raw_sources = cfg.get("cascade_sources", [])
    if not raw_sources:
        raise ValueError("use_cascade_inputs=True but cascade_sources is empty.")

    for source_cfg in raw_sources:
        if not source_cfg or not bool(source_cfg.get("enabled", True)):
            continue

        name = str(source_cfg.get("name", f"source{len(sources)}"))
        run_dir_text = _format_cfg_path(source_cfg.get("run_dir", ""), cfg)
        checkpoint_text = _format_cfg_path(source_cfg.get("checkpoint_path", ""), cfg)
        run_dir = _resolve_path(run_dir_text)
        checkpoint_path = _resolve_path(checkpoint_text)

        if run_dir is None and checkpoint_path is None:
            raise ValueError(
                f"Cascade source `{name}` needs run_dir or checkpoint_path. "
                "A run_dir is recommended because normalization stats live there."
            )
        if run_dir is None:
            run_dir = checkpoint_path.parent
        if checkpoint_path is not None and checkpoint_path.is_dir():
            run_dir = checkpoint_path
            checkpoint_path = None

        train_config_path = run_dir / "train_config.json"
        if not train_config_path.exists():
            raise FileNotFoundError(f"Missing train_config.json for cascade source `{name}`: {train_config_path}")
        train_cfg = _load_json_file(train_config_path)
        checkpoint_path = _find_checkpoint(run_dir, train_cfg, checkpoint_path=checkpoint_path)

        required_stats = ["input_mean.npy", "input_std.npy", "label_mean.npy", "label_std.npy"]
        missing = [fname for fname in required_stats if not (run_dir / fname).exists()]
        if missing:
            raise FileNotFoundError(f"Cascade source `{name}` missing stats in {run_dir}: {missing}")

        model_cfg = _model_cfg_from_train_cfg(train_cfg, run_dir)
        model = TCNModel(model_cfg).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        output_indices = source_cfg.get("output_indices", None)
        if output_indices is None:
            output_indices = list(range(model_cfg["output_size"]))
        elif isinstance(output_indices, int):
            output_indices = [int(output_indices)]
        else:
            output_indices = [int(v) for v in output_indices]
        if not output_indices:
            raise ValueError(f"Cascade source `{name}` has no output_indices.")
        bad_indices = [i for i in output_indices if i < 0 or i >= model_cfg["output_size"]]
        if bad_indices:
            raise ValueError(
                f"Cascade source `{name}` output_indices {bad_indices} out of range "
                f"for output_size={model_cfg['output_size']}."
            )

        sources.append(
            CascadeSource(
                name=name,
                model=model,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                input_mean=torch.tensor(np.load(run_dir / "input_mean.npy"), dtype=torch.float32, device=device),
                input_std=torch.tensor(np.load(run_dir / "input_std.npy"), dtype=torch.float32, device=device),
                label_mean=torch.tensor(np.load(run_dir / "label_mean.npy"), dtype=torch.float32, device=device),
                label_std=torch.tensor(np.load(run_dir / "label_std.npy"), dtype=torch.float32, device=device),
                output_indices=output_indices,
                input_size=model_cfg["input_size"],
                output_size=model_cfg["output_size"],
                window_size=model_cfg["window_size"],
            )
        )
        print(
            f"[Cascade] Loaded {name}: outputs={output_indices}, "
            f"window={model_cfg['window_size']}, checkpoint={checkpoint_path}"
        )

    if not sources:
        raise ValueError("use_cascade_inputs=True but no enabled cascade source was loaded.")
    return sources


def _adapt_cascade_window(raw_x: torch.Tensor, target_window_size: int, allow_adapter: bool, source_name: str) -> torch.Tensor:
    current_window_size = int(raw_x.shape[-1])
    target_window_size = int(target_window_size)
    if current_window_size == target_window_size:
        return raw_x
    if not allow_adapter:
        raise ValueError(
            f"Cascade source `{source_name}` expects window_size={target_window_size}, "
            f"but KJL window_size={current_window_size}. Set cascade_allow_window_adapter=True "
            "or use matching upstream checkpoints."
        )
    if current_window_size > target_window_size:
        return raw_x[:, :, -target_window_size:]

    pad_len = target_window_size - current_window_size
    pad = raw_x[:, :, :1].expand(-1, -1, pad_len)
    return torch.cat([pad, raw_x], dim=-1)


def _augment_with_cascade_predictions(
    x: torch.Tensor,
    cascade_sources: list[CascadeSource] | None = None,
    base_input_mean_t: torch.Tensor | None = None,
    base_input_std_t: torch.Tensor | None = None,
    prediction_mode: str = "normalized",
    allow_window_adapter: bool = True,
) -> torch.Tensor:
    if not cascade_sources:
        return x
    if base_input_mean_t is None or base_input_std_t is None:
        raise ValueError("base_input_mean_t/base_input_std_t are required for cascade inputs.")

    prediction_mode = str(prediction_mode).lower()
    if prediction_mode not in {"normalized", "denormalized"}:
        raise ValueError(f"Unsupported cascade_prediction_mode: {prediction_mode}")

    base_mean = base_input_mean_t.reshape(1, -1, 1)
    base_std = base_input_std_t.reshape(1, -1, 1)
    raw_x = x * base_std + base_mean

    extra_features = []
    for source in cascade_sources:
        if source.input_size != raw_x.shape[1]:
            raise ValueError(
                f"Cascade source `{source.name}` expects {source.input_size} IMU channels, "
                f"but KJL loader has {raw_x.shape[1]}. Keep exclude_input_cols consistent "
                "between KJL and upstream models, or retrain upstream models with the same inputs."
            )

        source_raw_x = _adapt_cascade_window(
            raw_x,
            target_window_size=source.window_size,
            allow_adapter=allow_window_adapter,
            source_name=source.name,
        )
        source_x = (source_raw_x - source.input_mean.reshape(1, -1, 1)) / source.input_std.reshape(1, -1, 1)
        pred = source.model(source_x)
        pred = pred[:, source.output_indices]
        if prediction_mode == "denormalized":
            pred = pred * source.label_std.reshape(1, -1)[:, source.output_indices] + source.label_mean.reshape(1, -1)[:, source.output_indices]
        extra_features.append(pred)

    cascade_feat = torch.cat(extra_features, dim=1)
    cascade_seq = cascade_feat.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.cat([x, cascade_seq], dim=1)


def _build_model_input(
    x: torch.Tensor,
    delay_ms: torch.Tensor | None = None,
    use_oracle_delay_input: bool = False,
    use_soft_delay_input: bool = False,
    delay_class_values: list[int] | None = None,
    delay_classifier: DelayClassifier | None = None,
    delay_temperature: float = 1.0,
    cascade_sources: list[CascadeSource] | None = None,
    base_input_mean_t: torch.Tensor | None = None,
    base_input_std_t: torch.Tensor | None = None,
    cascade_prediction_mode: str = "normalized",
    cascade_allow_window_adapter: bool = True,
) -> torch.Tensor:
    x_model = _augment_with_cascade_predictions(
        x,
        cascade_sources=cascade_sources,
        base_input_mean_t=base_input_mean_t,
        base_input_std_t=base_input_std_t,
        prediction_mode=cascade_prediction_mode,
        allow_window_adapter=cascade_allow_window_adapter,
    )

    if use_oracle_delay_input:
        return _augment_with_oracle_delay_onehot(x_model, delay_ms, delay_class_values)

    if use_soft_delay_input:
        if delay_classifier is None:
            return x_model
        with torch.no_grad():
            probs = predict_delay_probs(delay_classifier, x, temperature=delay_temperature)
        probs_seq = probs.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.cat([x_model, probs_seq], dim=1)

    return x_model


def _compute_loss(pred, y, criterion, loss_cfg=None, trial_idx=None, start_idx=None, delay_ms=None):
    loss = criterion(pred, y)
    if loss_cfg and loss_cfg.get("use_corr_loss", False):
        corr_loss = _batch_pearson_corr_loss(
            pred, y, eps=float(loss_cfg.get("corr_loss_eps", 1e-8))
        )
        loss = loss + float(loss_cfg.get("corr_loss_weight", 0.0)) * corr_loss
    if (
        loss_cfg
        and loss_cfg.get("use_deriv_loss", False)
        and trial_idx is not None
        and start_idx is not None
    ):
        deriv_loss = _batch_derivative_loss(
            pred,
            y,
            trial_idx,
            start_idx,
            delay_ms=delay_ms,
            min_delay_ms=int(loss_cfg.get("deriv_loss_min_delay_ms", 200)),
            beta=float(loss_cfg.get("deriv_loss_beta", 1.0)),
        )
        loss = loss + float(loss_cfg.get("deriv_loss_weight", 0.0)) * deriv_loss
    if (
        loss_cfg
        and loss_cfg.get("use_chunk_shape_loss", False)
        and trial_idx is not None
        and start_idx is not None
    ):
        chunk_shape_loss = _batch_chunk_shape_loss(
            pred,
            y,
            trial_idx,
            start_idx,
            delay_ms=delay_ms,
            chunk_size=int(loss_cfg.get("chunk_size", 32)),
            chunk_stride=int(loss_cfg.get("chunk_stride", 16)),
            min_delay_ms=int(loss_cfg.get("chunk_min_delay_ms", 200)),
            corr_weight=float(loss_cfg.get("chunk_corr_weight", 1.0)),
            fft_weight=float(loss_cfg.get("chunk_fft_weight", 0.5)),
            fft_bins=int(loss_cfg.get("chunk_fft_bins", 8)),
            fft_beta=float(loss_cfg.get("chunk_fft_beta", 1.0)),
            eps=float(loss_cfg.get("chunk_shape_eps", 1e-8)),
        )
        loss = loss + float(loss_cfg.get("chunk_shape_weight", 0.0)) * chunk_shape_loss
    return loss


def _eval_epoch(
    model,
    loader,
    criterion,
    device,
    label_mean_t,
    label_std_t,
    loss_cfg=None,
    use_oracle_delay_input: bool = False,
    use_soft_delay_input: bool = False,
    delay_class_values: list[int] | None = None,
    delay_classifier: DelayClassifier | None = None,
    delay_temperature: float = 1.0,
    cascade_sources: list[CascadeSource] | None = None,
    base_input_mean_t: torch.Tensor | None = None,
    base_input_std_t: torch.Tensor | None = None,
    cascade_prediction_mode: str = "normalized",
    cascade_allow_window_adapter: bool = True,
):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    skipped_nonfinite = 0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Eval", dynamic_ncols=True):
            x, y = batch[:2]
            trial_idx = batch[2].to(device) if len(batch) > 2 else None
            start_idx = batch[3].to(device) if len(batch) > 3 else None
            delay_ms = batch[4].to(device) if len(batch) > 4 else None
            x = x.to(device)
            y = y.to(device)
            x_model = _build_model_input(
                x,
                delay_ms=delay_ms,
                use_oracle_delay_input=use_oracle_delay_input,
                use_soft_delay_input=use_soft_delay_input,
                delay_class_values=delay_class_values,
                delay_classifier=delay_classifier,
                delay_temperature=delay_temperature,
                cascade_sources=cascade_sources,
                base_input_mean_t=base_input_mean_t,
                base_input_std_t=base_input_std_t,
                cascade_prediction_mode=cascade_prediction_mode,
                cascade_allow_window_adapter=cascade_allow_window_adapter,
            )
            pred = model(x_model)
            if not torch.isfinite(pred).all():
                skipped_nonfinite += 1
                continue
            loss = _compute_loss(
                pred, y, criterion, loss_cfg=loss_cfg,
                trial_idx=trial_idx, start_idx=start_idx, delay_ms=delay_ms
            )
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue

            pred_denorm = pred * label_std_t + label_mean_t
            y_denorm = y * label_std_t + label_mean_t
            rmse = torch.sqrt(torch.mean((pred_denorm - y_denorm) ** 2))

            total_loss += loss.item()
            total_rmse += rmse.item()
            n_batches += 1
            y_true_all.append(y_denorm.detach().cpu().numpy())
            y_pred_all.append(pred_denorm.detach().cpu().numpy())

    out = {
        "loss": float("nan") if n_batches == 0 else total_loss / n_batches,
        "rmse": float("nan") if n_batches == 0 else total_rmse / n_batches,
        "num_batches": n_batches,
        "skipped_nonfinite": skipped_nonfinite,
    }
    if y_true_all:
        out.update(_regression_metrics(np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)))
    else:
        out.update({"r2": float("nan"), "pearson_r": float("nan"), "nrmse_pct": float("nan")})
    return out


def _train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    label_mean_t,
    label_std_t,
    loss_cfg=None,
    use_oracle_delay_input: bool = False,
    use_soft_delay_input: bool = False,
    delay_class_values: list[int] | None = None,
    delay_classifier: DelayClassifier | None = None,
    delay_temperature: float = 1.0,
    cascade_sources: list[CascadeSource] | None = None,
    base_input_mean_t: torch.Tensor | None = None,
    base_input_std_t: torch.Tensor | None = None,
    cascade_prediction_mode: str = "normalized",
    cascade_allow_window_adapter: bool = True,
):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    skipped_nonfinite = 0

    for batch in tqdm(loader, leave=False, desc="Train", dynamic_ncols=True):
        x, y = batch[:2]
        trial_idx = batch[2].to(device) if len(batch) > 2 else None
        start_idx = batch[3].to(device) if len(batch) > 3 else None
        delay_ms = batch[4].to(device) if len(batch) > 4 else None
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        x_model = _build_model_input(
            x,
            delay_ms=delay_ms,
            use_oracle_delay_input=use_oracle_delay_input,
            use_soft_delay_input=use_soft_delay_input,
            delay_class_values=delay_class_values,
            delay_classifier=delay_classifier,
            delay_temperature=delay_temperature,
            cascade_sources=cascade_sources,
            base_input_mean_t=base_input_mean_t,
            base_input_std_t=base_input_std_t,
            cascade_prediction_mode=cascade_prediction_mode,
            cascade_allow_window_adapter=cascade_allow_window_adapter,
        )
        pred = model(x_model)
        if not torch.isfinite(pred).all():
            skipped_nonfinite += 1
            continue
        loss = _compute_loss(
            pred, y, criterion, loss_cfg=loss_cfg,
            trial_idx=trial_idx, start_idx=start_idx, delay_ms=delay_ms
        )
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred_denorm = pred * label_std_t + label_mean_t
            y_denorm = y * label_std_t + label_mean_t
            rmse = torch.sqrt(torch.mean((pred_denorm - y_denorm) ** 2))

        total_loss += loss.item()
        total_rmse += rmse.item()
        n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan"), {"num_batches": 0, "skipped_nonfinite": skipped_nonfinite}
    return total_loss / n_batches, total_rmse / n_batches, {"num_batches": n_batches, "skipped_nonfinite": skipped_nonfinite}


def _eval_per_trial(
    model,
    test_dataset,
    criterion,
    device,
    label_mean_t,
    label_std_t,
    batch_size,
    num_workers=0,
    save_dir=None,
    loss_cfg=None,
    use_oracle_delay_input: bool = False,
    use_soft_delay_input: bool = False,
    delay_class_values: list[int] | None = None,
    delay_classifier: DelayClassifier | None = None,
    delay_temperature: float = 1.0,
    cascade_sources: list[CascadeSource] | None = None,
    base_input_mean_t: torch.Tensor | None = None,
    base_input_std_t: torch.Tensor | None = None,
    cascade_prediction_mode: str = "normalized",
    cascade_allow_window_adapter: bool = True,
):
    """Evaluate the best model on each test trial individually and return per-trial metrics.

    If save_dir is provided, saves per-trial predictions as preds_{condition}.npz
    containing arrays 'y_pred' and 'y_true' (both denormalized, shape [N]).
    """
    results = []
    for tdir in test_dataset.trial_dirs:
        rng = test_dataset.row_ranges.get(str(tdir), (0, None))
        try:
            single_ds = WindowedTrialDataset(
                [tdir],
                window_size=test_dataset.window_size,
                input_mean=test_dataset.input_mean,
                input_std=test_dataset.input_std,
                label_mean=test_dataset.label_mean,
                label_std=test_dataset.label_std,
                target_col=test_dataset.target_col,
                row_ranges={str(tdir): rng},
                exclude_feature_cols=getattr(test_dataset, "exclude_feature_cols", None),
            )
        except ValueError:
            continue
        loader = DataLoader(single_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        metrics = _eval_epoch(
            model,
            loader,
            criterion,
            device,
            label_mean_t,
            label_std_t,
            loss_cfg=loss_cfg,
            use_oracle_delay_input=use_oracle_delay_input,
            use_soft_delay_input=use_soft_delay_input,
            delay_class_values=delay_class_values,
            delay_classifier=delay_classifier,
            delay_temperature=delay_temperature,
            cascade_sources=cascade_sources,
            base_input_mean_t=base_input_mean_t,
            base_input_std_t=base_input_std_t,
            cascade_prediction_mode=cascade_prediction_mode,
            cascade_allow_window_adapter=cascade_allow_window_adapter,
        )
        cond = Path(str(tdir)).parent.name  # e.g. "NoExo", "30p250ms"
        tag = f"{cond} [last 30%]" if rng[0] is not None and rng[0] > 0 else cond
        results.append({"trial": str(tdir), "condition": cond, "row_range": list(rng), **metrics})
        print(
            f"  {tag:22s}  R2={metrics['r2']:+.3f}  r={metrics['pearson_r']:.3f}"
            f"  nRMSE={metrics['nrmse_pct']:.2f}%  rmse={metrics['rmse']:.4f}"
        )
        if save_dir is not None:
            model.eval()
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for batch in loader:
                    x_batch = batch[0].to(device)
                    y_batch = batch[1].to(device)
                    delay_ms_batch = batch[4].to(device) if len(batch) > 4 else None
                    x_model = _build_model_input(
                        x_batch,
                        delay_ms=delay_ms_batch,
                        use_oracle_delay_input=use_oracle_delay_input,
                        use_soft_delay_input=use_soft_delay_input,
                        delay_class_values=delay_class_values,
                        delay_classifier=delay_classifier,
                        delay_temperature=delay_temperature,
                        cascade_sources=cascade_sources,
                        base_input_mean_t=base_input_mean_t,
                        base_input_std_t=base_input_std_t,
                        cascade_prediction_mode=cascade_prediction_mode,
                        cascade_allow_window_adapter=cascade_allow_window_adapter,
                    )
                    pred = model(x_model)
                    pred_denorm = (pred * label_std_t + label_mean_t).cpu().numpy()
                    y_denorm = (y_batch * label_std_t + label_mean_t).cpu().numpy()
                    y_pred_list.append(pred_denorm.reshape(-1))
                    y_true_list.append(y_denorm.reshape(-1))
            np.savez(
                Path(save_dir) / f"preds_{cond}.npz",
                y_pred=np.concatenate(y_pred_list),
                y_true=np.concatenate(y_true_list),
            )
    return results


def _make_run_name(cfg):
    channels_tag = "x".join(map(str, cfg["num_channels"]))
    loss_tag = cfg["loss_type"]
    if cfg["loss_type"].lower() == "huber":
        loss_tag = f"huber_b{cfg['huber_beta']}"
    corr_tag = ""
    if cfg.get("use_corr_loss", False):
        corr_w = str(cfg.get("corr_loss_weight", 0.0)).replace(".", "p")
        corr_tag = f"_corrw{corr_w}"
    deriv_tag = ""
    if cfg.get("use_deriv_loss", False):
        deriv_w = str(cfg.get("deriv_loss_weight", 0.0)).replace(".", "p")
        deriv_tag = f"_derivw{deriv_w}"
    wave_tag = ""
    if cfg.get("use_chunk_shape_loss", False):
        wave_w = str(cfg.get("chunk_shape_weight", 0.0)).replace(".", "p")
        wave_tag = f"_wavew{wave_w}"
    oracle_tag = "_oracleDelay" if cfg.get("use_oracle_delay_input", False) else ""
    soft_tag = "_softDelay" if cfg.get("use_soft_delay_input", False) else ""
    cascade_tag = "_cascade" if cfg.get("use_cascade_inputs", False) else ""
    drop_tag = ""
    raw_drop_cols = cfg.get("exclude_input_cols", [])
    if raw_drop_cols is None:
        raw_drop_cols = []
    elif isinstance(raw_drop_cols, str):
        raw_drop_cols = [raw_drop_cols]
    drop_cols = [str(c) for c in raw_drop_cols if c is not None and str(c)]
    if drop_cols:
        short_cols = [c.replace("_", "") for c in drop_cols[:3]]
        extra = f"+{len(drop_cols) - 3}" if len(drop_cols) > 3 else ""
        drop_tag = f"_drop{'-'.join(short_cols)}{extra}"
    return (
        f"{cfg['run_name']}_seed{cfg['seed']}"
        f"_w{cfg['window_size']}_bs{cfg['batch_size']}"
        f"_do{str(cfg['dropout']).replace('.', 'p')}"
        f"_{loss_tag}{corr_tag}{deriv_tag}{wave_tag}{oracle_tag}{soft_tag}{drop_tag}_ch{channels_tag}"
        f"{cascade_tag}"
    )


def _build_criterion(cfg):
    loss_type = str(cfg.get("loss_type", "mse")).lower()
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type in {"huber", "smoothl1", "smooth_l1"}:
        return torch.nn.SmoothL1Loss(beta=float(cfg.get("huber_beta", 1.0)))
    raise ValueError(f"Unsupported loss_type: {cfg['loss_type']}")


def train(cfg_override=None):
    cfg = copy.deepcopy(CONFIG)
    if cfg_override:
        cfg.update(cfg_override)

    # Normalize optional feature exclusion field so downstream logic can safely
    # assume a list-like container.
    exclude_input_cols = cfg.get("exclude_input_cols", [])
    if exclude_input_cols is None:
        exclude_input_cols = []
    elif isinstance(exclude_input_cols, str):
        exclude_input_cols = [exclude_input_cols]
    else:
        exclude_input_cols = [str(c) for c in exclude_input_cols if c is not None and str(c)]
    cfg["exclude_input_cols"] = exclude_input_cols

    set_seed(cfg["seed"])

    cfg["run_name"] = _make_run_name(cfg)
    out_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_kjl_ab03_dataloaders(
        dataset_root=cfg["dataset_root"],
        window_size=cfg["window_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        split_json=cfg["split_json"],
        target_col=cfg["target_col"],
        noexo_split_ratio=cfg.get("noexo_split_ratio", 0.7),
        trial_balanced_sampling=cfg.get("trial_balanced_sampling", True),
        delay_weight_map=cfg.get("delay_weight_map", {}),
        noexo_train_weight=cfg.get("noexo_train_weight", 1.0),
        noassi_train_weight=cfg.get("noassi_train_weight", 1.0),
        default_train_weight=cfg.get("default_train_weight", 1.0),
        sequential_train_batches=bool(
            cfg.get("use_deriv_loss", False) or cfg.get("use_chunk_shape_loss", False)
        ),
        exclude_feature_cols=cfg["exclude_input_cols"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_val = data.val_loader is not None
    print(f"Device: {device}")
    print(f"Train/Val/Test trials: {len(data.train_trials)}/{len(data.val_trials)}/{len(data.test_trials)}")
    print(f"Train/Val/Test windows: {len(data.train_dataset)}/{len(data.val_dataset) if has_val else 0}/{len(data.test_dataset)}")
    if cfg["exclude_input_cols"]:
        print(f"Excluded input cols: {cfg['exclude_input_cols']}")
    print(f"Input feature count: {data.input_size}")

    base_input_mean_t = torch.tensor(data.input_mean, dtype=torch.float32, device=device)
    base_input_std_t = torch.tensor(data.input_std, dtype=torch.float32, device=device)

    cascade_sources = _load_cascade_sources(cfg, device)
    cascade_feature_dim = sum(source.feature_dim for source in cascade_sources)
    if cascade_sources:
        print(
            f"Cascade input sources: {[source.name for source in cascade_sources]} "
            f"(extra channels={cascade_feature_dim}, mode={cfg.get('cascade_prediction_mode', 'normalized')})"
        )

    use_oracle_delay_input = bool(cfg.get("use_oracle_delay_input", False))
    use_soft_delay_input = bool(cfg.get("use_soft_delay_input", False))
    if use_oracle_delay_input and use_soft_delay_input:
        raise ValueError("Only one of use_oracle_delay_input/use_soft_delay_input can be True.")

    delay_class_values: list[int] = []
    delay_cond_dim = 0
    delay_classifier = None
    delay_clf_summary = None
    if use_oracle_delay_input or use_soft_delay_input:
        delay_class_values = _collect_delay_class_values(data)
        delay_cond_dim = len(delay_class_values)
        print(f"Delay conditioning classes: {delay_class_values}")
        if use_soft_delay_input:
            delay_classifier = DelayClassifier(
                input_channels=data.input_size,
                window_size=cfg["window_size"],
                num_classes=delay_cond_dim,
                hidden_dim=int(cfg.get("delay_clf_hidden_dim", 128)),
                dropout=float(cfg.get("delay_clf_dropout", 0.1)),
            ).to(device)
            delay_clf_summary = train_delay_classifier(
                delay_classifier,
                train_loader=data.train_loader,
                val_loader=data.val_loader if has_val else data.train_loader,
                class_values=delay_class_values,
                device=device,
                epochs=int(cfg.get("delay_clf_epochs", 12)),
                lr=float(cfg.get("delay_clf_lr", 1e-3)),
                weight_decay=float(cfg.get("delay_clf_weight_decay", 1e-4)),
            )
            print(
                f"[DelayClf] best_epoch={delay_clf_summary.best_epoch} "
                f"best_val_acc={delay_clf_summary.best_val_acc:.3f}"
            )
            delay_classifier.eval()

    model_cfg = {
        "input_size": data.input_size + delay_cond_dim + cascade_feature_dim,
        "output_size": 1,
        "dropout": cfg["dropout"],
        "number_of_layers": cfg["number_of_layers"],
        "num_channels": cfg["num_channels"],
        "kernel_size": cfg["kernel_size"],
        "dilations": cfg["dilations"],
        "window_size": cfg["window_size"],
    }
    model = TCNModel(model_cfg).to(device)

    criterion = _build_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    label_mean_t = torch.tensor(data.label_mean, dtype=torch.float32, device=device)
    label_std_t = torch.tensor(data.label_std, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        train_loss, train_rmse, train_stats = _train_epoch(
            model, data.train_loader, criterion, optimizer, device, label_mean_t, label_std_t,
            loss_cfg=cfg,
            use_oracle_delay_input=use_oracle_delay_input,
            use_soft_delay_input=use_soft_delay_input,
            delay_class_values=delay_class_values,
            delay_classifier=delay_classifier,
            delay_temperature=float(cfg.get("delay_softmax_temperature", 1.0)),
            cascade_sources=cascade_sources,
            base_input_mean_t=base_input_mean_t,
            base_input_std_t=base_input_std_t,
            cascade_prediction_mode=cfg.get("cascade_prediction_mode", "normalized"),
            cascade_allow_window_adapter=bool(cfg.get("cascade_allow_window_adapter", True)),
        )
        if has_val:
            val_metrics = _eval_epoch(
                model, data.val_loader, criterion, device, label_mean_t, label_std_t,
                loss_cfg=cfg,
                use_oracle_delay_input=use_oracle_delay_input,
                use_soft_delay_input=use_soft_delay_input,
                delay_class_values=delay_class_values,
                delay_classifier=delay_classifier,
                delay_temperature=float(cfg.get("delay_softmax_temperature", 1.0)),
                cascade_sources=cascade_sources,
                base_input_mean_t=base_input_mean_t,
                base_input_std_t=base_input_std_t,
                cascade_prediction_mode=cfg.get("cascade_prediction_mode", "normalized"),
                cascade_allow_window_adapter=bool(cfg.get("cascade_allow_window_adapter", True)),
            )
        else:
            val_metrics = {"loss": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                           "pearson_r": float("nan"), "nrmse_pct": float("nan"),
                           "num_batches": 1, "skipped_nonfinite": 0}
        test_metrics = _eval_epoch(
            model, data.test_loader, criterion, device, label_mean_t, label_std_t,
            loss_cfg=cfg,
            use_oracle_delay_input=use_oracle_delay_input,
            use_soft_delay_input=use_soft_delay_input,
            delay_class_values=delay_class_values,
            delay_classifier=delay_classifier,
            delay_temperature=float(cfg.get("delay_softmax_temperature", 1.0)),
            cascade_sources=cascade_sources,
            base_input_mean_t=base_input_mean_t,
            base_input_std_t=base_input_std_t,
            cascade_prediction_mode=cfg.get("cascade_prediction_mode", "normalized"),
            cascade_allow_window_adapter=bool(cfg.get("cascade_allow_window_adapter", True)),
        )
        val_loss, val_rmse = val_metrics["loss"], val_metrics["rmse"]
        test_loss, test_rmse = test_metrics["loss"], test_metrics["rmse"]

        # If non-finite values dominate and no valid batches remain, stop safely without corrupting best checkpoint selection.
        if (
            not np.isfinite(train_loss)
            or not np.isfinite(test_loss)
            or train_stats["num_batches"] == 0
            or test_metrics["num_batches"] == 0
            or (has_val and (not np.isfinite(val_loss) or val_metrics["num_batches"] == 0))
        ):
            print(
                "Non-finite training detected; stopping this run without updating best checkpoint. "
                f"(train valid/skipped: {train_stats['num_batches']}/{train_stats['skipped_nonfinite']}, "
                f"val valid/skipped: {val_metrics['num_batches']}/{val_metrics['skipped_nonfinite']}, "
                f"test valid/skipped: {test_metrics['num_batches']}/{test_metrics['skipped_nonfinite']})"
            )
            break

        if has_val:
            scheduler.step(val_loss)

        print(
            f"train_loss={train_loss:.4f} train_rmse={train_rmse:.4f} | "
            f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f} val_R2={val_metrics['r2']:.3f} val_r={val_metrics['pearson_r']:.3f} val_nRMSE={val_metrics['nrmse_pct']:.2f}% | "
            f"test_rmse={test_rmse:.4f} test_R2={test_metrics['r2']:.3f} test_r={test_metrics['pearson_r']:.3f} test_nRMSE={test_metrics['nrmse_pct']:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"[valid/skipped train={train_stats['num_batches']}/{train_stats['skipped_nonfinite']}, "
            f"val={val_metrics['num_batches']}/{val_metrics['skipped_nonfinite']}, "
            f"test={test_metrics['num_batches']}/{test_metrics['skipped_nonfinite']}]"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_r2": val_metrics["r2"],
                "val_pearson_r": val_metrics["pearson_r"],
                "val_nrmse_pct": val_metrics["nrmse_pct"],
                "test_loss": test_loss,
                "test_rmse": test_rmse,
                "test_r2": test_metrics["r2"],
                "test_pearson_r": test_metrics["pearson_r"],
                "test_nrmse_pct": test_metrics["nrmse_pct"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_num_batches": train_stats["num_batches"],
                "train_skipped_nonfinite": train_stats["skipped_nonfinite"],
                "val_num_batches": val_metrics["num_batches"],
                "val_skipped_nonfinite": val_metrics["skipped_nonfinite"],
                "test_num_batches": test_metrics["num_batches"],
                "test_skipped_nonfinite": test_metrics["skipped_nonfinite"],
            }
        )

        torch.save(model.state_dict(), out_dir / f"{cfg['run_name']}_epoch_{epoch}.pt")

        if has_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), out_dir / f"{cfg['run_name']}.pt")
            else:
                patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print("Early stopping triggered.")
                break

    # Final evaluation: use last epoch OR best-val checkpoint
    if cfg.get("use_last_epoch", False):
        # Model is already at last epoch state; also save it as the "final" checkpoint
        torch.save(model.state_dict(), out_dir / f"{cfg['run_name']}.pt")
        last_epoch = history[-1]["epoch"] if history else cfg["epochs"]
        print(f"[use_last_epoch=True] Using model from epoch {last_epoch} for final evaluation.")
    else:
        best_model_path = out_dir / f"{cfg['run_name']}.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_test_metrics = _eval_epoch(
        model, data.test_loader, criterion, device, label_mean_t, label_std_t,
        loss_cfg=cfg,
        use_oracle_delay_input=use_oracle_delay_input,
        use_soft_delay_input=use_soft_delay_input,
        delay_class_values=delay_class_values,
        delay_classifier=delay_classifier,
        delay_temperature=float(cfg.get("delay_softmax_temperature", 1.0)),
        cascade_sources=cascade_sources,
        base_input_mean_t=base_input_mean_t,
        base_input_std_t=base_input_std_t,
        cascade_prediction_mode=cfg.get("cascade_prediction_mode", "normalized"),
        cascade_allow_window_adapter=bool(cfg.get("cascade_allow_window_adapter", True)),
    )
    final_test_loss, final_test_rmse = final_test_metrics["loss"], final_test_metrics["rmse"]
    print(
        f"\nBest model test_loss={final_test_loss:.4f}, test_rmse={final_test_rmse:.4f}, "
        f"test_R2={final_test_metrics['r2']:.3f}, test_r={final_test_metrics['pearson_r']:.3f}, "
        f"test_nRMSE={final_test_metrics['nrmse_pct']:.2f}%"
    )

    print("\nPer-trial test breakdown:")
    per_trial_metrics = _eval_per_trial(
        model, data.test_dataset, criterion, device, label_mean_t, label_std_t,
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        save_dir=out_dir, loss_cfg=cfg,
        use_oracle_delay_input=use_oracle_delay_input,
        use_soft_delay_input=use_soft_delay_input,
        delay_class_values=delay_class_values,
        delay_classifier=delay_classifier,
        delay_temperature=float(cfg.get("delay_softmax_temperature", 1.0)),
        cascade_sources=cascade_sources,
        base_input_mean_t=base_input_mean_t,
        base_input_std_t=base_input_std_t,
        cascade_prediction_mode=cfg.get("cascade_prediction_mode", "normalized"),
        cascade_allow_window_adapter=bool(cfg.get("cascade_allow_window_adapter", True)),
    )
    with open(out_dir / "per_trial_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_trial_metrics, f, ensure_ascii=True, indent=2)

    np.save(out_dir / "input_mean.npy", data.input_mean)
    np.save(out_dir / "input_std.npy", data.input_std)
    np.save(out_dir / "label_mean.npy", data.label_mean)
    np.save(out_dir / "label_std.npy", data.label_std)

    with open(out_dir / "split_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_trials": data.train_trials,
                "val_trials": data.val_trials,
                "test_trials": data.test_trials,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg, **model_cfg}, f, ensure_ascii=True, indent=2)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    with open(out_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_metrics, f, ensure_ascii=True, indent=2)

    if use_oracle_delay_input or use_soft_delay_input:
        delay_info = {
            "delay_class_values": delay_class_values,
            "use_oracle_delay_input": use_oracle_delay_input,
            "use_soft_delay_input": use_soft_delay_input,
        }
        if delay_clf_summary is not None:
            delay_info["delay_classifier"] = {
                "best_epoch": delay_clf_summary.best_epoch,
                "best_val_acc": delay_clf_summary.best_val_acc,
                "last_val_acc": delay_clf_summary.last_val_acc,
                "train_acc": delay_clf_summary.train_acc,
                "train_loss": delay_clf_summary.train_loss,
            }
        with open(out_dir / "delay_conditioning.json", "w", encoding="utf-8") as f:
            json.dump(delay_info, f, ensure_ascii=True, indent=2)

    if cascade_sources:
        cascade_info = {
            "use_cascade_inputs": True,
            "cascade_prediction_mode": cfg.get("cascade_prediction_mode", "normalized"),
            "cascade_allow_window_adapter": bool(cfg.get("cascade_allow_window_adapter", True)),
            "sources": [
                {
                    "name": source.name,
                    "run_dir": str(source.run_dir),
                    "checkpoint_path": str(source.checkpoint_path),
                    "output_indices": source.output_indices,
                    "input_size": source.input_size,
                    "output_size": source.output_size,
                    "window_size": source.window_size,
                }
                for source in cascade_sources
            ],
        }
        with open(out_dir / "cascade_sources.json", "w", encoding="utf-8") as f:
            json.dump(cascade_info, f, ensure_ascii=True, indent=2)

    print(f"Saved outputs to: {out_dir}")
    return {
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "window_size": cfg["window_size"],
        "batch_size": cfg["batch_size"],
        "dropout": cfg["dropout"],
        "loss_type": cfg["loss_type"],
        "huber_beta": cfg.get("huber_beta", None),
        "use_corr_loss": bool(cfg.get("use_corr_loss", False)),
        "corr_loss_weight": float(cfg.get("corr_loss_weight", 0.0)),
        "use_deriv_loss": bool(cfg.get("use_deriv_loss", False)),
        "deriv_loss_weight": float(cfg.get("deriv_loss_weight", 0.0)),
        "deriv_loss_min_delay_ms": int(cfg.get("deriv_loss_min_delay_ms", 200)),
        "use_chunk_shape_loss": bool(cfg.get("use_chunk_shape_loss", False)),
        "chunk_shape_weight": float(cfg.get("chunk_shape_weight", 0.0)),
        "chunk_size": int(cfg.get("chunk_size", 32)),
        "chunk_stride": int(cfg.get("chunk_stride", 16)),
        "chunk_min_delay_ms": int(cfg.get("chunk_min_delay_ms", 200)),
        "chunk_corr_weight": float(cfg.get("chunk_corr_weight", 1.0)),
        "chunk_fft_weight": float(cfg.get("chunk_fft_weight", 0.5)),
        "chunk_fft_bins": int(cfg.get("chunk_fft_bins", 8)),
        "chunk_fft_beta": float(cfg.get("chunk_fft_beta", 1.0)),
        "use_oracle_delay_input": use_oracle_delay_input,
        "use_soft_delay_input": use_soft_delay_input,
        "delay_class_values": json.dumps(delay_class_values),
        "delay_clf_best_val_acc": (
            float(delay_clf_summary.best_val_acc) if delay_clf_summary is not None else float("nan")
        ),
        "use_cascade_inputs": bool(cascade_sources),
        "cascade_sources": json.dumps([source.name for source in cascade_sources]),
        "cascade_feature_dim": cascade_feature_dim,
        "cascade_prediction_mode": cfg.get("cascade_prediction_mode", "normalized"),
        "exclude_input_cols": json.dumps(cfg["exclude_input_cols"]),
        "target_col": cfg["target_col"],
        "num_channels": cfg["num_channels"],
        "final_test_loss": final_test_metrics["loss"],
        "final_test_rmse": final_test_metrics["rmse"],
        "final_test_r2": final_test_metrics["r2"],
        "final_test_pearson_r": final_test_metrics["pearson_r"],
        "final_test_nrmse_pct": final_test_metrics["nrmse_pct"],
        "out_dir": str(out_dir),
    }


def run_multi_seed():
    base_cfg = copy.deepcopy(CONFIG)
    seeds = base_cfg.get("seeds", None)
    if not seeds:
        seeds = [base_cfg.get("seed", 42)]

    root_out = Path(base_cfg["output_dir"])
    root_out.mkdir(parents=True, exist_ok=True)
    summary_csv = root_out / "summary.csv"

    results = []
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Starting seed {seed}")
        print("=" * 80)
        result = train({"seed": int(seed)})
        results.append(result)

        write_header = not summary_csv.exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)

    # Aggregate quick summary to stdout
    if results:
        rmse = np.array([r["final_test_rmse"] for r in results], dtype=float)
        r2 = np.array([r["final_test_r2"] for r in results], dtype=float)
        pr = np.array([r["final_test_pearson_r"] for r in results], dtype=float)
        nrmse = np.array([r["final_test_nrmse_pct"] for r in results], dtype=float)
        print("\n" + "=" * 80)
        print("Multi-seed summary")
        print(f"Seeds: {seeds}")
        print(f"RMSE mean±std: {np.nanmean(rmse):.4f} ± {np.nanstd(rmse):.4f}")
        print(f"R2   mean±std: {np.nanmean(r2):.4f} ± {np.nanstd(r2):.4f}")
        print(f"r    mean±std: {np.nanmean(pr):.4f} ± {np.nanstd(pr):.4f}")
        print(f"nRMSE mean±std: {np.nanmean(nrmse):.4f}% ± {np.nanstd(nrmse):.4f}%")
        print(f"Summary CSV: {summary_csv}")
        print("=" * 80)


if __name__ == "__main__":
    run_multi_seed()
