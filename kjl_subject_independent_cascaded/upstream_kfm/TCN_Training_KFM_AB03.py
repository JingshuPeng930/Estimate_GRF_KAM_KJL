import csv
import copy
import json
import os
from pathlib import Path

# In restricted environments without /dev/shm, Intel OpenMP may fail at runtime.
os.environ.setdefault("KMP_USE_SHM", "0")

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from TCN_Header_Model import TCNModel
from kfm_ab03_tcn_dataset import build_kfm_ab03_dataloaders, WindowedTrialDataset


CONFIG = {
    "run_name": "KFM_AB03_Amy_TCN_IMU",
    "seeds": [42, 123, 2026, 7, 99],
    "dataset_root": "training_code_IMUonly_KFM/data_kfm_ab03_id",
    "exclude_input_cols": [],
    "target_col": "kfm_bwbh",
    "split_json": None,
    "overlap_split_ratio": 0.7,
    "trial_balanced_sampling": True,
    "use_label_filter": True,
    "label_filter_cutoff_hz": 15.0,
    "label_filter_order": 4,
    "label_filter_fs_hz": 100.0,
    "seed": 42,
    "window_size": 150,
    "batch_size": 32,
    "num_workers": 0,
    "epochs": 30,
    "lr": 1e-5,
    "weight_decay": 1e-5,
    "dropout": 0.15,
    "loss_type": "huber",  # "mse" or "huber"
    "huber_beta": 10.0,
    "number_of_layers": 2,
    "num_channels": [32, 32, 32, 32],
    "kernel_size": 5,
    "dilations": [1, 2, 4, 8, 16],
    "patience": 4,
    "max_grad_norm": 1.0,
    "max_epoch_explosions": 3,
    "explosion_lr_decay": 0.5,
    "min_lr": 1e-7,
    "max_train_skipped_ratio": 0.2,
    "use_last_epoch": False,
    "output_dir": "training_code_IMUonly_KFM/runs_kfm_ab03_huber",
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


def _build_criterion(cfg):
    loss_type = str(cfg.get("loss_type", "mse")).lower()
    if loss_type == "mse":
        return torch.nn.MSELoss()
    if loss_type in {"huber", "smoothl1", "smooth_l1"}:
        return torch.nn.SmoothL1Loss(beta=float(cfg.get("huber_beta", 1.0)))
    raise ValueError(f"Unsupported loss_type: {cfg['loss_type']}")


def _eval_epoch(model, loader, criterion, device, label_mean_t, label_std_t):
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
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            if not torch.isfinite(pred).all():
                skipped_nonfinite += 1
                continue

            loss = criterion(pred, y)
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


def _train_epoch(model, loader, criterion, optimizer, device, label_mean_t, label_std_t, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    skipped_nonfinite = 0

    for batch in tqdm(loader, leave=False, desc="Train", dynamic_ncols=True):
        x, y = batch[:2]
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        if not torch.isfinite(pred).all():
            skipped_nonfinite += 1
            continue

        loss = criterion(pred, y)
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            continue

        loss.backward()

        grads_finite = True
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                grads_finite = False
                break
        if not grads_finite:
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
        if not torch.isfinite(grad_norm):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        params_finite = True
        for p in model.parameters():
            if not torch.isfinite(p).all():
                params_finite = False
                break
        if not params_finite:
            return float("nan"), float("nan"), {
                "num_batches": n_batches,
                "total_batches": n_batches + skipped_nonfinite + 1,
                "skipped_nonfinite": skipped_nonfinite,
                "exploded": True,
                "explode_reason": "param_nonfinite_after_step",
            }

        with torch.no_grad():
            pred_denorm = pred * label_std_t + label_mean_t
            y_denorm = y * label_std_t + label_mean_t
            rmse = torch.sqrt(torch.mean((pred_denorm - y_denorm) ** 2))

        total_loss += loss.item()
        total_rmse += rmse.item()
        n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan"), {
            "num_batches": 0,
            "total_batches": skipped_nonfinite,
            "skipped_nonfinite": skipped_nonfinite,
            "exploded": False,
            "explode_reason": None,
        }
    return total_loss / n_batches, total_rmse / n_batches, {
        "num_batches": n_batches,
        "total_batches": n_batches + skipped_nonfinite,
        "skipped_nonfinite": skipped_nonfinite,
        "exploded": False,
        "explode_reason": None,
    }


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
):
    results = []
    for tdir in test_dataset.kept_trial_dirs:
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
                label_filter_cutoff_hz=getattr(test_dataset, "label_filter_cutoff_hz", None),
                label_filter_order=getattr(test_dataset, "label_filter_order", 4),
                label_filter_fs_hz=getattr(test_dataset, "label_filter_fs_hz", 100.0),
            )
        except ValueError:
            continue

        loader = DataLoader(single_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        metrics = _eval_epoch(model, loader, criterion, device, label_mean_t, label_std_t)

        cond = Path(str(tdir)).parent.name
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
                    pred = model(x_batch)
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

    if bool(cfg.get("use_label_filter", True)):
        cutoff = cfg.get("label_filter_cutoff_hz", 15.0)
        lpf_tag = f"_lpf{str(cutoff).replace('.', 'p')}"
    else:
        lpf_tag = "_lpfOff"

    safe_target = str(cfg["target_col"]).replace("_norm_bw", "NBW")
    return (
        f"{cfg['run_name']}_seed{cfg['seed']}"
        f"_{safe_target}"
        f"_w{cfg['window_size']}_bs{cfg['batch_size']}"
        f"_do{str(cfg['dropout']).replace('.', 'p')}"
        f"_{loss_tag}{drop_tag}{lpf_tag}_ch{channels_tag}"
    )


def train(cfg_override=None):
    cfg = CONFIG.copy()
    if cfg_override:
        cfg.update(cfg_override)

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

    data = build_kfm_ab03_dataloaders(
        dataset_root=cfg["dataset_root"],
        window_size=cfg["window_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        split_json=cfg["split_json"],
        target_col=cfg["target_col"],
        trial_balanced_sampling=cfg.get("trial_balanced_sampling", True),
        overlap_split_ratio=cfg.get("overlap_split_ratio", 0.7),
        exclude_feature_cols=cfg["exclude_input_cols"],
        label_filter_cutoff_hz=(
            cfg.get("label_filter_cutoff_hz", 15.0) if cfg.get("use_label_filter", True) else None
        ),
        label_filter_order=int(cfg.get("label_filter_order", 4)),
        label_filter_fs_hz=float(cfg.get("label_filter_fs_hz", 100.0)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_val = data.val_loader is not None

    print(f"Device: {device}")
    print(f"Train/Val/Test trials: {len(data.train_trials)}/{len(data.val_trials)}/{len(data.test_trials)}")
    print(
        f"Train/Val/Test windows: {len(data.train_dataset)}/"
        f"{len(data.val_dataset) if has_val else 0}/{len(data.test_dataset)}"
    )
    if cfg["exclude_input_cols"]:
        print(f"Excluded input cols: {cfg['exclude_input_cols']}")
    if cfg.get("use_label_filter", True):
        print(
            "Label filter: on "
            f"(Butterworth LPF, cutoff={cfg.get('label_filter_cutoff_hz', 15.0)}Hz, "
            f"order={cfg.get('label_filter_order', 4)})"
        )
    else:
        print("Label filter: off")
    print(f"Input feature count: {data.input_size}")

    model_cfg = {
        "input_size": data.input_size,
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
    epoch_explosion_count = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        epoch_model_backup = copy.deepcopy(model.state_dict())
        epoch_opt_backup = copy.deepcopy(optimizer.state_dict())
        train_loss, train_rmse, train_stats = _train_epoch(
            model,
            data.train_loader,
            criterion,
            optimizer,
            device,
            label_mean_t,
            label_std_t,
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
        )
        train_total_batches = max(
            1, int(train_stats.get("total_batches", train_stats["num_batches"] + train_stats["skipped_nonfinite"]))
        )
        train_skipped_ratio = float(train_stats["skipped_nonfinite"]) / float(train_total_batches)
        train_unstable = (
            bool(train_stats.get("exploded", False))
            or (not np.isfinite(train_loss))
            or train_stats["num_batches"] == 0
            or train_skipped_ratio > float(cfg.get("max_train_skipped_ratio", 0.2))
        )
        if train_unstable:
            model.load_state_dict(epoch_model_backup)
            optimizer.load_state_dict(epoch_opt_backup)
            for g in optimizer.param_groups:
                g["lr"] = max(float(cfg.get("min_lr", 1e-7)), float(g["lr"]) * float(cfg.get("explosion_lr_decay", 0.5)))
            epoch_explosion_count += 1
            print(
                "Numerical instability detected in train epoch; "
                f"restored epoch-start checkpoint and reduced lr to {optimizer.param_groups[0]['lr']:.2e}. "
                f"[train valid/skipped={train_stats['num_batches']}/{train_stats['skipped_nonfinite']}, "
                f"skipped_ratio={train_skipped_ratio:.3f}] "
                f"[explosions={epoch_explosion_count}/{cfg.get('max_epoch_explosions', 3)}]"
            )
            if epoch_explosion_count >= int(cfg.get("max_epoch_explosions", 3)):
                print("Too many epoch explosions; stopping this run.")
                break
            continue

        if has_val:
            val_metrics = _eval_epoch(
                model, data.val_loader, criterion, device, label_mean_t, label_std_t
            )
        else:
            val_metrics = {
                "loss": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "pearson_r": float("nan"),
                "nrmse_pct": float("nan"),
                "num_batches": 1,
                "skipped_nonfinite": 0,
            }

        test_metrics = _eval_epoch(
            model, data.test_loader, criterion, device, label_mean_t, label_std_t
        )

        val_loss = val_metrics["loss"]
        eval_unstable = (
            (not np.isfinite(test_metrics["loss"]))
            or test_metrics["num_batches"] == 0
            or (has_val and ((not np.isfinite(val_loss)) or val_metrics["num_batches"] == 0))
        )
        if eval_unstable:
            model.load_state_dict(epoch_model_backup)
            optimizer.load_state_dict(epoch_opt_backup)
            for g in optimizer.param_groups:
                g["lr"] = max(float(cfg.get("min_lr", 1e-7)), float(g["lr"]) * float(cfg.get("explosion_lr_decay", 0.5)))
            epoch_explosion_count += 1
            print(
                "Numerical instability detected in eval epoch; "
                f"restored epoch-start checkpoint and reduced lr to {optimizer.param_groups[0]['lr']:.2e}. "
                f"(train valid/skipped: {train_stats['num_batches']}/{train_stats['skipped_nonfinite']}, "
                f"val valid/skipped: {val_metrics['num_batches']}/{val_metrics['skipped_nonfinite']}, "
                f"test valid/skipped: {test_metrics['num_batches']}/{test_metrics['skipped_nonfinite']}) "
                f"[explosions={epoch_explosion_count}/{cfg.get('max_epoch_explosions', 3)}]"
            )
            if epoch_explosion_count >= int(cfg.get("max_epoch_explosions", 3)):
                print("Too many epoch explosions; stopping this run.")
                break
            continue

        epoch_explosion_count = 0

        if has_val:
            scheduler.step(val_loss)

        print(
            f"train_loss={train_loss:.4f} train_rmse={train_rmse:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_rmse={val_metrics['rmse']:.4f} "
            f"val_R2={val_metrics['r2']:.3f} val_r={val_metrics['pearson_r']:.3f} "
            f"val_nRMSE={val_metrics['nrmse_pct']:.2f}% | "
            f"test_rmse={test_metrics['rmse']:.4f} test_R2={test_metrics['r2']:.3f} "
            f"test_r={test_metrics['pearson_r']:.3f} test_nRMSE={test_metrics['nrmse_pct']:.2f}% "
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
                "val_loss": val_metrics["loss"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_pearson_r": val_metrics["pearson_r"],
                "val_nrmse_pct": val_metrics["nrmse_pct"],
                "test_loss": test_metrics["loss"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_pearson_r": test_metrics["pearson_r"],
                "test_nrmse_pct": test_metrics["nrmse_pct"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_num_batches": train_stats["num_batches"],
                "train_total_batches": train_total_batches,
                "train_skipped_nonfinite": train_stats["skipped_nonfinite"],
                "train_skipped_ratio": train_skipped_ratio,
                "train_exploded": bool(train_stats.get("exploded", False)),
                "train_explode_reason": train_stats.get("explode_reason", None),
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

    if cfg.get("use_last_epoch", False):
        torch.save(model.state_dict(), out_dir / f"{cfg['run_name']}.pt")
        last_epoch = history[-1]["epoch"] if history else cfg["epochs"]
        print(f"[use_last_epoch=True] Using model from epoch {last_epoch} for final evaluation.")
    else:
        best_model_path = out_dir / f"{cfg['run_name']}.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_test_metrics = _eval_epoch(
        model, data.test_loader, criterion, device, label_mean_t, label_std_t
    )
    print(
        f"\nBest model test_loss={final_test_metrics['loss']:.4f}, "
        f"test_rmse={final_test_metrics['rmse']:.4f}, "
        f"test_R2={final_test_metrics['r2']:.3f}, test_r={final_test_metrics['pearson_r']:.3f}, "
        f"test_nRMSE={final_test_metrics['nrmse_pct']:.2f}%"
    )

    print("\nPer-trial test breakdown:")
    per_trial_metrics = _eval_per_trial(
        model,
        data.test_dataset,
        criterion,
        device,
        label_mean_t,
        label_std_t,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        save_dir=out_dir,
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

    print(f"Saved outputs to: {out_dir}")
    return {
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "target_col": cfg["target_col"],
        "window_size": cfg["window_size"],
        "batch_size": cfg["batch_size"],
        "dropout": cfg["dropout"],
        "loss_type": cfg["loss_type"],
        "huber_beta": cfg.get("huber_beta", None),
        "use_label_filter": bool(cfg.get("use_label_filter", True)),
        "label_filter_cutoff_hz": (
            float(cfg.get("label_filter_cutoff_hz", 15.0))
            if cfg.get("use_label_filter", True)
            else None
        ),
        "label_filter_order": int(cfg.get("label_filter_order", 4)),
        "label_filter_fs_hz": float(cfg.get("label_filter_fs_hz", 100.0)),
        "exclude_input_cols": json.dumps(cfg["exclude_input_cols"]),
        "num_channels": cfg["num_channels"],
        "final_test_loss": final_test_metrics["loss"],
        "final_test_rmse": final_test_metrics["rmse"],
        "final_test_r2": final_test_metrics["r2"],
        "final_test_pearson_r": final_test_metrics["pearson_r"],
        "final_test_nrmse_pct": final_test_metrics["nrmse_pct"],
        "out_dir": str(out_dir),
    }


def run_multi_seed():
    base_cfg = CONFIG.copy()
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

    if results:
        rmse = np.array([r["final_test_rmse"] for r in results], dtype=float)
        r2 = np.array([r["final_test_r2"] for r in results], dtype=float)
        pr = np.array([r["final_test_pearson_r"] for r in results], dtype=float)
        nrmse = np.array([r["final_test_nrmse_pct"] for r in results], dtype=float)
        print("\n" + "=" * 80)
        print("Multi-seed summary")
        print(f"Seeds: {seeds}")
        print(f"RMSE mean+-std: {np.nanmean(rmse):.4f} +- {np.nanstd(rmse):.4f}")
        print(f"R2   mean+-std: {np.nanmean(r2):.4f} +- {np.nanstd(r2):.4f}")
        print(f"r    mean+-std: {np.nanmean(pr):.4f} +- {np.nanstd(pr):.4f}")
        print(f"nRMSE mean+-std: {np.nanmean(nrmse):.4f}% +- {np.nanstd(nrmse):.4f}%")
        print(f"Summary CSV: {summary_csv}")
        print("=" * 80)


if __name__ == "__main__":
    run_multi_seed()
