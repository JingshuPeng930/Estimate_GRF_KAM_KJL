from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class DelayClassifier(nn.Module):
    """Small MLP classifier that predicts delay class from one IMU window."""

    def __init__(
        self,
        input_channels: int,
        window_size: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = int(input_channels) * int(window_size)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.net(x.flatten(start_dim=1))


def build_delay_classes(delay_values: Iterable[int]) -> List[int]:
    vals = sorted(set(int(v) for v in delay_values))
    if not vals:
        raise ValueError("No delay classes found.")
    return vals


def map_delay_to_index(delay_ms: torch.Tensor, class_to_idx: Dict[int, int]) -> torch.Tensor:
    out = torch.zeros_like(delay_ms, dtype=torch.long)
    mapped_any = torch.zeros_like(delay_ms, dtype=torch.bool)
    for delay_val, cls_idx in class_to_idx.items():
        mask = delay_ms == int(delay_val)
        if torch.any(mask):
            out[mask] = int(cls_idx)
            mapped_any[mask] = True
    if not torch.all(mapped_any):
        # fallback unknown delays to the first class
        out[~mapped_any] = 0
    return out


def predict_delay_probs(
    model: DelayClassifier,
    x: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    logits = model(x)
    temp = max(float(temperature), 1e-6)
    return F.softmax(logits / temp, dim=1)


@dataclass
class DelayTrainResult:
    class_values: List[int]
    best_epoch: int
    best_val_acc: float
    last_val_acc: float
    train_acc: float
    train_loss: float


def _classification_epoch(
    model: DelayClassifier,
    loader,
    class_to_idx: Dict[int, int],
    device: str,
    optimizer=None,
    criterion=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="DelayTrain" if is_train else "DelayEval", dynamic_ncols=True):
            x = batch[0].to(device)
            delay_ms = batch[4].to(device)
            y = map_delay_to_index(delay_ms, class_to_idx)

            logits = model(x)
            loss = criterion(logits, y) if criterion is not None else F.cross_entropy(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            pred_cls = torch.argmax(logits, dim=1)
            total_correct += int((pred_cls == y).sum().item())
            total_samples += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())

    acc = float(total_correct / max(total_samples, 1))
    mean_loss = float(total_loss / max(total_samples, 1))
    return mean_loss, acc


def train_delay_classifier(
    model: DelayClassifier,
    train_loader,
    val_loader,
    class_values: List[int],
    device: str,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> DelayTrainResult:
    class_to_idx = {int(v): i for i, v in enumerate(class_values)}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = 1
    last_val_acc = 0.0
    last_train_acc = 0.0
    last_train_loss = 0.0

    for epoch in range(1, int(epochs) + 1):
        train_loss, train_acc = _classification_epoch(
            model, train_loader, class_to_idx, device, optimizer=optimizer, criterion=criterion
        )
        val_loss, val_acc = _classification_epoch(
            model, val_loader, class_to_idx, device, optimizer=None, criterion=criterion
        )
        last_val_acc = val_acc
        last_train_acc = train_acc
        last_train_loss = train_loss

        print(
            f"[DelayClf] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return DelayTrainResult(
        class_values=[int(v) for v in class_values],
        best_epoch=int(best_epoch),
        best_val_acc=float(best_val_acc),
        last_val_acc=float(last_val_acc),
        train_acc=float(last_train_acc),
        train_loss=float(last_train_loss),
    )
