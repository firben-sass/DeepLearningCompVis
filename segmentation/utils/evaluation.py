"""Evaluation helpers for segmentation experiments."""
from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    loss_fn: Callable,
    metric_fns: Mapping[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    device: torch.device,
    threshold: float = 0.5,
):
    """Return average loss and metrics for the given loader."""
    if data_loader is None:
        raise ValueError("Data loader provided to evaluate_model cannot be None.")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    metrics_sums: Dict[str, float] = {name: 0.0 for name in metric_fns}

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            batch_loss = loss_fn(logits, y_batch)
            total_loss += batch_loss.item()

            predictions = (torch.sigmoid(logits) > threshold).float()
            for name, fn in metric_fns.items():
                metrics_sums[name] += fn(predictions, y_batch)

    num_batches = max(len(data_loader), 1)
    avg_loss = total_loss / num_batches
    metrics_avg = {name: total / num_batches for name, total in metrics_sums.items()}

    if was_training:
        model.train()

    return avg_loss, metrics_avg
