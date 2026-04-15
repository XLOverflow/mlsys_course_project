"""Training / inference loops for the cost model.

Loss = MSE on predicted latency  +  λ · pairwise-ranking hinge loss.

The ranking term is computed within each minibatch using the sign of target
latency differences, explicitly optimizing the model for strategy ordering
in addition to absolute latency accuracy.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from hetero_cost_model.training.config import TrainConfig
from hetero_cost_model.training.losses import pairwise_ranking_loss


def train(model: torch.nn.Module, dataset, config: TrainConfig) -> List[float]:
    """Fit ``model`` on ``dataset`` and return per-epoch mean training loss."""
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    model.to(config.device).train()

    history: List[float] = []
    for _ in range(config.epochs):
        history.append(_run_epoch(model, loader, optimizer, config))
    return history


def _run_epoch(model, loader, optimizer, config: TrainConfig) -> float:
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        batch = batch.to(config.device)
        pred = model(batch)
        loss = (
            F.mse_loss(pred, batch.y)
            + config.ranking_lambda * pairwise_ranking_loss(pred, batch.y)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.numel()
        total_items += pred.numel()
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataset,
    device: str = "cpu",
) -> Tuple[List[float], List[float]]:
    """Return ``(predictions, targets)`` on ``dataset``."""
    loader = DataLoader(dataset, batch_size=64)
    model.to(device).eval()

    preds: List[float] = []
    trues: List[float] = []
    for batch in loader:
        batch = batch.to(device)
        preds.extend(model(batch).cpu().tolist())
        trues.extend(batch.y.cpu().tolist())
    return preds, trues


__all__ = ["train", "predict"]
