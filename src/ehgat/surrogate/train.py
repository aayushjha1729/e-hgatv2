"""Training loop for the E-HGATv2 surrogate (MSE on normalised (C_max, E)).

Trains on decoded/evaluated random schedules and reports held-out R^2 / MAE per
objective in physical units. The trained model predicts physical (C_max, E) from a
raw graph (normalisation is baked into the model), ready for the guided search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from ehgat.environment.instance import Instance
from ehgat.surrogate.dataset import (
    fit_normalization,
    generate_graphs,
    split_graphs,
)
from ehgat.surrogate.ehgatv2 import EHGATv2, EHGATv2Config
from ehgat.utils.seeding import seed_everything

__all__ = ["TrainConfig", "TrainResult", "regression_metrics", "train_surrogate"]

_OBJECTIVE_NAMES = ("makespan", "energy")


@dataclass(frozen=True)
class TrainConfig:
    """Surrogate training hyper-parameters."""

    num_samples: int = 1000
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden: int = 64
    layers: int = 3
    heads: int = 4
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 0


@dataclass
class TrainResult:
    """Trained model, per-epoch history and held-out test metrics."""

    model: EHGATv2
    history: list[dict[str, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def regression_metrics(pred: Tensor, true: Tensor) -> dict[str, float]:
    """Per-objective and overall R^2 / MAE in physical units.

    pred/true are [num_graphs, 2] physical (C_max, E) tensors.
    """
    metrics: dict[str, float] = {}
    abs_err = (pred - true).abs()
    for i, name in enumerate(_OBJECTIVE_NAMES):
        y = true[:, i]
        ss_res = torch.sum((y - pred[:, i]) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2).clamp_min(1e-12)
        metrics[f"r2_{name}"] = float(1.0 - ss_res / ss_tot)
        metrics[f"mae_{name}"] = float(abs_err[:, i].mean())
    metrics["mae_overall"] = float(abs_err.mean())
    return metrics


def _normalise_targets(model: EHGATv2, y: Tensor) -> Tensor:
    return (y - model.target_mean) / model.target_std


@torch.no_grad()
def _predict_split(
    model: EHGATv2, graphs: list[HeteroData], device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    model.eval()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    preds, trues = [], []
    for batch in loader:
        if device is not None:
            batch = batch.to(device)
        preds.append(model.predict(batch).cpu())
        trues.append(batch.y.cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick the training device: explicit request, else CUDA when available, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_surrogate(
    instance: Instance,
    config: TrainConfig | None = None,
    *,
    device: str | torch.device | None = None,
    graphs: list | None = None,
) -> TrainResult:
    """Generate data, train the surrogate, and evaluate on a held-out test split.

    The core GNN training is the dominant cost in the fused pipeline; device moves the
    model + mini-batches onto the GPU (default: CUDA when available) so the H100 is the
    load-bearer for the heavy 80-epoch core fit. Data generation stays on the CPU (it is a
    Python combinatorial decode+simulate per sample, not GPU-addressable). The returned
    model is moved back to CPU so downstream fused fine-tuning (which encodes on CPU before
    its own device move) is unchanged.
    """
    config = config or TrainConfig()
    seed_everything(config.seed)
    dev = _resolve_device(device)

    # graphs lets a caller inject a pre-pooled dataset (e.g. samples pooled across a range
    # of instance sizes for a size-generalisation curriculum); otherwise generate from instance.
    if graphs is None:
        graphs = generate_graphs(instance, config.num_samples, seed=config.seed)
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs, val_frac=config.val_frac, test_frac=config.test_frac, seed=config.seed
    )

    model = EHGATv2(EHGATv2Config(hidden=config.hidden, layers=config.layers, heads=config.heads))
    fit_normalization(train_graphs).apply_to(model)
    model = model.to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_graphs, batch_size=config.batch_size, shuffle=True, generator=generator
    )

    result = TrainResult(model=model)
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            pred, _ = model(batch)
            loss = loss_fn(pred, _normalise_targets(model, batch.y))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * batch.num_graphs
        epoch_loss /= max(len(train_graphs), 1)

        record = {"epoch": float(epoch), "train_mse": epoch_loss}
        if val_graphs:
            val_pred, val_true = _predict_split(model, val_graphs, dev)
            record["val_mae_overall"] = regression_metrics(val_pred, val_true)["mae_overall"]
        result.history.append(record)

    eval_graphs = test_graphs or val_graphs or train_graphs
    test_pred, test_true = _predict_split(model, eval_graphs, dev)
    result.metrics = regression_metrics(test_pred, test_true)
    model.to("cpu")
    return result
