"""
Retrain-from-scratch baseline for UvS-TS unlearning evaluation.

Trains a fresh model on the retain distribution (no forget data ever seen).
This is the "gold standard" reference: a well-unlearned model should behave
like this retrained model on every metric. The gap between the unlearned
model and this retrained model is the real evaluation signal.

Data construction mirrors what _training_loop already uses internally:
    - NODE mode: retain_loader has the faulty node zeroed in both input and
                 target, and new_A_wave has the faulty row/col zeroed. The
                 retrain model sees the same isolated graph.
    - SUBSET mode: retain_loader holds windows that don't overlap any forget
                   motif. Adjacency is unchanged.

We reuse the loaders built inside UvSTS.unlearn_{node,subset} so that the
retrain sees exactly the same retain distribution the unlearning loop did.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.stgcn import STGCN
from models.stgat import STGAT
from models.gwn import gwnet


def _build_fresh_model(model_type: str, config: dict, device: str) -> nn.Module:
    """Construct a freshly-initialized model with the same config as the original."""
    config = dict(config)
    if model_type == "stgcn":
        return STGCN(**config).to(device)
    if model_type == "stgat":
        return STGAT(**config).to(device)
    if model_type == "gwnet":
        config["device"] = device
        return gwnet(**config).to(device)
    raise ValueError(f"Unknown model_type: {model_type}")


def retrain_from_scratch(
    model_type: str,
    config: dict,
    retain_loader: DataLoader,
    A_wave: torch.Tensor,
    device: str,
    epochs: int = 100,
    lr: float = 1e-3,
    faulty_node_idx: int = None,
    mask_faulty_in_retain: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Train a fresh model on the retain loader.

    Params mirror the original training script (train.py): Adam, lr=1e-3, MSE.
    If mask_faulty_in_retain is True, the loss is computed only on non-faulty
    nodes (matches the node-unlearning training objective).
    """
    model = _build_fresh_model(model_type, config, device)
    A_wave = A_wave.float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    N = None
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in retain_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(-1)

            if N is None:
                N = X_batch.shape[1]

            optimizer.zero_grad()
            pred = model(A_wave, X_batch)

            if mask_faulty_in_retain and faulty_node_idx is not None and pred.shape[1] > 1:
                keep = torch.ones(N, dtype=torch.bool, device=device)
                keep[faulty_node_idx] = False
                loss = mse(pred[:, keep, :, :], y_batch[:, keep, :, :])
            else:
                loss = mse(pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if verbose and (epoch + 1) % 10 == 0:
            avg = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"  [retrain] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    model.eval()
    return model


def parameter_distance_ratio(
    model_unlearned: nn.Module,
    model_retrained: nn.Module,
    model_original: nn.Module,
) -> float:
    """
    ||theta_unlearned - theta_retrained||_F / ||theta_original - theta_retrained||_F

    <1.0 : unlearned model is closer to the gold-standard retrain than the
           original was (unlearning moved params toward the ideal)
    ~0.0 : unlearned model matches retrain in parameter space (ideal)
    >1.0 : unlearning pushed further away from the retrain than starting point
    """
    num = 0.0
    den = 0.0
    u_params = dict(model_unlearned.named_parameters())
    r_params = dict(model_retrained.named_parameters())
    o_params = dict(model_original.named_parameters())

    for name in u_params:
        if name not in r_params or name not in o_params:
            continue
        num += (u_params[name].data - r_params[name].data).pow(2).sum().item()
        den += (o_params[name].data - r_params[name].data).pow(2).sum().item()

    if den < 1e-12:
        return 0.0
    return (num / den) ** 0.5
