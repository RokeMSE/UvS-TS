import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unlearning.pa_ewc import PopulationAwareEWC
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from typing import Dict, Tuple, Optional
import numpy as np

def get_model_predictions(model: nn.Module, data_loader: DataLoader,
                         A_wave: torch.Tensor, device: str, faulty_node_idx: int = None,
                         mask_faulty_node: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get model predictions.

    mask_faulty_node: when True AND faulty_node_idx is set AND both output and
    target contain all nodes, drop the faulty node from both tensors. Use this
    for retain-set metrics in NODE unlearning mode, where training zeroed the
    faulty node's row in the retain target — including it would score the model
    on predicting zeros rather than on retain performance.
    """
    model.eval()
    predictions = []
    ground_truth = []

    if isinstance(A_wave, np.ndarray):
        A_wave = torch.from_numpy(A_wave).float()
    A_wave = A_wave.to(device)

    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                X_batch, y_batch = batch_data
            else:
                X_batch = batch_data
                y_batch = batch_data

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(-1)

            output = model(A_wave, X_batch)

            if faulty_node_idx is not None and y_batch.shape[1] == 1 and output.shape[1] > 1:
                output = output[:, faulty_node_idx:faulty_node_idx+1, :, :]
            elif (mask_faulty_node and faulty_node_idx is not None
                  and output.shape[1] > 1 and y_batch.shape[1] == output.shape[1]):
                N = output.shape[1]
                keep = torch.arange(N, device=output.device) != faulty_node_idx
                output = output[:, keep, :, :]
                y_batch = y_batch[:, keep, :, :]

            predictions.append(output.cpu())
            ground_truth.append(y_batch.cpu())

    return torch.cat(predictions), torch.cat(ground_truth)

def fidelity_score(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader,
                   new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None,
                   mask_faulty_in_retain: bool = False):
    """
    Measure performance preservation on the retain set.
    Convention: mse_unlearned / mse_original.
      ~1.0  = retain-set performance unchanged (ideal)
      > 1.0 = retain-set degraded (bad)
      < 1.0 = retain-set improved (unusual, check for data leakage)
    """
    preds_unlearned, truth = get_model_predictions(
        model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx,
        mask_faulty_node=mask_faulty_in_retain,
    )
    preds_original, _ = get_model_predictions(
        model_original, retain_loader, A_wave, device, faulty_node_idx,
        mask_faulty_node=mask_faulty_in_retain,
    )

    mse_unlearned = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    mse_original = mean_squared_error(truth.numpy().flatten(), preds_original.numpy().flatten())

    return mse_unlearned / (mse_original + 1e-8)

def forgetting_efficacy(model_unlearned: nn.Module, model_original: nn.Module, forget_loader: DataLoader,
                        new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Unbounded forget-MSE ratio (kept for backward compatibility only).

    NOTE: this metric's "higher is better" convention contradicts the bounded
    margin-ascent objective the training loop actually optimises. Use
    `forget_margin_achievement` as the primary forgetting signal.
    Returns the relative increase in MSE on the forget set.
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, forget_loader, A_wave, device, faulty_node_idx)
    
    mse_unlearned = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    mse_original = mean_squared_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    # Return the ratio (how much worse the unlearned model is)
    return mse_unlearned / (mse_original + 1e-8)

def generalization_score(model_unlearned: nn.Module, model_original: nn.Module, test_loader: DataLoader,
                         new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure test-set performance to check for overfitting to the retain set.
    Convention: mse_unlearned / mse_original.
      ~1.0  = generalization unchanged (ideal)
      > 1.0 = unlearned model generalises worse
      < 1.0 = unlearned model generalises better (unlikely without retain-set shift)
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, test_loader, new_A_wave, device)
    preds_original, _ = get_model_predictions(model_original, test_loader, A_wave, device)

    mse_unlearned = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    mse_original = mean_squared_error(truth.numpy().flatten(), preds_original.numpy().flatten())

    return mse_unlearned / (mse_original + 1e-8)

def statistical_distance(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader,
                         new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None,
                         mask_faulty_in_retain: bool = False):
    """
    Measure the statistical similarity between the original and unlearned models on the retain set.
    Lower is better.
    """
    pa_ewc = PopulationAwareEWC(device=device)
    preds_unlearned, _ = get_model_predictions(
        model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx,
        mask_faulty_node=mask_faulty_in_retain,
    )
    preds_original, _ = get_model_predictions(
        model_original, retain_loader, A_wave, device, faulty_node_idx,
        mask_faulty_node=mask_faulty_in_retain,
    )
    
    # Calculate the L_pop loss between the two distributions of predictions
    return pa_ewc.calculate_L_pop(preds_unlearned, preds_original).item()


# ============ NEW METRICS ============

def spatial_correlation_divergence(model_unlearned: nn.Module, model_original: nn.Module, 
                                   forget_loader: DataLoader, new_A_wave: torch.Tensor, 
                                   A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> float:
    """
    Measure divergence in spatial correlations between nodes.
    Higher is better (more decorrelation from forget patterns).
    
    This measures how well the unlearned model has disrupted spatial dependencies
    that were present in the original model for the forget set.
    """
    # Call get_model_predictions with faulty_node_idx=None to get ALL nodes -> can compare nodes, which is the point of "spatial" correlation.
    preds_unlearned, truth = get_model_predictions(model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx=None)
    preds_original, _ = get_model_predictions(model_original, forget_loader, A_wave, device, faulty_node_idx=None)
    
    # Compute spatial correlation matrices
    # Shape: (B, N, T, F)
    B, N, T, F = preds_unlearned.shape
    
    # Add safeguard for correlation calculation
    if N < 2:
        # Cannot compute spatial correlation with fewer than 2 nodes
        return 0.0

    # Reshape logic to handle 4D tensor
    # Select first feature (:, :, :, 0) -> (B, N, T)
    # Permute to (B, T, N)
    # Reshape to (B*T, N)
    preds_unlearned_f0 = preds_unlearned[:, :, :, 0]
    preds_original_f0 = preds_original[:, :, :, 0]

    preds_unlearned_flat = preds_unlearned_f0.permute(0, 2, 1).reshape(B * T, N).cpu().numpy()
    preds_original_flat = preds_original_f0.permute(0, 2, 1).reshape(B * T, N).cpu().numpy()
    
    # Compute correlation matrices
    corr_unlearned = np.corrcoef(preds_unlearned_flat.T)
    corr_original = np.corrcoef(preds_original_flat.T)
    
    # Handle NaNs that occur if a node's data has zero variance
    corr_unlearned = np.nan_to_num(corr_unlearned)
    corr_original = np.nan_to_num(corr_original)
    
    # Measure Frobenius norm of difference
    corr_diff = np.linalg.norm(corr_unlearned - corr_original, 'fro')
    
    # Normalize by number of nodes
    return corr_diff / N


def temporal_pattern_divergence(model_unlearned: nn.Module, model_original: nn.Module,
                                forget_loader: DataLoader, new_A_wave: torch.Tensor,
                                A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> float:
    """
    Measure divergence in temporal patterns using autocorrelation.
    Higher is better (temporal patterns are more different).
    
    This checks if the unlearned model has disrupted temporal dependencies
    that the original model learned from the forget set.
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, forget_loader, A_wave, device, faulty_node_idx)
    
    def compute_autocorr(data, max_lag=3):
        """Compute average autocorrelation across nodes"""
        B, N, T, F = data.shape
        autocorrs = []
        
        for b in range(B):
            for n in range(N):
                series = data[b, n, :, 0].numpy()
                mean_val = np.mean(series)
                var_val = np.var(series)
                
                if var_val > 1e-8:
                    for lag in range(1, min(max_lag + 1, T)):
                        acf = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        if not np.isnan(acf):
                            autocorrs.append(acf)
        
        return np.mean(autocorrs) if autocorrs else 0.0
    
    acf_unlearned = compute_autocorr(preds_unlearned)
    acf_original = compute_autocorr(preds_original)
    
    # Return absolute difference
    return abs(acf_unlearned - acf_original)


def prediction_confidence_score(model_unlearned: nn.Module, forget_loader: DataLoader,
                                new_A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> float:
    """
    Measure prediction confidence/uncertainty on forget set.
    Lower variance/confidence is better (indicates uncertainty about forget data).
    
    Returns the coefficient of variation (std/mean) of predictions.
    Higher values indicate more uncertainty.
    """
    preds_unlearned, _ = get_model_predictions(model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx)
    
    # Compute coefficient of variation
    preds_flat = preds_unlearned.numpy().flatten()
    mean_pred = np.mean(np.abs(preds_flat))
    std_pred = np.std(preds_flat)
    
    # Coefficient of variation (normalized measure of dispersion)
    cv = std_pred / (mean_pred + 1e-8)
    
    return cv


def forget_set_mse(model_unlearned: nn.Module, forget_loader: DataLoader,
                   new_A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> float:
    """
    Direct MSE on the forget set.
    Higher is better (worse predictions on forget data).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx)
    
    mse = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    return mse


def forget_margin_achievement(model_unlearned: nn.Module, forget_loader: DataLoader,
                               new_A_wave: torch.Tensor, device: str,
                               forget_margin: float,
                               faulty_node_idx: int = None) -> float:
    """
    Margin-achievement indicator aligned with the method's bounded ascent.

    The training loop uses max(0, margin - L_forget) so it stops pushing once
    forget MSE reaches `forget_margin`. A useful metric therefore caps at the
    target instead of rewarding unbounded ascent:

        min(forget_mse, forget_margin) / forget_margin

      1.0  = forget MSE reached or exceeded the target (ideal)
      <1.0 = undershoot — forgetting incomplete relative to the stated goal

    Pair with `forget_set_mse` to see the raw value and whether the margin was
    set realistically.
    """
    if forget_margin is None or forget_margin <= 0:
        return float("nan")
    mse = forget_set_mse(model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx)
    return float(min(mse, forget_margin) / forget_margin)


def retain_set_mse(model_unlearned: nn.Module, retain_loader: DataLoader,
                   new_A_wave: torch.Tensor, device: str, faulty_node_idx: int = None,
                   mask_faulty_in_retain: bool = False) -> float:
    """
    Direct MSE on the retain set.
    Lower is better (good predictions on retain data).
    """
    preds_unlearned, truth = get_model_predictions(
        model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx,
        mask_faulty_node=mask_faulty_in_retain,
    )
    
    mse = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    return mse


def test_set_mse(model_unlearned: nn.Module, test_loader: DataLoader,
                 new_A_wave: torch.Tensor, device: str) -> float:
    """
    Direct MSE on the test set.
    Lower is better (maintains generalization).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, test_loader, new_A_wave, device)
    
    mse = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    return mse


def graph_structure_impact(model_unlearned: nn.Module, forget_loader: DataLoader,
                           new_A_wave: torch.Tensor, A_wave: torch.Tensor, 
                           device: str, faulty_node_idx: int) -> float:
    """
    Measure how much the graph structure change impacts predictions.
    This is specific to node unlearning scenarios.
    
    Returns the ratio of prediction errors with modified vs original adjacency.
    """
    if faulty_node_idx is None:
        return 0.0
    
    model_unlearned.eval()
    errors_new_graph = []
    errors_old_graph = []
    
    with torch.no_grad():
        for batch_data in forget_loader:
            if isinstance(batch_data, (list, tuple)):
                X_batch, y_batch = batch_data
            else:
                X_batch = batch_data
                y_batch = batch_data
                
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(-1)
            
            # Predictions with new graph structure
            output_new = model_unlearned(new_A_wave.to(device), X_batch)
            if y_batch.shape[1] == 1 and output_new.shape[1] > 1:
                output_new = output_new[:, faulty_node_idx:faulty_node_idx+1, :, :]
            error_new = torch.mean((output_new - y_batch) ** 2).item()
            
            # Predictions with original graph structure
            output_old = model_unlearned(A_wave.to(device), X_batch)
            if y_batch.shape[1] == 1 and output_old.shape[1] > 1:
                output_old = output_old[:, faulty_node_idx:faulty_node_idx+1, :, :]
            error_old = torch.mean((output_old - y_batch) ** 2).item()
            
            errors_new_graph.append(error_new)
            errors_old_graph.append(error_old)
    
    avg_error_new = np.mean(errors_new_graph)
    avg_error_old = np.mean(errors_old_graph)
    
    # Return ratio (how much worse with new graph structure)
    return avg_error_new / (avg_error_old + 1e-8)


def _per_sample_losses(model: nn.Module, data_loader: DataLoader,
                        A_wave: torch.Tensor, device: str,
                        faulty_node_idx: int = None) -> np.ndarray:
    """Per-sample MSE under model. If faulty_node_idx is set and the target is
    full-node, losses are computed on the faulty node only so member and
    non-member distributions in MIA are directly comparable."""
    model.eval()
    A_wave = A_wave.to(device)
    losses = []
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                X_batch, y_batch = batch_data
            else:
                X_batch = batch_data
                y_batch = batch_data

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(-1)

            out = model(A_wave, X_batch)

            if faulty_node_idx is not None and out.shape[1] > 1:
                out = out[:, faulty_node_idx:faulty_node_idx+1, :, :]
                if y_batch.shape[1] > 1:
                    y_batch = y_batch[:, faulty_node_idx:faulty_node_idx+1, :, :]

            diff = (out - y_batch) ** 2
            per_sample = diff.reshape(diff.shape[0], -1).mean(dim=1)
            losses.append(per_sample.cpu().numpy())
    return np.concatenate(losses) if losses else np.array([])


def membership_inference_auc(model: nn.Module, member_loader: DataLoader,
                              nonmember_loader: DataLoader, A_wave: torch.Tensor,
                              device: str, faulty_node_idx: int = None) -> float:
    """
    Threshold-based loss MIA.

    Under the null (model has no membership signal) the forget-set loss
    distribution should match a held-out non-member loss distribution and
    a classifier that scores samples by -loss should achieve AUC ~ 0.5.
    A well-unlearned model should approach 0.5; a model that still memorises
    forget data will be > 0.5.

    Compare unlearned AUC to the original model's AUC: the delta is the
    unlearning's privacy effect.
    """
    member_losses = _per_sample_losses(model, member_loader, A_wave, device, faulty_node_idx)
    nonmember_losses = _per_sample_losses(model, nonmember_loader, A_wave, device, faulty_node_idx)

    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        return 0.5

    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])

    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


# --------- Main evaluation function -----------
def evaluate_unlearning(model_unlearned: nn.Module, model_original: nn.Module,
                       retain_loader: DataLoader, forget_loader: DataLoader,
                       test_loader: DataLoader, new_A_wave: torch.Tensor,
                       A_wave: torch.Tensor, device: str, faulty_node_idx: int = None,
                       mask_faulty_in_retain: bool = False,
                       model_retrained: Optional[nn.Module] = None,
                       forget_margin: Optional[float] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation for ST-GNN unlearning.

    All ratio metrics use the convention  mse_unlearned / mse_original
    so that 1.0 always means "identical to the original model" and the
    direction of a good result is consistent:

    - Fidelity             retain-set ratio  ~1.0 ideal  (> 1.0 = degraded)
    - Forgetting Efficacy  forget-set ratio  >> 1.0 ideal (higher = more forgotten)
    - Generalization       test-set ratio    ~1.0 ideal  (> 1.0 = worse generalisation)
    - Statistical Distance MMD on retain-set predictions  lower is better
    - Spatial Correlation Divergence  forget-set spatial disruption  higher is better
    - Temporal Pattern Divergence     forget-set temporal disruption higher is better
    - Prediction Confidence           CoV on forget-set preds        higher = more uncertain
    - Forget Set MSE      raw MSE on forget data  higher is better
    - Retain Set MSE      raw MSE on retain data  lower is better
    - Test Set MSE        raw MSE on test data    lower is better
    - Graph Structure Impact  ratio with modified vs original graph  higher is better
    """
    results = {}

    # Original metrics (ratios) — retain-side metrics honour mask_faulty_in_retain
    results["fidelity_score"] = fidelity_score(
        model_unlearned, model_original, retain_loader, new_A_wave, A_wave, device,
        faulty_node_idx, mask_faulty_in_retain=mask_faulty_in_retain,
    )

    results["forgetting_efficacy"] = forgetting_efficacy(
        model_unlearned, model_original, forget_loader, new_A_wave, A_wave, device, faulty_node_idx
    )

    # Primary forgetting signal, aligned with the bounded margin-ascent objective.
    if forget_margin is not None:
        results["forget_margin_achievement"] = forget_margin_achievement(
            model_unlearned, forget_loader, new_A_wave, device, forget_margin, faulty_node_idx
        )

    results["generalization_score"] = generalization_score(
        model_unlearned, model_original, test_loader, new_A_wave, A_wave, device
    )

    results["statistical_distance"] = statistical_distance(
        model_unlearned, model_original, retain_loader, new_A_wave, A_wave, device,
        faulty_node_idx, mask_faulty_in_retain=mask_faulty_in_retain,
    )
    
    # New spatio-temporal specific metrics
    results["spatial_correlation_divergence"] = spatial_correlation_divergence(
        model_unlearned, model_original, forget_loader, new_A_wave, A_wave, device, faulty_node_idx
    )
    
    results["temporal_pattern_divergence"] = temporal_pattern_divergence(
        model_unlearned, model_original, forget_loader, new_A_wave, A_wave, device, faulty_node_idx
    )
    
    results["prediction_confidence"] = prediction_confidence_score(
        model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx
    )
    
    # Direct MSE metrics
    results["forget_set_mse"] = forget_set_mse(
        model_unlearned, forget_loader, new_A_wave, device, faulty_node_idx
    )
    
    results["retain_set_mse"] = retain_set_mse(
        model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx,
        mask_faulty_in_retain=mask_faulty_in_retain,
    )

    results["test_set_mse"] = test_set_mse(
        model_unlearned, test_loader, new_A_wave, device
    )

    # Graph structure impact (only for node unlearning)
    if faulty_node_idx is not None:
        results["graph_structure_impact"] = graph_structure_impact(
            model_unlearned, forget_loader, new_A_wave, A_wave, device, faulty_node_idx
        )

    # ---- MIA: loss-threshold attack using forget as members, test as non-members ----
    # For node mode, per-sample losses are restricted to the faulty node so member
    # and non-member distributions are comparable.
    mia_node = faulty_node_idx if faulty_node_idx is not None else None
    results["mia_auc_original"] = membership_inference_auc(
        model_original, forget_loader, test_loader, A_wave, device, mia_node
    )
    results["mia_auc_unlearned"] = membership_inference_auc(
        model_unlearned, forget_loader, test_loader, new_A_wave, device, mia_node
    )
    # How much closer to 0.5 (no membership signal) the unlearned model is.
    # Positive = unlearning reduced the membership signal (good).
    results["mia_auc_reduction"] = (
        abs(results["mia_auc_original"] - 0.5) - abs(results["mia_auc_unlearned"] - 0.5)
    )

    # ---- Retrain-from-scratch gap metrics ----
    # The retrained model is the "gold standard". A successful unlearned model
    # behaves like this reference on each metric — so we report the absolute
    # gap per metric. Closer to 0 is better.
    if model_retrained is not None:
        retrain_retain = retain_set_mse(
            model_retrained, retain_loader, new_A_wave, device, faulty_node_idx,
            mask_faulty_in_retain=mask_faulty_in_retain,
        )
        retrain_forget = forget_set_mse(
            model_retrained, forget_loader, new_A_wave, device, faulty_node_idx
        )
        retrain_test = test_set_mse(
            model_retrained, test_loader, new_A_wave, device
        )
        retrain_mia = membership_inference_auc(
            model_retrained, forget_loader, test_loader, new_A_wave, device, mia_node
        )

        results["retrain_retain_mse"] = retrain_retain
        results["retrain_forget_mse"] = retrain_forget
        results["retrain_test_mse"] = retrain_test
        results["retrain_mia_auc"] = retrain_mia

        results["gap_retain_mse"] = abs(results["retain_set_mse"] - retrain_retain)
        results["gap_forget_mse"] = abs(results["forget_set_mse"] - retrain_forget)
        results["gap_test_mse"] = abs(results["test_set_mse"] - retrain_test)
        results["gap_mia_auc"] = abs(results["mia_auc_unlearned"] - retrain_mia)

        # Parameter-space distance: ||theta_u - theta_r|| / ||theta_o - theta_r||
        # <1.0 = unlearning moved params toward retrain (good)
        from retrain_baseline import parameter_distance_ratio
        results["param_distance_ratio"] = parameter_distance_ratio(
            model_unlearned, model_retrained, model_original
        )

    return results