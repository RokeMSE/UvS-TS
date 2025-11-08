import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unlearning.pa_ewc import PopulationAwareEWC
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple
import numpy as np

def get_model_predictions(model: nn.Module, data_loader: DataLoader, 
                         A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get model predictions """
    model.eval()
    predictions = []
    ground_truth = []
    
    # Ensure A_wave is on the correct device and is a tensor
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
            
            # Ensure X_batch is 4D for STGCN: (B, N, T, F)
            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(-1)
            
            # Get model output - always use forward() with A_wave
            output = model(A_wave, X_batch)

            # If evaluating a single node, slice the output to match the ground truth shape
            if faulty_node_idx is not None and y_batch.shape[1] == 1 and output.shape[1] > 1:
                output = output[:, faulty_node_idx:faulty_node_idx+1, :, :]
            
            predictions.append(output.cpu())
            ground_truth.append(y_batch.cpu())
            
    return torch.cat(predictions), torch.cat(ground_truth)

def fidelity_score(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, 
                   new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure the performance preservation on the retain set.
    Higher is better (closer to 1.0 means similar performance).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, retain_loader, A_wave, device, faulty_node_idx)
    
    mse_unlearned = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    mse_original = mean_squared_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    # Return the ratio of MSEs, should ~ 1
    return mse_original / (mse_unlearned + 1e-8)

def forgetting_efficacy(model_unlearned: nn.Module, model_original: nn.Module, forget_loader: DataLoader, 
                        new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure how much the model has forgotten the forget set.
    Higher is better (unlearned model has a much higher error).
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
    Measure the performance on the test set to check for overfitting to the retain set.
    Should ~1.0 (unlearned model performs similarly to original).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, test_loader, new_A_wave, device)
    preds_original, _ = get_model_predictions(model_original, test_loader, A_wave, device)
    
    mse_unlearned = mean_squared_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    mse_original = mean_squared_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    return mse_original / (mse_unlearned + 1e-8)

def statistical_distance(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, 
                         new_A_wave: torch.Tensor, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure the statistical similarity between the original and unlearned models on the retain set.
    Lower is better.
    """
    pa_ewc = PopulationAwareEWC(device=device)
    preds_unlearned, _ = get_model_predictions(model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, retain_loader, A_wave, device, faulty_node_idx)
    
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


def retain_set_mse(model_unlearned: nn.Module, retain_loader: DataLoader,
                   new_A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> float:
    """
    Direct MSE on the retain set.
    Lower is better (good predictions on retain data).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx)
    
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


# --------- Main evaluation function -----------
def evaluate_unlearning(model_unlearned: nn.Module, model_original: nn.Module, 
                       retain_loader: DataLoader, forget_loader: DataLoader, 
                       test_loader: DataLoader, new_A_wave: torch.Tensor, 
                       A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> Dict[str, float]:
    """
    Comprehensive evaluation with improved metrics for STGCN unlearning.
    
    Metrics:
    - Fidelity: Performance preservation on retain set (higher is better, ~1.0 ideal)
    - Forgetting Efficacy: Error increase on forget set (higher is better, >>1.0 ideal)
    - Generalization: Performance on test set (higher is better, ~1.0 ideal)
    - Statistical Distance: Distribution similarity on retain set (lower is better)
    
    New Metrics:
    - Spatial Correlation Divergence: Disruption of spatial patterns (higher is better)
    - Temporal Pattern Divergence: Disruption of temporal patterns (higher is better)
    - Prediction Confidence: Uncertainty on forget set (higher is better)
    - Forget Set MSE: Direct error on forget data (higher is better)
    - Retain Set MSE: Direct error on retain data (lower is better)
    - Test Set MSE: Direct error on test data (lower is better)
    - Graph Structure Impact: Effect of graph modification (higher is better for node unlearning)
    """
    results = {}
    
    # Original metrics (ratios)
    results["fidelity_score"] = fidelity_score(
        model_unlearned, model_original, retain_loader, new_A_wave, A_wave, device, faulty_node_idx
    )
    
    results["forgetting_efficacy"] = forgetting_efficacy(
        model_unlearned, model_original, forget_loader, new_A_wave, A_wave, device, faulty_node_idx
    )
    
    results["generalization_score"] = generalization_score(
        model_unlearned, model_original, test_loader, new_A_wave, A_wave, device
    )
    
    results["statistical_distance"] = statistical_distance(
        model_unlearned, model_original, retain_loader, new_A_wave, A_wave, device, faulty_node_idx
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
        model_unlearned, retain_loader, new_A_wave, device, faulty_node_idx
    )
    
    results["test_set_mse"] = test_set_mse(
        model_unlearned, test_loader, new_A_wave, device
    )
    
    # Graph structure impact (only for node unlearning)
    if faulty_node_idx is not None:
        results["graph_structure_impact"] = graph_structure_impact(
            model_unlearned, forget_loader, new_A_wave, A_wave, device, faulty_node_idx
        )
    
    return results