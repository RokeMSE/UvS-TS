import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple

from src.unlearning.pa_ewc import PopulationAwareEWC 

def get_model_predictions(model: nn.Module, data_loader: DataLoader, A_wave: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get model predictions for a given data loader.
    """
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            A_wave = A_wave.to(device)
            
            # Get model output
            output = model(A_wave, X_batch)
            
            predictions.append(output.cpu())
            ground_truth.append(y_batch.cpu())
            
    return torch.cat(predictions), torch.cat(ground_truth)

def fidelity_score(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, A_wave: torch.Tensor, device: str):
    """
    Measure the performance preservation on the retain set.
    Higher is better.
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, retain_loader, A_wave, device)
    preds_original, _ = get_model_predictions(model_original, retain_loader, A_wave, device)
    
    error_unlearned = mean_absolute_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    error_original = mean_absolute_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    # Return the ratio of errors, should ~ 1
    return error_original / (error_unlearned + 1e-8) # + 1e-8 to avoid division by zero (can be removed ?)

def forgetting_efficacy(model_unlearned: nn.Module, model_original: nn.Module, forget_loader: DataLoader, A_wave: torch.Tensor, device: str):
    """
    Measure how much the model has forgotten the forget set.
    Higher is better (unlearned model has a much higher error).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, forget_loader, A_wave, device)
    preds_original, _ = get_model_predictions(model_original, forget_loader, A_wave, device)
    
    error_unlearned = mean_absolute_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    error_original = mean_absolute_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    # Return the difference in errors
    return error_unlearned - error_original

def generalization_score(model_unlearned: nn.Module, model_original: nn.Module, test_loader: DataLoader, A_wave: torch.Tensor, device: str):
    """
    Measure the performance on the test set to check for overfitting to the retain set.
    Should ~1.0 (unlearned model performs similarly to original), if not there might be overfitting?
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, test_loader, A_wave, device)
    preds_original, _ = get_model_predictions(model_original, test_loader, A_wave, device)
    
    error_unlearned = mean_absolute_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    error_original = mean_absolute_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    return error_original / (error_unlearned + 1e-8)

def statistical_distance(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, A_wave: torch.Tensor, device: str):
    """
    Measure the statistical similarity between the original and unlearned models on the retain set.
    Lower is better.
    """
    pa_ewc = PopulationAwareEWC(device=device)
    preds_unlearned, _ = get_model_predictions(model_unlearned, retain_loader, A_wave, device)
    preds_original, _ = get_model_predictions(model_original, retain_loader, A_wave, device)
    
    # Calculate the L_pop loss between the two distributions of predictions
    return pa_ewc.calculate_L_pop(preds_unlearned, preds_original).item()


def MIA_score(model_unlearned: nn.Module, model_original: nn.Module, forget_loader: DataLoader, A_wave: torch.Tensor, device: str):
    """
    Measure the vulnerability to Membership Inference Attacks (MIA) on the forget set.
    Higher is better, should be > 1 (unlearned model should be less vulnerable).
    """
    # MIA typically requires a separate attack model, not sure how to implement yet. ON HOLD
    return 


# --------- Main evaluation function -----------
def evaluate_unlearning(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, forget_loader: DataLoader, test_loader: DataLoader,A_wave: torch.Tensor, device: str) -> Dict[str, float]:
    """
    Run everything.
    """
    results = {}
    # 1. Fidelity Score
    results["fidelity_score"] = fidelity_score(
        model_unlearned, model_original, retain_loader, A_wave, device
    )
    # 2. Forgetting Efficacy
    results["forgetting_efficacy"] = forgetting_efficacy(
        model_unlearned, model_original, forget_loader, A_wave, device
    )
    # 3. Generalization Score
    results["generalization_score"] = generalization_score(
        model_unlearned, model_original, test_loader, A_wave, device
    )
    # 4. Statistical Distance
    results["statistical_distance"] = statistical_distance(
        model_unlearned, model_original, retain_loader, A_wave, device
    )
    return results