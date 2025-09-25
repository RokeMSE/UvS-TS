import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unlearning.pa_ewc import PopulationAwareEWC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_recall_fscore_support
from typing import Dict, Tuple

def get_model_predictions(model: nn.Module, data_loader: DataLoader, 
                         A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get model predictions """
    model.eval()
    predictions = []
    ground_truth = []
    
    # Store adjacency matrix if model supports it
    if hasattr(model, 'set_adjacency_matrix'):
        model.set_adjacency_matrix(A_wave.to(device))
    
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                X_batch, y_batch = batch_data
            else:
                X_batch = batch_data
                y_batch = batch_data
                
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) 
            A_wave = A_wave.to(device)
            
            # Ensure X_batch is 4D for STGCN: (B, N, T, F)
            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(-1)
            
            # Get model output
            if hasattr(model, 'forward_unlearning'):
                output = model.forward_unlearning(X_batch)
            else:
                output = model(A_wave, X_batch)

            # If evaluating a single node, slice the output to match the ground truth shape
            if faulty_node_idx is not None and y_batch.shape[1] == 1 and output.shape[1] > 1:
                output = output[:, faulty_node_idx:faulty_node_idx+1, :, :]
            
            predictions.append(output.cpu())
            ground_truth.append(y_batch.cpu())
            
    return torch.cat(predictions), torch.cat(ground_truth)

def fidelity_score(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure the performance preservation on the retain set.
    Higher is better.
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, retain_loader, A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, retain_loader, A_wave, device, faulty_node_idx)
    
    error_unlearned = mean_absolute_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    error_original = mean_absolute_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    # Return the ratio of errors, should ~ 1
    return error_original / (error_unlearned + 1e-8) # + 1e-8 to avoid division by zero (can be removed ?)

def forgetting_efficacy(model_unlearned: nn.Module, model_original: nn.Module, forget_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure how much the model has forgotten the forget set.
    Higher is better (unlearned model has a much higher error).
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, forget_loader, A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, forget_loader, A_wave, device, faulty_node_idx)
    
    error_unlearned = mean_absolute_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    error_original = mean_absolute_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    # Return the difference in errors
    return error_unlearned - error_original

def generalization_score(model_unlearned: nn.Module, model_original: nn.Module, test_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure the performance on the test set to check for overfitting to the retain set.
    Should ~1.0 (unlearned model performs similarly to original), if not there might be overfitting?
    """
    preds_unlearned, truth = get_model_predictions(model_unlearned, test_loader, A_wave, device) # No faulty_node_idx needed for test set
    preds_original, _ = get_model_predictions(model_original, test_loader, A_wave, device)
    
    error_unlearned = mean_absolute_error(truth.numpy().flatten(), preds_unlearned.numpy().flatten())
    error_original = mean_absolute_error(truth.numpy().flatten(), preds_original.numpy().flatten())
    
    return error_original / (error_unlearned + 1e-8) 

def statistical_distance(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Measure the statistical similarity between the original and unlearned models on the retain set.
    Lower is better.
    """
    pa_ewc = PopulationAwareEWC(device=device)
    preds_unlearned, _ = get_model_predictions(model_unlearned, retain_loader, A_wave, device, faulty_node_idx)
    preds_original, _ = get_model_predictions(model_original, retain_loader, A_wave, device, faulty_node_idx)
    
    # Calculate the L_pop loss between the two distributions of predictions
    return pa_ewc.calculate_L_pop(preds_unlearned, preds_original).item()


def get_model_losses(model: nn.Module, data_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None):
    """
    Get the per-sample loss for a given data loader.
    """
    model.eval()
    losses = []
    loss_criterion = nn.MSELoss(reduction='none') # We want per-sample loss

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            A_wave = A_wave.to(device)
            
            output = model(A_wave, X_batch)
            
            # If evaluating a single node, slice the output to match the ground truth shape
            if faulty_node_idx is not None and y_batch.shape[1] == 1 and output.shape[1] > 1:
                output = output[:, faulty_node_idx:faulty_node_idx+1, :, :]

            # Calculate loss for each sample in the batch
            # Reshape to (batch_size, -1) and take the mean over the last dimension
            loss = loss_criterion(output, y_batch).mean(dim=[1, 2, 3]) # Mean over all but batch dim
            losses.append(loss.cpu())
            
    return torch.cat(losses)

def membership_inference_attack(model_unlearned: nn.Module, forget_loader: DataLoader, test_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> Dict[str, float]:
    """
    Performs a Membership Inference Attack to evaluate unlearning effectiveness.
    
    Returns:
        A dictionary with the attack model's accuracy, precision, recall, and f1-score.
    """
    # 1. Get losses for the unlearned model on forget and test sets
    forget_losses = get_model_losses(model_unlearned, forget_loader, A_wave, device, faulty_node_idx)
    test_losses = get_model_losses(model_unlearned, test_loader, A_wave, device) # No faulty_node_idx for test set
    
    # 2. Create training data for the attack model
    # Members (from forget set) are labeled 1, Non-members (from test set) are labeled 0
    member_labels = torch.ones(len(forget_losses))
    non_member_labels = torch.zeros(len(test_losses))
    
    # Combine the data and labels
    all_losses = torch.cat([forget_losses, test_losses]).numpy().reshape(-1, 1)
    all_labels = torch.cat([member_labels, non_member_labels]).numpy()
    
    # 3. Train the attack model (a simple logistic regression model)
    # Split data to train and test the attack model itself
    X_train, X_test, y_train, y_test = train_test_split(
        all_losses, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    
    attack_model = LogisticRegression()
    attack_model.fit(X_train, y_train)
    
    # 4. Evaluate the attack model's performance
    y_pred = attack_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    return {
        "mia_accuracy": accuracy,
        "mia_precision": precision,
        "mia_recall": recall,
        "mia_f1_score": f1
    }

# --------- Main evaluation function -----------"
def evaluate_unlearning(model_unlearned: nn.Module, model_original: nn.Module, retain_loader: DataLoader, forget_loader: DataLoader, test_loader: DataLoader, A_wave: torch.Tensor, device: str, faulty_node_idx: int = None) -> Dict[str, float]:
    """
    Run everything.
    """
    results = {}
    # 1. Fidelity Score
    results["fidelity_score"] = fidelity_score(
        model_unlearned, model_original, retain_loader, A_wave, device, faulty_node_idx
    )
    # 2. Forgetting Efficacy
    results["forgetting_efficacy"] = forgetting_efficacy(
        model_unlearned, model_original, forget_loader, A_wave, device, faulty_node_idx
    )
    # 3. Generalization Score
    results["generalization_score"] = generalization_score(
        model_unlearned, model_original, test_loader, A_wave, device
    )
    # 4. Statistical Distance
    results["statistical_distance"] = statistical_distance(
        model_unlearned, model_original, retain_loader, A_wave, device, faulty_node_idx
    )
    # 5. Membership Inference Attack
    mia_results = membership_inference_attack(
        model_unlearned, forget_loader, test_loader, A_wave, device, faulty_node_idx
    )
    results.update(mia_results) # Merge the dictionaries
    return results
