# Calculates L_pop, MMD, and the Population-Aware FIM
import torch
from src.utils.statistic import (get_marginal_distribution, get_cross_correlations, 
                                 get_power_spectral_density, get_autocorrelation)
from src.utils.losses import mmd_loss

def calculate_L_pop(model_output, ground_truth_data):
    """
    Calculates the Generalized Population Loss (L_pop) == weighted sum of MMDs over several key statistics.
    """
    # model_output and ground_truth_data shape: (batch, seq_len, num_nodes)
    # 1. Get statistics for ground truth data
    gt_marginal = get_marginal_distribution(ground_truth_data)
    # NOTE: can be computationally intensive. For speed, pre-calculate and save | use a subset of statistics.
    # gt_acf = get_autocorrelation(ground_truth_data)
    
    # 2. Get statistics for model-generated data
    gen_marginal = get_marginal_distribution(model_output)
    # gen_acf = get_autocorrelation(model_output)
    
    # 3. Calculate MMD for each statistic
    # equal weights (w_s = 1) for simplicity
    loss = mmd_loss(gt_marginal, gen_marginal) #+ mmd_loss(gt_acf, gen_acf)
    
    return loss

def calculate_pa_fim(model, retain_data_loader, device):
    """
    Calculates the diagonal of the Population-Aware Fisher Information Matrix.
    """
    model.eval()
    fim_diagonal = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    for data_batch in retain_data_loader:
        data_batch = data_batch.to(device)
        model.zero_grad()
        
        # Get model output for the batch
        model_output = model(data_batch)
        
        # Calculate L_pop loss
        loss = calculate_L_pop(model_output, data_batch)
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fim_diagonal[name] += param.grad.data.pow(2)

    # Average over the number of samples
    num_samples = len(retain_data_loader.dataset)
    for name in fim_diagonal:
        fim_diagonal[name] /= num_samples
        
    # NOTE: this might take forever and memory expensive -> TASK: optimize this somehow (approximate, subset of data, etc.)    
        
    return fim_diagonal
