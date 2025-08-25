""" write functions to compute the required statistics: 
1. Marginal Value Distributions 
2. Cross-Correlations (CC)
3. Power Spectral Densities (PSD) 
4. Autocorrelation Functions (ACF).
"""

# NOTE: Check the proposal for info about these
import torch
import numpy as np
from scipy.signal import welch # for power spectral density
from scipy.stats import pearsonr # for cross-correlation

def get_marginal_distribution(data_tensor):
    # Flatten the tensor except for the node dimension
    return data_tensor.view(-1, data_tensor.shape[-1])

def get_cross_correlations(data_tensor):
    """distribution of pairwise cross-correlations."""
    # data_tensor shape: (batch, seq_len, num_nodes)
    batch_size, _, num_nodes = data_tensor.shape
    correlations = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Calculate correlation/item in the batch
            for b in range(batch_size):
                corr, _ = pearsonr(data_tensor[b, :, i].cpu().numpy(), data_tensor[b, :, j].cpu().numpy()) # .cpu().numpy(): convert to numpy
                correlations.append(corr)
    return torch.tensor(correlations, dtype=torch.float32)

def get_power_spectral_density(data_tensor):
    """distribution of PSD"""
    # data_tensor shape: (batch, seq_len, num_nodes)
    psds = []
    for b in range(data_tensor.shape[0]):
        for n in range(data_tensor.shape[2]):
            # Use Welch's method to get a smooth PSD estimate ()
            _, psd = welch(data_tensor[b, :, n].cpu().numpy())
            psds.append(psd)
    return torch.tensor(np.array(psds), dtype=torch.float32).mean(dim=0)

def get_autocorrelation(data_tensor, lag=1):
    """distribution of ACF"""
    # data_tensor shape: (batch, seq_len, num_nodes)
    acf_vals = []
    for b in range(data_tensor.shape[0]):
        for n in range(data_tensor.shape[2]):
            series = data_tensor[b, :, n]
            mean = torch.mean(series)
            var = torch.var(series)
            if var == 0: continue
            
            cov = torch.mean((series[:-lag] - mean) * (series[lag:] - mean))
            acf_vals.append(cov / var)
            
    return torch.tensor(acf_vals, dtype=torch.float32)