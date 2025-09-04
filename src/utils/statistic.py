""" write functions to compute the required statistics: 
1. Marginal Value Distributions 
2. Cross-Correlations (CC)
3. Power Spectral Densities (PSD) 
4. Autocorrelation Functions (ACF).
"""

# NOTE: Check the proposal for info about these
import torch
import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore') # Ignore warnings for cleaner output

def get_marginal_distribution(data_tensor):
    """Get flattened distribution of all values"""
    if data_tensor.numel() == 0:
        return torch.empty(0)
    return data_tensor.reshape(-1)

def get_cross_correlations(data_tensor, max_pairs=50):
    """Compute cross-correlations with memory optimization"""
    batch_size, seq_len, num_nodes = data_tensor.shape
    
    if num_nodes < 2 or seq_len < 3:
        return torch.empty(0)
    
    correlations = []
    pair_count = 0
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if pair_count >= max_pairs:  # Limit to prevent memory issues
                break
                
            for b in range(min(batch_size, 8)):  # Limit batch size
                try:
                    x_i = data_tensor[b, :, i].cpu().numpy()
                    x_j = data_tensor[b, :, j].cpu().numpy()
                    
                    if np.var(x_i) > 1e-8 and np.var(x_j) > 1e-8:
                        corr, _ = pearsonr(x_i, x_j)
                        if not np.isnan(corr):
                            correlations.append(corr)
                except:
                    continue
            
            pair_count += 1
            if pair_count >= max_pairs:
                break
        
        if pair_count >= max_pairs:
            break
    
    return torch.tensor(correlations, dtype=torch.float32) if correlations else torch.empty(0)

def get_power_spectral_density(data_tensor, max_samples=100):
    """Compute PSD with memory optimization"""
    batch_size, seq_len, num_nodes = data_tensor.shape
    
    if seq_len < 8:  # Not enough samples for meaningful PSD
        return torch.empty(0)
    
    psds = []
    sample_count = 0
    
    for b in range(batch_size):
        for n in range(num_nodes):
            if sample_count >= max_samples:
                break
                
            try:
                signal = data_tensor[b, :, n].cpu().numpy()
                if np.var(signal) > 1e-8:  # Only if signal has variance
                    _, psd = welch(signal, nperseg=min(seq_len//2, 32))
                    psds.extend(psd)
                    sample_count += 1
            except:
                continue
                
        if sample_count >= max_samples:
            break
    
    return torch.tensor(psds, dtype=torch.float32) if psds else torch.empty(0)

def get_autocorrelation(data_tensor, max_lag=5, max_samples=200):
    """Compute autocorrelation with optimization"""
    batch_size, seq_len, num_nodes = data_tensor.shape
    
    if seq_len <= max_lag:
        return torch.empty(0)
    
    acf_vals = []
    sample_count = 0
    
    for lag in range(1, min(max_lag + 1, seq_len // 2)):
        for b in range(batch_size):
            for n in range(num_nodes):
                if sample_count >= max_samples:
                    break
                    
                try:
                    series = data_tensor[b, :, n]
                    mean_val = torch.mean(series)
                    var_val = torch.var(series)
                    
                    if var_val > 1e-8:
                        cov = torch.mean((series[:-lag] - mean_val) * (series[lag:] - mean_val))
                        acf = cov / var_val
                        if not torch.isnan(acf):
                            acf_vals.append(acf.item())
                            sample_count += 1
                except:
                    continue
                    
            if sample_count >= max_samples:
                break
        if sample_count >= max_samples:
            break
    
    return torch.tensor(acf_vals, dtype=torch.float32) if acf_vals else torch.empty(0)