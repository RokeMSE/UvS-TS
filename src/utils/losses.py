import torch

def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel for MMD with numerical stability"""
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.tensor(0.0)
    
    try:
        # Ensure proper dimensions
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        # Limit sample size for memory
        max_samples = 500
        if x.shape[0] > max_samples:
            indices = torch.randperm(x.shape[0])[:max_samples]
            x = x[indices]
        if y.shape[0] > max_samples:
            indices = torch.randperm(y.shape[0])[:max_samples]
            y = y[indices]
        
        # Compute distances with numerical stability
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        
        # Multiple bandwidth Gaussian kernel (more robust)
        sigmas = [sigma * 0.5, sigma, sigma * 2.0]
        kernel_vals = []
        
        for s in sigmas:
            beta = 1. / (2. * s**2)
            kernel_vals.append(torch.exp(-beta * dist_sq))
        
        return torch.stack(kernel_vals).mean(dim=0)
    
    except Exception as e:
        print(f"Warning in gaussian_kernel: {e}")
        return torch.tensor(1e-6)

def mmd_loss(x, y, sigma=1.0):
    """MMD loss with enhanced stability"""
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(1e-6, requires_grad=True)
    
    try:
        k_xx = gaussian_kernel(x, x, sigma).mean()
        k_yy = gaussian_kernel(y, y, sigma).mean()
        k_xy = gaussian_kernel(x, y, sigma).mean()
        
        mmd_val = k_xx + k_yy - 2 * k_xy
        
        # Ensure non-negative
        mmd_val = torch.clamp(mmd_val, min=1e-8)
        
        return mmd_val
    
    except Exception as e:
        print(f"Warning in mmd_loss: {e}")
        return torch.tensor(1e-6, requires_grad=True)
