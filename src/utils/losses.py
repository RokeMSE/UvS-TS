# Define loss functions AKA unlearning objectives
import torch

def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel for MMD."""
    beta = 1. / (2. * sigma**2)
    dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-beta * dist)

def mmd_loss(x, y, sigma=1.0):
    """
        sigma (float): Bandwidth of the Gaussian kernel.
    """
    if x.dim() == 1: x = x.unsqueeze(1) # x (Tensor): Samples from the first distribution.
    if y.dim() == 1: y = y.unsqueeze(1) # y (Tensor): Samples from the second distribution.
        
    k_xx = gaussian_kernel(x, x, sigma).mean() # E_{x,x'} ~ P, P is the distribution of x 
    k_yy = gaussian_kernel(y, y, sigma).mean() # E_{y,y'} ~ Q, Q is the distribution of y
    k_xy = gaussian_kernel(x, y, sigma).mean() 
    
    return k_xx + k_yy - 2 * k_xy # MMD^2
