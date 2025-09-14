import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple

class TemporalGenerativeReplay:
    """
    Self-Contained Temporal Generative Replay (T-GR) implementation
    Uses the model's own capabilities for reconstruction and neutralization
    """
    def __init__(self, model_type: str = "stgcn"): # Change the model type for different tests
        """
        Params:
            model_type: Type of model ("stgcn", "rnn_vae", "diffusion")
        """
        self.model_type = model_type
        
    def create_mask(self, data: torch.Tensor, d_f: Union[int, list], 
                   mask_type: str = "node") -> torch.Tensor:
        """
        Create mask for the data to be reconstructed
        
        Params:
            data: Input data tensor (batch, seq_len, num_nodes) or (batch, seq_len, features)
            d_f: Indices of nodes/features to mask
            mask_type: "node" for spatial masking, "temporal" for time masking
        
        Returns:
            Masked data tensor
        """
        masked_data = data.clone()
        
        if isinstance(d_f, int):
            d_f = [d_f]
            
        if mask_type == "node":
            # Mask entire node(s) across all time steps
            for idx in d_f:
                masked_data[:, :, idx] = 0.0  # or torch.nan for explicit missing data
        elif mask_type == "temporal":
            # Mask temporal segments containing motifs
            for idx in d_f:
                masked_data[:, idx[0]:idx[1], :] = 0.0
                
        return masked_data
    
# ----------------- Model reconstruction --------------------
    def reconstruct_stgcn(self, model: nn.Module, masked_data: torch.Tensor,
                        faulty_node_idx: int, A_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct using STGCN model"""
        model.eval()
        
        # Ensure masked_data is 4D: (B, N, T, F)
        if masked_data.dim() == 3:
            masked_data = masked_data.unsqueeze(-1)
        elif masked_data.dim() == 2:
            masked_data = masked_data.unsqueeze(0).unsqueeze(-1)
        
        with torch.no_grad():
            if hasattr(model, 'forward_unlearning_compatible'):
                reconstructed = model.forward_unlearning_compatible(masked_data)
            elif A_hat is not None:
                reconstructed = model(A_hat, masked_data)
            else:
                raise ValueError("Either model must have forward_unlearning or A_hat must be provided")
                
            # Handle shape matching for reconstruction
            if reconstructed.shape[2] != masked_data.shape[2]:
                target_len = masked_data.shape[2]
                if reconstructed.shape[2] < target_len:
                    pad_size = target_len - reconstructed.shape[2]
                    last_vals = reconstructed[:, :, -1:].repeat(1, 1, pad_size)
                    reconstructed = torch.cat([reconstructed, last_vals], dim=2)
                else:
                    reconstructed = reconstructed[:, :, :target_len]
                    
        return reconstructed

# --------------------- Main Steps -------------------------
    def _apply_graph_constraints(self, data: torch.Tensor, edge_index: torch.Tensor,
                                faulty_node_idx: int) -> torch.Tensor:
        """Apply graph structure constraints during reconstruction"""
        # Get neighbors of faulty node
        neighbors = edge_index[1][edge_index[0] == faulty_node_idx]
        
        if len(neighbors) > 0:
            # Weighted average of neighbor values
            neighbor_data = data[:, neighbors, :]
            data[:, faulty_node_idx, :] = neighbor_data.mean(dim=1)
            
        return data
    

    def _denoise_step(self, x_t: torch.Tensor, noise_pred: torch.Tensor, 
                     t: int, num_steps: int) -> torch.Tensor:
        """Single denoising step for diffusion model"""
        # Simplified DDPM step
        beta_t = 0.0001 + (0.02 - 0.0001) * t / num_steps
        alpha_t = 1 - beta_t
        alpha_cumprod_t = alpha_t ** t
        
        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        
        x_prev = coeff1 * (x_t - coeff2 * noise_pred)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            x_prev += torch.sqrt(beta_t) * noise
            
        return x_prev
    

    def add_error_minimizing_noise(self, data: torch.Tensor, 
                                  d_f: Union[int, list],
                                  noise_scale: float = 0.01) -> torch.Tensor:
        """
        Add imperceptible error-minimizing noise to promote unlearning
        Inspired by unlearnable examples literature
        What this will do is make the model less sensitive to the forgotten patterns because it introduces noise in a controlled manner.
        """
        noisy_data = data.clone()
        
        if isinstance(d_f, int):
            d_f = [d_f] # Convert to list if single index is provided

        for idx in d_f: # Currently all samples in d_f is added with noise, this might cause a bottleneck, idk how to avoid this yet
            # Generate adversarial noise that minimizes learning signal
            noise = torch.randn_like(data[:, :, idx]) * noise_scale
            
            # Apply smoothing to make noise less detectable
            if data.shape[1] > 1:  # temporal smoothing
                noise = torch.nn.functional.avg_pool1d(
                    noise.unsqueeze(1), kernel_size=3, stride=1, padding=1
                ).squeeze(1)
            
            noisy_data[:, :, idx] += noise
            
        return noisy_data
    

    def perform_temporal_generative_replay(self, model: nn.Module, 
                                         forget_sample: torch.Tensor,
                                         d_f: Union[int, list],
                                         edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main T-GR function implementing Reconstruction and Neutralization
        
        Params:
            model: The model being unlearned
            forget_sample: Sample from forget set D_f
            d_f: Indices to be neutralized
            edge_index: Graph edges (for STGCN)
            
        Returns:
            Neutralized surrogate data
        """
        # Determine device from model and move input tensor to it
        device = next(model.parameters()).device
        forget_sample = forget_sample.to(device)

        # Step 1: Create mask for unwanted patterns
        masked_data = self.create_mask(forget_sample, d_f)
        
        # Step 2: Model-specific reconstruction
        if self.model_type == "stgcn":
            if isinstance(d_f, list) and len(d_f) == 1:
                faulty_node_idx = d_f[0]
            else:
                faulty_node_idx = d_f
            reconstructed = self.reconstruct_stgcn(model, masked_data, faulty_node_idx, edge_index)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Step 3: Create surrogate by replacing only the faulty parts
        surrogate_sample = forget_sample.clone()
        if isinstance(d_f, int):
            d_f = [d_f]
            
        for idx in d_f:
            if isinstance(idx, int):  # Node-level replacement
                surrogate_sample[:, :, idx] = reconstructed[:, :, idx]
            else:  # Temporal segment replacement
                start, end = idx
                surrogate_sample[:, start:end, :] = reconstructed[:, start:end, :]
        
        # Step 4: Add error-minimizing noise
        surrogate_sample = self.add_error_minimizing_noise(
            surrogate_sample, d_f, noise_scale=0.01 # Noise scale can be adjusted to control the amount 
        )
        
        return surrogate_sample


# -------------- Example usage function -------------------------
def create_surrogate_dataset(model: nn.Module, forget_loader: torch.utils.data.DataLoader,
                           d_f: Union[int, list], model_type: str = "stgcn",
                           edge_index: Optional[torch.Tensor] = None) -> torch.utils.data.TensorDataset: # edge_index is only for STGCN, it is the adjacency matrix
    """
    Create a dataset of surrogate samples for the entire forget set
    """
    tgr = TemporalGenerativeReplay(model_type)
    surrogate_samples = []
    
    model.eval()
    with torch.no_grad():
        for batch in forget_loader:
            if isinstance(batch, (list, tuple)):
                data_batch = batch[0]
            else:
                data_batch = batch
                
            surrogate_batch = tgr.perform_temporal_generative_replay(
                model, data_batch, d_f, edge_index
            )
            surrogate_samples.append(surrogate_batch)
    
    surrogate_data = torch.cat(surrogate_samples, dim=0)
    return torch.utils.data.TensorDataset(surrogate_data)