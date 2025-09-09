# src/unlearning/t_gr.py - Enhanced Implementation
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
                         faulty_node_idx: int, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct using STGCN model with spatial-temporal inpainting
        """
        model.eval()
        
        with torch.no_grad():
            if hasattr(model, 'forward_with_mask'):
                # If model supports masked forward pass
                reconstructed = model.forward_with_mask(masked_data, faulty_node_idx)
            else:
                # Standard forward pass - model should handle missing data
                reconstructed = model(masked_data)
                
            # For STGCN -> might need to handle graph structure
            if edge_index is not None:
                # Apply graph convolution constraints
                reconstructed = self._apply_graph_constraints(reconstructed, edge_index, faulty_node_idx)
                
        return reconstructed
    

    def reconstruct_rnn_vae(self, model: nn.Module, masked_data: torch.Tensor,
                           d_f: list) -> torch.Tensor:
        """
        Reconstruct using RNN-VAE with latent space interpolation
        """
        model.eval()
        
        with torch.no_grad():
            # Encode the partially observed sequence
            if hasattr(model, 'encode'):
                mu, logvar = model.encode(masked_data)
                z = model.reparameterize(mu, logvar)
                reconstructed = model.decode(z)
            else:
                # Standard VAE forward
                reconstructed, mu, logvar = model(masked_data)
                
        return reconstructed
    

    def reconstruct_diffusion(self, model: nn.Module, masked_data: torch.Tensor,
                             d_f: list, num_steps: int = 50) -> torch.Tensor:
        """
        Reconstruct using diffusion model with conditional generation
        Following CSDI approach for time series imputation
        """
        model.eval()
        
        # Create conditioning mask
        cond_mask = torch.ones_like(masked_data)
        if isinstance(d_f[0], int):
            # Node-level masking
            for idx in d_f:
                cond_mask[:, :, idx] = 0.0
        else:
            # Temporal segment masking
            for start, end in d_f:
                cond_mask[:, start:end, :] = 0.0
        
        with torch.no_grad():
            # Initialize with noise for masked regions
            x_t = torch.randn_like(masked_data)
            x_t = masked_data * cond_mask + x_t * (1 - cond_mask)
            
            # Reverse diffusion process
            for t in reversed(range(num_steps)):
                t_tensor = torch.full((masked_data.shape[0],), t, dtype=torch.long)
                
                if hasattr(model, 'p_sample'):
                    x_t = model.p_sample(x_t, t_tensor, cond_mask, masked_data)
                else:
                    # Standard denoising step
                    noise_pred = model(x_t, t_tensor, cond_mask)
                    x_t = self._denoise_step(x_t, noise_pred, t, num_steps)
                    
                # Apply conditioning
                x_t = masked_data * cond_mask + x_t * (1 - cond_mask)
        
        return x_t
    

# --------------------- Main Steps -------------------------
    def _apply_graph_constraints(self, data: torch.Tensor, edge_index: torch.Tensor,
                                faulty_node_idx: int) -> torch.Tensor:
        """Apply graph structure constraints during reconstruction"""
        # Get neighbors of faulty node
        neighbors = edge_index[1][edge_index[0] == faulty_node_idx]
        
        if len(neighbors) > 0:
            # Weighted average of neighbor values
            neighbor_data = data[:, :, neighbors]
            data[:, :, faulty_node_idx] = neighbor_data.mean(dim=2)
            
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
        # Step 1: Create mask for unwanted patterns
        masked_data = self.create_mask(forget_sample, d_f)
        
        # Step 2: Model-specific reconstruction
        if self.model_type == "stgcn":
            if isinstance(d_f, list) and len(d_f) == 1:
                faulty_node_idx = d_f[0]
            else:
                faulty_node_idx = d_f
            reconstructed = self.reconstruct_stgcn(model, masked_data, faulty_node_idx, edge_index)
            
        elif self.model_type == "rnn_vae":
            reconstructed = self.reconstruct_rnn_vae(model, masked_data, d_f)
            
        elif self.model_type == "diffusion":
            reconstructed = self.reconstruct_diffusion(model, masked_data, d_f)
            
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
            surrogate_sample, d_f, noise_scale=0.01
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