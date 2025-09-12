import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union
import numpy as np
from torch.utils.data import DataLoader
from utils.statistic import (get_marginal_distribution, get_cross_correlations,
                                get_power_spectral_density, get_autocorrelation)
from utils.losses import mmd_loss

class PopulationAwareEWC:
    """
    Population-Aware Elastic Weight Consolidation (PA-EWC)
    Implements distribution-space regularization instead of parameter-space
    """
    
    def __init__(self, model_type: str = "stgcn", device: str = "cuda"):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Default weights for different statistics
        self.default_weights = {
            'marginal': 1.0,
            'acf': 0.8,
            'psd': 0.6,
            'cc': 0.4  # Most expensive, lowest weight by default (cc stands for cross-correlation)
        }
    

    def calculate_L_pop(self, model_output: torch.Tensor, 
                       ground_truth_data: torch.Tensor,
                       weights: Optional[Dict[str, float]] = None,
                       statistics_subset: Optional[List[str]] = None) -> torch.Tensor:
        """
        L_pop calculation with better error handling and efficiency options
        
        Params:
            model_output: Generated data (batch, seq_len, num_nodes)
            ground_truth_data: Real retain data (batch, seq_len, num_nodes)
            weights: Custom weights for statistics
            statistics_subset: Subset of statistics to compute for efficiency
        """
        # Convert shapes for statistical analysis - Expected: (B, T, N)
        if model_output.dim() == 3:
            # Check if it's (B, N, T) and convert to (B, T, N)
            if model_output.shape[1] > model_output.shape[2]:
                model_output = model_output.transpose(1, 2)
        
        if ground_truth_data.dim() == 3:
            if ground_truth_data.shape[1] > ground_truth_data.shape[2]:
                ground_truth_data = ground_truth_data.transpose(1, 2)

        if not isinstance(weights, (dict, type(None))):
            # If weights is not a dict or None, default to the class weights.
            # This handles cases where arguments might be passed incorrectly.
            print(f"Warning: `weights` argument in calculate_L_pop was not a dictionary. Defaulting. Got type: {type(weights)}")
            weights = self.default_weights.copy()
        elif weights is None:
            weights = self.default_weights.copy()
                
        # Default to all statistics if not specified
        if statistics_subset is None:
            statistics_subset = ['marginal', 'acf']  # Start with most important ones
            
        total_loss = 0.0
        computed_losses = {}
        
        # Ensure tensors are on the same device
        model_output = model_output.to(self.device)
        ground_truth_data = ground_truth_data.to(self.device) 
        
        try:
            # 1. Marginal Value Distributions (most fundamental)
            if 'marginal' in statistics_subset and weights.get('marginal', 0.0) > 0:
                gt_marginal = self._safe_get_marginal_distribution(ground_truth_data)
                gen_marginal = self._safe_get_marginal_distribution(model_output)
                
                if gt_marginal is not None and gen_marginal is not None:
                    loss_marginal = self._safe_mmd_loss(gt_marginal, gen_marginal)
                    if loss_marginal is not None:
                        computed_losses['marginal'] = loss_marginal
                        total_loss += weights['marginal'] * loss_marginal

            # 2. Autocorrelation Functions (temporal dependencies)
            if 'acf' in statistics_subset and weights.get('acf', 0.0) > 0:
                gt_acf = self._safe_get_autocorrelation(ground_truth_data)
                gen_acf = self._safe_get_autocorrelation(model_output)
                
                if gt_acf is not None and gen_acf is not None and len(gt_acf) > 0:
                    loss_acf = self._safe_mmd_loss(gt_acf, gen_acf)
                    if loss_acf is not None:
                        computed_losses['acf'] = loss_acf
                        total_loss += weights['acf'] * loss_acf

            # 3. Power Spectral Densities (frequency content)
            if 'psd' in statistics_subset and weights.get('psd', 0.0) > 0:
                gt_psd = self._safe_get_power_spectral_density(ground_truth_data)
                gen_psd = self._safe_get_power_spectral_density(model_output)
                
                if gt_psd is not None and gen_psd is not None:
                    loss_psd = self._safe_mmd_loss(gt_psd, gen_psd)
                    if loss_psd is not None:
                        computed_losses['psd'] = loss_psd
                        total_loss += weights['psd'] * loss_psd

            # 4. Cross-Correlations (inter-node dependencies)
            if 'cc' in statistics_subset and weights.get('cc', 0.0) > 0:
                # Only compute if there are multiple nodes and reasonable batch size
                if ground_truth_data.shape[-1] > 1 and ground_truth_data.shape[0] <= 32:
                    gt_cc = self._safe_get_cross_correlations(ground_truth_data)
                    gen_cc = self._safe_get_cross_correlations(model_output)
                    
                    if gt_cc is not None and gen_cc is not None and len(gt_cc) > 0:
                        loss_cc = self._safe_mmd_loss(gt_cc, gen_cc)
                        if loss_cc is not None:
                            computed_losses['cc'] = loss_cc
                            total_loss += weights['cc'] * loss_cc

        except Exception as e:
            print(f"Warning: Error in L_pop calculation: {e}")
            # Return a small positive loss to avoid training issues
            return torch.tensor(1e-6, device=self.device, requires_grad=True)
        
        # Store computed losses for debugging
        self.last_computed_losses = computed_losses
        
        return total_loss if total_loss > 0 else torch.tensor(1e-6, device=self.device, requires_grad=True)
    

# ------------------- Wrappers for Statistics -------------------
    def _safe_get_marginal_distribution(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Wrapper for marginal distribution calculation"""
        try:
            return get_marginal_distribution(data)
        except Exception as e:
            print(f"Warning: Failed to compute marginal distribution: {e}")
            return None
    

    def _safe_get_autocorrelation(self, data: torch.Tensor, max_lag: int = 5) -> Optional[torch.Tensor]:
        """Wrapper for autocorrelation calculation"""
        try:
            if data.shape[1] <= max_lag:  # Not enough data for meaningful ACF
                return None
            return get_autocorrelation(data, max_lag=min(max_lag, data.shape[1] // 4))
        except Exception as e:
            print(f"Warning: Failed to compute autocorrelation: {e}")
            return None
    

    def _safe_get_power_spectral_density(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Wrapper for PSD calculation"""
        try:
            if data.shape[1] < 8:  # Not enough samples for meaningful PSD
                return None
            return get_power_spectral_density(data)
        except Exception as e:
            print(f"Warning: Failed to compute PSD: {e}")
            return None
    

    def _safe_get_cross_correlations(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Wrapper for cross-correlation calculation"""
        try:
            if data.shape[-1] < 2:  # Need at least 2 nodes
                return None
            return get_cross_correlations(data)
        except Exception as e:
            print(f"Warning: Failed to compute cross-correlations: {e}")
            return None
    

    def _safe_mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        """Wrapper for MMD loss calculation"""
        try:
            if x.numel() == 0 or y.numel() == 0:
                return None
            return mmd_loss(x, y)
        except Exception as e:
            print(f"Warning: Failed to compute MMD loss: {e}")
            return None
    

# ------------------- FIM Calculation -------------------
    def calculate_pa_fim(self, model: nn.Module, 
                        retain_data_loader: DataLoader,
                        A_wave: torch.Tensor,  # Adjacency matrix for STGCN
                        weights: Optional[Dict[str, float]] = None,
                        max_samples: int = 1000,
                        batch_accumulation: int = 4) -> Dict[str, torch.Tensor]:
        """
        Enhanced PA-FIM calculation for STGCN with proper data handling
        
        Params:
            model: The STGCN model to calculate FIM for
            retain_data_loader: DataLoader for retain set 
            A_wave: Normalized adjacency matrix for STGCN
            weights: Weights for L_pop statistics
            max_samples: Maximum samples to use (for memory management)
            batch_accumulation: Accumulate gradients over multiple batches
        """
        model.eval()
        
        # Initialize FIM diagonal storage
        fim_diagonal = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim_diagonal[name] = torch.zeros_like(param, device=self.device)
        
        sample_count = 0
        batch_count = 0
        
        # Use subset of statistics for FIM computation (efficiency)
        statistics_subset = ['marginal', 'acf']  # Most important ones
        
        try:
            for data_batch in retain_data_loader:
                if sample_count >= max_samples:
                    break
                    
                # Extract training input and target from batch
                # Expected format from your generate_dataset: (input, target)
                if isinstance(data_batch, (list, tuple)):
                    X_batch, y_batch = data_batch
                    X_batch = X_batch.to(self.device)  # (batch, num_nodes, num_timesteps_input, num_features)
                    y_batch = y_batch.to(self.device)  # (batch, num_nodes, num_timesteps_output)
                else:
                    # Single tensor - need to extract input/target somehow
                    data_batch = data_batch.to(self.device)
                    X_batch = data_batch
                    y_batch = data_batch  # Use same data as target (reconstruction task)
                
                # Skip if batch is too small
                if X_batch.shape[0] < 2:
                    continue
                
                model.zero_grad()
                
                # Forward pass through STGCN model
                model_output = model(A_wave, X_batch)  # Output: (batch, num_nodes, num_timesteps_output)
                
                # Calculate L_pop loss
                loss = self.calculate_L_pop(
                    model_output, y_batch, weights, statistics_subset
                )
                
                if loss.requires_grad:
                    loss.backward()
                    
                    # Accumulate squared gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fim_diagonal[name] += param.grad.data.pow(2)
                
                sample_count += X_batch.shape[0]
                batch_count += 1
                
                # Clear intermediate computations
                if batch_count % batch_accumulation == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        except Exception as e:
            print(f"Warning: Error during FIM calculation: {e}")
            # Return small positive values to avoid numerical issues
            for name in fim_diagonal:
                fim_diagonal[name].fill_(1e-6)
            return fim_diagonal
        
        # Average over number of samples
        if sample_count > 0:
            for name in fim_diagonal:
                fim_diagonal[name] /= sample_count
        
        # Add small regularization to avoid zero values
        for name in fim_diagonal:
            fim_diagonal[name] += 1e-8
        
        return fim_diagonal
    
    def _forward_rnn_vae(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Handle RNN-VAE specific forward pass"""
        try:
            output = model(data)
            if isinstance(output, tuple):
                return output[0]  # Return reconstruction
            return output
        except:
            return model(data)
    

    def _forward_diffusion(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Handle Diffusion model specific forward pass"""
        try:
            # For diffusion models sampling is required 
            if hasattr(model, 'sample'):
                return model.sample(data.shape)
            elif hasattr(model, 'forward'):
                # Add some noise and denoise
                t = torch.randint(0, 1000, (data.shape[0],), device=data.device)
                return model(data, t)
            else:
                return model(data)
        except:
            return model(data)
    

    def apply_ewc_penalty(self, model: nn.Module, 
                         fim_diagonal: Dict[str, torch.Tensor],
                         original_params: Dict[str, torch.Tensor],
                         lambda_ewc: float = 1000.0) -> torch.Tensor:
        """
        Apply the EWC penalty term
        
        Params:
            model: Current model
            fim_diagonal: Computed FIM diagonal
            original_params: Original model parameters (θ*)
            lambda_ewc: EWC regularization strength
        """
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if name in fim_diagonal and name in original_params:
                penalty += (fim_diagonal[name] * (param - original_params[name]).pow(2)).sum()
        
        return lambda_ewc * penalty / 2.0
    

    def get_statistics_summary(self) -> Dict[str, float]:
        """Get summary of last computed losses for debugging"""
        return getattr(self, 'last_computed_losses', {})


# ---------------- Utility function to save/load FIM (to not blow up the PC) -------------------
def save_fim(fim_diagonal: Dict[str, torch.Tensor], filepath: str):
    """Save FIM diagonal to file"""
    torch.save({k: v.cpu() for k, v in fim_diagonal.items()}, filepath)

def load_fim(filepath: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load FIM diagonal from file"""
    fim_data = torch.load(filepath, map_location=device)
    return {k: v.to(device) for k, v in fim_data.items()}