import torch
import torch.nn as nn
import numpy as np
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
from typing import Optional, Union, Tuple

class TemporalGenerativeReplay:
    """
    Enhanced Temporal Generative Replay (T-GR) with multi-stage noise injection
    """
    def __init__(self, model_type: str = "stgcn"):
        self.model_type = model_type
        
    # ============ STAGE 1: Pre-reconstruction Noise ============
    def add_input_perturbation(self, masked_data: torch.Tensor, 
                               noise_scale: float = 0.05) -> torch.Tensor:
        """
        Add noise to input BEFORE reconstruction to encourage different trajectories
        """
        noise = torch.randn_like(masked_data) * noise_scale
        perturbed = masked_data + noise
        return perturbed
    
    # ============ STAGE 2: Reconstruction with Smoothing ============
    def surrogate_stgcn_enhanced(self, model: nn.Module, node_dataset,
                                 forget_indices: list, faulty_node_idx: int, 
                                 num_timesteps_input, num_timesteps_output, 
                                 device, A_hat: Optional[torch.Tensor] = None,
                                 reconstruction_noise: float = 0.02) -> torch.Tensor:
        """Enhanced STGCN reconstruction with intermediate noise injection"""
        model.eval()
        
        node_input, node_target = generate_dataset(node_dataset, num_timesteps_input, num_timesteps_output)
        node_input = node_input.float()
        node_target = node_target.float()
        
        # STAGE 1: Add input perturbation
        node_input = self.add_input_perturbation(node_input, noise_scale=0.03)

        with torch.no_grad():
            if A_hat is not None:
                batch_size = 256
                all_outputs = []
                num_samples = node_input.size(0)

                for i in range(0, num_samples, batch_size):
                    batch = node_input[i:i+batch_size].to(device)
                    batch_out = model(A_hat, batch)
                    
                    # STAGE 2: Add noise during reconstruction
                    batch_out = batch_out + torch.randn_like(batch_out) * reconstruction_noise
                    
                    all_outputs.append(batch_out.detach())

                project_output = torch.cat(all_outputs, dim=0).detach().cpu()
            else:
                raise ValueError("A_hat must be provided")
            
        surrogate = []
        num_outputs, _, _, _ = project_output.shape
        
        for item in forget_indices:
            subset = []
            for i in range(item[0], item[1]):
                row = i - num_timesteps_input
                col = 0
                count = 0
                value = torch.zeros(3)
                
                while row >= num_outputs:
                    row = row - 1
                    col = col + 1
                    
                while row >= 0 and col < num_timesteps_output:
                    value += project_output[row, faulty_node_idx, col, :]
                    count += 1
                    row = row - 1
                    col = col + 1

                if count > 0:
                    value = value / count

                subset.append(value.unsqueeze(1))
                
            if subset:
                seg_tensor = torch.cat(subset, dim=1).unsqueeze(0)
                surrogate.append(seg_tensor.numpy())

        return surrogate

    # ============ STAGE 3: Statistical Distortion ============
    def apply_statistical_distortion(self, data: list, 
                                    forget_indices: list,
                                    distortion_strength: float = 0.15) -> list:
        """
        Apply statistical distortions to break correlation patterns
        """
        distorted_data = []
        
        for i, seq in enumerate(data):
            seq_copy = seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq)
            
            if i in forget_indices:
                # 1. Add frequency-domain noise
                if seq_copy.ndim >= 2 and seq_copy.shape[-1] > 4:
                    # Convert to frequency domain
                    fft_vals = torch.fft.rfft(seq_copy, dim=-1)
                    
                    # Add noise to high frequencies (preserves general trend)
                    noise_mask = torch.randn_like(fft_vals.real) * distortion_strength
                    fft_vals = fft_vals + noise_mask * (1 + 1j)
                    
                    # Convert back
                    seq_copy = torch.fft.irfft(fft_vals, n=seq_copy.shape[-1], dim=-1)
                
                # 2. Random time-shift (breaks temporal correlation)
                if seq_copy.shape[-1] > 3:
                    shift = np.random.randint(-2, 3)
                    seq_copy = torch.roll(seq_copy, shifts=shift, dims=-1)
                
                # 3. Scale perturbation (changes magnitude distribution)
                scale_factor = 1.0 + torch.randn(1).item() * 0.1
                seq_copy = seq_copy * scale_factor
            
            distorted_data.append(seq_copy)
        
        return distorted_data

    # ============ STAGE 4: Temporal Smoothing with Randomization ============
    def apply_adaptive_smoothing(self, data: list, 
                                 forget_indices: list,
                                 base_kernel_size: int = 3) -> list:
        """
        Apply adaptive temporal smoothing with randomization
        """
        smoothed_data = []
        
        for i, seq in enumerate(data):
            seq_copy = seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq)
            
            if i in forget_indices and seq_copy.ndim >= 2:
                # Randomize kernel size
                kernel_size = np.random.choice([3, 5, 7])
                kernel = torch.ones(kernel_size, device=seq_copy.device) / kernel_size
                
                # Add random weights to kernel
                kernel = kernel + torch.randn_like(kernel) * 0.05
                kernel = kernel / kernel.sum()
                
                padding = kernel_size // 2
                
                # Apply convolution
                smoothed = torch.nn.functional.conv1d(
                    seq_copy.unsqueeze(1), 
                    kernel.view(1, 1, -1), 
                    padding=padding
                )
                seq_copy = smoothed.squeeze(1)
            
            smoothed_data.append(seq_copy)
        
        return smoothed_data

    # ============ STAGE 5: Multi-Scale Noise Injection ============
    def add_multiscale_noise(self, data: list, 
                            forget_indices: list,
                            noise_scales: list = [0.02, 0.05, 0.1]) -> list:
        """
        Add noise at multiple scales to ensure diversity
        """
        noisy_data = []
        
        for i, seq in enumerate(data):
            seq_copy = seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq)
            
            if i in forget_indices:
                # Low frequency noise (trend)
                low_freq_noise = torch.randn_like(seq_copy) * noise_scales[0]
                if seq_copy.shape[-1] > 3:
                    # Smooth it
                    kernel = torch.ones(5) / 5.0
                    low_freq_noise = torch.nn.functional.conv1d(
                        low_freq_noise.unsqueeze(1),
                        kernel.view(1, 1, -1),
                        padding=2
                    ).squeeze(1)
                
                # Medium frequency noise
                med_freq_noise = torch.randn_like(seq_copy) * noise_scales[1]
                
                # High frequency noise (details)
                high_freq_noise = torch.randn_like(seq_copy) * noise_scales[2]
                
                # Combine
                seq_copy = seq_copy + low_freq_noise + med_freq_noise + high_freq_noise
            
            noisy_data.append(seq_copy)
        
        return noisy_data

    # ============ STAGE 6: Distribution Matching Prevention ============
    def prevent_distribution_matching(self, data: list, 
                                     forget_indices: list,
                                     shift_magnitude: float = 0.3) -> list:
        """
        Deliberately shift the distribution away from the forget set
        """
        shifted_data = []
        
        for i, seq in enumerate(data):
            seq_copy = seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq)
            
            if i in forget_indices:
                # Add a systematic bias
                bias = torch.randn(seq_copy.shape[:-1]).unsqueeze(-1) * shift_magnitude
                seq_copy = seq_copy + bias
                
                # Add random outliers
                outlier_mask = torch.rand_like(seq_copy) > 0.95
                seq_copy[outlier_mask] += torch.randn_like(seq_copy[outlier_mask]) * 0.5
            
            shifted_data.append(seq_copy)
        
        return shifted_data

    # ============ Main Pipeline ============
    def perform_temporal_generative_replay_subset(self, model: nn.Module, 
                                                  node_dataset,
                                                  forget_indices: Union[int, list],
                                                  faulty_node_idx: int,
                                                  num_timesteps_input,
                                                  num_timesteps_output,
                                                  device,
                                                  A_wave: Optional[torch.Tensor] = None,
                                                  aggressive_unlearning: bool = True) -> list:
        """
        Multi-stage T-GR pipeline with comprehensive noise injection
        """
        
        if self.model_type == "stgcn":
            # STAGE 1-2: Enhanced reconstruction with noise
            surrogate_sample = self.surrogate_stgcn_enhanced(
                model, node_dataset, forget_indices, faulty_node_idx, 
                num_timesteps_input, num_timesteps_output, device, A_wave,
                reconstruction_noise=0.03
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if aggressive_unlearning:
            # STAGE 3: Statistical distortion
            surrogate_sample = self.apply_statistical_distortion(
                surrogate_sample, forget_indices, distortion_strength=0.2
            )
            
            # STAGE 4: Adaptive smoothing
            surrogate_sample = self.apply_adaptive_smoothing(
                surrogate_sample, forget_indices, base_kernel_size=3
            )
            
            # STAGE 5: Multi-scale noise
            surrogate_sample = self.add_multiscale_noise(
                surrogate_sample, forget_indices, 
                noise_scales=[0.03, 0.06, 0.12]
            )
            
            # STAGE 6: Distribution shift
            surrogate_sample = self.prevent_distribution_matching(
                surrogate_sample, forget_indices, shift_magnitude=0.25
            )
        else:
            # Conservative approach (original)
            surrogate_sample = self.add_multiscale_noise(
                surrogate_sample, forget_indices,
                noise_scales=[0.01, 0.02, 0.05]
            )
        
        return surrogate_sample
    
    def perform_temporal_generative_replay_node(self, model: nn.Module, 
                                               dataset,
                                               faulty_node_idx: int,
                                               num_timesteps_input,
                                               num_timesteps_output,
                                               device,
                                               A_wave: Optional[torch.Tensor] = None,
                                               aggressive_unlearning: bool = True) -> list:
        """Node-level unlearning with multi-stage noise"""
        
        _, _, timestep = dataset[0].shape
        forget_indices = [[0, timestep]]
        surrogate_sample = []
        
        if self.model_type == "stgcn":
            for node_dataset in dataset:
                surr = self.surrogate_stgcn_enhanced(
                    model, node_dataset, forget_indices, faulty_node_idx, 
                    num_timesteps_input, num_timesteps_output, device, A_wave,
                    reconstruction_noise=0.03
                )
                surrogate_sample.append(surr)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        surrogate_sample = np.stack(surrogate_sample, axis=0)
        surrogate_sample = surrogate_sample.mean(axis=0)
        
        # Squeeze out unnecessary dimensions before converting to a tensor list
        surrogate_sample_squeezed = np.squeeze(surrogate_sample)
        # Ensure surrogate_list contains the entire squeezed tensor as one element
        surrogate_list = [torch.from_numpy(surrogate_sample_squeezed)]
        
        if aggressive_unlearning:
            surrogate_list = self.apply_statistical_distortion(
                surrogate_list, list(range(len(surrogate_list))), 0.2
            )
            surrogate_list = self.apply_adaptive_smoothing(
                surrogate_list, list(range(len(surrogate_list)))
            )
            surrogate_list = self.add_multiscale_noise(
                surrogate_list, list(range(len(surrogate_list))),
                noise_scales=[0.03, 0.06, 0.12]
            )
            surrogate_list = self.prevent_distribution_matching(
                surrogate_list, list(range(len(surrogate_list))), 0.25
            )
        else:
            surrogate_list = self.add_multiscale_noise(
                surrogate_list, list(range(len(surrogate_list))),
                noise_scales=[0.01, 0.02, 0.05]
            )
        
        return surrogate_list

class TGRConfig:
    """Configuration for T-GR noise injection stages"""
    
    # Preset configurations
    CONSERVATIVE = {
        'input_perturbation': 0.01,
        'reconstruction_noise': 0.01,
        'distortion_strength': 0.05,
        'noise_scales': [0.01, 0.02, 0.03],
        'distribution_shift': 0.1,
        'use_frequency_noise': False,
        'use_time_shift': False,
        'use_outliers': False,
    }
    
    MODERATE = {
        'input_perturbation': 0.03,
        'reconstruction_noise': 0.02,
        'distortion_strength': 0.15,
        'noise_scales': [0.02, 0.05, 0.08],
        'distribution_shift': 0.2,
        'use_frequency_noise': True,
        'use_time_shift': True,
        'use_outliers': False,
    }
    
    AGGRESSIVE = {
        'input_perturbation': 0.05,
        'reconstruction_noise': 0.03,
        'distortion_strength': 0.25,
        'noise_scales': [0.03, 0.06, 0.12],
        'distribution_shift': 0.35,
        'use_frequency_noise': True,
        'use_time_shift': True,
        'use_outliers': True,
    }
    
    EXTREME = {
        'input_perturbation': 0.08,
        'reconstruction_noise': 0.05,
        'distortion_strength': 0.4,
        'noise_scales': [0.05, 0.1, 0.2],
        'distribution_shift': 0.5,
        'use_frequency_noise': True,
        'use_time_shift': True,
        'use_outliers': True,
    }