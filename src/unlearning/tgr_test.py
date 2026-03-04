import torch
import torch.nn as nn
import numpy as np
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
from typing import Optional, Union, Tuple
import time

class TemporalGenerativeReplay:
    """
    Self-Contained Temporal Generative Replay (T-GR) implementation
    Uses the model's own capabilities for reconstruction and neutralization
    """
    def __init__(self, model_type: str = "stgcn"): # Change the model type for different tests
        """
        Params:
            model_type: Type of model ("stgcn", "", "")
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
                
        return masked_data.to(torch.float32)
    
# ----------------- Model reconstruction --------------------
    def surrogate(self, model: nn.Module, forget_input, 
                        forget_indices: list, faulty_node_idx: int, num_timesteps_input, num_timesteps_output, 
                        device, A_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct using STGCN model"""
        model.eval()

        with torch.no_grad():
            if hasattr(model, 'forward_unlearning_compatible'):
                project_output = model.forward_unlearning_compatible(forget_input)
            elif A_hat is not None:
                batch_size = 256
                all_outputs = []
                num_samples = forget_input.size(0)

                for i in range(0, num_samples, batch_size):
                    batch = forget_input[i:i+batch_size].to(device)
                    batch_out = model(A_hat, batch)
                    all_outputs.append(batch_out.detach())

                project_output = torch.cat(all_outputs, dim=0).detach().cpu()
                # (B, N, T, F)
            else:
                raise ValueError("Either model must have forward_unlearning or A_hat must be provided")
            
        B, N, T_out, F = project_output.shape
        row_ids = torch.arange(B).unsqueeze(1)                # (B,1)
        col_ids = torch.arange(T_out).unsqueeze(0)            # (1,T_out)

        global_times = row_ids + num_timesteps_input + col_ids

        global_times_flat = global_times.reshape(-1)
        # outputs_flat = project_output.reshape(-1, N, F)

        node_outputs = project_output[:, faulty_node_idx, :, :] 
        outputs_flat = node_outputs.reshape(-1, F)
        
        surrogate = []
        for start, end in forget_indices:
            interval_mask = (global_times_flat >= start) & (global_times_flat < end)

            times = global_times_flat[interval_mask]
            outputs = outputs_flat[interval_mask]

            unique_times = torch.unique(times)
            unique_times, _ = torch.sort(unique_times)
            
            seg = torch.zeros(F, unique_times.shape[0])

            for idx, t in enumerate(unique_times):
                seg[:, idx] = outputs[times == t].mean(0)

            surrogate.append(seg)        


        return surrogate

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
        """Single denoising step for  model"""
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
    

    def add_error_minimizing_noise(self, data: list, 
                                  forget_indices: Union[int, list],
                                  noise_scale: float = 0.01) -> torch.Tensor:
        """
        Add imperceptible error-minimizing noise to promote unlearning
        Inspired by unlearnable examples literature
        What this will do is make the model less sensitive to the forgotten patterns because it introduces noise in a controlled manner.
        """      

        noisy_data = []

        for i, seq in enumerate(data):
            seq_copy = seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq)

            if i in forget_indices:
                # Thêm noise Gaussian cùng shape
                noise = torch.randn_like(seq_copy) * noise_scale
                seq_copy = seq_copy + noise

                # Optionally: smooth theo chiều thời gian (axis=-1)
                if seq_copy.ndim >= 2:
                    kernel = torch.ones(3, device=seq_copy.device) / 3.0
                    padding = (1,)  # same padding cho conv1d
                    # reshape về (batch=features, channel=1, length=time)
                    smoothed = torch.nn.functional.conv1d(
                        seq_copy.unsqueeze(1), 
                        kernel.view(1, 1, -1), 
                        padding=padding
                    )
                    seq_copy = smoothed.squeeze(1)

            noisy_data.append(seq_copy)

        return noisy_data
    

    def perform_temporal_generative_replay_subset(self, model: nn.Module, 
                                         forget_input,
                                         forget_indices: Union[int, list],
                                         faulty_node_idx: int,
                                         num_timesteps_input,
                                         num_timesteps_output,
                                         device,
                                         A_wave: Optional[torch.Tensor] = None) -> list:
        """
        Main T-GR function implementing Reconstruction and Neutralization
        
        Params:
            model: The model being unlearned
            node_dataset: Dataset of faulty node
            forget_indices: Indices to be neutralized
            faulty_node_idx: Index of faulty node
            num_timesteps_input: number of sample input
            num_timesteps_output: number of sample output
            device,
            A_wave: Graph edges (for STGCN)
            
        Returns:
            Neutralized surrogate data
        """
        

        surrogate_sample = self.surrogate(model, forget_input, forget_indices, faulty_node_idx, num_timesteps_input, num_timesteps_output, device, A_wave)
        # [(F, T)]
            

        surrogate_sample = self.add_error_minimizing_noise(
            surrogate_sample, forget_indices, noise_scale=1.5 # Change accordingly to fit the desired 
        )
        # [(F, T)]
        
        return [i.transpose(0, 1) for i in surrogate_sample]
    
    def perform_temporal_generative_replay_node(self, model: nn.Module, 
                                         dataset,
                                         faulty_node_idx: int,
                                         num_timesteps_input,
                                         num_timesteps_output,
                                         device,
                                         A_wave: Optional[torch.Tensor] = None) -> list:
        """
        Main T-GR function implementing Reconstruction and Neutralization
        
        Params:
            model: The model being unlearned
            dataset: Dataset of faulty node
            forget_indices: Indices to be neutralized
            faulty_node_idx: Index of faulty node
            num_timesteps_input: number of sample input
            num_timesteps_output: number of sample output
            device,
            A_wave: Graph edges (for STGCN)
            
        Returns:
            Neutralized surrogate data
        """
        
        _, _, timestep = dataset[0].shape
        forget_indices = [[0, timestep]]
        surrogate_sample = []
        print(forget_indices)
        if self.model_type == "stgcn":
            for node_dataset in dataset:
                surrogate_sample.append(self.surrogate(model, node_dataset, forget_indices, faulty_node_idx, 
                                                             num_timesteps_input, num_timesteps_output, device, A_wave))
            # (B, N, F, T)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        surrogate_sample = np.stack(surrogate_sample, axis=0)
        surrogate_sample = surrogate_sample.mean(axis=0)
        
        surrogate_sample = self.add_error_minimizing_noise(
            surrogate_sample, forget_indices, noise_scale=1.5 # Change accordingly to fit the desired 
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