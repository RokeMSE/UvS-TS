""" Run the 3 Orders from the Initial Models using files from the unlearning folder
- Combine Components: Load the pre-trained model ($θ*$).
- Partition Data: Use your PEPA implementation to get $D_f$ and $D_r$.
- Calculate FIM: Compute the PA-FIM ($F^T$) using $D_r$ and your PA-EWC module. """
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os

# Components
from models.stgcn import STGCN
from utils.data_loader import load_data_PEMS_BAY
from utils.data_utils import prepare_unlearning_data
from unlearning.pa_ewc import PopulationAwareEWC
from unlearning.t_gr import TemporalGenerativeReplay
from unlearning.motif_def import discover_motifs_proxy
from data.preprocess_pemsbay import get_normalized_adj
import sys
sys.path.append('src')

class SATimeSeries:
    """Complete SA-TS Framework Integration"""
    def __init__(self, model, A_hat, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Set adjacency matrix for STGCN
        if hasattr(self.model, 'set_adjacency_matrix'):
            self.model.set_adjacency_matrix(A_hat.to(self.device))
        
        # Store original parameters
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()
        
        # Initialize components
        self.pa_ewc = PopulationAwareEWC("stgcn", device)
        self.t_gr = TemporalGenerativeReplay("stgcn")
        self.fim_diagonal = None
        
    def unlearn_faulty_node(self, dataset, faulty_node_idx, 
                           num_epochs=50, learning_rate=1e-4, 
                           lambda_ewc=100.0, batch_size=32):
        """
        Main unlearning process for faulty node
        
        Args:
            dataset: Training dataset  
            faulty_node_idx: Index of node to forget
            num_epochs: Training epochs
            learning_rate: Learning rate
            lambda_ewc: EWC strength
            batch_size: Batch size
        """
        print(f"Starting unlearning for faulty node {faulty_node_idx}")
        
        # --- Partition data
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, faulty_node_idx, window_size=100
        )
        
        print(f"Forget samples: {len(forget_indices)}, Retain samples: {len(retain_indices)}")
        
        # Create data loaders
        forget_data = torch.stack([dataset[i][0] if isinstance(dataset[i], tuple) 
                                 else dataset[i] for i in forget_indices])
        retain_data = torch.stack([dataset[i][0] if isinstance(dataset[i], tuple) 
                                 else dataset[i] for i in retain_indices])
        
        forget_dataset = torch.utils.data.TensorDataset(forget_data)
        retain_dataset = torch.utils.data.TensorDataset(retain_data)
        
        forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=False)
        
        # --- Compute FIM
        print("Computing Population-Aware FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, max_samples=500
        )
        
        # --- Create surrogate data using T-GR
        print("Creating surrogate data...")
        surrogate_samples = []
        
        for batch in forget_loader:
            data_batch = batch[0].to(self.device)
            surrogate_batch = self.t_gr.perform_temporal_generative_replay(
                self.model, data_batch, faulty_node_idx
            )
            surrogate_samples.append(surrogate_batch)
        
        surrogate_data = torch.cat(surrogate_samples, dim=0)
        surrogate_dataset = torch.utils.data.TensorDataset(surrogate_data)
        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        
        # --- Unlearning optimization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        history = {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}
        
        for epoch in range(num_epochs):
            epoch_losses = {key: [] for key in history.keys()}
            
            # Iterate through both surrogate and retain data
            surrogate_iter = iter(surrogate_loader)
            retain_iter = iter(retain_loader)
            
            max_batches = max(len(surrogate_loader), len(retain_loader))
            
            for _ in range(max_batches):
                # Get batches
                try:
                    surrogate_batch = next(surrogate_iter)[0].to(self.device)
                except StopIteration:
                    surrogate_iter = iter(surrogate_loader)
                    surrogate_batch = next(surrogate_iter)[0].to(self.device)
                    
                try:
                    retain_batch = next(retain_iter)[0].to(self.device)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)[0].to(self.device)
                
                optimizer.zero_grad()
                
                # Compute losses
                losses = self.compute_sa_ts_objective(
                    surrogate_batch, retain_batch, lambda_ewc
                )
                
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Record losses
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key].item())
            
            # Average epoch losses
            for key in history:
                history[key].append(np.mean(epoch_losses[key]))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Total: {history['total_loss'][-1]:.4f}")
                print(f"  Surrogate: {history['surrogate_loss'][-1]:.4f}")
                print(f"  EWC: {history['ewc_penalty'][-1]:.4f}")
                print(f"  Retain: {history['retain_loss'][-1]:.4f}")
        
        return history
    
    def compute_sa_ts_objective(self, surrogate_batch, retain_batch, lambda_ewc):
        """Compute SA-TS objective"""
        # Ensure batches are 4D for model input
        if surrogate_batch.dim() == 3:
            surrogate_batch = surrogate_batch.unsqueeze(-1)
        if retain_batch.dim() == 3:
            retain_batch = retain_batch.unsqueeze(-1)
        
        # Forward passes
        surrogate_pred = self.model.forward_unlearning(surrogate_batch)
        retain_pred = self.model.forward_unlearning(retain_batch)
        
        # Prepare targets - use last time steps matching prediction length
        surrogate_target = surrogate_batch[:, :, -surrogate_pred.shape[2]:, 0]  # First feature
        retain_target = retain_batch[:, :, -retain_pred.shape[2]:, 0]
        
        # Calculate losses
        surrogate_loss = nn.MSELoss()(surrogate_pred, surrogate_target)
        retain_loss = nn.MSELoss()(retain_pred, retain_target)
        
        # EWC penalty
        if self.fim_diagonal is not None:
            ewc_penalty = self.pa_ewc.apply_ewc_penalty(
                self.model, self.fim_diagonal, self.original_params, lambda_ewc
            )
        else:
            ewc_penalty = torch.tensor(0.0, device=self.device)
        
        # Total SA-TS objective
        total_loss = -surrogate_loss + ewc_penalty + retain_loss
        
        return {
            'total_loss': total_loss,
            'surrogate_loss': surrogate_loss, 
            'ewc_penalty': ewc_penalty,
            'retain_loss': retain_loss
        }

def main():
    """Main execution function"""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_timesteps_input = 12
    num_timesteps_output = 3
    faulty_node_idx = 42  # Example faulty node
    
    # Load data
    print("Loading PEMS-BAY data...")
    A, X, means, stds = load_data_PEMS_BAY("data/PEMSBAY")
    
    # Prepare data
    split_line = int(X.shape[2] * 0.1)
    train_data = X[:, :, :split_line]
    
    # Convert to proper format
    training_input, training_target = fix_data_shapes(
        torch.from_numpy(train_data), num_timesteps_input, num_timesteps_output
    )
    
    print(f"Training input shape: {training_input.shape}")
    print(f"Training target shape: {training_target.shape}")
    
    # Create adjacency matrix
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float()
    
    # Create model
    print("Creating STGCN model...")
    model = STGCN(
        num_nodes=A_wave.shape[0],
        num_features=training_input.shape[3],
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output
    )
    
    # Load pre-trained weights if available
    checkpoint_path = "checkpoints/pretrained_stgcn.pth"
    if os.path.exists(checkpoint_path):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Warning: No pre-trained model found. Using randomly initialized model.")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(training_input, training_target)
    
    # Initialize SA-TS framework
    sa_ts = SATimeSeries(model, A_wave, device)
    
    # Run unlearning
    print(f"\nStarting unlearning for faulty node {faulty_node_idx}...")
    history = sa_ts.unlearn_faulty_node(
        dataset=dataset,
        faulty_node_idx=faulty_node_idx,
        num_epochs=50,
        learning_rate=1e-4,
        lambda_ewc=100.0,
        batch_size=16
    )
    
    # Save results
    print("\nSaving results...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'faulty_node_idx': faulty_node_idx
    }, f"results/unlearned_model_node_{faulty_node_idx}.pth")
    
    print("Unlearning completed!")

if __name__ == "__main__":
    main()