""" Run the 3 Orders from the Initial Models using files from the unlearning folder
- Combine Components: Load the pre-trained model ($θ*$).
- Partition Data: Use your PEPA implementation to get $D_f$ and $D_r$.
- Calculate FIM: Compute the PA-FIM ($F^T$) using $D_r$ and your PA-EWC module. """
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse

# Components
from models.stgcn import STGCN
from utils.data_loader import load_data_PEMS_BAY
from utils.data_utils import prepare_unlearning_data
from unlearning.pa_ewc import PopulationAwareEWC
from unlearning.t_gr import TemporalGenerativeReplay
from unlearning.motif_def import discover_motifs_proxy
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
import sys
sys.path.append('src')

parser = argparse.ArgumentParser(description='Unlearning')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('--input', type=str, required=True,
                    help='Path to the directory containing dataset')
parser.add_argument('--model', type=str, required=True,
                    help='Path to the directory containing weights of origin model')
parser.add_argument('--forget_set', type=str, required=True,
                    help='Path to the directory containing forget dataset')

args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

class SATimeSeries:
    """Complete SA-TS Framework Integration"""
    def __init__(self, model, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Set adjacency matrix for STGCN
        # if hasattr(self.model, 'set_adjacency_matrix'):
        #     self.model.set_adjacency_matrix(A_hat.to(self.device))
        
        # Store original parameters
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()
        
        # Initialize components
        self.pa_ewc = PopulationAwareEWC("stgcn", device)
        self.t_gr = TemporalGenerativeReplay("stgcn")
        self.fim_diagonal = None
        
    def unlearn_faulty_node(self, dataset, forget_ex, faulty_node_idx, A_wave, 
                           num_timesteps_input, num_timesteps_output,threshold=0.3,
                           num_epochs=50, learning_rate=1e-4, 
                           lambda_ewc=100.0, batch_size=32):
        """
        Main unlearning process for faulty node
        
        Args:
            dataset: Training dataset (N, F, T) 
            faulty_node_idx: Index of node to forget
            num_epochs: Training epochs
            learning_rate: Learning rate
            lambda_ewc: EWC strength
            batch_size: Batch size
        """
        print(f"Starting unlearning for faulty node {faulty_node_idx}")
        
        # --- Partition data
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, forget_ex, faulty_node_idx, threshold
        )

        #[[0, 4]]

        print(f"Forget samples: {len(forget_indices)}, Retain samples: {len(retain_indices)}")
        forget_data = []
        for item in forget_indices:
            forget_data.append(dataset[faulty_node_idx:faulty_node_idx+1, :,item[0]:item[1]]) 

        retain_data = []
        for item in retain_indices:
            retain_data.append(dataset[faulty_node_idx:faulty_node_idx+1, :,item[0]:item[1]])

        all_features = []
        all_targets = []
        for item in retain_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output) #(B, N, T, F), (B, N, T, F)
            all_features.append(feature)
            all_targets.append(target)
        

        all_features = torch.cat(all_features, dim=0)  # (Tổng sample, num_vertices, num_timesteps_input, num_features)
        all_targets = torch.cat(all_targets, dim=0)

        retain_dataset = TensorDataset(all_features, all_targets)
        retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True)

        all_features = []
        all_targets = []
        for item in forget_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            all_features.append(feature)
            all_targets.append(target)
        
        all_features = torch.cat(all_features, dim=0)  # (Tổng sample, num_vertices, num_timesteps_input, num_features)
        all_targets = torch.cat(all_targets, dim=0)

        forget_dataset = TensorDataset(all_features, all_targets)
        forget_loader = DataLoader(forget_dataset, batch_size=64, shuffle=True)
        
        # --- Compute FIM
        print("Computing Population-Aware FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, A_wave, max_samples=500
        )

        #--- Create surrogate data using T-GR
        

        print("Creating surrogate data...")

        surrogate_data = self.t_gr.perform_temporal_generative_replay(self.model, dataset[faulty_node_idx:faulty_node_idx+1, :, :],
                                                     forget_indices, faulty_node_idx, num_timesteps_input, num_timesteps_output, self.device, A_wave)
        
        all_features = []
        all_targets = []
        for item in surrogate_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            all_features.append(feature)
            all_targets.append(target)
        
        all_features = torch.cat(all_features, dim=0)  # (Tổng sample, num_vertices, num_timesteps_input, num_features)
        all_targets = torch.cat(all_targets, dim=0)

        surrogate_dataset = TensorDataset(all_features, all_targets)
        surrogate_loader = DataLoader(surrogate_dataset, batch_size=64, shuffle=True)

        # --- Unlearning optimization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print('Unlearn training...')

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
                    surrogate_batch = next(surrogate_iter)
                except StopIteration:
                    surrogate_iter = iter(surrogate_loader)
                    surrogate_batch = next(surrogate_iter)
                    
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_indices)
                    retain_batch = next(retain_iter)
                
                optimizer.zero_grad()
                
                # Compute losses
                losses = self.compute_sa_ts_objective(
                    surrogate_batch, retain_batch, lambda_ewc, A_wave
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
                print("\n===================================================\n")
        
        return history
    
    def compute_sa_ts_objective(self, surrogate_batch, retain_batch, lambda_ewc, A_wave):
        """Compute SA-TS objective"""
        # # Ensure batches are 4D for model input
        # if surrogate_batch.dim() == 3:
        #     surrogate_batch = surrogate_batch.unsqueeze(-1)
        # if retain_batch.dim() == 3:
        #     retain_batch = retain_batch.unsqueeze(-1)

        surrogate_features, surrogate_target = surrogate_batch
        retain_features, retain_target = retain_batch

        surrogate_features = surrogate_features.float().to(self.device)
        surrogate_target = surrogate_target.float().to(self.device)
        retain_features = retain_features.float().to(self.device)
        retain_target = retain_target.float().to(self.device)
        # Forward passes
        surrogate_pred = self.model(A_wave, surrogate_features)
        retain_pred = self.model(A_wave, retain_features)
        
        # # Prepare targets - use last time steps matching prediction length
        # surrogate_target = surrogate_batch[:, :, -surrogate_pred.shape[2]:, 0]  # First feature
        # retain_target = retain_batch[:, :, -retain_pred.shape[2]:, 0]
        
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
    
    faulty_node_idx = 0  # Example faulty node

    with open(args.forget_set, "r") as f:
        content = f.read().strip()
        values = content.split(",")
        forget_array = np.array([int(v) for v in values], dtype=np.float32)
        
    # Load data
    print("Loading PEMS-BAY data...")
    A, X, means, stds = load_data_PEMS_BAY(args.input)
    #(N, F, T)
    forget_array = (forget_array - means[1]) / stds[1]
    
    # # Prepare data
    # split_line = int(X.shape[2] * 0.1)
    # train_data = X[:, :, :split_line]
    
    # # Convert to proper format
    # training_input, training_target = fix_data_shapes(
    #     torch.from_numpy(train_data), num_timesteps_input, num_timesteps_output
    # )
    
    # print(f"Training input shape: {training_input.shape}")
    # print(f"Training target shape: {training_target.shape}")
    
    # # Create adjacency matrix
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float()
    A_wave = A_wave.to(device=args.device)

    # # Create model
    # print("Creating STGCN model...")
    # model = STGCN(
    #     num_nodes=A_wave.shape[0],
    #     num_features=training_input.shape[3],
    #     num_timesteps_input=num_timesteps_input,
    #     num_timesteps_output=num_timesteps_output
    # )
    
    # # Load pre-trained weights if available
    # checkpoint_path = "checkpoints/pretrained_stgcn.pth"
    # if os.path.exists(checkpoint_path):
    #     print("Loading pre-trained model...")
    #     model.load_state_dict(torch.load(checkpoint_path))
    # else:
    #     print("Warning: No pre-trained model found. Using randomly initialized model.")
    
    checkpoint = torch.load(args.model + "/model.pt", map_location=args.device)
    model = STGCN(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    config = checkpoint["config"]

    num_nodes = config["num_nodes"]
    num_features = config["num_features"]
    num_timesteps_input = config["num_timesteps_input"]
    num_timesteps_output = config["num_timesteps_output"]
    num_features_output = config["num_features_output"]

    # Create dataset
    #dataset = torch.utils.data.TensorDataset(training_input, training_target)
    
    # Initialize SA-TS framework
    sa_ts = SATimeSeries(model,  args.device)
    # Run unlearning
    # print(f"\nStarting unlearning for faulty node {faulty_node_idx}...")
    history = sa_ts.unlearn_faulty_node(
        X,
        forget_array,
        faulty_node_idx,
        A_wave, 
        num_timesteps_input, 
        num_timesteps_output,
        threshold=0.3,
        num_epochs=50,
        learning_rate=1e-4,
        lambda_ewc=100.0,
        batch_size=128
    )
    
    # Save results
    print("\nSaving results...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'faulty_node_idx': faulty_node_idx
    }, args.model + "/unlearned_model.pt")
    
    print("Unlearning completed!")

if __name__ == "__main__":
    main()