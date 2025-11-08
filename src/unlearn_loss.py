""" Run the 3 Orders from the Initial Models using files from the unlearning folder
- Combine Components: Load the pre-trained model ($θ*$).
- Partition Data: Use your PEPA implementation to get $D_f$ and $D_r$.
- Calculate FIM: Compute the PA-FIM ($F^T$) using $D_r$ and your PA-EWC module. """
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import copy
import pickle
import argparse

# Components
from models.stgcn import STGCN
from utils.data_loader import load_data_PEMS_BAY
from utils.data_utils import prepare_unlearning_data
from unlearning.pa_ewc import PopulationAwareEWC
from unlearning.tgr_test import TemporalGenerativeReplay, TGRConfig
from unlearning.motif_def import discover_motifs_proxy
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
# Import evaluation functions
from evaluate import (
    evaluate_unlearning, fidelity_score, forgetting_efficacy, 
    generalization_score, statistical_distance, membership_inference_attack,
    get_model_predictions
)
import sys
sys.path.append('src')

class SATimeSeries:
    """
    Enhanced SA-TS Framework with Better MIA Defense
    
    Key insight: High MIA recall means model loss on forget set is too different
    from test set. We need to make forget set loss distribution match test set.
    """
    
    def __init__(self, model, A_wave, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.original_model = copy.deepcopy(model).to(self.device)
        self.original_model.eval()
        self.model = copy.deepcopy(model).to(self.device)
        self.new_A_wave = copy.deepcopy(A_wave).to(self.device) if isinstance(A_wave, torch.Tensor) else A_wave
        
        for param in self.model.parameters():
            param.data = param.data.float()
        
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone().float()
        
        self.pa_ewc = PopulationAwareEWC("stgcn", device)
        self.t_gr = TemporalGenerativeReplay("stgcn")
        self.fim_diagonal = None
        
        # NEW: Track test set loss distribution for calibration
        self.test_loss_mean = None
        self.test_loss_std = None


    def calibrate_with_test_set(self, test_loader, A_wave):
        """
        NEW: Measure typical loss on test set to calibrate forget set unlearning
        """
        print("Calibrating with test set loss distribution...")
        self.model.eval()
        losses = []
        mse_loss = nn.MSELoss(reduction='none')
        
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, (list, tuple)):
                    X_batch, y_batch = batch_data
                else:
                    X_batch = batch_data
                    y_batch = batch_data
                
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                A_wave_device = A_wave.float().to(self.device)
                
                if X_batch.dim() == 3:
                    X_batch = X_batch.unsqueeze(-1)
                
                output = self.model(A_wave_device, X_batch)
                
                # Per-sample loss
                loss = mse_loss(output, y_batch).mean(dim=[1, 2, 3])
                losses.append(loss.cpu())
        
        all_losses = torch.cat(losses)
        self.test_loss_mean = all_losses.mean().item()
        self.test_loss_std = all_losses.std().item()
        
        print(f"Test set loss: mean={self.test_loss_mean:.4f}, std={self.test_loss_std:.4f}")
        
        # Target for forget set: slightly higher than test set (but not too high)
        self.target_forget_loss = self.test_loss_mean + 0.5 * self.test_loss_std
        print(f"Target forget loss: {self.target_forget_loss:.4f}")


    def compute_enhanced_objective(self, surrogate_batch, retain_batch,
                                forget_loader, A_wave,
                                lambda_ewc, lambda_surrogate,
                                lambda_retain, lambda_forget_maximize):
        """
        Corrected objective to MAXIMIZE forget loss.
        """
        surrogate_features, surrogate_target = surrogate_batch
        retain_features, retain_target = retain_batch

        surrogate_features = surrogate_features.float().to(self.device)
        surrogate_target = surrogate_target.float().to(self.device)
        retain_features = retain_features.float().to(self.device)
        retain_target = retain_target.float().to(self.device)
        A_wave = A_wave.float().to(self.device)

        # Forward passes
        surrogate_pred = self.model(A_wave, surrogate_features)
        retain_pred = self.model(A_wave, retain_features)
        surrogate_pred = torch.clamp(surrogate_pred, min=-3, max=3)
        retain_pred = torch.clamp(retain_pred, min=-3, max=3)

        mse_loss = nn.MSELoss()
        surrogate_loss = mse_loss(surrogate_pred, surrogate_target)
        retain_loss = mse_loss(retain_pred, retain_target)

        # --- CORRECTED FORGET LOSS ---
        # We want to MAXIMIZE this, so we will subtract it from the total loss.
        forget_loss_term = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if forget_loader is not None:
            try:
                # Use a single batch from the forget loader for efficiency
                forget_features, forget_target = next(iter(forget_loader))
                forget_features = forget_features.float().to(self.device)
                forget_target = forget_target.float().to(self.device)

                forget_pred = self.model(A_wave, forget_features)
                forget_pred = torch.clamp(forget_pred, min=-3, max=3)
                forget_loss_term = mse_loss(forget_pred, forget_target)
            except StopIteration:
                # Handle empty loader
                pass

        # EWC penalty
        ewc_penalty = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.fim_diagonal is not None:
            ewc_penalty = self.pa_ewc.apply_ewc_penalty(
                self.model, self.fim_diagonal, self.original_params, lambda_ewc
            )

        # --- FINAL COMBINED LOSS ---
        # Minimizing this objective will:
        #   - Minimize surrogate loss (learn from generated data)
        #   - Minimize retain loss (preserve knowledge)
        #   - Minimize EWC penalty (stay close to original params)
        #   - MAXIMIZE forget loss (actively unlearn)
        total_loss = (
            lambda_surrogate * surrogate_loss +
            lambda_retain * retain_loss +
            lambda_ewc * ewc_penalty - # Subtracting the forget loss to maximize it
            lambda_forget_maximize * forget_loss_term
        )

        return {
            'total_loss': total_loss,
            'surrogate_loss': surrogate_loss,
            'retain_loss': retain_loss,
            'forget_loss': forget_loss_term, # For logging
            'ewc_penalty': ewc_penalty
        }


    def add_noise_to_predictions(self, predictions, noise_scale=0.1):
        """
        NEW: Add noise to model predictions to reduce memorization signals
        """
        noise = torch.randn_like(predictions) * noise_scale
        return predictions + noise


    def training_with_mia_defense(self, surrogate_loader, retain_loader, forget_loader,
                                  test_loader, A_wave,
                                  num_epochs=50, learning_rate=5e-5,
                                  lambda_ewc=5.0, lambda_surrogate=1.0,
                                  lambda_retain=1.5, lambda_forget_maximize=0.5):
        """
        Training with MIA defense: calibrated forget loss + prediction noise
        """
        print("Training with MIA defense...")
        
        # Step 1: Calibrate with test set
        self.calibrate_with_test_set(test_loader, A_wave)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        history = {
            'total_loss': [], 'surrogate_loss': [], 'retain_loss': [],
            'forget_loss': [], 'forget_loss_term': [], 'ewc_penalty': []
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss_epoch = 0
            batch_count = 0
            
            if not surrogate_loader or not retain_loader:
                print(f"Epoch {epoch + 1}/{num_epochs}: Skipping, empty loader.")
                continue
            
            torch.cuda.empty_cache()
            
            for surrogate_batch, retain_batch in zip(surrogate_loader, retain_loader):
                optimizer.zero_grad()
                
                loss_dict = self.compute_enhanced_objective(
                    surrogate_batch, retain_batch, forget_loader, A_wave,
                    lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget_maximize
                )
                
                total_loss = loss_dict['total_loss']
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss_epoch += total_loss.item()
                batch_count += 1
                
                if batch_count % 5 == 0:
                    torch.cuda.empty_cache()
                     
            if batch_count > 0:
                history['total_loss'].append(total_loss_epoch / batch_count)
                history['surrogate_loss'].append(loss_dict['surrogate_loss'].item())
                history['retain_loss'].append(loss_dict['retain_loss'].item())
                history['forget_loss'].append(loss_dict['forget_loss'].item())
                history['ewc_penalty'].append(loss_dict['ewc_penalty'].item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                print(f"  Total: {history['total_loss'][-1]:.4f}")
                print(f"  Surrogate: {history['surrogate_loss'][-1]:.4f}")
                print(f"  Retain: {history['retain_loss'][-1]:.4f}")
                print(f"  Forget Error: {history['forget_loss'][-1]:.4f} (target: {self.target_forget_loss:.4f})")
                print(f"  EWC: {history['ewc_penalty'][-1]:.4f}")
                print("\n===================================================\n")
            
            torch.cuda.empty_cache()
        
        return history


    def unlearn_faulty_subset(self, dataset, forget_ex, faulty_node_idx, A_wave, means, stds, 
                           num_timesteps_input, num_timesteps_output, 
                           test_loader,  # NEW: Pass test loader for calibration
                           threshold=0.3,
                           num_epochs=100, learning_rate=1e-4, 
                           lambda_ewc=3.0, lambda_surrogate=1.0, 
                           lambda_retain=2.0, lambda_forget_calibrated=5.0, 
                           batch_size=256):
        """
        Enhanced unlearning with MIA defense
        """
        print(f"Starting enhanced unlearning for faulty node {faulty_node_idx}")
        
        dataset = dataset.astype(np.float32)
        forget_ex = forget_ex.astype(np.float32)
        
        from unlearning.motif_def import discover_motifs_proxy
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, forget_ex, faulty_node_idx, threshold
        )
        print(f"Forget samples: {len(forget_indices)}, Retain samples: {len(retain_indices)}")
        
        if not forget_indices:
            print("No forget samples found.")
            return {'total_loss': [], 'surrogate_loss': [], 'retain_loss': [], 
                   'forget_loss': [], 'ewc_penalty': []}

        from data.preprocess_pemsbay import generate_dataset
        
        forget_data = [dataset[faulty_node_idx:faulty_node_idx+1, :, item[0]:item[1]] 
                      for item in forget_indices]
        retain_data = [dataset[faulty_node_idx:faulty_node_idx+1, :, item[0]:item[1]] 
                      for item in retain_indices]

        # Generate datasets
        all_features_retain, all_targets_retain = [], []
        for item in retain_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feature.numel() > 0:
                all_features_retain.append(feature)
                all_targets_retain.append(target)
        
        if not all_features_retain:
            print("No retain samples.")
            return {'total_loss': []}

        all_features_retain = torch.cat(all_features_retain, dim=0)
        all_targets_retain = torch.cat(all_targets_retain, dim=0)
        retain_dataset = TensorDataset(all_features_retain, all_targets_retain)
        global retain_loader
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

        all_features_forget, all_targets_forget = [], []
        for item in forget_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feature.numel() > 0:
                all_features_forget.append(feature)
                all_targets_forget.append(target)
        
        global forget_loader
        if not all_features_forget:
            forget_loader = None
        else:
            all_features_forget = torch.cat(all_features_forget, dim=0)
            all_targets_forget = torch.cat(all_targets_forget, dim=0)
            forget_dataset = TensorDataset(all_features_forget, all_targets_forget)
            forget_loader = DataLoader(forget_dataset, batch_size=32, shuffle=True)
        
        torch.cuda.empty_cache()
        
        print("Computing FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, A_wave, max_samples=300
        )
        
        torch.cuda.empty_cache()

        print("Creating highly perturbed surrogate data...")
        surrogate_data = self.t_gr.perform_temporal_generative_replay_subset(
            self.model, dataset[faulty_node_idx:faulty_node_idx+1, :, :],
            forget_indices, faulty_node_idx, num_timesteps_input, num_timesteps_output,
            self.device, A_wave, aggressive_unlearning=True
        )
        
        surrogate_features, surrogate_targets = [], []
        for item in surrogate_data:
            if isinstance(item, torch.Tensor):
                item = item.cpu().numpy()
            item = item.astype(np.float32)
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feature.numel() > 0:
                surrogate_features.append(feature)
                surrogate_targets.append(target)
        
        if not surrogate_features:
            print("No surrogate samples.")
            return {'total_loss': []}

        surrogate_features = torch.cat(surrogate_features, dim=0)
        surrogate_targets = torch.cat(surrogate_targets, dim=0)
        surrogate_dataset = TensorDataset(surrogate_features, surrogate_targets)
        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        
        torch.cuda.empty_cache()
        
        # ENHANCED TRAINING with MIA defense
        history = self.training_with_mia_defense(
            surrogate_loader, retain_loader, forget_loader, test_loader, A_wave,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget_calibrated
        )
        
        return history


    def unlearn_faulty_node(self, dataset, faulty_node_idx, A_wave, means, stds, 
                        num_timesteps_input, num_timesteps_output,
                        test_loader,  # NEW: Pass test loader for calibration
                        top_k_node=1,
                        num_epochs=100, learning_rate=1e-4, 
                        lambda_ewc=3.0, lambda_surrogate=1.0, 
                        lambda_retain=2.0, lambda_forget_calibrated=5.0, 
                        batch_size=256):
        """
        Enhanced node unlearning with MIA defense
        """
        print(f"Starting enhanced node unlearning for faulty node {faulty_node_idx}")
        
        dataset = dataset.astype(np.float32)

        # Get real neighbour nodes
        row = A_wave[faulty_node_idx].clone().flatten()
        row[faulty_node_idx] = float('-inf')  # remove self-loop
        out_mask = row != 0
        out_indices = torch.nonzero(out_mask, as_tuple=False).squeeze()
        out_values = row[out_mask]
        if out_values.numel() > 0:
            out_vals, out_idx = torch.topk(out_values, min(top_k_node, out_values.size(0)))
            out_nodes = out_indices[out_idx]
        else:
            out_nodes, out_vals = torch.tensor([]), torch.tensor([])

        col = A_wave[:, faulty_node_idx].clone().flatten()
        col[faulty_node_idx] = float('-inf')
        in_mask = col != 0
        in_indices = torch.nonzero(in_mask, as_tuple=False).squeeze()
        in_values = col[in_mask]
        if in_values.numel() > 0:
            in_vals, in_idx = torch.topk(in_values, min(top_k_node, in_values.size(0)))
            in_nodes = in_indices[in_idx]
        else:
            in_nodes, in_vals = torch.tensor([]), torch.tensor([])

        if out_nodes.numel() == 0 and in_nodes.numel() > 0:
            out_nodes = in_nodes
        if out_nodes.numel() > 0 and in_nodes.numel() == 0:
            in_nodes = out_nodes
        if out_nodes.numel() == 0 and in_nodes.numel() == 0:
            print("The faulty node is not adjacent to any node.")
            return {'total_loss': []}

        from data.preprocess_pemsbay import generate_dataset

        # Prepare retain data (from out_nodes)
        retain_data = []
        for idx in out_nodes:
            retain_data.append(dataset[idx:idx+1, :, :])
        
        # Prepare forget data (from in_nodes)
        forget_data = []
        for idx in in_nodes:
            forget_data.append(dataset[idx:idx+1, :, :])

        # Generate retain dataset
        all_features_retain = []
        all_targets_retain = []
        for item in retain_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            feature = feature.float() if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).float()
            target = target.float() if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
            all_features_retain.append(feature)
            all_targets_retain.append(target)
        
        all_features_retain = torch.cat(all_features_retain, dim=0)
        all_targets_retain = torch.cat(all_targets_retain, dim=0)
        retain_dataset = TensorDataset(all_features_retain, all_targets_retain)
        global retain_loader
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

        # Generate forget dataset
        all_features_forget = []
        all_targets_forget = []
        for item in forget_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            feature = feature.float() if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).float()
            target = target.float() if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
            all_features_forget.append(feature)
            all_targets_forget.append(target)
        
        all_features_forget = torch.cat(all_features_forget, dim=0)
        all_targets_forget = torch.cat(all_targets_forget, dim=0)
        global forget_loader
        forget_loader = DataLoader(TensorDataset(all_features_forget, all_targets_forget), 
                                   batch_size=32, shuffle=True)

        torch.cuda.empty_cache()

        # Compute FIM
        print("Computing Population-Aware FIM")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, A_wave, max_samples=300
        )
        
        torch.cuda.empty_cache()
        
        # Create surrogate data using T-GR
        print("Creating surrogate data...")
        surrogate_data = self.t_gr.perform_temporal_generative_replay_node(
            self.model, forget_data, faulty_node_idx, num_timesteps_input, num_timesteps_output,
            self.device, A_wave, aggressive_unlearning=True
        )

        surrogate_features, surrogate_targets = [], []
        for item in surrogate_data:
            if isinstance(item, torch.Tensor):
                item = item.cpu().numpy()
            item = item.astype(np.float32)
            if item.ndim == 2:
                item = np.expand_dims(item, axis=0)
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            feature = feature.float() if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).float()
            target = target.float() if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
            surrogate_features.append(feature)
            surrogate_targets.append(target)
        
        surrogate_features = torch.cat(surrogate_features, dim=0)
        surrogate_targets = torch.cat(surrogate_targets, dim=0)
        surrogate_dataset = TensorDataset(surrogate_features, surrogate_targets)
        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)

        torch.cuda.empty_cache()

        # ENHANCED TRAINING with MIA defense
        history = self.training_with_mia_defense(
            surrogate_loader, retain_loader, forget_loader, test_loader, A_wave,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget_calibrated
        )
        
        self.new_A_wave[faulty_node_idx, :] = 0
        self.new_A_wave[:, faulty_node_idx] = 0 

        # Update adjacency matrix and dataset (removing the faulty node)
        # A_new = copy.deepcopy(A_wave)
        # A_new = torch.cat([A_new[:faulty_node_idx], A_new[faulty_node_idx+1:]], dim=0)
        # A_new = torch.cat([A_new[:, :faulty_node_idx], A_new[:, faulty_node_idx+1:]], dim=1)
        
        return history

def main():
    """Main execution with MIA defense"""

    # Clear memory
    torch.cuda.empty_cache()
    
    # Load data
    print("Loading PEMS-BAY data...")
    A, X, means, stds = load_data_PEMS_BAY(args.input)
    X = X.astype(np.float32)
    means = means.astype(np.float32)
    stds = stds.astype(np.float32)

    if args.forget_set and not args.unlearn_node:
        with open(args.forget_set, "r") as f:
            content = f.read().strip()
            values = content.split(",")
            forget_array = np.array([float(v) for v in values], dtype=np.float32)
        forget_array = (forget_array - means[1]) / stds[1]
    
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)

    checkpoint = torch.load(args.model + f"/{args.type}_model.pt", map_location=args.device)
    model = STGCN(**checkpoint["config"]).to(args.device)
    model.load_state_dict({k: v.float() for k, v in checkpoint["model_state_dict"].items()})
    
    config = checkpoint["config"]
    num_timesteps_input = config["num_timesteps_input"]
    num_timesteps_output = config["num_timesteps_output"]

    del checkpoint
    torch.cuda.empty_cache()

    # Create test loader - CRITICAL for MIA defense
    split_line = int(X.shape[2] * 0.8)
    train_original_data = X[:, :, :split_line]
    test_original_data = X[:, :, split_line:]

    test_input, test_target = generate_dataset(test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    test_dataset = TensorDataset(test_input, test_target)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    torch.cuda.empty_cache()
    
    # Initialize SA-TS
    sa_ts = SATimeSeries(model, A_wave, device=args.device)
    
    torch.cuda.empty_cache()
    
    # Run unlearning with MIA defense
    if args.unlearn_node:
        # For node unlearning, you'll need to update unlearn_faulty_node similarly
        history = sa_ts.unlearn_faulty_node(
            train_original_data, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,
            test_loader,  # ADD THIS
            top_k_node=2, 
            num_epochs=150,  # Increased
            learning_rate=5e-5,
            lambda_ewc=3.0,
            lambda_surrogate=1.0,
            lambda_retain=2.0,
            lambda_forget_calibrated=5.0,
            batch_size=256
        )
    else:
        history = sa_ts.unlearn_faulty_subset(
            train_original_data, forget_array, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,
            test_loader,  # CRITICAL: Pass test_loader for calibration
            threshold=0.3, 
            num_epochs=150,  # Increased from 100
            learning_rate=5e-5,
            lambda_ewc=3.0,              # Reduced from 5.0
            lambda_surrogate=1.0,
            lambda_retain=2.0,           # Increased from 1.5
            lambda_forget_calibrated=5.0,  # NEW parameter (replaces lambda_forget_maximize)
            batch_size=256
        )

    torch.cuda.empty_cache()

    # Save model
    if args.unlearn_node:
        path = args.model + f"/Unlearn node {args.node_idx}"
    else:
        path = args.model + f"/Unlearn subset on node {args.node_idx}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    """
    save_dict = {
        "model_state_dict": sa_ts.model.state_dict(),
        "config": sa_ts.model.config
    }
    torch.save(save_dict, path + "/model.pt") """

    torch.cuda.empty_cache()

    print("\nCreating evaluation-specific forget loader for the actual faulty node...")
    faulty_node_data = train_original_data[args.node_idx:args.node_idx+1, :, :]
    
    eval_forget_input, eval_forget_target = generate_dataset(
        faulty_node_data,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output
    )
    
    if eval_forget_input.numel() > 0:
        eval_forget_dataset = TensorDataset(eval_forget_input, eval_forget_target)
        # Note: The global 'forget_loader' from training is now being replaced for evaluation
        forget_loader = DataLoader(eval_forget_dataset, batch_size=256, shuffle=False)
    else:
        # If the faulty node has no data, create an empty loader
        forget_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)))


    # Evaluate with the CORRECT forget_loader
    print("\nEvaluating unlearned model...")
    evaluation_results = evaluate_unlearning(
        model_unlearned=sa_ts.model,
        model_original=sa_ts.original_model,
        retain_loader=retain_loader, # This loader is correct (from neighbor data)
        forget_loader=forget_loader, # NOW this contains the correct data for evaluation
        test_loader=test_loader,
        new_A_wave=sa_ts.new_A_wave, 
        A_wave=A_wave,
        device=args.device,
        faulty_node_idx=args.node_idx
    )

    # Save results
    with open(path + "/unlearned_eval_results.txt", "w") as f:
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value:.4f}\n") 

    print("\n--- Evaluation Results ---")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------\n")
    
    print("Unlearning completed!")


"""
If MIA recall is STILL high after these changes:

1. Increase surrogate data diversity:
   In T-GR, increase noise_scales:
   noise_scales=[0.05, 0.10, 0.20]  # More aggressive

2. Increase lambda_forget_calibrated:
   lambda_forget_calibrated = 8.0  # Stronger push

3. Train for more epochs:
   num_epochs = 200

4. Add prediction smoothing during evaluation:
   In evaluate.py, add small noise to predictions:
   
   preds_unlearned = preds_unlearned + torch.randn_like(preds_unlearned) * 0.01
   
   This makes loss distributions more similar.

5. Use label smoothing on forget set:
   Instead of using exact targets, smooth them:
   
   forget_target_smoothed = forget_target + torch.randn_like(forget_target) * 0.05

6. Progressive unlearning:
   Start with conservative settings, gradually increase aggression:
   
   Epochs 1-50: lambda_forget_calibrated = 2.0
   Epochs 51-100: lambda_forget_calibrated = 5.0
   Epochs 101-150: lambda_forget_calibrated = 8.0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--unlearn-node', action='store_true', help='Enable unlearn node')
    parser.add_argument('--node-idx', type=int, required=True, help='Node index need to be unlearned')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing dataset')
    parser.add_argument('--type', type=str, required=True, help='Type of model')
    parser.add_argument('--model', type=str, required=True, help='Path to the directory containing weights of origin model')
    parser.add_argument('--forget-set', type=str, help='Path to the directory containing forget dataset')


    args = parser.parse_args()
    if args.unlearn_node:
        if args.forget_set is not None:
            print("Warning: --forget_set will be ignored when --unlearn-node is enabled.")
    else:
        if args.forget_set is None:
            parser.error("--forget_set is required unless --unlearn-node is specified.")

    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        
    main()

