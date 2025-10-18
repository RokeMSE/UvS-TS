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
from unlearning.t_gr import TemporalGenerativeReplay
from unlearning.motif_def import discover_motifs_proxy
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
# Import evaluation functions
from evaluate import evaluate_unlearning
import sys
sys.path.append('src')

class SATimeSeries:
    """Complete SA-TS Framework Integration"""
    def __init__(self, model, A_wave, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # --- Store the original model state for evaluation ---
        self.original_model = copy.deepcopy(model).to(self.device)
        self.original_model.eval()
        
        self.model = copy.deepcopy(model).to(self.device) # This is the model that will be modified during unlearning
        self.new_A_wave = copy.deepcopy(A_wave).to(self.device)
        # Ensure model parameters are float32
        for param in self.model.parameters():
            param.data = param.data.float()
        
        # Store original parameters for EWC
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone().float()
        
        # Initialize components
        self.pa_ewc = PopulationAwareEWC("stgcn", device)
        self.t_gr = TemporalGenerativeReplay("stgcn")
        self.fim_diagonal = None
        
    def unlearn_faulty_subset(self, dataset, forget_ex, faulty_node_idx, A_wave, means, stds, 
                           num_timesteps_input, num_timesteps_output, threshold=0.3,
                           num_epochs=50, learning_rate=5e-5, 
                           lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0, batch_size=512):
        """
        Main unlearning process for faulty node
        """
        print(f"Starting unlearning for faulty node {faulty_node_idx}")
        
        # Ensure dataset is float32
        dataset = dataset.astype(np.float32)
        forget_ex = forget_ex.astype(np.float32)
        
        # --- Partition data
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, forget_ex, faulty_node_idx, threshold
        )
        print(f"Forget samples: {len(forget_indices)}, Retain samples: {len(retain_indices)}")
        
        global forget_loader, retain_loader

        if not forget_indices:
            print("No forget samples found to unlearn. Skipping training.")
            forget_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0))) # Empty dataloader
            # Create a loader with all training data for retain_loader as nothing is forgotten
            training_input, training_target = generate_dataset(dataset, num_timesteps_input, num_timesteps_output)
            retain_loader = DataLoader(TensorDataset(training_input, training_target), batch_size=batch_size, shuffle=True)
            return {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}

        forget_data = []
        for item in forget_indices:
            forget_data.append(dataset[faulty_node_idx:faulty_node_idx+1, :,item[0]:item[1]]) 

        retain_data = []
        for item in retain_indices:
            retain_data.append(dataset[faulty_node_idx:faulty_node_idx+1, :,item[0]:item[1]])

        all_features_retain = []
        all_targets_retain = []
        for item in retain_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feature.numel() > 0:
                all_features_retain.append(feature)
                all_targets_retain.append(target)
        
        if not all_features_retain:
            print("No retain samples generated. Skipping training.")
            return {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}

        all_features_retain = torch.cat(all_features_retain, dim=0)
        all_targets_retain = torch.cat(all_targets_retain, dim=0)
        retain_dataset = TensorDataset(all_features_retain, all_targets_retain)
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

        all_features_forget = []
        all_targets_forget = []
        for item in forget_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feature.numel() > 0:
                all_features_forget.append(feature)
                all_targets_forget.append(target)
        
        if not all_features_forget:
            print("No forget samples generated. Unlearning will not be performed.")
            forget_loader = [] # Empty loader is ok, training will be skipped
        else:
            all_features_forget = torch.cat(all_features_forget, dim=0)
            all_targets_forget = torch.cat(all_targets_forget, dim=0)
            forget_dataset = TensorDataset(all_features_forget, all_targets_forget)
            forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
        
        # --- Compute FIM
        print("Computing Population-Aware FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, A_wave, max_samples=500
        )

        # --- Create surrogate data using T-GR
        print("Creating surrogate data...")
        surrogate_data = self.t_gr.perform_temporal_generative_replay_subset(
            self.model, dataset[faulty_node_idx:faulty_node_idx+1, :, :],
            forget_indices, faulty_node_idx, num_timesteps_input, num_timesteps_output,
            self.device, A_wave
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
            print("Warning: No surrogate samples generated. Skipping training loop.")
            return {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}

        surrogate_features = torch.cat(surrogate_features, dim=0)
        surrogate_targets = torch.cat(surrogate_targets, dim=0)
        surrogate_dataset = TensorDataset(surrogate_features, surrogate_targets)
        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        
        history = self.training(surrogate_loader, retain_loader, A_wave, 
                                num_epochs, learning_rate,
                                lambda_ewc, lambda_surrogate, lambda_retain)
        
        dataset_new = copy.deepcopy(dataset)
        for i in range(len(forget_indices)):
            dataset_new[faulty_node_idx, :, forget_indices[i][0]:forget_indices[i][1]] = surrogate_data[i].squeeze(0)
        dataset_new = dataset_new * stds.reshape(1, -1, 1) + means.reshape(1, -1, 1)
        
        """ if not os.path.exists(args.input + f"/Unlearn_subset_node_{faulty_node_idx}"):
            os.makedirs(args.input + f"/Unlearn_subset_node_{faulty_node_idx}")
        np.save(args.input + f"/Unlearn_subset_node_{faulty_node_idx}/PEMSBAY.npy", dataset_new.transpose(2, 0, 1)) """

        return history
    
    
    def unlearn_faulty_node(self, dataset, faulty_node_idx, A_wave, means, stds, 
                        num_timesteps_input, num_timesteps_output, top_k_node=1,
                        num_epochs=50, learning_rate=5e-5, 
                        lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0, batch_size=512):
        
        dataset = dataset.astype(np.float32)

        # Get real neighbour node
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
            return []

        retain_data = []
        for idx in out_nodes:
            retain_data.append(dataset[idx:idx+1, :, :])
        forget_data = []
        for idx in in_nodes:
            forget_data.append(dataset[idx:idx+1, :, :])

        all_features = []
        all_targets = []
        for item in retain_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output) #(B, N, T, F), (B, N, T, F)
            feature = feature.float() if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).float()
            target = target.float() if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
            all_features.append(feature)
            all_targets.append(target)
        
        all_features = torch.cat(all_features, dim=0)  # (Total samples, num_vertices, num_timesteps_input, num_features)
        all_targets = torch.cat(all_targets, dim=0)
        retain_dataset = TensorDataset(all_features, all_targets)
        global retain_loader  # For evaluation in main
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

        all_features = []
        all_targets = []
        for item in forget_data:
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            feature = feature.float() if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).float()
            target = target.float() if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
            all_features.append(feature)
            all_targets.append(target)
        
        all_features = torch.cat(all_features, dim=0)  # (Total samples, num_vertices, num_timesteps_input, num_features)
        all_targets = torch.cat(all_targets, dim=0)
        global forget_loader  # For evaluation in main
        forget_loader = DataLoader(TensorDataset(all_features, all_targets), batch_size=batch_size, shuffle=True)

        #--- Compute FIM
        print("Computing Population-Aware FIM...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, A_wave, max_samples=500
        )
        print("Creating surrogate data...")
        surrogate_data = self.t_gr.perform_temporal_generative_replay_node(
            self.model, forget_data, faulty_node_idx, num_timesteps_input, num_timesteps_output,
            self.device, A_wave
        )

        surrogate_features, surrogate_targets = [], []
        for item in surrogate_data:
            if isinstance(item, torch.Tensor):
                item = item.cpu().numpy()
            item = item.astype(np.float32)
            feature, target = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            feature = feature.float() if isinstance(feature, torch.Tensor) else torch.from_numpy(feature).float()
            target = target.float() if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
            surrogate_features.append(feature)
            surrogate_targets.append(target)
        surrogate_features = torch.cat(surrogate_features, dim=0)
        surrogate_targets = torch.cat(surrogate_targets, dim=0)
        surrogate_dataset = TensorDataset(surrogate_features, surrogate_targets)
        surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)

        history = self.training(surrogate_loader, retain_loader, A_wave, 
                                num_epochs, learning_rate,
                                lambda_ewc, lambda_surrogate, lambda_retain)
        
        self.new_A_wave[faulty_node_idx, :] = 0
        self.new_A_wave[:, faulty_node_idx] = 0 

        # dataset_new = copy.deepcopy(dataset)
        # dataset_new = torch.from_numpy(dataset_new)
        # dataset_new = torch.cat([dataset_new[:faulty_node_idx], dataset_new[faulty_node_idx+1:]], dim=0)

        """ if not os.path.exists(args.input + f"/Unlearn_node_{faulty_node_idx}"):
            os.makedirs(args.input + f"/Unlearn_node_{faulty_node_idx}")
        np.save(args.input + f"/Unlearn_node_{faulty_node_idx}/PEMSBAY.npy", dataset_new.transpose(2, 0, 1))
        with open(args.input + f"/Unlearn_node_{faulty_node_idx}/adj_mx_bay.pkl", "wb") as f:
            pickle.dump(A_new, f) """

        return history, dataset


    def training(self, surrogate_loader, retain_loader, A_wave,
                num_epochs=50, learning_rate=5e-5, 
                lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0):
        print("Unlearn training...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        history = {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [], 'retain_loss': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss_epoch = 0
            
            if not surrogate_loader or not retain_loader:
                print(f"Epoch {epoch + 1}/{num_epochs}: Skipping, empty loader.")
                history['total_loss'].append(0)
                history['surrogate_loss'].append(0)
                history['ewc_penalty'].append(0)
                history['retain_loss'].append(0)
                continue
            
            for surrogate_batch, retain_batch in zip(surrogate_loader, retain_loader):
                optimizer.zero_grad()
                loss_dict = self.compute_sa_ts_objective(surrogate_batch, retain_batch, lambda_ewc, lambda_surrogate, lambda_retain, A_wave)
                total_loss = loss_dict['total_loss']
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss_epoch += total_loss.item()
            
            history['total_loss'].append(total_loss_epoch / len(surrogate_loader))
            history['surrogate_loss'].append(loss_dict['surrogate_loss'].item())
            history['ewc_penalty'].append(loss_dict['ewc_penalty'].item())
            history['retain_loss'].append(loss_dict['retain_loss'].item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                print(f"  Total: {history['total_loss'][-1]:.4f}")
                print(f"  Surrogate: {history['surrogate_loss'][-1]:.4f}")
                print(f"  EWC: {history['ewc_penalty'][-1]:.4f}")
                print(f"  Retain: {history['retain_loss'][-1]:.4f}")
                print("\n===================================================\n")
        
        return history
    

    def compute_sa_ts_objective(self, surrogate_batch, retain_batch, lambda_ewc, lambda_surrogate, lambda_retain, A_wave):
        """Compute SA-TS objective with corrected loss"""
        surrogate_features, surrogate_target = surrogate_batch
        retain_features, retain_target = retain_batch

        # Ensure float32 and device
        surrogate_features = surrogate_features.float().to(self.device)
        surrogate_target = surrogate_target.float().to(self.device)
        retain_features = retain_features.float().to(self.device)
        retain_target = retain_target.float().to(self.device)
        A_wave = A_wave.float().to(self.device)

        # Forward passes
        surrogate_pred = self.model(A_wave, surrogate_features)
        retain_pred = self.model(A_wave, retain_features)

        # Clamp predictions to prevent instability
        surrogate_pred = torch.clamp(surrogate_pred, min=-3, max=3)
        retain_pred = torch.clamp(retain_pred, min=-3, max=3)

        # Calculate losses
        mse_loss = nn.MSELoss()
        surrogate_loss = mse_loss(surrogate_pred, surrogate_target)
        retain_loss = mse_loss(retain_pred, retain_target)
        
        # EWC penalty
        ewc_penalty = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.fim_diagonal is not None:
            ewc_penalty = self.pa_ewc.apply_ewc_penalty(
                self.model, self.fim_diagonal, self.original_params, lambda_ewc
            )
        
        # CHANGED THE LOSS FUNCTION
        total_loss = lambda_surrogate * surrogate_loss + lambda_ewc * ewc_penalty + lambda_retain * retain_loss
        
        return {
            'total_loss': total_loss,
            'surrogate_loss': surrogate_loss,
            'ewc_penalty': ewc_penalty,
            'retain_loss': retain_loss
        }

def main():
    """Main execution function"""
    torch.cuda.empty_cache()
    # Load data
    print("Loading PEMS-BAY data...")
    A, X, means, stds = load_data_PEMS_BAY(args.input)
    # N, F, T
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

    checkpoint = torch.load(args.model + "/model.pt", map_location=args.device)
    model = STGCN(**checkpoint["config"]).to(args.device)
    model.load_state_dict({k: v.float() for k, v in checkpoint["model_state_dict"].items()})
    
    config = checkpoint["config"]
    num_timesteps_input = config["num_timesteps_input"]
    num_timesteps_output = config["num_timesteps_output"]

    # --- Create a test loader for evaluation ---
    split_line = int(X.shape[2] * 0.8)
    train_original_data = X[:, :, :split_line]
    test_original_data = X[:, :, split_line:]

    test_input, test_target = generate_dataset(test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    test_dataset = TensorDataset(test_input, test_target)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    
    # --- Initialize SA-TS framework, which now stores the original model ---
    sa_ts = SATimeSeries(model, A_wave, args.device)
    
    # Run unlearning on the training data portion
    # TESTING FOR BEST PARAMETERS   
    if args.unlearn_node:
        history = sa_ts.unlearn_faulty_node(
            train_original_data, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,
            top_k_node=2, num_epochs=100, learning_rate=1e-5,
            lambda_ewc=5.0, lambda_surrogate=2.0, lambda_retain=1.0, batch_size=512
        )
    else:
        history = sa_ts.unlearn_faulty_subset(
            train_original_data, forget_array, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,
            threshold=10, num_epochs=100, learning_rate=1e-5,
            lambda_ewc=5.0, lambda_surrogate=2.0, lambda_retain=1.0, batch_size=512
        )

    if history == []:
        return 
     
    if args.unlearn_node:
        path = args.model + f"/Unlearn node {args.node_idx}"
    else:
        path = args.model + f"/Unlearn subset on node {args.node_idx}"
    if not os.path.exists(path):
        os.makedirs(path)
    save_dict = {
        "model_state_dict": sa_ts.model.state_dict(),
        "config": sa_ts.model.config
    }
    torch.save(save_dict, path + "/model.pt")

    # --- Evaluate the unlearned model ---
    print("\nEvaluating unlearned model...")
    evaluation_results = evaluate_unlearning(
        model_unlearned=sa_ts.model,
        model_original=sa_ts.original_model, # Use the copy from the class
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        test_loader=test_loader,
        A_wave=A_wave,
        device=args.device,
        faulty_node_idx=args.node_idx
    )

    # Save eval results
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    with open(args.model + "/unlearned_eval_results.txt", "w") as f:
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("\n--- Evaluation Results ---")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------\n")
    
    # # Save results
    # print("\nSaving results...")
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'history': history,
    #     'faulty_node_idx': args.node_idx
    # }, args.model + "/unlearned_model.pt")
    
    print("Unlearning completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--unlearn-node', action='store_true', help='Enable unlearn node')
    parser.add_argument('--node-idx', type=int, required=True, help='Node index need to be unlearned')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing dataset')
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

