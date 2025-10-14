import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import copy
import pickle
import random
import argparse
from tslearn.metrics import dtw

# Components
from models.stgcn import STGCN
from utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
# Import evaluation functions
from evaluate import (
    evaluate_unlearning, fidelity_score, forgetting_efficacy, 
    generalization_score, statistical_distance, membership_inference_attack,
    get_model_predictions
)
from train import train_epoch
import sys
sys.path.append('src')


epochs = 100
batch_size = 64


def fill_missing_with_node_mean(data):
    '''
    Fill position have feature [0, 0, 0] by mean
    '''
    data_filled = data.copy()
    total_timesteps, total_nodes, num_features = data.shape

    node_means = np.zeros((total_nodes, num_features))
    node_stds = np.zeros((total_nodes, num_features))
    for node in range(total_nodes):
        valid_mask = ~np.all(data[:, node, :] == 0, axis=1) 
        if valid_mask.sum() > 0:
            node_means[node] = data[valid_mask, node, :].mean(axis=0)
            node_stds[node] = data[valid_mask, node, :].std(axis=0)
        else:
            node_means[node] = 0
            node_stds[node] = 0

    for t in range(total_timesteps):
        for node in range(total_nodes):
            if np.all(data_filled[t, node, :] == 0):
                data_filled[t, node, :] = node_means[node] + random.choice([-1, 1]) * random.randint(0, 10) * node_stds[node] / 10

    return data_filled

def fix_data_for_subset(dataset, u, faulty_node_idx, num_timesteps_input, num_timesteps_output, threshold):
    new_dataset = dataset.copy()
    S = dataset[:, 1, :]
    u_mean, u_std = u.mean(), u.std() or 1.0
    u_norm = (u - u_mean) / u_std
    _, time_step = S.shape
    forget_indices = []
    window_size = len(u)
    for i in range(time_step - window_size + 1):
        segment = S[faulty_node_idx, i:i+window_size]
        segment_norm = (segment - u_mean) / u_std
        if dtw(u_norm, segment_norm) <= threshold:
            if forget_indices and i <= forget_indices[-1][1]:
                forget_indices[-1][1] = i + window_size
            else:
                forget_indices.append([i, i + window_size])

    retain_indices = []
    # Handle case where no motifs are found
    if not forget_indices:
        print("Warning: No motifs found with the given threshold. Entire dataset is considered 'retain'.")
        retain_indices.append([0, time_step])
    else:
        last_forget_end = 0
        for start, end in forget_indices:
            if start > last_forget_end:
                retain_indices.append([last_forget_end, start])
            last_forget_end = end
        
        if last_forget_end < time_step:
            retain_indices.append([last_forget_end, time_step])
    
    for item in forget_indices:
        new_dataset[faulty_node_idx, :, item[0] : item[1]] = 0

    # CREATE RETAIN LOADER, FORGET LOADER
    global forget_loader, retain_loader
    if not forget_indices:
        print("No forget samples found to unlearn. Skipping training.")
        forget_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0))) # Empty dataloader
        # Create a loader with all training data for retain_loader as nothing is forgotten
        training_input, training_target = generate_dataset(dataset, num_timesteps_input, num_timesteps_output)
        retain_loader = DataLoader(TensorDataset(training_input, training_target), batch_size=batch_size, shuffle=True)
    else:
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

    result = fill_missing_with_node_mean(new_dataset)

    return result, forget_indices, retain_indices

def fix_data_for_node(A_wave, faulty_node_idx):

    new_A_wave = A_wave.copy()
    new_A_wave[faulty_node_idx, :] = 0
    new_A_wave[:, faulty_node_idx] = 0  

    return new_A_wave

def main():
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

    checkpoint = torch.load(args.model + "/model.pt", map_location=args.device)
    original_model = STGCN(**checkpoint["config"]).to(args.device)
    original_model.load_state_dict({k: v.float() for k, v in checkpoint["model_state_dict"].items()})
    
    config = checkpoint["config"]
    num_timesteps_input = config["num_timesteps_input"]
    num_timesteps_output = config["num_timesteps_output"]
    
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)
    new_A_wave = A_wave

    split_line = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line]
    new_train_original_data = train_original_data
    train_original_data = train_original_data * stds.reshape(1, -1, 1) + means.reshape(1, -1, 1)
    test_original_data = X[:, :, split_line:]

    if args.unlearn_node:
        print("UNLEARNING NODE...\n")
        new_A_wave = fix_data_for_node(A_wave, args.node_idx)
        
    else:
        print("UNLEARNING SUBSET...\n")
        new_train_original_data, forget_indices, retain_indices = fix_data_for_subset(train_original_data, forget_array, args.node_idx, 
                                                                                      num_timesteps_input, num_timesteps_output, 10)
        if forget_indices == []:
            print("NOT FIND SUBSET TO UNLEARN")
            return

        means = np.mean(new_train_original_data, axis=(0, 2))
        new_train_original_data = new_train_original_data - means.reshape(1, -1, 1)
        stds = np.std(new_train_original_data, axis=(0, 2))
        new_train_original_data = new_train_original_data / stds.reshape(1, -1, 1)
    

    training_input, training_target = generate_dataset(new_train_original_data,
                                                        num_timesteps_input=num_timesteps_input,
                                                        num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    test_dataset = TensorDataset(test_input, test_target)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    
    new_model = STGCN(new_A_wave.shape[0],
                    training_input.shape[3],
                    num_timesteps_input,
                    num_timesteps_output, 
                    num_features_output=3).to(device=args.device)

    optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    for epoch in range(epochs):
        loss = train_epoch(new_model, new_A_wave, loss_criterion, optimizer ,training_input, training_target,
                            batch_size=batch_size, device=args.device)
        training_losses.append(loss)
        print(f"Epoch {epoch} training loss: {format(training_losses[-1])}")
    
    print("\nEvaluating unlearned model...")
    evaluation_results = evaluate_unlearning(
        model_unlearned=new_model,
        model_original=original_model, # Use the copy from the class
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        test_loader=test_loader,
        new_A_wave=new_A_wave,
        A_wave=A_wave,
        device=args.device,
        faulty_node_idx=args.node_idx
    )

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