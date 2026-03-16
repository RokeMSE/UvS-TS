import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from models.stgcn import STGCN
from models.stgat import STGAT
from models.gwn import gwnet
from utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import generate_dataset, get_normalized_adj

num_timesteps_input = 12
num_timesteps_output = 4

epochs = 3
batch_size = 68

def train_epoch(model, A_wave, loss_criterion, optimizer, training_input, training_target, batch_size, device):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        model.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.float().to(device)
        y_batch = y_batch.float().to(device)

        out = model(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())

    return sum(epoch_training_losses)/len(epoch_training_losses)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable-cuda', action='store_true',
                        help='Enable CUDA')
    
    parser.add_argument('--all', action='store_true',
                        help='Train all model')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the directory containing dataset')
    parser.add_argument('--type', type=str,
                        help='Type of model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the directory containing weights of model')

    args = parser.parse_args()
    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')


    torch.manual_seed(3)

    A, train_original_data, test_original_data, means, stds = load_data_PEMS_BAY(args.input) # (N, F, T)
    # split_line = int(X.shape[2] * 0.1)

    # train_original_data = X[:, :, :split_line]
    # test_original_data = X[:, :, split_line:]

    training_input, training_target = generate_dataset(train_original_data,
                                                        num_timesteps_input=num_timesteps_input,
                                                        num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    
    print(training_input.shape)
    print(training_target.shape)

    # (B, N, T, F), (B, N, T)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float()
    A_wave = A_wave.to(device=args.device)
    print("Initialize model...")
    supports = [A_wave]
    aptinit = supports[0]
    MODEL_FARM = {
        "stgcn": lambda: STGCN(nums_node=A_wave.shape[0], nums_step_in=num_timesteps_input, nums_step_out=num_timesteps_output, nums_feature_in=training_input.shape[3], nums_feature_out=training_target.shape[3]),
        "stgat": lambda: STGAT(nums_node=A_wave.shape[0], nums_step_in=num_timesteps_input, nums_step_out=num_timesteps_output, nums_feature_in=training_input.shape[3], nums_feature_out=training_target.shape[3], n_heads=8),
        "gwnet": lambda: gwnet(nums_node=A_wave.shape[0], nums_step_in=num_timesteps_input, nums_step_out=num_timesteps_output, nums_feature_in=training_input.shape[3], nums_feature_out=training_target.shape[3], device=args.device, supports=supports, aptinit=aptinit)

    }
    path = args.model 
    os.makedirs(path, exist_ok=True)
    
    if not args.all:
        if args.type == 'stgcn':
            model = STGCN(A_wave.shape[0],
                            nums_step_in=num_timesteps_input,
                            nums_step_out=num_timesteps_output, 
                            nums_feature_in=training_input.shape[3],
                            nums_feature_out=3)
            
        elif args.type == 'stgat':
            model = STGAT(A_wave.shape[0],
                          num_timesteps_input,
                          num_timesteps_output, 
                          training_input.shape[3],
                          nums_feature_output=3, 
                          n_heads=8)
            
        elif args.type == 'gwnet':
            model = gwnet(num_nodes=A_wave.shape[0], 
                          nums_step_in=num_timesteps_input,
                          nums_step_out=num_timesteps_output,
                          nums_feature_in=3, 
                          nums_feature_out=3,
                          device=args.device,
                          supports=supports,
                          aptinit=aptinit)
            
        model = model.to(device=args.device)
        print(f"Training on model {args.type}...")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_criterion = nn.MSELoss()

        training_losses = []
        print("TRAINING...\n")
        for epoch in range(epochs):
            loss = train_epoch(model, A_wave, loss_criterion, optimizer, training_input, training_target,
                                batch_size=batch_size, device=args.device)
            training_losses.append(loss)
            print(f"Epoch {epoch} training loss: {format(training_losses[-1])}")

        

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config
        }
        torch.save(save_dict, path + f"/{args.type}_model.pt")
    else:
        list_models = ['stgcn', 'stgat', 'gwnet']
        
        
        for model_name in list_models:
            model = MODEL_FARM[model_name]()

            model = model.to(device=args.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_criterion = nn.MSELoss()

            training_losses = []
            print(f"\nTRAINING MODEL {model_name.upper()}...\n")
            for epoch in range(epochs):
                loss = train_epoch(model, A_wave, loss_criterion, optimizer, training_input, training_target,
                                    batch_size=batch_size, device=args.device)
                training_losses.append(loss)
                print(f"Epoch {epoch} training loss: {format(training_losses[-1])}")

            

            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model.config
            }
            torch.save(save_dict, path + f"/{model_name}_model.pt")

            del model
            del optimizer
            del loss_criterion
            torch.cuda.empty_cache()                                                                                                                                                                                                                                                                                                                        