import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from src.models.stgcn import STGCN
from src.utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import generate_dataset, get_normalized_adj

num_timesteps_input = 12
num_timesteps_output = 3

epochs = 100
batch_size = 256

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('--input', type=str, required=True,
                    help='Path to the directory containing dataset')
parser.add_argument('--model', type=str, required=True,
                    help='Path to the directory containing weights of model')

args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

def train_epoch(training_input, training_target, batch_size):
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
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = model(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

if __name__ == '__main__':
    torch.manual_seed(3)

    A, X, means, stds = load_data_PEMS_BAY(args.input) # (N, F, T)
    split_line = int(X.shape[2] * 0.1)

    train_original_data = X[:, :, :split_line]
    test_original_data = X[:, :, split_line:]

    training_input, training_target = generate_dataset(train_original_data,
                                                        num_timesteps_input=num_timesteps_input,
                                                        num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)

    # (B, N, T, F), (B, N, T)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)

    model = STGCN(A_wave.shape[0],
                    training_input.shape[3],
                    num_timesteps_input,
                    num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []

    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                            batch_size=batch_size)
        training_losses.append(loss)
        print("Training loss: {}".format(training_losses[-1]))

    path = args.model 
    if not os.path.exists(path):
        os.makedirs(path)
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config
    }
    torch.save(save_dict, path + "/model.pt")
