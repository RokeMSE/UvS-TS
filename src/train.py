import numpy as np
import torch
import torch.nn as nn
from src.models.stgcn import STGCN
from src.utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import generate_dataset, get_normalized_adj

num_timesteps_input = 12
num_timesteps_output = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 1000
batch_size = 128

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
        print("Batch: ", i / batch_size)
        model.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        out = model(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

print("START!!!\n")
torch.manual_seed(3)

A, X, means, stds = load_data_PEMS_BAY("PEMSBAY") # (N, F, T)
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
A_wave = A_wave.to(device)

model = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_criterion = nn.MSELoss()

training_losses = []

for epoch in range(epochs):
    loss = train_epoch(training_input, training_target,
                        batch_size=batch_size)
    training_losses.append(loss)
    print("Training loss: {}".format(training_losses[-1]))
