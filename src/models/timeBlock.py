import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        # Adjust padding for 'same' convolution
        padding = (0, (kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)  # (B, C_in, N, T)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)  # (B, N, T, C_out)
        return out
    
if __name__ == "__main__":
    # Simple test
    batch_size = 2
    num_nodes = 5
    num_timesteps = 10
    in_channels = 3
    out_channels = 4

    x = torch.randn(batch_size, num_nodes, num_timesteps, in_channels)
    time_block = TimeBlock(in_channels, out_channels)
    out = time_block(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # Should be (2, 5, 10, 4)
