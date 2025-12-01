import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeBlock import TimeBlock

class STSAGE_Block(nn.Module):
    """
    STSAGE block: Temporal conv -> SAGE-style spatial conv -> Temporal conv
    """
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STSAGE_Block, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        
        # SAGE-style spatial aggregation
        # We concatenate self-features and aggregated neighbor features
        self.W_spatial = nn.Linear(out_channels * 2, spatial_channels)
        
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        self.W_spatial.reset_parameters()

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features=in_channels).
        :param A_hat: Normalized adjacency matrix (num_nodes x num_nodes).
        :return: Output data of shape (batch_size, num_nodes, num_timesteps_out, num_features=out_channels).
        """
        t1 = self.temporal1(X)  # (B, N, T, C_out)
        
        # SAGE spatial aggregation
        # t1 permuted: (N, B, T, C_out)
        # A_hat: (N, N)
        # 'ij,jklm->iklm' -> (N, N) @ (N, B, T, C_out) -> (N, B, T, C_out)
        neighbor_features = torch.einsum("ij,jklm->iklm", [A_hat, t1.permute(1, 0, 2, 3)])
        # Permute back to (B, N, T, C_out)
        neighbor_features = neighbor_features.permute(1, 0, 2, 3)
        
        # Concatenate self-features and neighbor features
        combined_features = torch.cat([t1, neighbor_features], dim=-1) # (B, N, T, 2*C_out)
        
        # Apply linear transformation (W_spatial)
        spatial_out = F.relu(self.W_spatial(combined_features)) # (B, N, T, C_spatial)

        t2 = self.temporal2(spatial_out) # (B, N, T, C_out)
        return self.batch_norm(t2)


class STSAGE(nn.Module):
    """
    Spatio-Temporal GraphSAGE Network
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, num_features_output=3):
        super(STSAGE, self).__init__()
        self.block1 = STSAGE_Block(in_channels=num_features, spatial_channels=16,
                                   out_channels=64, num_nodes=num_nodes)
        self.block2 = STSAGE_Block(in_channels=64, spatial_channels=16,
                                   out_channels=64, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        
        self.fully = nn.Linear(num_timesteps_input * 64,
                               num_timesteps_output * num_features_output)
        
        self.num_timesteps_output = num_timesteps_output
        self.num_features_output = num_features_output
        
        self.config = {
            "model_type": "STSAGE",
            "num_nodes": num_nodes,
            "num_features": num_features,
            "num_timesteps_input": num_timesteps_input,
            "num_timesteps_output": num_timesteps_output,
            "num_features_output": num_features_output
        }

    def forward(self, A_hat, X):
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)  # (B, N, T_in, 64)

        B, N, T, C = out3.shape
        flat = out3.reshape((B, N, -1))  # (B, N, T_in * 64)
        out4 = self.fully(flat)  # (B, N, T_out * F_out)

        out4 = out4.view(B, N, self.num_timesteps_output, self.num_features_output)
        return out4