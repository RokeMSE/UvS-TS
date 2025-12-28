import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeBlock import TimeBlock

class GATv2Layer(nn.Module):
    """
    Pure PyTorch GATv2 layer compatible with (B, N, T, F) inputs
    """
    def __init__(self, in_features, out_features, n_heads=8, alpha=0.2, dropout=0.1):
        super(GATv2Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features // n_heads
        self.n_heads = n_heads
        
        self.W = nn.Linear(in_features, n_heads * self.out_features, bias=False)
        self.a = nn.Linear(self.out_features, 1, bias=False) # Single attention vector
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, A_hat):
        # X shape: (B, N, T, C_in)
        B, N, T, _ = X.shape
        
        # 1. Linear transformation
        h = self.W(X) # (B, N, T, n_heads * C_out)
        h = h.view(B, N, T, self.n_heads, self.out_features) # (B, N, T, H, C_out)
        
        # 2. Compute attention coefficients (GATv2 style)
        # Permute for easier broadcasting: (B, T, H, N, C_out)
        h_permuted = h.permute(0, 2, 3, 1, 4)
        
        # Create (h_i || h_j) -> (B, T, H, N, N, 2*C_out)
        h_i = h_permuted.unsqueeze(4).expand(-1, -1, -1, -1, N, -1)
        h_j = h_permuted.unsqueeze(3).expand(-1, -1, -1, N, -1, -1)
        
        # GATv2: e = a^T * LeakyReLU(W * [h_i || h_j])
        # Here we just apply LeakyReLU to the features
        e_input = self.leaky_relu(h_i + h_j) # (B, T, H, N, N, C_out)
        
        # Apply attention vector 'a'
        e = self.a(e_input).squeeze(-1) # (B, T, H, N, N)
        
        # 3. Masking
        adj_mask = (A_hat > 0).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        e = e.masked_fill(adj_mask == 0, -float('inf'))
        
        # 4. Softmax
        alpha = F.softmax(e, dim=-1) # (B, T, H, N, N)
        alpha = self.dropout(alpha)
        
        # 5. Aggregate features
        # (B, T, H, N, N) @ (B, T, H, N, C_out) -> (B, T, H, N, C_out)
        h_prime = torch.einsum('bthnn,bthnc->bthnc', [alpha, h_permuted])
        
        # Permute back: (B, N, T, H, C_out)
        h_prime = h_prime.permute(0, 3, 1, 2, 4)
        
        # Concatenate heads
        h_prime = h_prime.reshape(B, N, T, self.n_heads * self.out_features) # (B, N, T, C_out_total)
        
        return h_prime

class STGATv2_Block(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, n_heads=8):
        super(STGATv2_Block, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.gat_layer = GATv2Layer(in_features=out_channels, 
                                    out_features=spatial_channels, 
                                    n_heads=n_heads)
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
    
    def forward(self, X, A_hat):
        t1 = self.temporal1(X)
        spatial_out = F.relu(self.gat_layer(t1, A_hat))
        t2 = self.temporal2(spatial_out)
        return self.batch_norm(t2)

class STGATv2(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, num_features_output=3):
        super(STGATv2, self).__init__()
        n_heads = 4 # You can make this an argument
        
        self.block1 = STGATv2_Block(in_channels=num_features, spatial_channels=64,
                                    out_channels=64, num_nodes=num_nodes, n_heads=n_heads)
        self.block2 = STGATv2_Block(in_channels=64, spatial_channels=64,
                                    out_channels=64, num_nodes=num_nodes, n_heads=n_heads)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        
        self.fully = nn.Linear(num_timesteps_input * 64,
                               num_timesteps_output * num_features_output)
        
        self.num_timesteps_output = num_timesteps_output
        self.num_features_output = num_features_output

        self.config = {
            "model_type": "STGATv2",
            "num_nodes": num_nodes,
            "num_features": num_features,
            "num_timesteps_input": num_timesteps_input,
            "num_timesteps_output": num_timesteps_output,
            "num_features_output": num_features_output
        }

    def forward(self, A_hat, X):
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)

        B, N, T, C = out3.shape
        flat = out3.reshape((B, N, -1))
        out4 = self.fully(flat)

        out4 = out4.view(B, N, self.num_timesteps_output, self.num_features_output)
        return out4