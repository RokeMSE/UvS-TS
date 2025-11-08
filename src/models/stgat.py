import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj_mat):
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        g_i = g.unsqueeze(1)      # (N,1,h,n_hidden)
        g_j = g.unsqueeze(0)      # (1,N,h,n_hidden)
        g_concat = torch.cat([g_i.expand(n_nodes,n_nodes,-1,-1), g_j.expand(n_nodes,n_nodes,-1,-1)], dim=-1)
        
        e = self.activation(self.attn(g_concat)).squeeze(-1)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


class STGAT(nn.Module):
    def __init__(self, num_nodes, nums_feature_input, nums_step_in, nums_step_out, nums_feature_output=3, n_heads=8):
        super().__init__()
        self.hidden_dim = 32
        self.num_timesteps_output = nums_step_out
        self.num_features_output = nums_feature_output
        # spatial
        self.gat = GraphAttentionLayer(nums_feature_input, self.hidden_dim, n_heads, dropout=0)

        # temporal: stacked LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,     # 2 layers LSTM
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, nums_feature_output * nums_step_out)

        self.config = {
            "num_nodes": num_nodes,
            "num_features": nums_feature_input,
            "num_timesteps_input": nums_step_in,
            "num_timesteps_output": nums_step_out,
            "num_features_output": nums_feature_output,
            "n_head": n_heads
        }

    def forward(self, A_hat, X):

        B, N, T_in, F = X.shape

        outs = []
        for t in range(T_in):
            h_t = []
            for b in range(B):
                h_t.append(self.gat(X[b, :, t, :], A_hat.unsqueeze(-1)))
            h_t = torch.stack(h_t, dim=0)  # (B, N, hidden_dim)
            outs.append(h_t.unsqueeze(2))   # add time dim

        # stack timesteps → (B, N, T_in, hidden_dim)
        outs = torch.cat(outs, dim=2)

        # merge batch & nodes for LSTM
        lstm_in = outs.reshape(B * N, T_in, self.hidden_dim)
        lstm_out, _ = self.lstm(lstm_in)
        h_last = lstm_out[:, -1, :]

        # FC to predict F * T_out
        out = self.fc(h_last)
        out = out.reshape(B, N, self.num_timesteps_output, F)  # (B, N, T_out, F)
        return out