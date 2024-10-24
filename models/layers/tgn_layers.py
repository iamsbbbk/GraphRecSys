# models/layers/tgn_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class TimeEmbedding(nn.Module):
    """
    时间编码器，将时间戳编码为向量。
    """
    def __init__(self, time_dim):
        super(TimeEmbedding, self).__init__()
        self.freqs = nn.Parameter(torch.randn(time_dim))

    def forward(self, t):
        """
        参数：
        - t: 时间戳，形状为 [num_edges]
        """
        t = t.unsqueeze(-1) * self.freqs  # [num_edges, time_dim]
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # [num_edges, 2 * time_dim]

class TemporalConv(MessagePassing):
    def __init__(self, in_channels, out_channels, time_dim):
        super(TemporalConv, self).__init__(aggr='add')  # or 'mean', 'max'
        self.linear = nn.Linear(in_channels + 2 * time_dim, out_channels)
        self.time_encoder = TimeEmbedding(time_dim)

    def forward(self, x, edge_index, edge_timestamps):
        """
        参数：
        - x: 节点特征矩阵，形状为 [num_nodes, in_channels]
        - edge_index: 边索引矩阵，形状为 [2, num_edges]
        - edge_timestamps: 边的时间戳，形状为 [num_edges]
        """
        t_enc = self.time_encoder(edge_timestamps)  # [num_edges, 2 * time_dim]
        return self.propagate(edge_index, x=x, t_enc=t_enc)

    def message(self, x_j, t_enc):
        """
        参数：
        - x_j: 源节点的特征，形状为 [num_edges, in_channels]
        - t_enc: 边的时间编码，形状为 [num_edges, 2 * time_dim]
        """
        msg = torch.cat([x_j, t_enc], dim=-1)
        msg = self.linear(msg)
        return msg

    def update(self, aggr_out):
        return aggr_out

class TGNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(TGNLayer, self).__init__()
        self.temporal_conv = TemporalConv(in_channels, out_channels, time_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_timestamps):
        out = self.temporal_conv(x, edge_index, edge_timestamps)
        out = self.activation(out)
        return out