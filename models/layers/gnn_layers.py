# models/layers/gnn_layers.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv,GATConv,HeteroConv

class RGCNLayer(nn.Module):
    """
    关系图卷积网络层（Relational Graph Convolutional Network Layer）。
    """
    def __init__(self, in_channels, out_channels, num_relations, num_bases=None, activation=None, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = RGCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=num_bases
        )

    def forward(self, x, edge_index, edge_type):
        """
        前向传播。

        参数：
        - x: 节点特征矩阵，形状为 [num_nodes, in_channels]。
        - edge_index: 边索引矩阵，形状为 [2, num_edges]。
        - edge_type: 每条边的关系类型，形状为 [num_edges]。
        """
        out = self.conv(x, edge_index, edge_type)
        if self.activation:
            out = self.activation(out)
        out = self.dropout(out)
        return out
class RGATLayer(nn.Module):
    """
    关系图注意力网络层（Relational Graph Attention Network Layer）。
    """
    def __init__(self, in_channels, out_channels, num_relations, heads=4, activation=None, dropout=0.0):
        super(RGATLayer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

        # 为每种关系类型定义一个 GATConv 层
        self.convs = nn.ModuleDict()
        for rel in range(num_relations):
            conv = GATConv(
                in_channels=in_channels,
                out_channels=out_channels // heads,
                heads=heads,
                dropout=dropout,
                concat=True
            )
            self.convs[str(rel)] = conv

    def forward(self, x, edge_index_dict):
        """
        前向传播。

        参数：
        - x (dict of Tensor): 节点特征字典，键为节点类型，值为特征张量。
        - edge_index_dict (dict of Tensor): 边索引字典，键为关系类型（如 'interacts'），值为对应的边索引张量。
        """
        h = {}
        for rel_type, conv in self.convs.items():
            edge_index = edge_index_dict[rel_type]
            h_rel = conv(x, edge_index)
            if self.activation:
                h_rel = self.activation(h_rel)
            h_rel = self.dropout(h_rel)
            # 假设目标节点类型为 'item'，根据实际情况调整
            if 'item' in h:
                h['item'] += h_rel
            else:
                h['item'] = h_rel
        return h