# models/layers/kg_layers.py

import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

class KGATLayer(nn.Module):
    """
    知识图谱融合层，使用图注意力网络（GAT）处理异质图。
    """
    def __init__(self, in_channels_dict, out_channels, heads=4, dropout=0.0):
        super(KGATLayer, self).__init__()
        self.convs = HeteroConv({
            ('entity', 'relates_to', 'entity'): GATConv(
                in_channels_dict['entity'],
                out_channels,
                heads=heads,
                dropout=dropout,
                concat=True
            ),
            # 可以根据需要添加更多的关系类型
        })

    def forward(self, x_dict, edge_index_dict):
        """
        参数：
        - x_dict: 节点特征字典，键为节点类型，值为特征张量。
        - edge_index_dict: 边索引字典，键为 (src_type, rel_type, dst_type)，值为边索引张量。
        """
        h_dict = self.convs(x_dict, edge_index_dict)
        return h_dict