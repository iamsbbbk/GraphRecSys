# scripts/utils.py

import torch
import random
from torch_geometric.data import HeteroData

def sample_pos_neg_pairs(data: HeteroData, output: dict, device):
    """
    从图数据中采样正负样本对，计算正负样本的评分。

    参数：
    - data: HeteroData 对象，包含图数据。
    - output: 模型前向传播的输出，包含节点表示。
    - device: 计算设备。

    返回：
    - pos_scores: 正样本的评分张量。
    - neg_scores: 负样本的评分张量。
    """
    user_emb = output['user']
    item_emb = output['item']

    # 获取正样本的边（用户-商品交互）
    edge_index = data['user', 'interacts', 'item'].edge_index
    user_indices = edge_index[0]
    item_indices = edge_index[1]

    # 计算正样本评分
    pos_user_emb = user_emb[user_indices]
    pos_item_emb = item_emb[item_indices]
    pos_scores = (pos_user_emb * pos_item_emb).sum(dim=1)

    # 随机采样负样本
    num_edges = edge_index.size(1)
    neg_item_indices = torch.randint(0, item_emb.size(0), (num_edges,), device=device)
    neg_item_emb = item_emb[neg_item_indices]
    neg_user_emb = pos_user_emb  # 使用相同的用户

    # 计算负样本评分
    neg_scores = (neg_user_emb * neg_item_emb).sum(dim=1)

    return pos_scores, neg_scores

def compute_metrics(output: dict, data: HeteroData):
    """
    计算模型的评估指标，例如 AUC、准确率等。

    参数：
    - output: 模型前向传播的输出，包含节点表示。
    - data: HeteroData 对象，包含图数据。

    返回：
    - metrics: 包含评估指标的字典。
    """
    from sklearn.metrics import roc_auc_score

    user_emb = output['user']
    item_emb = output['item']

    # 获取所有真实的正样本
    edge_index = data['user', 'interacts', 'item'].edge_index
    user_indices = edge_index[0].cpu().numpy()
    item_indices = edge_index[1].cpu().numpy()

    # 计算正样本评分
    pos_user_emb = user_emb[user_indices]
    pos_item_emb = item_emb[item_indices]
    pos_scores = (pos_user_emb * pos_item_emb).sum(dim=1).cpu().numpy()

    # 构建标签，正样本为 1
    labels = [1] * len(pos_scores)

    # 生成与用户的负样本
    neg_item_indices = random.sample(range(item_emb.size(0)), len(pos_scores))
    neg_item_emb = item_emb[neg_item_indices]
    neg_user_emb = pos_user_emb
    neg_scores = (neg_user_emb * neg_item_emb).sum(dim=1).cpu().numpy()

    # 负样本标签为 0
    labels += [0] * len(neg_scores)
    all_scores = list(pos_scores) + list(neg_scores)

    # 计算 AUC
    auc = roc_auc_score(labels, all_scores)

    metrics = {'AUC': auc}
    return metrics