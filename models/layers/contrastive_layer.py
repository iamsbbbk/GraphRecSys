# models/layers/contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveContrastiveLoss(nn.Module):
    """
    自适应对比学习的损失函数。

    参数：
    - temperature (float): 温度参数，用于调整 logits 的分布。
    """
    def __init__(self, temperature=0.5):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, h_dict, augmented_h_dict):
        """
        计算对比学习损失。

        参数：
        - h_dict (dict): 原始节点表示字典，键为节点类型，值为节点表示张量。
        - augmented_h_dict (dict): 增强后的节点表示字典。

        返回：
        - loss (Tensor): 对比学习损失值。
        """
        losses = []
        for ntype in h_dict:
            h = h_dict[ntype]
            h_aug = augmented_h_dict[ntype]
            batch_size = h.size(0)
            h = F.normalize(h, dim=1)
            h_aug = F.normalize(h_aug, dim=1)

            representations = torch.cat([h, h_aug], dim=0)  # [2N, D]
            similarity_matrix = torch.matmul(representations, representations.T)  # [2N, 2N]

            # 去除自身相似度
            mask = torch.eye(2 * batch_size, device=h.device).bool()
            similarity_matrix = similarity_matrix / self.temperature
            similarity_matrix.masked_fill_(mask, -1e9)

            # 构建标签
            labels = torch.arange(batch_size, device=h.device)
            labels = torch.cat([labels, labels], dim=0)

            # 计算对比损失
            loss = F.cross_entropy(similarity_matrix, labels)
            losses.append(loss)

        total_loss = sum(losses) / len(losses)
        return total_loss