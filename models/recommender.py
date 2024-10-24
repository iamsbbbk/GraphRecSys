# models/recommender.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .layers import RGCNLayer, RGATLayer, AdaptiveContrastiveLoss, Generator, Discriminator
from torch_geometric.nn import HeteroConv

class RecommenderModel(BaseModel):
    def __init__(self, config, num_nodes_dict, num_relations):
        super(RecommenderModel, self).__init__(config)
        self.config = config
        self.num_nodes_dict = num_nodes_dict
        self.num_relations = num_relations

        embed_size = config['model']['embed_size']
        activation_name = config['model']['activation']
        self.activation = getattr(F, activation_name)

        # 初始化节点嵌入
        self.node_embeddings = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes, embed_size)
            for ntype, num_nodes in num_nodes_dict.items()
        })

        # 定义图神经网络层（可选择 RGCN 或 RGAT）
        gnn_type = config['model'].get('gnn_type', 'rgcn')
        if gnn_type == 'rgcn':
            self.gnn_layer = RGCNLayer(
                in_channels=embed_size,
                out_channels=embed_size,
                num_relations=num_relations,
                activation=self.activation,
                dropout=config['model'].get('dropout', 0.0)
            )
        elif gnn_type == 'rgat':
            self.gnn_layer = RGATLayer(
                in_channels=embed_size,
                out_channels=embed_size,
                num_relations=num_relations,
                heads=config['model'].get('n_heads', 4),
                activation=self.activation,
                dropout=config['model'].get('dropout', 0.0)
            )
        else:
            raise ValueError("Invalid GNN type: choose 'rgcn' or 'rgat'")

        # 对比学习损失函数
        self.use_cl = config['model'].get('use_cl', False)
        if self.use_cl:
            self.cl_loss_fn = AdaptiveContrastiveLoss(
                temperature=config['contrastive_learning'].get('temperature', 0.5)
            )

        # GAN 模块
        self.use_gan = config['model'].get('use_gan', False)
        if self.use_gan:
            noise_dim = config['gan'].get('noise_dim', embed_size)
            self.generator = Generator(
                noise_dim=noise_dim,
                output_dim=embed_size
            )
            self.discriminator = Discriminator(
                input_dim=embed_size
            )

    def forward(self, data):
        """
        前向传播。

        参数：
        - data: 包含图数据的对象，通常是 HeteroData。
        """
        x_dict = {}
        for ntype, embedding in self.node_embeddings.items():
            num_nodes = data[ntype].num_nodes
            x = embedding(torch.arange(num_nodes, device=self.device))
            x_dict[ntype] = x

        # 构建边索引和边类型
        edge_index = data['user', 'interacts', 'item'].edge_index
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=self.device)  # 假设只有一种关系
        # 如果有多种关系，需要根据边类型构建 edge_type

        # GNN 层
        h = self.gnn_layer(x_dict['item'], edge_index, edge_type)

        output = {'item': h}
        return output

    def calculate_loss(self, pos_scores, neg_scores, h_dict):
        """
        计算总的损失，包括推荐损失、对比学习损失、GAN 损失。

        参数：
        - pos_scores: 正样本的评分。
        - neg_scores: 负样本的评分。
        - h_dict: 节点表示的字典。

        返回：
        - total_loss: 总损失。
        - rec_loss: 推荐损失。
        - cl_loss: 对比学习损失（如果使用）。
        - gan_loss: GAN 损失（如果使用）。
        """
        # 推荐损失
        rec_loss = F.margin_ranking_loss(pos_scores, neg_scores, target=torch.ones_like(pos_scores))

        total_loss = rec_loss
        cl_loss = torch.tensor(0.0, device=self.device)
        gan_loss = torch.tensor(0.0, device=self.device)

        # 对比学习损失
        if self.use_cl:
            augmented_h_dict = self.augment(h_dict)  # 需要实现数据增强方法
            cl_loss = self.cl_loss_fn(h_dict, augmented_h_dict)
            cl_weight = self.config['training']['loss_weights'].get('cl_weight', 1.0)
            total_loss += cl_weight * cl_loss

        # GAN 损失
        if self.use_gan:
            gan_loss = self.calculate_gan_loss(h_dict)
            gan_weight = self.config['training']['loss_weights'].get('gan_weight', 1.0)
            total_loss += gan_weight * gan_loss

        return total_loss, rec_loss, cl_loss, gan_loss

    def predict(self, user_ids, item_ids):
        """
        预测用户对商品的评分或偏好度。

        参数：
        - user_ids: 用户索引的张量。
        - item_ids: 商品索引的张量。

        返回：
        - scores: 预测的评分。
        """
        # 获取用户和商品的嵌入
        user_emb = self.node_embeddings['user'](user_ids)
        item_emb = self.node_embeddings['item'](item_ids)

        # 计算评分
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

    def augment(self, h_dict):
        """
        对节点表示进行数据增强，返回增强后的节点表示。

        参数：
        - h_dict: 原始节点表示的字典。

        返回：
        - augmented_h_dict: 增强后的节点表示字典。
        """
        # 实现数据增强方法，例如添加噪声、Dropout 等
        augmented_h_dict = {}
        for ntype, h in h_dict.items():
            augmented_h = h + torch.randn_like(h) * 0.1  # 添加高斯噪声
            augmented_h_dict[ntype] = augmented_h
        return augmented_h_dict

    def calculate_gan_loss(self, h_dict):
        """
        计算 GAN 的损失。

        参数：
        - h_dict: 节点表示的字典。

        返回：
        - gan_loss: GAN 损失值。
        """
        real_embeddings = h_dict['item']

        # 生成器生成假嵌入
        noise = torch.randn(real_embeddings.size(0), self.config['gan']['noise_dim'], device=self.device)
        fake_embeddings = self.generator(noise)

        # 判别器判别真实和假嵌入
        real_scores = self.discriminator(real_embeddings)
        fake_scores = self.discriminator(fake_embeddings.detach())

        # 计算判别器的损失
        d_loss_real = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
        d_loss_fake = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
        d_loss = d_loss_real + d_loss_fake

        # 生成器的损失
        gen_scores = self.discriminator(fake_embeddings)
        g_loss = F.binary_cross_entropy(gen_scores, torch.ones_like(gen_scores))

        # 返回 GAN 的总损失
        gan_loss = d_loss + g_loss
        return gan_loss