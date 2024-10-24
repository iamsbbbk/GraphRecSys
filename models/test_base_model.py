# test_recommender.py

import unittest
import torch
from models.recommender import RecommenderModel

class TestRecommenderModel(unittest.TestCase):
    def setUp(self):
        # 模拟配置和数据
        self.config = {
            'model': {
                'embed_size': 16,
                'memory_dim': 16,
                'time_dim': 8,
                'activation': 'relu',
                'use_gan': False,
                'use_cl': False,
                'use_kg': False,
                'n_heads': 2,
            },
            'training': {
                'loss_weights': {
                    'cl_weight': 0.1,
                    'gan_weight': 0.1,
                }
            },
            'gan': {
                'noise_dim': 16,
                'hidden_dim': 32,
                'num_layers': 2,
                'activation': 'leaky_relu',
            },
            'contrastive_learning': {
                'temperature': 0.5,
            }
        }
        self.num_nodes_dict = {'user': 10, 'item': 15, 'entity': 20}
        self.num_relations = 5
        self.model = RecommenderModel(self.config, self.num_nodes_dict, self.num_relations)
        self.device = torch.device('cpu')
        self.model.to(self.device)

    def test_forward(self):
        # 模拟事件批次
        event_batches = [
            [
                {'src_id': 0, 'dst_id': 5, 'src_type': 'user', 'dst_type': 'item', 'timestamp': 1.0},
                {'src_id': 1, 'dst_id': 6, 'src_type': 'user', 'dst_type': 'item', 'timestamp': 2.0},
            ],
            [
                {'src_id': 2, 'dst_id': 7, 'src_type': 'user', 'dst_type': 'item', 'timestamp': 3.0},
            ],
        ]
        # 模拟知识图谱图（空图，用于测试）
        kg_graph = None

        h_dict = self.model.forward(event_batches, kg_graph)
        for ntype in h_dict:
            self.assertEqual(h_dict[ntype].shape[0], self.num_nodes_dict[ntype])
            self.assertEqual(h_dict[ntype].shape[1], self.config['model']['embed_size'])

    def test_predict(self):
        user_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        item_ids = torch.tensor([5, 6, 7], dtype=torch.long)
        # 创建假设的 h_dict
        h_dict = {
            'user': torch.randn(self.num_nodes_dict['user'], self.config['model']['embed_size']),
            'item': torch.randn(self.num_nodes_dict['item'], self.config['model']['embed_size']),
        }
        scores = self.model.predict(user_ids, item_ids, h_dict)
        self.assertEqual(scores.shape[0], user_ids.shape[0])

    def test_calculate_loss(self):
        pos_scores = torch.tensor([3.0, 2.5, 4.0])
        neg_scores = torch.tensor([1.0, 1.5, 0.5])
        h_dict = {
            'user': torch.randn(self.num_nodes_dict['user'], self.config['model']['embed_size']),
            'item': torch.randn(self.num_nodes_dict['item'], self.config['model']['embed_size']),
        }
        total_loss, rec_loss, cl_loss, gan_loss = self.model.calculate_loss(pos_scores, neg_scores, h_dict)
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(rec_loss, float)

if __name__ == '__main__':
    unittest.main()