# scripts/evaluate.py

import torch
from data.load_data import load_amazon_dataset
from models import RecommenderModel
from utils.helper_functions import load_config
from utils.metrics import compute_metrics
import os
import logging
from logs.log_config import setup_logging

def evaluate():
    # 读取配置文件
    config = load_config('configs/config.yaml')

    # 设置日志记录器
    logger = setup_logging(config)

    # 设置设备
    device_type = config['device']['type']
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    logger.info("Loading dataset...")
    _, valid_graph, test_graph, user2id, item2id = load_amazon_dataset(config['data']['root_dir'])
    logger.info("Dataset loaded.")

    # 获取节点和关系数量信息
    num_nodes_dict = {ntype: valid_graph[ntype].num_nodes for ntype in valid_graph.node_types}
    num_relations = len(valid_graph.edge_types)

    # 初始化模型
    logger.info("Initializing model...")
    model = RecommenderModel(config, num_nodes_dict, num_relations)
    model.to(device)
    logger.info("Model initialized.")

    # 加载训练好的模型参数
    model_path = os.path.join(config['logging']['save_dir'], 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}")
        return

    # 开始评估
    model.eval()
    with torch.no_grad():
        test_graph = test_graph.to(device)
        output = model(test_graph)
        metrics = compute_metrics(output, test_graph)
        logger.info(f"Evaluation metrics on test set: {metrics}")

if __name__ == '__main__':
    evaluate()