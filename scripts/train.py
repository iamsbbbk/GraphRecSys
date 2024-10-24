# scripts/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from data.load_data import load_amazon_dataset
from models import RecommenderModel
from utils.helper_functions import load_config,set_random_seed
from logs.log_config import setup_logging
from trainers import Trainer, EarlyStopping
from torch.utils.data import DataLoader
import os

def train():
    # 读取配置文件
    config = load_config('config/config.yaml')

    # 设置日志记录器
    logger = setup_logging(config)

    # 设置随机种子
    seed = config['other'].get('seed', 42)
    set_random_seed(seed)

    # 设置设备
    device_type = config['device']['type']
    if torch.cuda.is_available() and device_type == 'cuda':
        gpu_id = config['device'].get('gpu_id', 0)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device('cpu')
        logger.info("CUDA is not available or not requested. Using CPU")

    # 加载数据集
    logger.info("Loading dataset...")
    train_graph, valid_graph, _, user2id, item2id = load_amazon_dataset(config['data']['root_dir'])
    logger.info("Dataset loaded.")

    # 获取节点和关系数量信息
    num_nodes_dict = {ntype: train_graph[ntype].num_nodes for ntype in train_graph.node_types}
    num_relations = len(train_graph.edge_types)

    # 初始化模型
    logger.info("Initializing model...")
    model = RecommenderModel(config, num_nodes_dict, num_relations)
    model.to(device)
    logger.info("Model initialized.")

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    logger.info("Optimizer initialized.")

    # 定义损失函数
    loss_fn = nn.BCEWithLogitsLoss()
    logger.info("Loss function defined.")

    # 创建 DataLoader（需要实现 HeteroData 的自定义数据集）
    train_loader = DataLoader([train_graph], batch_size=1, shuffle=True)
    valid_loader = DataLoader([valid_graph], batch_size=1)

    # 定义回调函数
    early_stopping = EarlyStopping(
        patience=config['training'].get('patience', 10),
        monitor_metric='AUC',  # 示例，需改为实际指标
        mode='max',
        save_path=os.path.join(config['logging']['save_dir'], 'best_model.pth')
    )

    # 创建 Trainer 对象
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,  # 传入损失函数
        device=device,
        train_graph=train_graph,
        valid_graph=valid_graph,
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        logger=logger
    )

    # 创建 Trainer 对象
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,    # 传入损失函数
        device=device,
        train_graph=train_graph,
        valid_graph=valid_graph,
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        logger=logger
    )

    # 开始训练
    trainer.train(train_loader)

if __name__ == '__main__':
    train()