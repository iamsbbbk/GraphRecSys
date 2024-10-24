# utils/helper_functions.py

import yaml
import random
import numpy as np
import torch
import logging
import os

def load_config(config_path):
    """
    加载 YAML 格式的配置文件。

    参数：
    - config_path: 配置文件的路径。

    返回：
    - config: 配置参数的字典。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def set_random_seed(seed=42):
    """
    设置随机种子，确保实验可重复。

    参数：
    - seed: 随机种子数值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保在有确定性算法的情况下运行，以保证可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    保存模型的检查点。

    参数：
    - model: 模型。
    - optimizer: 优化器。
    - epoch: 当前的 epoch 数。
    - loss: 当前的损失值。
    - save_path: 保存路径。
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, save_path)
    logging.info(f"Checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, load_path, device):
    """
    加载模型的检查点。

    参数：
    - model: 模型。
    - optimizer: 优化器。
    - load_path: 检查点文件路径。
    - device: 设备。

    返回：
    - epoch: 上次训练的 epoch 数。
    - loss: 上次保存的损失值。
    """
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f"Checkpoint loaded from {load_path}")
        return epoch, loss
    else:
        logging.error(f"No checkpoint found at {load_path}")
        return None, None