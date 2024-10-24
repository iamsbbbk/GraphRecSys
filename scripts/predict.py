# scripts/predict.py

import torch
from models import RecommenderModel
from utils.helper_functions import load_config
import logging
from logs.log_config import setup_logging
import os

def predict(user_ids, item_ids):
    # 读取配置文件
    config = load_config('configs/config.yaml')

    # 设置日志记录器
    logger = setup_logging(config)

    # 设置设备
    device_type = config['device']['type']
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')

    # 加载模型
    logger.info("Initializing model...")
    # 需要获取 num_nodes_dict 和 num_relations，可以从保存的模型文件中加载，或者提前保存这些信息
    num_nodes_dict = ...  # 从文件中加载
    num_relations = ...   # 从文件中加载

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

    # 模型预测
    model.eval()
    with torch.no_grad():
        user_ids = torch.tensor(user_ids, device=device)
        item_ids = torch.tensor(item_ids, device=device)
        scores = model.predict(user_ids, item_ids)
        logger.info(f"Predicted scores: {scores}")

    return scores.cpu().numpy()

if __name__ == '__main__':
    # 示例用户和商品 ID 列表
    user_ids = [0, 1, 2]
    item_ids = [10, 20, 30]
    predict(user_ids, item_ids)