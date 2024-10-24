# trainers/evaluator.py

import torch
import logging
from utils.metrics import compute_metrics

class Evaluator:
    """
    模型评估器，封装评估过程。

    参数：
    - model: 要评估的模型。
    - device: 计算设备。
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)

    def evaluate(self, data_loader):
        """
        在给定的数据集上评估模型。

        参数：
        - data_loader: 数据集的 DataLoader。

        返回：
        - metrics: 包含评估指标的字典。
        """
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)
                output = self.model(batch_data)
                labels = self.model.get_labels(batch_data)
                all_outputs.append(output)
                all_labels.append(labels)

        # 将所有批次的数据合并
        outputs = torch.cat(all_outputs, dim=0)
        labels = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(outputs, labels)
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics