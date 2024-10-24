# trainers/predictor.py

import torch
import logging

class Predictor:
    """
    模型预测器，封装预测过程。

    参数：
    - model: 要用于预测的模型。
    - device: 计算设备。
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)

    def predict(self, data_loader):
        """
        对给定的数据集进行预测。

        参数：
        - data_loader: 数据集的 DataLoader。

        返回：
        - predictions: 模型的预测结果。
        """
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)
                output = self.model(batch_data)
                predictions = self.model.get_predictions(output)
                all_predictions.append(predictions)

        # 将所有批次的预测结果合并
        predictions = torch.cat(all_predictions, dim=0)
        return predictions.cpu().numpy()