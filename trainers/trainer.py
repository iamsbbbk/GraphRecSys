# trainers/trainer.py

import torch
import logging
from utils.metrics import compute_metrics
from utils.helper_functions import save_checkpoint
from torch.utils.data import DataLoader

class Trainer:
    """
    模型训练器，封装训练过程。

    参数：
    - model: 要训练的模型。
    - optimizer: 优化器。
    - device: 计算设备。
    - config: 配置参数字典。
    """
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,      # 添加 loss_fn 参数
                 device,
                 train_graph,
                 valid_graph,
                 num_epochs,
                 batch_size,
                 logger):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn  # 保存损失函数
        self.device = device
        self.train_graph = train_graph
        self.valid_graph = valid_graph
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.logger = logger

    def train(self, train_loader, valid_loader=None, epochs=10, callbacks=[]):
        """
        训练模型。

        参数：
        - train_loader: 训练数据的 DataLoader。
        - valid_loader: 验证数据的 DataLoader，可选。
        - epochs: 训练的轮数。
        - callbacks: 回调函数列表，例如 EarlyStopping。

        返回：
        - None
        """
        self.logger.info("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_data in train_loader:
                # 将数据移动到设备
                batch_data = batch_data.to(self.device)

                # 前向传播
                output = self.model(batch_data)

                # 计算损失
                loss = self.model.calculate_batch_loss(output, batch_data)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

            # 验证
            if valid_loader is not None:
                metrics = self.evaluate(valid_loader)
                self.logger.info(f"Validation metrics at epoch {epoch+1}: {metrics}")

                # 检查回调函数
                for callback in callbacks:
                    callback_metric = metrics.get(callback.monitor_metric, None)
                    if callback_metric is not None:
                        callback(callback_metric, self.model)
                        if callback.early_stop:
                            self.logger.info("Early stopping triggered.")
                            return

        self.logger.info("Training completed.")

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
        return metrics