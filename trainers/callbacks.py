# trainers/callbacks.py

import numpy as np
import torch
import os
import logging
from utils.helper_functions import save_checkpoint

class EarlyStopping:
    """
    实现早停（Early Stopping）机制。

    参数：
    - patience: 在验证指标没有提升的情况下，等待的 epoch 数量。
    - monitor_metric: 要监控的验证指标名称。
    - mode: 监控指标的模式，'min' 或 'max'。
    - save_path: 如果模型改进，保存模型的路径。
    """
    def __init__(self, patience=10, monitor_metric='loss', mode='min', save_path='best_model.pth'):
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logging.getLogger(__name__)

        if self.mode == 'min':
            self.score_func = lambda current, best: current < best
            self.best_score = np.Inf
        elif self.mode == 'max':
            self.score_func = lambda current, best: current > best
            self.best_score = -np.Inf
        else:
            raise ValueError("Mode must be 'min' or 'max'")

    def __call__(self, current_score, model):
        if self.score_func(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            self.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """保存模型"""
        save_checkpoint(model, None, None, self.best_score, self.save_path)
        self.logger.info(f"Model improved. Saved to {self.save_path}")