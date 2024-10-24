# models/base_model.py

import torch.nn as nn
import torch
class BaseModel(nn.Module):
    """
    模型的基类，所有模型都继承自该类。
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.device = torch.device(config['device']['type'] if torch.cuda.is_available() else 'cpu')

    def forward(self, *input):
        """
        前向传播方法，需要在子类中实现。
        """
        raise NotImplementedError

    def calculate_loss(self, *input):
        """
        计算损失函数的方法，需要在子类中实现。
        """
        raise NotImplementedError

    def predict(self, *input):
        """
        预测方法，需要在子类中实现。
        """
        raise NotImplementedError