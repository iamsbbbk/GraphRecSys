# utils/seed.py

import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    设置全局随机种子。

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