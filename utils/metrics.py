# utils/metrics.py

import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(outputs, labels):
    """
    计算模型的评估指标。

    参数：
    - outputs: 模型的输出，预测值，张量或 NumPy 数组。
    - labels: 真实标签，张量或 NumPy 数组。

    返回：
    - metrics: 包含评估指标的字典。
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 二分类情况下，计算概率值或 logits 的 AUC
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        # 多分类，取正类的概率
        probs = outputs[:, 1]
    else:
        probs = outputs

    auc = roc_auc_score(labels, probs)
    # 将概率转换为类别
    preds = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    metrics = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    return metrics