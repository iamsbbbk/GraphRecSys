# utils/data_processing.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

def process_raw_data(raw_data_path):
    """
    处理原始数据，生成用于模型训练的数据集。

    参数：
    - raw_data_path: 原始数据文件路径。

    返回：
    - processed_data: 处理后的数据。
    """
    # 示例：加载 CSV 文件
    df = pd.read_csv(raw_data_path)

    # 数据清洗
    df = df.dropna()

    # 编码分类变量
    label_encoders = {}
    for col in ['user_id', 'item_id']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 提取特征和标签
    features = df.drop('label', axis=1)
    labels = df['label']

    # 将数据转换为 PyTorch 张量
    features = torch.tensor(features.values, dtype=torch.float)
    labels = torch.tensor(labels.values, dtype=torch.long)

    # 返回处理后的数据
    processed_data = {
        'features': features,
        'labels': labels,
        'label_encoders': label_encoders
    }

    return processed_data

def create_heterodata_from_dataframe(df):
    """
    从 Pandas DataFrame 创建 HeteroData 对象。

    参数：
    - df: 包含用户-商品交互的 DataFrame，列包括 'user_id'、'item_id'、'label' 等。

    返回：
    - data: HeteroData 对象。
    """
    data = HeteroData()

    # 编码用户和商品节点
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['item_id'] = item_encoder.fit_transform(df['item_id'])
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    # 添加节点，该示例未包含节点特征
    data['user'].num_nodes = num_users
    data['item'].num_nodes = num_items

    # 添加边（用户与商品的交互）
    interactions = torch.tensor(df[['user_id', 'item_id']].values, dtype=torch.long).t()
    data['user', 'interacts', 'item'].edge_index = interactions

    # 添加边属性，例如评级或标签
    if 'rating' in df.columns:
        data['user', 'interacts', 'item'].edge_attr = torch.tensor(df['rating'].values, dtype=torch.float)
    if 'label' in df.columns:
        data['user', 'interacts', 'item'].edge_label = torch.tensor(df['label'].values, dtype=torch.long)

    # 返回 HeteroData 对象
    return data