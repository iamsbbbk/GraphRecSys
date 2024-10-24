# data/preprocess.py

import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    对原始数据进行预处理，处理时间戳、用户和商品 ID 等。

    参数：
    - df: 包含原始数据的 Pandas DataFrame，必须包含以下列：
        - 'user_id': 用户 ID
        - 'item_id': 商品 ID
        - 'rating': 评分
        - 'timestamp': 时间戳

    返回：
    - df: 预处理后的 DataFrame
    - user2id: 用户 ID 映射字典
    - item2id: 商品 ID 映射字典
    """

    # 检查 DataFrame 是否包含必要的列
    required_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 缺少必要的列：{col}")

    # 处理缺失值，删除包含缺失值的行
    df = df.dropna(subset=required_columns)
    df = df.reset_index(drop=True)

    # 检查 'timestamp' 列的值
    print("Timestamp column info:")
    print(df['timestamp'].describe())
    print("Minimum timestamp:", df['timestamp'].min())
    print("Maximum timestamp:", df['timestamp'].max())
    print("First few timestamps:", df['timestamp'].head())

    # 判断时间戳的单位，根据时间戳数字的位数
    timestamp_max = df['timestamp'].max()
    timestamp_digits = len(str(int(timestamp_max)))

    if timestamp_digits >= 16:
        time_unit = 'ns'  # 纳秒
    elif timestamp_digits >= 13:
        time_unit = 'ms'  # 毫秒
    elif timestamp_digits >= 10:
        time_unit = 's'  # 秒
    else:
        time_unit = 's'  # 默认单位为秒

    print(f"Detected timestamp unit: {time_unit}")

    # 将时间戳转换为日期时间类型
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit=time_unit)

    # 对用户 ID 和商品 ID 进行编码，从 0 开始
    user_encoder = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
    item_encoder = {iid: idx for idx, iid in enumerate(df['item_id'].unique())}

    # 应用编码
    df['user_id'] = df['user_id'].map(user_encoder)
    df['item_id'] = df['item_id'].map(item_encoder)

    # 创建映射的反向字典（可选）
    user_decoder = {idx: uid for uid, idx in user_encoder.items()}
    item_decoder = {idx: iid for iid, idx in item_encoder.items()}

    # 打印编码后的信息
    print(f"Number of unique users: {len(user_encoder)}")
    print(f"Number of unique items: {len(item_encoder)}")

    # 返回预处理后的数据和编码映射
    return df, user_encoder, item_encoder

def split_data(df, test_size=0.2, random_state=42):
    """
    将数据集拆分为训练集和验证集。

    参数：
    - df: 预处理后的 DataFrame
    - test_size: 测试集所占比例
    - random_state: 随机种子

    返回：
    - train_df: 训练集 DataFrame
    - valid_df: 验证集 DataFrame
    """
    from sklearn.model_selection import train_test_split

    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['user_id']
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f"Train data size: {len(train_df)}")
    print(f"Validation data size: {len(valid_df)}")

    return train_df, valid_df