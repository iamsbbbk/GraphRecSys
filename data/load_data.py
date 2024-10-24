# data/load_data.py

import os
import pandas as pd
from .preprocess import preprocess_data
from utils.data_processing import create_heterodata_from_dataframe

def load_data_from_directory(data_dir):
    """
    加载指定目录下的所有 CSV 文件，并合并成一个 DataFrame。
    """
    data_frames = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            # 重命名列名
            df = df.rename(columns={
                'parent_asin': 'item_id',
                # 如果其他列名需要映射，也可以在这里添加
            })
            data_frames.append(df)
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
    else:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到 CSV 文件")
    return combined_df

def load_amazon_dataset(root_dir):
    """
    加载 Amazon 数据集，返回训练图、验证图和测试图，以及用户和商品的编码映射。
    """
    # 加载训练集数据
    train_dir = os.path.join(root_dir, 'train')
    train_df = load_data_from_directory(train_dir)

    # 预处理训练集数据，获得编码映射
    train_df, user2id, item2id = preprocess_data(train_df)

    # 加载验证集数据
    valid_dir = os.path.join(root_dir, 'valid')
    valid_df = load_data_from_directory(valid_dir)
    # 重命名验证集的列名
    valid_df = valid_df.rename(columns={'parent_asin': 'item_id'})
    # 使用训练集的编码映射对验证集进行编码
    valid_df['user_id'] = valid_df['user_id'].map(user2id)
    valid_df['item_id'] = valid_df['item_id'].map(item2id)
    # 删除无法映射的记录
    valid_df.dropna(subset=['user_id', 'item_id'], inplace=True)
    valid_df = valid_df.astype({'user_id': int, 'item_id': int})

    # 加载测试集数据
    test_dir = os.path.join(root_dir, 'test')
    test_df = load_data_from_directory(test_dir)
    # 重命名测试集的列名
    test_df = test_df.rename(columns={'parent_asin': 'item_id'})
    # 使用训练集的编码映射对测试集进行编码
    test_df['user_id'] = test_df['user_id'].map(user2id)
    test_df['item_id'] = test_df['item_id'].map(item2id)
    # 删除无法映射的记录
    test_df.dropna(subset=['user_id', 'item_id'], inplace=True)
    test_df = test_df.astype({'user_id': int, 'item_id': int})

    # 将 DataFrame 转换为图数据
    train_graph = create_heterodata_from_dataframe(train_df)
    valid_graph = create_heterodata_from_dataframe(valid_df)
    test_graph = create_heterodata_from_dataframe(test_df)

    # 返回结果
    return train_graph, valid_graph, test_graph, user2id, item2id