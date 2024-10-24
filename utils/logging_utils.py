# utils/logging_utils.py

import logging
import os

def setup_logging(config):
    """
    设置日志记录器。

    参数：
    - config: 配置参数字典，包含日志相关的设置。
    """
    log_dir = config['logging']['log_dir']
    log_filename = config['logging']['log_filename']
    log_level = config['logging'].get('log_level', 'INFO').upper()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件的完整路径
    log_file = os.path.join(log_dir, log_filename)

    # 创建 Logger 对象
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # 创建 Formatter 对象，定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建 FileHandler 对象，将日志写入文件
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建 StreamHandler 对象，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger