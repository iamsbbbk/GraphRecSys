# Logs/log_config.py

import logging
import os

def setup_logging(config):
    """
    设置日志记录器。

    参数：
    - config: 配置参数字典，包含日志相关的设置。
    """
    # 从配置中获取日志相关的参数
    log_dir = config['logging']['log_dir']
    log_filename = config['logging']['log_filename']
    log_level = config['logging'].get('log_level', 'INFO').upper()

    # 调试打印，查看 log_dir 的类型和值
    print(f"log_dir type: {type(log_dir)}, value: {log_dir}")

    # 如果日志目录不存在，创建目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件的完整路径
    log_file = os.path.join(log_dir, log_filename)

    # 创建 Logger 对象
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # 如果已经存在处理器，清除旧的处理器，防止重复日志
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建 Formatter 对象，定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建 FileHandler 对象，将日志写入文件
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建 StreamHandler 对象，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger