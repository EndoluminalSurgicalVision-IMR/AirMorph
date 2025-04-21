# -*- coding: utf-8 -*-



from typing import List, Tuple, Dict, Callable, Union, Any
import sys
import os
import numpy as np
import pandas as pd

__all__ = [
    'cosine_similarity_replace_nan_with_zero',
    'read_array_from_csv',
    'read_selected_array_from_csv'

]


def read_array_from_csv(csv_path, start_row=1, start_col=1):
    df = pd.read_csv(csv_path, header=None)
    df_trimmed = df.iloc[start_row:, start_col:]
    df_array = np.array(df_trimmed.values, dtype=float)
    return df_array


def read_selected_array_from_csv(csv_path, keywords, start_row=0):
    """
    从 CSV 文件中读取包含指定关键词的列，并从指定行开始提取为 NumPy 数组。

    参数:
        csv_path (str): CSV 文件路径。
        keywords (list of str): 要筛选的列标题中的关键词列表。
        start_row (int): 从指定的行开始（以 0 为基准）。

    返回:
        np.ndarray: 包含筛选列数据的 NumPy 数组。
    """
    # 读取整个 CSV 文件
    df = pd.read_csv(csv_path)

    # 获取匹配的列名
    matching_columns = [col for col in df.columns if any(
        keyword in col for keyword in keywords)]

    if not matching_columns:
        raise ValueError(f"No matching columns found for keywords: {keywords}")

    # 筛选匹配列并从指定行开始
    df_trimmed = df.loc[start_row:, matching_columns]

    # 转换为 NumPy 数组
    df_array = df_trimmed.to_numpy(dtype=float)

    return df_array


def cosine_similarity_replace_nan_with_zero(A, B):
    """
    计算两个矩阵 A 和 B 的余弦相似度矩阵，将nan记作0

    参数:
    A (numpy.ndarray): 形状为 (n, m) 的矩阵
    B (numpy.ndarray): 形状为  (n, m)  的矩阵

    返回:
    C (numpy.ndarray): 形状为  (n, n)  的余弦相似度矩阵
    """
    # 检查输入矩阵的形状是否一致
    if A.shape[1] != B.shape[1]:
        raise ValueError("矩阵 A 和 B 必须具有相同的列数")

    # 获取矩阵的形状
    n, d = A.shape
    m, d_B = B.shape

    # 扩展矩阵 A 和 B 以便进行广播
    # A_expanded: (n, 1, d)
    # B_expanded: (1, m, d)
    A_expanded = A[:, np.newaxis, :]  # Shape: (127, 1, 50)
    B_expanded = B[np.newaxis, :, :]  # Shape: (1, 127, 50)

    # 创建掩码，标记哪些维度在 A 和 B 中均不为 NaN
    # mask: (127, 127, 50)
    mask = (~np.isnan(A_expanded)) & (~np.isnan(B_expanded))

    # 用 0 替换 NaN，以便进行乘法运算
    A_filled = np.where(mask, A_expanded, 0)
    B_filled = np.where(mask, B_expanded, 0)

    # 计算内积：对最后一个维度求和
    # inner_product: (127, 127)
    inner_product = np.sum(A_filled * B_filled, axis=2)

    # 计算 A 的范数：对每对向量，计算有效维度上的平方和再开方
    # norm_A: (127, 127)
    norm_A = np.sqrt(np.sum(A_filled ** 2, axis=2))

    # 计算 B 的范数：同上
    # norm_B: (127, 127)
    norm_B = np.sqrt(np.sum(B_filled ** 2, axis=2))

    # 计算余弦相似度
    # similarity: (127, 127)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = inner_product / (norm_A * norm_B)
        # 如果 norm_A 或 norm_B 为 0，则结果为 NaN
        # similarity[(norm_A == 0) | (norm_B == 0)] = np.nan
        similarity[(norm_A == 0) | (norm_B == 0)] = 0.0
    return similarity
