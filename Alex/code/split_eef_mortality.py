import pandas as pd
import numpy as np
import os

# 数据路径
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/data_cleaned/allinfo_dead.csv')
try:
    df = pd.read_csv(DATA_PATH, encoding='gbk')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

# 去除字段名首尾空格
if hasattr(df, 'columns'):
    df.columns = df.columns.str.strip()

# eef分两类（中位数切分）
eef_median = df['eef'].median()
df['eef_flag'] = np.where(df['eef'] >= eef_median, 1, 0)
print(f"eef中位数: {eef_median:.4f}，高于等于为1，低于为0")

# Mortality_rate分两类（0.05为阈值）
df['Mortality_flag'] = np.where(df['Mortality_rate'] <= 0.05, 1, 0)
print(f"Mortality_rate阈值: 0.05，小于等于为1，大于为0")

# 保存新文件
save_path = os.path.join(os.path.dirname(__file__), '../data/data_cleaned/allinfo_dead_with_flags.csv')
df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"已保存切分后数据到: {save_path}")

# 可选：输出分布统计
print('\neef_flag 分布:')
print(df['eef_flag'].value_counts())
print('\nMortality_flag 分布:')
print(df['Mortality_flag'].value_counts())
