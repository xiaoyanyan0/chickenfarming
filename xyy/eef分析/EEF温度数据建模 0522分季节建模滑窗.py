
#########加上内外温差湿差等数据
import numpy as np
# import math
import pandas as pd
import lightgbm as lgb
# import pickle
import matplotlib.pyplot as plt
# from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import *
from sklearn.metrics import *
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import *

import optbinning
from optbinning import BinningProcess
import warnings

import seaborn as sns
import os
# import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
all_info_temdata=pd.read_csv('./data/data_cleaned/sliding_temdata0521.csv',encoding='gbk')
all_info_temdata.shape
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')
import warnings
warnings.filterwarnings("ignore")
# all_info_temdata
col=[a.upper()for a in all_info_temdata.columns.to_list()]
all_info_temdata.columns=col

df=all_info_temdata.set_index('ID_NUM')


df=df.rename({'MORTALITY_RATE_X':'MORTALITY_RATE'},axis=1)
# 查看各个月份的分布
# 将日期列转换为 datetime 类型

df.columns.to_list()
#筛除0占比超过阈值特征
drop_columns_Mortality=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','livability_pct'
                        ,'yield_per_m2'
                        ]
drop_columns_Mortality=[a.upper()for a in drop_columns_Mortality]
drop_columns_Mortality=[i for i in df.columns.to_list() if i in drop_columns_Mortality]

marketingdata_columns = [
    'house',                    # 栋舍 House
    'birds_placed',             # 进雏只数\nBird placed No.
    'gender',                   # 公母\nGender
    'house_area_m2',            # 鸡舍面积m2 \nHouse Area
    'stocking_density',         # 出栏密度\nDensity
    'birds_hung',               # 挂鸡只数\nHang No.
    'total_hung_weight_kg',     # 挂鸡总重（kg）\nTotal hung weight
    'avg_weight_kg',            # 均重（kg）\n Average weight
    'small_broilers_count',     # 小毛鸡数量\nSmall broilers No.
    'small_broilers_weight_kg', # 小毛鸡总重（kg）\n Total weight of small broilers
    'pp_dead_culled_count',     # PP死淘鸡只\nPP cull and dead No.
    'dead_culled_weight_kg',    # 死淘总重（kg）\nDead and Cull Weight
    'pp_rejects_count',         # PP不合格淘汰鸡\nPP Cull
    'pp_rejects_weight_kg',     # PP淘汰鸡总重\nPP Cull bird Weight
    'age_days',                 # 日龄Age
    'dead_during_catch_count',  # 出栏造成死亡只数 \nDead while catching
    'birds_caught_count',       # 出鸡只数\nCatching No.
    'livability_pct',           # 成活率\nLivability (%)
    'total_caught_weight_kg',   # 出鸡总重（kg）\nTotal Catched weight
    'yield_per_m2',             # 单位面积产肉率\nDensity, Yield(KG)/m2
    'final_avg_weight_kg',      # 均重（kg）\nAverage weight
    'total_feed_kg',            # 累计耗料（kg）\nFeed cons. Cum.
    'fcr',                      # 料肉比 FCR
    'adjusted_fcr',             # Adjust FCR (base 2.45KG)
    'eef',                      # 欧洲指数 EEF
    'revenue',                  # 毛鸡销售收入（元'）
    'profit_per_house',         # 每栋纯利润（元）
    'medicine_per_bird',        # 药品（元/只）
    'vaccine_per_bird',         # 疫苗（元/只）
    'mv_cost_per_bird',         # M&V费（元/只）
    'disinfectant_per_bird',    # 消毒药费（元/只）
    'feed_per_bird',            # 饲料（元/只）
    'electricity_per_bird',     # 用电（元/只）
    'gas_per_bird',             # 燃气（元/只）
    'labor_per_bird',           # 人工（元/只）
    'consumables_per_bird',     # 低值易耗品（元/只）
    'depreciation_per_bird',    # 折旧费（元/只）
    'chick_cost_per_bird',      # 雏鸡成本（元/元）
    'cost_per_bird',            # 每只鸡成本（元）
    'chick_cost',               # 雏鸡成本（元）
    'total_cost',               # 总成本（元）
    'cost_per_kg',              # 每公斤成本（元）
    'feed_cost',                # 饲料成本（元）
]
marketingdata_columns=[a.upper()for a in marketingdata_columns]
drop_columns_marketingdata=[i for i in df.columns.to_list() if i in marketingdata_columns]

#把细分的死淘数据和出栏数据指标去掉
df2=df.drop(columns=drop_columns_marketingdata+drop_columns_Mortality,axis=1)

df2.shape
###字段处理
# 将日期列转换为 datetime 类型
date_columns = ['DOCDATE', 'ESTIMATEDSLAUGHTERDATE ', 'HARVESTSTATUS']
for col in date_columns:
    df2[col] = pd.to_datetime(df2[col])
    df2[f'{col}_month'] = df2[col].dt.month
    df2[f'{col}_month']=df2[f'{col}_month'].astype(str)


df2=df2.drop(date_columns,axis=1)

df2.shape

def calculate_window_differences(sliding_features_df):
    """
    计算滑动窗口特征之间的差值和变化率
    
    参数:
    sliding_features_df: 包含滑动窗口特征的DataFrame
    
    返回:
    包含差值和变化率的新DataFrame
    """
    # 复制原始DataFrame以避免修改原始数据
    result_df = sliding_features_df.copy()
    
    # 获取所有窗口特征列(mean列)
    mean_cols = [col for col in sliding_features_df.columns if '_MEAN' in col]
    
    # 按窗口大小分组
    window_groups = {}
    for col in mean_cols:
        # 提取窗口信息 (如 'W7_0-6天_体温_mean' -> ('7', '0-6'))
        parts = col.split('_')
        ws = parts[0][1:]  # 窗口大小 (去掉'W')
        day_range = parts[1]  # 日龄范围
        
        if ws not in window_groups:
            window_groups[ws] = []
        window_groups[ws].append((day_range, col))
    
    # 对每个窗口大小组计算差值和变化率
    for ws, windows in window_groups.items():
        # 按日龄范围排序窗口
        windows.sort(key=lambda x: int(x[0].split('-')[0]))
        
        # 计算相邻窗口的差值和变化率
        for i in range(1, len(windows)):
            prev_range, prev_col = windows[i-1]
            curr_range, curr_col = windows[i]
            
            # 差值特征名
            diff_name = f"Diff_{ws}_{prev_range}_to_{curr_range}"
            # 变化率特征名
            pct_name = f"PctChange_{ws}_{prev_range}_to_{curr_range}"
            
            # 计算差值
            result_df[diff_name] = result_df[curr_col] - result_df[prev_col]
            
            # 计算变化率(避免除以0)
            result_df[pct_name] = (result_df[curr_col] - result_df[prev_col]) / \
                                 (result_df[prev_col].replace(0, np.nan) + 1e-10)
    
    return result_df


# col='W3_0-2天_鸡舍温度-平均_MEAN'
# parts = col.split('_')
df3=calculate_window_differences(df2)
df3.columns.to_list()
# df2.reset_index(drop=False)[['ID_NUM']]
def prepare_seasonal_data(df, date_col='HARVESTSTATUS_month', target_col='MORTALITY_RATE', quantile_threshold=0.8):
    """
    将数据按季节分割并创建二分类目标变量

    """
    # 按季节分割数据
    seasonal_dfs = {
        'winter': df[df[date_col].isin(['12', '1', '2'])].copy(),
        'spring': df[df[date_col].isin(['3', '4'])].copy(),
        'summer': df[df[date_col].isin(['5','6', '7', '8','9'])].copy(),
        'autumn': df[df[date_col].isin(['10', '11'])].copy()
    }
    
    # 为每个季节创建二分类目标变量
    for season, season_df in seasonal_dfs.items():
        # 检查季节数据是否为空
        if season_df.empty:
            print(f"警告: {season.capitalize()}数据量为0，跳过处理")
            continue
            
        # 检查目标列是否有有效数据
        valid_values = season_df[target_col].dropna()
        if len(valid_values) == 0:
            print(f"警告: {season.capitalize()}的{target_col}列无有效数据，跳过处理")
            continue
            
        # 计算分位数并创建标签
        quantile = np.quantile(valid_values, quantile_threshold)
        print(f"{season.capitalize()}分位数: {quantile:.4f}")
        season_df['Mortality_flg'] = season_df[target_col].apply(
            lambda x: 1 if pd.notna(x) and x >= quantile else 0
        )
        
        # 打印各季节数据量
        print(f"{season.capitalize()}数据量: {season_df.shape[0]}, 正样本比例: {season_df['Mortality_flg'].mean():.2%}")
        
    return seasonal_dfs

def prepare_seasonal_data2(df, date_col='HARVESTSTATUS_month', target_col='MORTALITY_RATE', quantile_threshold=0.8):
    """
    将数据按季节分割并为每个月创建二分类目标变量，最终按季节合并

    """
    # 按月份分割数据
    monthly_dfs = {}
    for month in df[date_col].unique():
        monthly_dfs[month] = df[df[date_col] == month].copy()
    
    # 为每个月创建二分类目标变量
    for month, month_df in monthly_dfs.items():
        # 检查月份数据是否为空
        if month_df.empty:
            print(f"警告: 月份{month}数据量为0，跳过处理")
            continue
            
        # 检查目标列是否有有效数据
        valid_values = month_df[target_col].dropna()
        if len(valid_values) == 0:
            print(f"警告: 月份{month}的{target_col}列无有效数据，跳过处理")
            continue
            
        # 计算该月分位数并创建标签
        quantile = np.quantile(valid_values, quantile_threshold)
        month_df['Mortality_flg'] = month_df[target_col].apply(
            lambda x: 1 if pd.notna(x) and x >= quantile else 0
        )
        
        # 打印各月数据量
        print(f"月份{month}数据量: {month_df.shape[0]}, 正样本比例: {month_df['Mortality_flg'].mean():.2%}")
    
    # 按季节合并月份数据
    seasonal_dfs = {
        'winter': pd.concat([monthly_dfs.get('12', pd.DataFrame()), 
                             monthly_dfs.get('1', pd.DataFrame()), 
                             monthly_dfs.get('2', pd.DataFrame())], ignore_index=True),
        'spring': pd.concat([monthly_dfs.get('3', pd.DataFrame()), 
                             monthly_dfs.get('4', pd.DataFrame())], ignore_index=True),
        'summer': pd.concat([ monthly_dfs.get('5', pd.DataFrame()),
                             monthly_dfs.get('6', pd.DataFrame()), 
                             monthly_dfs.get('7', pd.DataFrame()), 
                             monthly_dfs.get('8', pd.DataFrame()),
                             monthly_dfs.get('9', pd.DataFrame())], ignore_index=True),
        'autumn': pd.concat([monthly_dfs.get('10', pd.DataFrame()), 
                             monthly_dfs.get('11', pd.DataFrame())], ignore_index=True)
    }
    
    # 打印各季节数据量
    for season, season_df in seasonal_dfs.items():
        if not season_df.empty:
            print(f"{season.capitalize()}数据量: {season_df.shape[0]}, 正样本比例: {season_df['Mortality_flg'].mean():.2%}")
    
    return seasonal_dfs

####建模函数

def split_data_m(df, model_type,validation_month_list):
    
    validation_data = df[df['HARVESTSTATUS_month'].isin(validation_month_list)]
    training_test_data = df[~df['HARVESTSTATUS_month'].isin(validation_month_list)]
   # 对比目标变量分布
    plt.figure(figsize=(10, 4))
    sns.kdeplot(training_test_data['MORTALITY_RATE'], label='Train')
    sns.kdeplot(validation_data['MORTALITY_RATE'], label='Validation')
    plt.title("MORTALITY_RATE Distribution Comparison")
    plt.legend()
    plt.show()
    if model_type == 'binary':
        X = training_test_data.drop(columns=['MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg'])
        y = training_test_data['Mortality_flg']
    elif model_type =='regression':
        X = training_test_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE_21','MORTALITY_RATE'])
        y = training_test_data['MORTALITY_RATE']
    else:
        raise ValueError("model_type 必须为 'binary' 或'regression'")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == 'binary':
        X_validation = validation_data.drop(columns=['MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg'])
        y_validation = validation_data['Mortality_flg']
    elif model_type =='regression':
        X_validation = validation_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE_21','MORTALITY_RATE'])
        y_validation = validation_data['MORTALITY_RATE']

    return X_train, X_test, y_train, y_test, X_validation, y_validation

def split_data_s(df, model_type, validation_size=0.2, test_size=0.3, random_state=42):
    """
    将数据按比例随机拆分为训练集、验证集和测试集
    
    """
    # 先将数据分为训练+测试集和验证集
    train_test_data, validation_data = train_test_split(
        df, 
        test_size=validation_size, 
        random_state=random_state,
        stratify=df['Mortality_flg'] if model_type == 'binary' else None
    )
    
    # 对比目标变量分布
    plt.figure(figsize=(10, 4))
    sns.kdeplot(train_test_data['MORTALITY_RATE'], label='Train')
    sns.kdeplot(validation_data['MORTALITY_RATE'], label='Validation')
    plt.title("MORTALITY_RATE Distribution Comparison")
    plt.legend()
    plt.show()
    
    # 根据模型类型选择目标变量
    if model_type == 'binary':
        X = train_test_data.drop(columns=['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg'])
        y = train_test_data['Mortality_flg']
        X_validation = validation_data.drop(columns=['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg'])
        y_validation = validation_data['Mortality_flg']
    elif model_type =='regression':
        X = train_test_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE_21', 'MORTALITY_RATE'])
        y = train_test_data['MORTALITY_RATE']
        X_validation = validation_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE_21', 'MORTALITY_RATE'])
        y_validation = validation_data['MORTALITY_RATE']
    else:
        raise ValueError("model_type 必须为 'binary' 或'regression'")
    
    # 再将训练+测试集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if model_type == 'binary' else None
    )
    
    # 打印各集合大小
    print("\n数据分割结果：")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"验证集大小: {X_validation.shape}")
    
    if model_type == 'binary':
        print(f"训练集正样本比例: {y_train.mean():.2%}")
        print(f"测试集正样本比例: {y_test.mean():.2%}")
        print(f"验证集正样本比例: {y_validation.mean():.2%}")
        print(f"训练集样本月份分布：{X_train['HARVESTSTATUS_month'].value_counts(normalize=True)}")
        print(f"测试集样本月份分布：{X_test['HARVESTSTATUS_month'].value_counts(normalize=True)}")
        print(f"验证集样本月份分布：{X_validation['HARVESTSTATUS_month'].value_counts(normalize=True)}")
    else:
        print(f"训练集目标变量均值: {y_train.mean():.4f}")
        print(f"测试集目标变量均值: {y_test.mean():.4f}")
        print(f"验证集目标变量均值: {y_validation.mean():.4f}")
    
    return X_train, X_test, y_train, y_test, X_validation, y_validation


def stratified_split_data(df, model_type, validation_size=0.2, test_size=0.3, random_state=42):
    """
    改进版分层抽样数据分割函数
    """
    # 参数校验
    if model_type not in ['binary', 'regression']:
        raise ValueError("model_type 必须为 'binary' 或 'regression'")
    
    # 创建分层依据
    if model_type == 'binary':
        stratify_col = df['Mortality_flg']
        print("二分类任务 - 按目标变量分层")
    else:
        # 对回归任务，将连续目标变量分箱后分层
        num_bins = min(5, len(df) // 20)  # 自适应分箱数
        stratify_col = pd.qcut(df['MORTALITY_RATE'], q=num_bins, duplicates='drop')
        print(f"回归任务 - 按目标变量分{len(stratify_col.unique())}层")
    
    # 第一次分割：训练测试集 vs 验证集
    train_test, val = train_test_split(
        df,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # 第二次分割：训练集 vs 测试集
    if model_type == 'binary':
        new_stratify = train_test['Mortality_flg']
    else:
        new_stratify = pd.qcut(train_test['MORTALITY_RATE'], 
                             q=num_bins, duplicates='drop')
    
    train, test = train_test_split(
        train_test,
        test_size=test_size,
        random_state=random_state,
        stratify=new_stratify
    )
    
    # 可视化分布对比
    plt.figure(figsize=(15, 5))
    
    if model_type == 'binary':
        # 二分类任务：展示正样本比例
        proportions = pd.DataFrame({
            'Train': [train['Mortality_flg'].mean()],
            'Test': [test['Mortality_flg'].mean()],
            'Validation': [val['Mortality_flg'].mean()]
        })
        sns.barplot(data=proportions)
        plt.title("Class Distribution Across Splits")
        plt.ylabel("Positive Class Proportion")
    else:
        # 回归任务：展示目标变量分布
        sns.kdeplot(train['MORTALITY_RATE'], label='Train')
        sns.kdeplot(test['MORTALITY_RATE'], label='Test')
        sns.kdeplot(val['MORTALITY_RATE'], label='Validation')
        plt.title("Target Variable Distribution Comparison")
        plt.legend()
    
    plt.show()
    
    # 准备特征和目标变量
    if model_type == 'binary':
        drop_cols = ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg']
        y_col = 'Mortality_flg'
    else:
        drop_cols = ['Mortality_flg', 'MORTALITY_RATE_21', 'MORTALITY_RATE']
        y_col = 'MORTALITY_RATE'
    
    X_train = train.drop(columns=drop_cols)
    y_train = train[y_col]
    X_test = test.drop(columns=drop_cols)
    y_test = test[y_col]
    X_val = val.drop(columns=drop_cols)
    y_val = val[y_col]
    
    # 打印分割结果
    print("\n=== 数据分割结果 ===")
    print(f"训练集: {X_train.shape[0]} samples")
    print(f"测试集: {X_test.shape[0]} samples")
    print(f"验证集: {X_val.shape[0]} samples")
    
    if model_type == 'binary':
        print(f"\n正样本比例:")
        print(f" - 训练集: {y_train.mean():.2%}")
        print(f" - 测试集: {y_test.mean():.2%}")
        print(f" - 验证集: {y_val.mean():.2%}")
        print(f"训练集样本月份分布：{X_train['HARVESTSTATUS_month'].value_counts(normalize=True)}")
        print(f"测试集样本月份分布：{X_test['HARVESTSTATUS_month'].value_counts(normalize=True)}")
        print(f"验证集样本月份分布：{X_val['HARVESTSTATUS_month'].value_counts(normalize=True)}")
    else:
        print(f"\n目标变量统计:")
        print(f" - 训练集均值: {y_train.mean():.4f} ± {y_train.std():.4f}")
        print(f" - 测试集均值: {y_test.mean():.4f} ± {y_test.std():.4f}")
        print(f" - 验证集均值: {y_val.mean():.4f} ± {y_val.std():.4f}")
    
    return X_train, X_test, y_train, y_test, X_val, y_val

def get_non_collinear_vars(cutoff, df):
    # 移除零方差特征
    df = df.loc[:, df.var() > 0]
    if df.empty:
        return []
        
    corr_matrix = df.corr().abs()
    selected_cols = []
    dropped_cols = set()
    
    # 按与目标变量的相关性排序（如果有目标变量）
    for col in corr_matrix.columns:
        if col in dropped_cols:
            continue
        selected_cols.append(col)
        high_corr_cols = corr_matrix.index[
            (corr_matrix[col] > cutoff) & 
            (corr_matrix.index != col)
        ].tolist()
        dropped_cols.update(high_corr_cols)
    return selected_cols
def data_preprocessing(df, thres_zeros=0.7, cutoff=0.8):
   
    # 处理缺失值
    missing_p = np.sum(df.isnull(), axis=0) / df.shape[0]
    low_missing = missing_p[missing_p < thres_zeros].index.tolist()
    df_keep = df[low_missing].copy()
    for col in ['HOUSEID', 'HEAGE']:
        if col in df_keep.columns:
            df_keep[col] = df_keep[col].astype(int).astype(str)
        else:
            print(f"不存在{col}列，跳过转换")

    # 区分数值型和对象型变量
    numeric_columns = []
    object_columns = []
    for column in df_keep.columns:
        if np.issubdtype(df_keep[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            object_columns.append(column)

    # 处理非有限值
    if np.any(~np.isfinite(df_keep[numeric_columns])):
        df_keep[numeric_columns] = df_keep[numeric_columns].replace([np.inf, -np.inf], np.nan)
        print("替换了非有限值")
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    df_keep[numeric_columns] = df_keep[numeric_columns].clip(lower=min_float32, upper=max_float32)

    # 共线性处理
    target_cols = [col for col in ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg'] 
                  if col in df_keep.columns]
    other_numeric = [col for col in numeric_columns if col not in target_cols]
    
    if other_numeric:
        selected_cols = get_non_collinear_vars(cutoff, df_keep[other_numeric])
        print(f"共线性处理后保留 {len(selected_cols)}/{len(other_numeric)} 个数值特征")
    else:
        selected_cols = []
        print("警告: 无有效数值特征进行共线性分析")

    # selected_cols = get_non_collinear_vars(cutoff, df_keep[numeric_columns].drop(['MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg'], axis=1))
    
    # 保留目标变量
    target_cols = ['MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg']
    
    # 合并选择的数值列、对象列和目标变量
    
    df_keep2 = df_keep[selected_cols + object_columns + target_cols].copy()
    
    # 将object类型转换为category类型
    for col in object_columns:
        df_keep2[col] = df_keep2[col].astype('category')
    
    return df_keep2
def data_preprocessing2(df, thres_zeros=0.7, cutoff=0.8):
   
    # 处理缺失值
    missing_p = np.sum(df.isnull(), axis=0) / df.shape[0]
    low_missing = missing_p[missing_p < thres_zeros].index.tolist()
    print(f"原始的变量数量: {len(df.columns)}")
    print(f"低缺失率的变量数量: {len(low_missing)}")
    df_keep = df[low_missing].copy()
    for col in ['HOUSEID', 'HEAGE']:
        if col in df_keep.columns:
            df_keep[col] = df_keep[col].astype(int).astype(str)
        else:
            print(f"不存在{col}列，跳过转换")

    # 区分数值型和对象型变量
    numeric_columns = []
    object_columns = []
    for column in df_keep.columns:
        if np.issubdtype(df_keep[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            object_columns.append(column)

    # 处理非有限值
    if np.any(~np.isfinite(df_keep[numeric_columns])):
        df_keep[numeric_columns] = df_keep[numeric_columns].replace([np.inf, -np.inf], np.nan)
        print("替换了非有限值")

    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    df_keep[numeric_columns] = df_keep[numeric_columns].clip(lower=min_float32, upper=max_float32)

    # # 共线性处理
    # target_cols = [col for col in ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg'] 
    #               if col in df_keep.columns]
    # other_numeric = [col for col in numeric_columns if col not in target_cols]
    
    # if other_numeric:
    #     selected_cols = get_non_collinear_vars(cutoff, df_keep[other_numeric])
    #     print(f"共线性处理后保留 {len(selected_cols)}/{len(other_numeric)} 个数值特征")
    # else:
    #     selected_cols = []
    #     print("警告: 无有效数值特征进行共线性分析")

    # # selected_cols = get_non_collinear_vars(cutoff, df_keep[numeric_columns].drop(['MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg'], axis=1))
    
    # # 保留目标变量
    # target_cols = ['MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg']
    
    # 合并选择的数值列、对象列和目标变量
    
    df_keep2 = df_keep[  object_columns + numeric_columns].copy()
    
    # 将object类型转换为category类型
    for col in object_columns:
        df_keep2[col] = df_keep2[col].astype('category')
    
    return df_keep2
# df_keep2 = data_preprocessing(seasonal_dfs['winter'])
# has_inf = seasonal_dfs['winter'].apply(lambda x: np.isfinite(x).any())
                       
def lightgbm_modeling(df, validation_month_list, split_model, model_type='binary', random_state=42, quantile_threshold=0.8):
    # 数据预处理和分割保持不变
    df_keep2 = data_preprocessing(df)
    print('保留的字段数量', df_keep2.shape)
    
    if 'ESTIMATEDSLAUGHTERDATE _month' in df_keep2.columns: 
        df_keep2 = df_keep2.drop(columns=['ESTIMATEDSLAUGHTERDATE _month', 'DOCDATE_month'], axis=1)
    
    if split_model == 'month':
        X_train, X_test, y_train, y_test, X_validation, y_validation = split_data_m(df_keep2, model_type, validation_month_list)
    elif split_model == 'sample':
        X_train, X_test, y_train, y_test, X_validation, y_validation = split_data_s(df_keep2, model_type)
    else:
        X_train, X_test, y_train, y_test, X_validation, y_validation = stratified_split_data(df_keep2, model_type)
    
    # print("\n数据分割结果：")
    # print(f"训练集大小: {X_train.shape}")
    # print(f"测试集大小: {X_test.shape}")
    # print(f"验证集大小: {X_validation.shape}")
 
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 3,
        'num_leaves': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': 42,
        'verbose': -1,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5
    }

    # 训练模型
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    lgb_baseline = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=1000
    )

    # 预测概率和类别
    y_pred_prob = lgb_baseline.predict(X_test, num_iteration=lgb_baseline.best_iteration)
    quantile = np.quantile(y_pred_prob, quantile_threshold)
    y_pred_class = (y_pred_prob >= quantile).astype(int)

    # 新增评估指标
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
    
    print("\n=== 测试集评估 ===")
    print("AUC Score:", roc_auc_score(y_test, y_pred_prob))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_class))
    print("F1-Score:", f1_score(y_test, y_pred_class))
    print("MCC:", matthews_corrcoef(y_test, y_pred_class))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))
    # print("\n分类报告:\n", classification_report(y_test, y_pred_class))

    # 跨期验证集评估
    y_pred_prob_validation = lgb_baseline.predict(X_validation, num_iteration=lgb_baseline.best_iteration)
    y_pred_class_validation = (y_pred_prob_validation >= quantile).astype(int)
    
    print("\n=== 跨期验证集评估 ===")
    print("Validation AUC Score:", roc_auc_score(y_validation, y_pred_prob_validation))
    print("Validation Balanced Accuracy:", balanced_accuracy_score(y_validation, y_pred_class_validation))
    print("Validation F1-Score:", f1_score(y_validation, y_pred_class_validation))
    print("Validation MCC:", matthews_corrcoef(y_validation, y_pred_class_validation))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_validation, y_pred_class_validation))
    # print("\n验证集分类报告:\n", classification_report(y_validation, y_pred_class_validation))

    # 特征重要性保持不变
    feature_imp = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': lgb_baseline.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    top_importantcol = list(feature_imp.head(50)['Feature'])

    return lgb_baseline, feature_imp, top_importantcol

def regression_modeling(df,validation_month_list, split_model,model_type='regression', random_state=42):
 
    df_keep2 = data_preprocessing(df)

    print('保留的字段数量',df_keep2.shape)
    if 'ESTIMATEDSLAUGHTERDATE _month' in df_keep2.columns: 
        df_keep2=df_keep2.drop(columns=['ESTIMATEDSLAUGHTERDATE _month','DOCDATE_month'],axis=1)
    if split_model=='month':
        X_train, X_test, y_train, y_test, X_validation, y_validation = split_data_m(df_keep2,model_type,validation_month_list)
    if split_model=='sample':
        X_train, X_test, y_train, y_test, X_validation, y_validation = split_data_s(df_keep2, model_type)
    if split_model=='stratified':
        X_train, X_test, y_train, y_test, X_validation, y_validation = stratified_split_data(df_keep2, model_type)
    print("\n数据分割结果：")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"验证集大小: {X_validation.shape}")
    # X_train=X_train.drop(columns=['HARVESTSTATUS_month'],axis=1)
    # X_test=X_test.drop(columns=['HARVESTSTATUS_month'],axis=1)
    # X_validation=X_validation.drop(columns=['HARVESTSTATUS_month'],axis=1)
    # 定义LightGBM参数（针对回归问题优化）
    params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 修改为回归任务
    'metric': 'rmse',          # 回归常用指标
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'seed': 42,
    'verbose': -1
}

    # 转换为LightGBM数据集格式
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 训练模型
    lgb_regressor = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=1000
    )

    # 预测连续值
    y_pred = lgb_regressor.predict(X_test, num_iteration=lgb_regressor.best_iteration)

    # 评估指标（回归问题）
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)

    # 绘制实际值 vs 预测值散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    y=df[~df['HARVESTSTATUS_month'].isin(validation_month_list)]['MORTALITY_RATE']
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想对角线
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

    # 输出具体重要性值
    feature_imp = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': lgb_regressor.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    top_important_cols = list(feature_imp.head(50)['Feature'])

    # 对验证集进行预测
    y_val_pred = lgb_regressor.predict(X_validation, num_iteration=lgb_regressor.best_iteration)

    # 评估验证集效果
    val_rmse = root_mean_squared_error(y_validation, y_val_pred)
    val_mae = mean_absolute_error(y_validation, y_val_pred)
    val_r2 = r2_score(y_validation, y_val_pred)

    print("=== 跨期验证结果 ===")
    print(f"Validation RMSE: {val_rmse:.4f} (Test RMSE: {rmse:.4f})")
    print(f"Validation MAE: {val_mae:.4f} (Test MAE: {mae:.4f})")
    print(f"Validation R2: {val_r2:.4f} (Test R2: {r2:.4f})")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_validation, y_val_pred, alpha=0.5, color='blue')
    plt.plot([y_validation.min(), y_validation.max()],
             [y_validation.min(), y_validation.max()], 'r--')
    plt.xlabel('Actual Values (Validation)')
    plt.ylabel('Predicted Values (Validation)')
    plt.title('Validation: Actual vs Predicted')

    # 测试集对比图
    plt.subplots_adjust(wspace=0.3)  # 调整子图间距
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values (Test)')
    plt.ylabel('Predicted Values (Test)')
    plt.title('Test: Actual vs Predicted')
    plt.tight_layout()
    plt.show()

    return lgb_regressor, feature_imp, top_important_cols

# 按季度总样本分割数据
all_columns=df2.columns.to_list()
# 筛选以'W'开头的列（窗口特征列）
# window_features = [col for col in all_columns if col.startswith('W')]
# # 剩余列（非窗口特征列）
# remaining_columns = [col for col in all_columns if col not in window_features]

# window_cols=remaining_columns+[col for col in window_features if col.startswith('W3')]


seasonal_dfs = prepare_seasonal_data(df2,date_col='HARVESTSTATUS_month', target_col='MORTALITY_RATE', quantile_threshold=0.8)
seasonal_dfs['winter'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
seasonal_dfs['autumn'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
seasonal_dfs['summer'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
seasonal_dfs['spring'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
df2['HARVESTSTATUS_month'].value_counts()
# df_keep2 = data_preprocessing(seasonal_dfs['winter'])
# 冬天
lgb_baseline_w, feature_imp_w, top_importantcol_w=lightgbm_modeling(seasonal_dfs['winter'],validation_month_list=['2'],split_model='stratified')
top_importantcol_w[:20]
len(feature_imp_w[feature_imp_w['Importance']>0]['Feature'].tolist())
important_cols=list(dict.fromkeys(top_importantcol_w + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['winter'][important_cols],validation_month_list=['2'],split_model='stratified')
top_importantcol2[:20]
# 回归模型建立
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['winter'],validation_month_list=['2'],split_model='stratified')
top_important_cols3[:20]


# 秋天
lgb_baseline_a, feature_imp_a, top_importantcol_a=lightgbm_modeling(seasonal_dfs['autumn'],validation_month_list=['11'],split_model='stratified')
top_importantcol_a[:20]
important_cols=list(dict.fromkeys(top_importantcol_a + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['autumn'][important_cols],validation_month_list=['11'],split_model='stratified')
top_importantcol2[:20]
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['autumn'],validation_month_list=['11'],split_model='stratified')
top_important_cols3[:20]

# 夏天
lgb_baseline_s, feature_imp_s, top_importantcol_s=lightgbm_modeling(seasonal_dfs['summer'],validation_month_list=['9'],split_model='stratified')
top_importantcol_s[:20]
important_cols=list(dict.fromkeys(top_importantcol_s + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['summer'][important_cols],validation_month_list=['9'],split_model='stratified')
top_importantcol2[:20]
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['summer'],validation_month_list=['9'],split_model='stratified')
top_important_cols3[:20]
# 春天
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(seasonal_dfs['spring'],validation_month_list=['4'],split_model='stratified')
important_cols=list(dict.fromkeys(top_importantcol + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['spring'][important_cols],validation_month_list=['11'])
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['spring'],validation_month_list=['11'])


# 分箱
from sklearn.exceptions import DataConversionWarning
def feature_binning(top_importantcol, object_columns, X, y, file_prefix):
    """
    对特征进行分箱分析，计算统计信息并可视化
    
    Parameters:
    -----------
    top_importantcol: list
        需要分箱的特征列表（不包含Mortality_rate列）
    object_columns: list
        分类特征列名列表
    X: DataFrame
        特征数据（包含Mortality_rate列）
    y: Series
        目标变量（如死亡标志0/1）
    file_prefix: str
        输出文件前缀
    
    Returns:
    --------
    binning_sum: DataFrame
        分箱汇总统计信息
    bin_table: DataFrame
        所有特征的分箱明细表（含Mortality_rate均值）
    """
    # 0. 确保Mortality_rate列存在
    if 'MORTALITY_RATE' not in X.columns:
        raise ValueError("X中必须包含'Mortality_rate'列")
    
    # 1. 识别分类特征
    cat_f = [col for col in top_importantcol if col in object_columns]
    
    # 2. 设置分箱选择标准
    selection_criteria = {"gini": {"min": 0.15, "max": 1}}
    
    # 3. 初始化分箱过程
    binning_process = BinningProcess(top_importantcol,
                                   categorical_variables=cat_f,
                                   selection_criteria=selection_criteria)
    
    # 4. 拟合分箱模型
    binning_process.fit(X[top_importantcol], y)
    
    # 5. 获取分箱汇总信息
    binning_sum = binning_process.summary()
    binning_sum = binning_sum.sort_values(by='gini', ascending=False)
    
    # 6. 构建完整分箱明细表
    bin_table = pd.DataFrame()
    for col in top_importantcol:
        try:
            optb = binning_process.get_binned_variable(col)
            temp = optb.binning_table.build()
            temp['feature'] = col  # 添加特征名列
            print(f"分箱阈值: {optb.splits}")
            X_binned = optb.transform(X[[col]],metric='bins').squeeze()  # 获取分箱结果
            bin_mortality = X.groupby(X_binned)['MORTALITY_RATE'].mean()
            bin_mortality=bin_mortality.reset_index(drop=False)
            temp2 = pd.merge(temp, bin_mortality, left_on='Bin', right_on='index', how='left').drop('index', axis=1)
            bin_table = pd.concat([bin_table, temp2])
            print(f"\n=== 死淘具体情况: {col} ===")
            print(bin_mortality)
             # 打印分箱详情
            print(f"\n=== 变量: {col} ===")
            display_cols = ['Bin', 'Count', 'Count (%)', 'Event rate', 'MORTALITY_RATE']
            print(temp2[display_cols])
            print('-'*50)
            
            
        except Exception as e:
            print(f"处理变量 {col} 时出错: {str(e)}")
            continue
    
    # 7. 保存结果
    # output_path = '.\\xyy\\死淘分析\\output'
    # os.makedirs(output_path, exist_ok=True)
    # bin_table_file = os.path.join(output_path, f"{file_prefix}bin_table.csv")
    # bin_table.to_csv(bin_table_file, index=False, encoding='gbk')
    
    # 8. 打印和可视化每个变量（修正后的可视化部分）
    warnings.filterwarnings("ignore")
    for col in top_importantcol:
        try:
            optb = binning_process.get_binned_variable(col)
            bin_table_var = optb.binning_table.build()
            optb.binning_table.plot(metric='event_rate')
            plt.show()
            print('')
            
        except Exception as e:
            print(f"可视化变量 {col} 时出错: {str(e)}")
            continue
    
    return binning_sum, bin_table

numeric_columns = []
object_columns = []
for column in df2.columns:
    if np.issubdtype(df2[column].dtype, np.number):
        numeric_columns.append(column)
    else:
        object_columns.append(column)


top_importantcol_w1=['HARVESTSTATUS_month', 'W3_15-17天_外部-平均_MEAN', 'DENSITY', 'W3_18-20天_平均温度变化率_MEAN', 
                     'AGE', 'W3_0-2天_外部-平均_MEAN', 'W3_9-11天_平均温度变化率_MEAN', 'W3_9-11天_最低温度变化率_RANGE',
                       'W3_18-20天_每日温差_RANGE', 'W3_15-17天_最低温度变化率_RANGE', 'DOCAMOUNT', 
                       'W3_6-8天_最低温度变化率_MEAN', 'W3_18-20天_外部-平均_RANGE', 'W3_9-11天_鸡舍温度-最高_RANGE',
                         'W3_18-20天_湿度内部平均_RANGE', 'W3_15-17天_平均温度变化率_MEAN', 
                         'W3_6-8天_平均温度变化率_RANGE', 'W3_15-17天_每日温差_RANGE', 'W3_21-23天_外部-平均_MEAN', 
                         'W3_3-5天_平均温度变化率_MEAN']
top_importantcol_s1=['W3_0-2天_鸡舍温度-平均_MEAN', 'W3_21-23天_鸡舍温度-最高_MEAN', 'W3_21-23天_最低温度变化率_RANGE',
                      'GAS_COST', 'W3_6-8天_鸡舍温度-平均_MEAN', 'W3_21-23天_外部-平均_RANGE', 
                      'W3_18-20天_外部-平均_RANGE', 'W3_6-8天_平均温度变化率_RANGE', 'W3_15-17天_鸡舍温度-平均_MEAN', 
                      'W3_12-14天_每日温差_MEAN', 'W3_12-14天_每日温差_RANGE', 'DENSITY', 'HOUSEAMOUNT',
                        'W3_3-5天_鸡舍温度-最低_RANGE', 'W3_12-14天_湿度内部平均_MEAN', 'W3_21-23天_湿度内部平均_RANGE',
                          'W3_21-23天_鸡舍温度-平均_MEAN', 'W3_15-17天_最高温度变化率_RANGE',
                            'W3_12-14天_外部-平均_RANGE', 'W3_15-17天_鸡舍温度-最低_MEAN']
top_importantcol_a1=['W3_3-5天_每日温差_MEAN', 'W3_9-11天_鸡舍温度-平均_MEAN', 'W3_12-14天_平均温度变化率_RANGE',
                      'W3_6-8天_湿度内部平均_RANGE', 'W3_21-23天_最低温度变化率_MEAN', 'W3_6-8天_鸡舍温度-最低_MEAN', 
                      'W3_12-14天_湿度内部平均_RANGE', 'W3_0-2天_每日温差_RANGE', 'W3_12-14天_每日温差_RANGE', 
                      'W3_15-17天_每日温差_RANGE', 'ELECTRICITY_COST', 'W3_3-5天_鸡舍温度-最低_RANGE', 
                      'W3_6-8天_外部-平均_MEAN', 'W3_18-20天_平均温度变化率_RANGE', 'W3_18-20天_最高温度变化率_RANGE', 
                      'W3_0-2天_最低温度变化率_MEAN', 'W3_6-8天_平均温度变化率_MEAN', 'W3_9-11天_最高温度变化率_MEAN', 
                      'W3_21-23天_平均温度变化率_RANGE', 'W3_21-23天_最高温度变化率_RANGE']
same_col=list(set(top_importantcol_w) & set(top_importantcol_s) & set(top_importantcol_a))


numeric_columns = []
object_columns = []
for column in df2.columns:
    if np.issubdtype(df2[column].dtype, np.number):
        numeric_columns.append(column)
    else:
        object_columns.append(column)

X=seasonal_dfs['winter'].drop(columns=[ 'MORTALITY_RATE_21', 'Mortality_flg'])
y=seasonal_dfs['winter']['Mortality_flg']
rate_cols=[ i for i in X.columns if '变化率' in i]
X[rate_cols]=X[rate_cols]*100
top_importantcol=['DENSITY']
binning_sum, bin_table=feature_binning(top_importantcol, object_columns, X, y,file_prefix="winter")


