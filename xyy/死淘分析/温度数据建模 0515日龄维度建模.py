
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
all_info_temdata=pd.read_csv('./data/data_cleaned/dead_HumTem_byage0603_2.csv',encoding='gbk')
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')

all_info_temdata.columns.to_list()

#把细分的死淘数据和出栏数据指标去掉
df2=all_info_temdata
# df2['Date']
df2.shape
###字段处理
# 将日期列转换为 datetime 类型
# 将Date列转换为datetime类型
df2['Date'] = pd.to_datetime(df2['Date'])

# 提取月份
df2['Month'] = df2['Date'].dt.month
df2=df2.drop('Date',axis=1)

df2=df2.set_index(['ID_NUM'])
df2.shape
# df2['Mortality_rate'].isna().sum()


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

    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    df_keep[numeric_columns] = df_keep[numeric_columns].clip(lower=min_float32, upper=max_float32)

    # 共线性处理
    target_cols = [col for col in ['Mortality_rate'] 
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
    target_cols = ['Mortality_rate']
    
    # 合并选择的数值列、对象列和目标变量
    
    df_keep2 = df_keep[selected_cols + object_columns + target_cols].copy()
    
    # 将object类型转换为category类型
    for col in object_columns:
        df_keep2[col] = df_keep2[col].astype('category')
    
    return df_keep2


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
        stratify_col = pd.qcut(df['Mortality_rate'], q=num_bins, duplicates='drop')
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
        new_stratify = pd.qcut(train_test['Mortality_rate'], 
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
        sns.kdeplot(train['Mortality_rate'], label='Train')
        sns.kdeplot(test['Mortality_rate'], label='Test')
        sns.kdeplot(val['Mortality_rate'], label='Validation')
        plt.title("Target Variable Distribution Comparison")
        plt.legend()
    
    plt.show()
    
    # 准备特征和目标变量
    if model_type == 'binary':
        drop_cols = ['Mortality_rate', 'Mortality_flg']
        y_col = 'Mortality_flg'
    else:
        drop_cols = ['Mortality_flg', 'Mortality_rate']
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
    else:
        print(f"\n目标变量统计:")
        print(f" - 训练集均值: {y_train.mean():.4f} ± {y_train.std():.4f}")
        print(f" - 测试集均值: {y_test.mean():.4f} ± {y_test.std():.4f}")
        print(f" - 验证集均值: {y_val.mean():.4f} ± {y_val.std():.4f}")
    
    return X_train, X_test, y_train, y_test, X_val, y_val

  
# X_train.columns.to_list()
def lightgbm_modeling(df, model_type='binary', random_state=42, quantile_threshold=0.8):
    # 数据预处理和分割保持不变
    df_keep2 = data_preprocessing(df)
    print('保留的字段数量', df_keep2.shape)
    df_keep2['Mortality_flg']=df_keep2['Mortality_rate'].apply(lambda x:1 if x>=np.quantile(df_keep2['Mortality_rate'],0.8) else 0)
  
    X_train, X_test, y_train, y_test, X_validation, y_validation = stratified_split_data(df_keep2, model_type)
    
    print("\n数据分割结果：")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"验证集大小: {X_validation.shape}")
    print(f"训练集目标分布: {pd.Series(y_train).value_counts() if model_type == 'binary' else pd.Series(y_train).describe()}")
    print(f"测试集目标分布: {pd.Series(y_test).value_counts() if model_type == 'binary' else pd.Series(y_test).describe()}")
    print(f"验证集目标分布: {pd.Series(y_validation).value_counts() if model_type == 'binary' else pd.Series(y_validation).describe()}")

    # LightGBM参数保持不变
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

df2.columns.to_list()
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(df2.drop(['Month'],axis=1), model_type='binary', random_state=42, quantile_threshold=0.8)


df_keep2=data_preprocessing(df2, thres_zeros=0.7, cutoff=0.8)
df_keep2.columns.to_list()
np.quantile(df_keep2['Mortality_rate'],0.8)
df_keep2['Mortality_rate'].describe()
df_keep2['Mortality_flg']=df_keep2['Mortality_rate'].apply(lambda x:1 if x>=np.quantile(df_keep2['Mortality_rate'],0.8) else 0)

df_keep2.groupby('Month')['Mortality_flg'].sum()
df_keep2.groupby('Age')['Mortality_flg'].sum()

df_keep2['Mortality_flg'].value_counts()
X_train, X_test, y_train, y_test, X_validation, y_validation = stratified_split_data(df_keep2, model_type='binary')


import matplotlib.pyplot as plt

# 计算每个 Age 的 Mortality_flg 总和
age_mortality = df_keep2.groupby('Age')['Mortality_flg'].sum()

# 创建折线图
plt.figure(figsize=(12, 6))
plt.plot(age_mortality.index, age_mortality.values, 
         marker='o', linestyle='-', color='b', linewidth=2)

# 添加标题和标签
plt.title('Daily Mortality by Age', fontsize=16)
plt.xlabel('Age (days)', fontsize=14)
plt.ylabel('高死淘样本数', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 优化显示
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()