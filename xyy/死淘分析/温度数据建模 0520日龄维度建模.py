
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
all_info_temdata=pd.read_csv('./data/data_cleaned/dead_HumTem_byage0521.csv',encoding='gbk')
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')
all_info_temdata.shape
all_info_temdata.columns.to_list()

#把细分的死淘数据和出栏数据指标去掉
df2=all_info_temdata.copy()
# df2['Date']
all_info_temdata.shape
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
def prepare_seasonal_data(df, date_col='Month', target_col='Mortality_rate', quantile_threshold=0.8):
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
df2['Month']=df2['Month'].astype(str)
# season_df=prepare_seasonal_data(df2, date_col='Month', target_col='Mortality_rate', quantile_threshold=0.8)
# df2['Month'].value_counts()
##去掉通风系数等字段
Ventilation_Coefficien_cols=[df2.columns[i] for i in range(len(df2.columns)) if 'Ventilation_Coefficient' in df2.columns[i]]
temp_diff_col=[df2.columns[i] for i in range(len(df2.columns)) if '内外温差' in df2.columns[i]]

df3=df2.drop(columns=Ventilation_Coefficien_cols+temp_diff_col,axis=1)
df2.columns.to_list()
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(df3.drop(['Month','Age'],axis=1), model_type='binary', random_state=42, quantile_threshold=0.8)
top_importantcol[:20]
len(feature_imp[feature_imp['Importance']>0]['Feature'].tolist())
important_cols=list(dict.fromkeys(top_importantcol + ['Mortality_rate']))
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(df2[important_cols], model_type='binary', random_state=42, quantile_threshold=0.8)



# lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(season_df['winter'].drop(['Month'],axis=1), model_type='binary', random_state=42, quantile_threshold=0.8)

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


# 重要变量分箱


# 分箱
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
    if 'Mortality_rate' not in X.columns:
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
            bin_mortality = X.groupby(X_binned)['Mortality_rate'].mean()
            bin_mortality=bin_mortality.reset_index(drop=False)
            temp2 = pd.merge(temp, bin_mortality, left_on='Bin', right_on='index', how='left').drop('index', axis=1)
            bin_table = pd.concat([bin_table, temp2])
             # 打印分箱详情
            print(f"\n=== 变量: {col} ===")
            display_cols = ['Bin', 'Count', 'Count (%)', 'Event rate', 'Mortality_rate']
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

df_binning=df2.copy()
df_binning['Mortality_flg']=df_binning['Mortality_rate'].apply(lambda x:1 if x>=np.quantile(df_binning['Mortality_rate'],0.8) else 0)

X=df_binning
y=df_binning['Mortality_flg']
rate_cols=[ i for i in X.columns if '变化率' in i]
X[rate_cols]=X[rate_cols]*100
top_importantcol=['Water_前7天变化百分比', 'Lowest_Temn_前7天', 'Lowest_Temn_前3天', 'Feed_前7天变化百分比', 
                  'Ventilation_Coefficient_Cold_前1天', 'Lowest_Temn_前5天', 'Ventilation_Coefficient_Warm_前3天',
                    'Ventilation_Coefficient_Warm_前5天', 'Water', 'Ventilation_Coefficient_Warm', 
                    'Ventilation_Coefficient_Cold_前3天', 'Water_前7天变化', 'Lowest_Temp_Outside_前5天变化百分比', 
                    'Lowest_Temp_Outside_前3天变化百分比', 'Lowest_Temn_前7天变化', '最高温度变化率_前1天', '每日温差',
                      'Ventilation_Coefficient_Warm_前5天变化百分比', 'Highest_Temn_前5天', 'Water_前3天变化']
top_importantcol2=['Water_前7天变化百分比', 'Lowest_Temn_前7天', 'Lowest_Temn_前3天', 'Feed_前7天变化百分比', 
                   'Water', 'Lowest_Temn_前5天', 'Highest_Temp_Outside_前7天', 'Highest_Temn_前7天', 
                   'Lowest_Temp_Outside_前3天变化百分比', 'Lowest_Temn_前7天变化', 
                   'Lowest_Temp_Outside_前5天变化百分比', '最高温度变化率_前1天', 'Highest_humidity_前3天',
                     'Water_前7天变化', 'Highest_Temn_前5天', '内外温差_max', '每日温差', 
                     'Highest_Temp_Outside_前3天变化百分比', '温度1-平均_max', '最低温度变化率_前5天']
top_importantcol3=['Lowest_Temn_前7天', 'Water_前7天变化百分比', 'Feed_前7天变化百分比', 'Lowest_Temn_前3天',
                   'Water', 'Lowest_Temn_前5天', 'Highest_Temp_Outside_前3天', 'Highest_Temn_前7天', 
                   'Highest_humidity_前3天', 'Lowest_Temp_Outside_前3天变化百分比', 'Lowest_Temn_前7天变化', 
                   'Highest_Temn_前5天', 'Lowest_Temp_Outside_前5天变化百分比', '每日温差', '平均温度变化率_前7天',
                     '最高温度变化率_前7天', 'Highest_Temp_Outside_前3天变化百分比', 'Highest_Temn_前1天', 
                     'Water_前3天变化', 'Water_前7天变化']




top_importantcol4=[i for i in top_importantcol3 if i not in top_importantcol + top_importantcol2]
# 设置pandas显示更多小数位（全局设置）
pd.set_option('display.float_format', lambda x: '%.6f' % x)  # 显示6位小数

top_importantcol3=['Water_前7天变化百分比']
binning_sum, bin_table=feature_binning(top_importantcol3, object_columns, X, y,file_prefix="Age2_")

binning_sum.to_csv('./xyy/死淘分析/output/Age2_binning_sum.csv', index=False, encoding='gbk')

X['最高温度变化率_前7天'].describe()
top_importantcol=['Water']

# 1. 识别分类特征
cat_f = [col for col in top_importantcol if col in object_columns]

# 2. 设置分箱选择标准
selection_criteria = {"gini": {"min": 0.15, "max": 1}}

# 3. 初始化分箱过程（只对top_importantcol中的特征分箱）
binning_process = BinningProcess(top_importantcol,
                                categorical_variables=cat_f,
                                selection_criteria=selection_criteria)

# 4. 拟合分箱模型（使用y作为目标变量）
binning_process.fit(X[top_importantcol], y)

# 5. 获取分箱汇总信息
binning_sum = binning_process.summary()
binning_sum = binning_sum.sort_values(by='gini', ascending=False)

# 6. 构建完整分箱明细表
bin_table = pd.DataFrame()
for col in top_importantcol:
    optb = binning_process.get_binned_variable(col)
    temp = optb.binning_table.build()
    temp['feature'] = col  # 添加特征名列
    temp['WoE'] = temp['WoE'].astype(float).round(6)  # 将WoE列转换为字符串格式
    # 计算当前特征分箱后的mortality_rate均值
    X_binned = optb.transform(X[[col]],metric='bins').squeeze()  # 获取分箱结果
    bin_mortality = X.groupby(X_binned)['Mortality_rate'].mean()
    bin_mortality=bin_mortality.reset_index(drop=False)
    temp2 = pd.merge(temp, bin_mortality, left_on='Bin', right_on='index', how='left').drop('index', axis=1)
    temp.index
    bin_table = pd.concat([bin_table, temp2])

# 7. 打印和可视化每个变量
warnings.filterwarnings("ignore")
for col in top_importantcol:
    optb = binning_process.get_binned_variable(col)
    bin_table_var = optb.binning_table.build()
    
    # 计算当前特征的mortality_rate分箱均值
    X_binned = optb.transform(X[[col]]).squeeze()
    bin_mortality = X.groupby(X_binned)['Mortality_rate'].mean()
    bin_mortality=bin_mortality.reset_index(drop=False)
    all_bins = range(len(bin_table_var))
    # bin_mortality = bin_mortality.reindex(all_bins, fill_value=0)
    bin_table_var['bin_mortality_rate'] = bin_mortality['Mortality_rate']