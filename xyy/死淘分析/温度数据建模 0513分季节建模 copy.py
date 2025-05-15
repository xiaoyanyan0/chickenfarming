
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
all_info_temdata=pd.read_csv('./data/data_cleaned/all_info_temdata0512.csv',encoding='gbk')
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')

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


def prepare_seasonal_data(df, date_col='HARVESTSTATUS_month', target_col='MORTALITY_RATE', quantile_threshold=0.8):
    """
    将数据按季节分割并创建二分类目标变量
    """
    # 按季节分割数据
    seasonal_dfs = {
        'winter': df[df[date_col].isin(['12', '1', '2'])].copy(),
       'spring': df[df[date_col].isin(['3', '4'])].copy(),
       'summer': df[df[date_col].isin(['5', '6', '7', '8', '9'])].copy(),
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
        season_df['Mortality_flg'] = season_df[target_col].apply(
            lambda x: 1 if pd.notna(x) and x >= quantile else 0
        )

        # 打印各季节数据量
        print(f"{season.capitalize()}数据量: {season_df.shape[0]}, 正样本比例: {season_df['Mortality_flg'].mean():.2%}")

    return seasonal_dfs


def split_data(df, model_type, test_size=0.3, random_state=42):
    """
    将数据按比例随机拆分为训练集和测试集
    """
    if model_type == 'binary':
        X = df.drop(columns=['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg','ESTIMATEDSLAUGHTERDATE _month','DOCDATE_month','HARVESTSTATUS_month'])
        y = df['Mortality_flg']
    elif model_type =='regression':
        X = df.drop(columns=['Mortality_flg', 'MORTALITY_RATE_21', 'MORTALITY_RATE','ESTIMATEDSLAUGHTERDATE _month','DOCDATE_month','HARVESTSTATUS_month'])
        y = df['MORTALITY_RATE']
    else:
        raise ValueError("model_type 必须为 'binary' 或'regression'")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y if model_type == 'binary' else None)

    # 对比目标变量分布
    plt.figure(figsize=(10, 4))
    sns.kdeplot(y_train, label='Train')
    sns.kdeplot(y_test, label='Test')
    plt.title("目标变量分布比较")
    plt.legend()
    plt.show()

    # 打印各集合大小
    print("\n数据分割结果：")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    if model_type == 'binary':
        print(f"训练集正样本比例: {y_train.mean():.2%}")
        print(f"测试集正样本比例: {y_test.mean():.2%}")
    else:
        print(f"训练集目标变量均值: {y_train.mean():.4f}")
        print(f"测试集目标变量均值: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


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
    target_cols = [col for col in ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg']
                   if col in df_keep.columns]
    other_numeric = [col for col in numeric_columns if col not in target_cols]

    if other_numeric:
        selected_cols = get_non_collinear_vars(cutoff, df_keep[other_numeric])
        print(f"共线性处理后保留 {len(selected_cols)}/{len(other_numeric)} 个数值特征")
    else:
        selected_cols = []
        print("警告: 无有效数值特征进行共线性分析")

    # 合并选择的数值列、对象列和目标变量
    df_keep2 = df_keep[selected_cols + object_columns + target_cols].copy()

    # 将object类型转换为category类型
    for col in object_columns:
        df_keep2[col] = df_keep2[col].astype('category')

    return df_keep2


def lightgbm_modeling(df, model_type='binary', random_state=42, quantile_threshold=0.8):
    df_keep2 = data_preprocessing(df)
    print('保留的字段数量', df_keep2.shape)

    X_train, X_test, y_train, y_test = split_data(df_keep2, model_type)

    # 定义LightGBM参数（针对二分类优化）
    if model_type == 'binary':
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
           'metric': 'auc',
           'max_depth': 3,  # 保持浅树
            'num_leaves': 10,  # 2^3=8，留一定余量设为10
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
           'min_data_in_leaf': 10,  # 每个叶子至少10个样本
           'reg_alpha': 0.1,  # L1正则化
           'reg_lambda': 0.1,  # L2正则化
           'seed': 42,
           'verbose': -1
        }

        # 转换为LightGBM数据集格式
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # 训练模型
        lgb_baseline = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=1000
        )

        # 预测概率
        y_pred_prob = lgb_baseline.predict(X_test, num_iteration=lgb_baseline.best_iteration)
        quantile = np.quantile(y_pred_prob, quantile_threshold)
        y_pred_class = (y_pred_prob >= quantile).astype(int)

        # 评估指标
        print("AUC Score:", roc_auc_score(y_test, y_pred_prob))
        print("Accuracy:", accuracy_score(y_test, y_pred_class))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))

        # 绘制ROC曲线
        RocCurveDisplay.from_predictions(y_test, y_pred_prob)
        plt.title("ROC Curve")
        plt.show()

        # 输出具体重要性值
        feature_imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': lgb_baseline.feature_importance(importance_type='gain')
        }).sort_values('Importance', ascending=False)

        top_importantcol = list(feature_imp.head(50)['Feature'])
        print(feature_imp)
        print(top_importantcol)

        return lgb_baseline, feature_imp, top_importantcol

    elif model_type =='regression':
        params = {
            'boosting_type': 'gbdt',
            'objective':'regression',  # 修改为回归任务
           'metric': 'rmse',  # 回归常用指标
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
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 理想对角线
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

        return lgb_regressor, feature_imp, top_important_cols
    


# 按季度总样本分割数据

seasonal_dfs = prepare_seasonal_data(df2,date_col='HARVESTSTATUS_month', target_col='MORTALITY_RATE', quantile_threshold=0.8)
seasonal_dfs['winter'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
seasonal_dfs['autumn'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
seasonal_dfs['summer'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
seasonal_dfs['spring'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
df2['HARVESTSTATUS_month'].value_counts()

# 冬天
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(seasonal_dfs['winter'])
important_cols=list(dict.fromkeys(top_importantcol + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['winter'][important_cols],validation_month_list=['2'])
# 回归模型建立
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['winter'],validation_month_list=['2'])

# 秋天
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(seasonal_dfs['autumn'],validation_month_list=['11'],split_model='sample')
important_cols=list(dict.fromkeys(top_importantcol + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['autumn'][important_cols],validation_month_list=['11'])
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['autumn'],validation_month_list=['11'])


# 按月度样本分割数据

seasonal_dfs = prepare_seasonal_data2(df2,date_col='HARVESTSTATUS_month', target_col='MORTALITY_RATE', quantile_threshold=0.80)
seasonal_dfs['winter']['HARVESTSTATUS_month'].value_counts()
# 冬天
seasonal_dfs['winter'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(seasonal_dfs['winter'],validation_month_list=['2'])
important_cols=list(dict.fromkeys(top_importantcol + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['winter'][important_cols],validation_month_list=['2'])
# 回归模型建立
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['winter'],validation_month_list=['2'])

# 秋天
seasonal_dfs['autumn'].groupby('HARVESTSTATUS_month')['Mortality_flg'].sum()
lgb_baseline, feature_imp, top_importantcol=lightgbm_modeling(seasonal_dfs['autumn'],validation_month_list=['11'])
important_cols=list(dict.fromkeys(top_importantcol + ['MORTALITY_RATE', 'MORTALITY_RATE_21', 'Mortality_flg', 'HARVESTSTATUS_month']))
#重要变量建模
lgb_model, feature_imp2, top_importantcol2=lightgbm_modeling(seasonal_dfs['autumn'][important_cols],validation_month_list=['11'])
lgb_regressor, feature_imp3, top_important_cols3=regression_modeling(seasonal_dfs['autumn'],validation_month_list=['11'])












def feature_binning(top_importantcol, object_columns, X, y):
    cat_f = [col for col in top_importantcol if col in object_columns]
    selection_criteria = {"gini": {"min": 0.15, "max": 1}}

    binning_process = BinningProcess(top_importantcol,
                                     categorical_variables=cat_f,
                                     selection_criteria=selection_criteria)

    binning_process.fit(X[top_importantcol], y)
    binning_sum = binning_process.summary()
    binning_sum = binning_sum.sort_values(by='gini', ascending=False)

    bin_table = pd.DataFrame()
    for i in top_importantcol:
        optb = binning_process.get_binned_variable(i)
        temp = optb.binning_table.build()
        temp['name'] = i
        bin_table = pd.concat([bin_table, temp])

    bin_table.to_csv('.\\xyy\\死淘分析\\bin_table0513.csv', index=False, encoding='gbk')

    warnings.filterwarnings("ignore")
    for var in top_importantcol:
        optb = binning_process.get_binned_variable(var)
        bin_table_var = optb.binning_table.build()
        optb.binning_table.plot(metric='event_rate')
        plt.show()
        print(bin_table_var.iloc[:, :-3])
        print('')

    return binning_sum, bin_table
