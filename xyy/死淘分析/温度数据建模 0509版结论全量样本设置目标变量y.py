
#########在全量样本范围内设置目标变量
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
all_info_temdata=pd.read_csv('./data/data_cleaned/all_info_temdata0509.csv',encoding='gbk')
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')

# all_info_temdata
col=[a.upper()for a in all_info_temdata.columns.to_list()]
all_info_temdata.columns=col

df=all_info_temdata.set_index('ID_NUM')
# df=df.fillna(0)
# df[['MORTALITY_RATE_X','MORTALITY_RATE_21']]
df.columns.to_list()
#筛除0占比超过阈值特征
drop_columns_Mortality=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','livability_pct'
                        ,'yield_per_m2'
                        ]
drop_columns_Mortality=[a.upper()for a in drop_columns_Mortality]
drop_columns_Mortality=[i for i in df.columns.to_list() if i in drop_columns_Mortality]

df=df.rename({'MORTALITY_RATE_X':'MORTALITY_RATE'},axis=1)

# quantile_10 = np.quantile(df['MORTALITY_RATE'], 0.1)
# df['Mortality_flg']=df['MORTALITY_RATE'].apply(lambda x:1 if x<=quantile_10 else 0)


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
####缺失过高的去掉

thres_zeros=0.7
# f=df[X].replace(0,np.nan)
missing_p=np.sum(df2.isnull(),axis=0)/df2.shape[0]
low_missing=missing_p[missing_p<thres_zeros].index.tolist()
df_keep=df2[low_missing].copy()

df_keep.shape

df_keep['HOUSEID']=df_keep['HOUSEID'].astype(int).astype(str)
df_keep['HEAGE']=df_keep['HEAGE'].astype(int).astype(str)
df_keep['AGE']=df_keep['AGE'].astype(int).astype(str)
# 数值类别型变量
numeric_columns = []
object_columns = []
# special_columns=['HOUSEID']

for column in df_keep.columns:
    if np.issubdtype(df_keep[column].dtype, np.number):
        numeric_columns.append(column)
    else:
        object_columns.append(column)
# numeric_columns=[i for i in numeric_columns if i not in special_columns]
# object_columns=object_columns+special_columns

len(numeric_columns)+len(object_columns)

#数据处理

if np.any(~np.isfinite(df_keep[numeric_columns])):
    df_keep[numeric_columns] = df_keep[numeric_columns].replace([np.inf, -np.inf], np.nan)
    # 可以选择删除包含 NaN 的行或者填充 NaN
 
max_float32 = np.finfo(np.float32).max
min_float32 = np.finfo(np.float32).min

df_keep[numeric_columns] = df_keep[numeric_columns].clip(lower=min_float32, upper=max_float32)


#### 共线性处理
def get_non_collinear_vars(cutoff, df):
    """
    去除数据框中高度共线性（相关性 > cutoff）的变量，保留代表性变量。
    
    参数:
        cutoff (float): 相关性阈值（0 < cutoff < 1）
        df (pd.DataFrame): 数值型数据框
        
    返回:
        list: 筛选后的列名列表
    """
    # 计算相关性矩阵
    corr_matrix = df.corr().abs()  # 取绝对值
    
    # 初始化保留的列
    selected_cols = []
    dropped_cols = set()
    
    # 遍历相关性矩阵的列
    for i, col in enumerate(corr_matrix.columns):
        if col in dropped_cols:
            continue  # 如果已标记为删除，跳过
            
        selected_cols.append(col)
        
        # 找到与当前列高度相关的其他列
        high_corr_cols = corr_matrix.index[
            (corr_matrix[col] > cutoff) & 
            (corr_matrix.index != col)
        ].tolist()
        
        # 标记这些列为待删除
        dropped_cols.update(high_corr_cols)
    
    return selected_cols

selected_cols=get_non_collinear_vars(0.8,df_keep[numeric_columns].drop(['MORTALITY_RATE','MORTALITY_RATE_21'],axis=1))

df_keep2=df_keep[selected_cols+object_columns+['MORTALITY_RATE','MORTALITY_RATE_21']].copy()
df_keep2.shape
for col in df_keep2.columns:
    if df_keep2[col].dtypes=='object':
        df_keep2[col]=df_keep2[col].astype('category')


# df_keep2['HOUSENO']

# X=df_keep2.drop(columns=['Mortality_flg','MORTALITY_RATE','MORTALITY_RATE_21'])

# y=df_keep2['Mortality_flg']
# 筛选出 2、3 月份的数据作为跨期验证数据
# def assign_mortality_flag(group):
#     quantile_10 = np.quantile(group['MORTALITY_RATE'], 0.85)
#     group['Mortality_flg'] = group['MORTALITY_RATE'].apply(lambda x: 1 if x >= quantile_10 else 0)
#     return group

# df_keep2 = df_keep2.groupby('DOCDATE_month').apply(assign_mortality_flag).drop('DOCDATE_month',axis=1)

# df_keep2.groupby('DOCDATE_month')['MORTALITY_RATE_21'].mean()

# print(result)

# df_keep2=df_keep2.reset_index()
# df_keep2=df_keep2.set_index('ID_NUM')

df_keep2['Mortality_flg']=df_keep2['MORTALITY_RATE'].apply(lambda x:1 if x>=np.quantile(df_keep2['MORTALITY_RATE'],0.85) else 0)
###不进行共线性去除
# def assign_mortality_flag(group):
#     quantile_10 = np.quantile(group['MORTALITY_RATE'], 0.85)
#     group['Mortality_flg'] = group['MORTALITY_RATE'].apply(lambda x: 1 if x >= quantile_10 else 0)
#     return group

# df_keep = df_keep.groupby('DOCDATE_month').apply(assign_mortality_flag).drop('DOCDATE_month',axis=1)

# df_keep=df_keep.reset_index()
# df_keep=df_keep.set_index('ID_NUM')

# for col in df_keep.columns:
#     if df_keep[col].dtypes=='object':
#         df_keep[col]=df_keep[col].astype('category')
# 选取样本
validation_data = df_keep2[df_keep2['DOCDATE_month'].isin(['2', '3'])]

validation_data.shape
# 剩余的数据用于训练和测试集划分
training_test_data = df_keep2[~df_keep2['DOCDATE_month'].isin(['2', '3'])]
training_test_data.shape

# 对训练和测试数据进行特征和目标变量的划分

X = training_test_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21'])
y = training_test_data['Mortality_flg']
# X.columns.to_list()
y.value_counts()
# 进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 对跨期验证数据进行特征和目标变量的划分
X_validation = validation_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21'])
y_validation = validation_data['Mortality_flg']

y_validation.value_counts()
# 定义LightGBM参数（针对二分类优化）
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    # 'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # 处理类别不平衡
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

# 预测类别（默认阈值0.5）
f=np.quantile(y_pred_prob, 0.85)
y_pred_class = (y_pred_prob >= f).astype(int)

# 评估指标
print("AUC Score:", roc_auc_score(y_test, y_pred_prob))
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))

# y_test.value_counts()
# 绘制ROC曲线
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve")
plt.show()


# # 特征重要性（增益）
# lgb.plot_importance(lgb_baseline, importance_type='gain', max_num_features=15, figsize=(10, 6))
# plt.title("Feature Importance (Gain)")
# plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_baseline.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

feature_imp[feature_imp['Importance']>0]['Feature']
top_importantcol=list(feature_imp.head(20)['Feature'])
feature_imp.head(20)
print(feature_imp)
print(top_importantcol)


# 跨期验证部分
# 跨期验证集预测概率
y_pred_prob_validation = lgb_baseline.predict(X_validation, num_iteration=lgb_baseline.best_iteration)
# 跨期验证集预测类别（默认阈值 0.5）
f=np.quantile(y_pred_prob_validation, 0.85)
y_pred_class_validation = (y_pred_prob_validation >= f).astype(int)

# y_pred_prob_validation.mean()
# 跨期验证集评估指标
print("Validation AUC Score:", roc_auc_score(y_validation, y_pred_prob_validation))
print("Validation Accuracy:", accuracy_score(y_validation, y_pred_class_validation))
print("Validation Confusion Matrix:\n", confusion_matrix(y_validation, y_pred_class_validation))
# 跨期验证集 ROC 曲线
RocCurveDisplay.from_predictions(y_validation, y_pred_prob_validation)
plt.title("ROC Curve - Validation Set")
plt.show()
    


from optbinning import BinningProcess
import optbinning

cat_f=[col for col in top_importantcol if col in object_columns]

selection_criteria={"gini":{"min":0.15, "max":1}
                   }

binning_process=BinningProcess(top_importantcol,
                              categorical_variables=cat_f
                              ,selection_criteria=selection_criteria
                              )

binning_process.fit(X[top_importantcol],y)
binning_sum=binning_process.summary()
binning_sum=binning_sum.sort_values(by='gini', ascending=False)


#分箱结果
bin_table=pd.DataFrame()
for i in top_importantcol:
    optb=binning_process.get_binned_variable(i)
    temp=optb.binning_table.build()
    temp['name']=i
    bin_table=pd.concat([bin_table,temp])

bin_table.to_csv('.\\xyy\\死淘分析\\bin_table0506.csv',index=False,encoding='gbk')
import warnings
warnings.filterwarnings("ignore")
for var in top_importantcol:
    optb=binning_process.get_binned_variable(var)
    bin_table_var=optb.binning_table.build()
    optb.binning_table.plot(metric='event_rate')
    plt.show()
    print(bin_table_var.iloc[:,:-3])
    print('')




###################### 回归模型#########################################
# df_keep2['MORTALITY_RATE'].hist(bins=100)
# plt.show()


# 对训练和测试数据进行特征和目标变量的划分
# 选取样本
validation_data = df_keep2[df_keep2['DOCDATE_month'].isin(['2', '3'])]

validation_data.shape
# 剩余的数据用于训练和测试集划分
training_test_data = df_keep2[~df_keep2['DOCDATE_month'].isin(['2', '3'])]
training_test_data.shape


#######################################################################333
X = training_test_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21'])
y = training_test_data['MORTALITY_RATE']

# y.value_counts()
# 进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 对跨期验证数据进行特征和目标变量的划分
X_validation = validation_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21'])
y_validation = validation_data['MORTALITY_RATE']

# y_validation.value_counts()



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
    num_boost_round=1000,
    # early_stopping_rounds=50  # 添加早停防止过拟合
)

# 预测连续值
y_pred = lgb_regressor.predict(X_test, num_iteration=lgb_regressor.best_iteration)
# np.expm1(model.predict(X_test))
# 评估指标（回归问题）
from sklearn.metrics import root_mean_squared_error  # scikit-learn >= 1.4

print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 绘制实际值 vs 预测值散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想对角线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# 残差图
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# # 特征重要性（增益）
# lgb.plot_importance(lgb_regressor, importance_type='gain', max_num_features=15, figsize=(10, 6))
# plt.title("Feature Importance (Gain)")
# plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_regressor.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
top_important_cols = list(feature_imp.head(15)['Feature'])

feature_imp.head(20)

#  对验证集进行预测
y_val_pred = lgb_regressor.predict(X_validation, num_iteration=lgb_regressor.best_iteration)

# 评估验证集效果
val_rmse = root_mean_squared_error(y_validation, y_val_pred)
val_mae = mean_absolute_error(y_validation, y_val_pred)
val_r2 = r2_score(y_validation, y_val_pred)

print("=== 跨期验证结果 ===")
print(f"Validation RMSE: {val_rmse:.4f} (Test RMSE: {root_mean_squared_error(y_test, y_pred):.4f})")
print(f"Validation MAE: {val_mae:.4f} (Test MAE: {mean_absolute_error(y_test, y_pred):.4f})")
print(f"Validation R2: {val_r2:.4f} (Test R2: {r2_score(y_test, y_pred):.4f})")



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_validation, y_val_pred, alpha=0.5, color='blue')
plt.plot([y_validation.min(), y_validation.max()], 
         [y_validation.min(), y_validation.max()], 'r--')
plt.xlabel('Actual Values (Validation)')
plt.ylabel('Predicted Values (Validation)')
plt.title('Validation: Actual vs Predicted')

# 测试集对比图
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values (Test)')
plt.ylabel('Predicted Values (Test)')
plt.title('Test: Actual vs Predicted')
plt.tight_layout()
plt.show()

# 对比目标变量分布
plt.figure(figsize=(10, 4))
sns.kdeplot(y_train, label='Train')
sns.kdeplot(y_validation, label='Validation')
plt.title("MORTALITY_RATE Distribution Comparison")
plt.show()


def binning_with_target_analysis(df, feature, target='MORTALITY_RATE', n_bins=5, binning_method='equal_width'):
    """
    分箱并分析每个分箱与目标变量的关系
    参数:
        df: 数据框
        feature: 要分箱的特征名
        target: 目标变量名(默认'eef')
        n_bins: 分箱数量(默认5)
        binning_method: 分箱方法('equal_width', 'equal_freq'或'kmeans')
    返回:
        分箱统计结果数据框
        并显示可视化图表
    """
    # 复制数据避免修改原数据
    df_analysis = df[[feature, target]].copy()
    
    # 根据选择的方法进行分箱
    if binning_method == 'equal_width':
        df_analysis[f'{feature}_bin'] = pd.cut(df_analysis[feature], bins=n_bins)
    elif binning_method == 'equal_freq':
        df_analysis[f'{feature}_bin'] = pd.qcut(df_analysis[feature], q=n_bins, duplicates='drop')
    elif binning_method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_bins, random_state=42)
        kmeans.fit(df_analysis[[feature]])
        df_analysis[f'{feature}_bin'] = pd.cut(df_analysis[feature], 
                                            bins=np.unique(kmeans.cluster_centers_).sort(), 
                                            include_lowest=True)
    else:
        raise ValueError("binning_method必须是'equal_width', 'equal_freq'或'kmeans'")
    
    # 计算每个分箱的统计信息
    bin_stats = df_analysis.groupby(f'{feature}_bin').agg({
        target: ['count', 'mean', 'median', 'std', 'min', 'max'],
        feature: ['min', 'max', 'mean']
    }).round(2)
    
    bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns.values]
    bin_stats = bin_stats.rename(columns={
        f'{target}_count': '样本数',
        f'{target}_mean': 'Mortality_rate均值',
        f'{target}_median': 'EMortality_rate中位数',
        f'{target}_std': 'Mortality_rate标准差',
        f'{target}_min': 'Mortality_rate最小值',
        f'{target}_max': 'Mortality_rate最大值',
        f'{feature}_min': '分箱最小值',
        f'{feature}_max': '分箱最大值',
        f'{feature}_mean': '分箱均值'
    })
    
    # 可视化
    plt.figure(figsize=(15, 6))
    
    # 子图1: 箱线图
    plt.subplot(1, 2, 1)
    sns.boxplot(x=f'{feature}_bin', y=target, data=df_analysis)
    plt.title(f'{feature}分箱与{target}分布')
    plt.xticks(rotation=45)
    
    # 子图2: 均值条形图
    plt.subplot(1, 2, 2)
    sns.barplot(x=f'{feature}_bin', y=target, data=df_analysis, estimator=np.mean, ci=None)
    plt.title(f'各分箱{target}均值比较')
    plt.xticks(rotation=45)
    plt.ylabel('Mortality_rate均值')
    
    plt.tight_layout()
    plt.show()
    
    return bin_stats
for col in top_important_cols:
    age_bin_stats = binning_with_target_analysis(df_keep2, 'HARVESTSTATUS_month', n_bins=5, binning_method='equal_freq')
    print("密度分箱统计:")
    print(age_bin_stats)


#############################21日死淘为目标分析
df_keep3=df_keep2[df_keep2['MORTALITY_RATE_21'].notna()]
def assign_mortality_flag(group):
    quantile_10 = np.quantile(group['MORTALITY_RATE_21'], 0.85)
    group['Mortality_flg_21'] = group['MORTALITY_RATE_21'].apply(lambda x: 1 if x >= quantile_10 else 0)
    return group

df_keep3 = df_keep3.groupby('DOCDATE_month').apply(assign_mortality_flag).drop('DOCDATE_month',axis=1)
# df_keep3['Mortality_flg_21'].value_counts()
# df_keep2.groupby('DOCDATE_month')['MORTALITY_RATE_21'].mean()
# df_keep['MORTALITY_RATE_21'].isna().sum()
# print(result)

df_keep3=df_keep3.reset_index()
df_keep3=df_keep3.set_index('ID_NUM')


# 选取样本
validation_data = df_keep3[df_keep3['DOCDATE_month'].isin(['2', '3'])]

validation_data.shape
# 剩余的数据用于训练和测试集划分
training_test_data = df_keep3[~df_keep3['DOCDATE_month'].isin(['2', '3'])]
training_test_data.shape

# 对训练和测试数据进行特征和目标变量的划分

X = training_test_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg_21'])
y = training_test_data['Mortality_flg_21']

y.value_counts()
# 进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# y_test.value_counts()
# 对跨期验证数据进行特征和目标变量的划分
X_validation = validation_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg_21'])
y_validation = validation_data['Mortality_flg_21']

y_validation.value_counts()
# 定义LightGBM参数（针对二分类优化）
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    # 'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # 处理类别不平衡
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

# 预测类别（默认阈值0.5）
f=np.quantile(y_pred_prob, 0.85)
y_pred_class = (y_pred_prob >= f).astype(int)

# 评估指标
print("AUC Score:", roc_auc_score(y_test, y_pred_prob))
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))

# y_test.value_counts()
# 绘制ROC曲线
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve")
plt.show()


# # 特征重要性（增益）
# lgb.plot_importance(lgb_baseline, importance_type='gain', max_num_features=15, figsize=(10, 6))
# plt.title("Feature Importance (Gain)")
# plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_baseline.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

feature_imp[feature_imp['Importance']>0]['Feature']
top_importantcol=list(feature_imp.head(20)['Feature'])
feature_imp.head(20)
print(feature_imp)
print(top_importantcol)


# 跨期验证部分
# 跨期验证集预测概率
y_pred_prob_validation = lgb_baseline.predict(X_validation, num_iteration=lgb_baseline.best_iteration)
# 跨期验证集预测类别（默认阈值 0.5）
f=np.quantile(y_pred_prob_validation, 0.85)
y_pred_class_validation = (y_pred_prob_validation >= f).astype(int)

# y_pred_prob_validation.mean()
# 跨期验证集评估指标
print("Validation AUC Score:", roc_auc_score(y_validation, y_pred_prob_validation))
print("Validation Accuracy:", accuracy_score(y_validation, y_pred_class_validation))
print("Validation Confusion Matrix:\n", confusion_matrix(y_validation, y_pred_class_validation))
# 跨期验证集 ROC 曲线
RocCurveDisplay.from_predictions(y_validation, y_pred_prob_validation)
plt.title("ROC Curve - Validation Set")
plt.show()



#######################################################################
# 21日龄后死淘回归模型


    
# 对训练和测试数据进行特征和目标变量的划分
# 选取样本
validation_data = df_keep3[df_keep3['DOCDATE_month'].isin(['2', '3'])]

validation_data.shape
# 剩余的数据用于训练和测试集划分
training_test_data = df_keep3[~df_keep3['DOCDATE_month'].isin(['2', '3'])]
training_test_data.shape


X = training_test_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg_21'])
y = training_test_data['MORTALITY_RATE_21']

# y.value_counts()
# 进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 对跨期验证数据进行特征和目标变量的划分
X_validation = validation_data.drop(columns=['Mortality_flg', 'MORTALITY_RATE', 'MORTALITY_RATE_21','Mortality_flg_21'])
y_validation = validation_data['MORTALITY_RATE_21']

# y_validation.value_counts()



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
    num_boost_round=1000,
    # early_stopping_rounds=50  # 添加早停防止过拟合
)

# 预测连续值
y_pred = lgb_regressor.predict(X_test, num_iteration=lgb_regressor.best_iteration)
# np.expm1(model.predict(X_test))
# 评估指标（回归问题）
from sklearn.metrics import root_mean_squared_error  # scikit-learn >= 1.4

print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 绘制实际值 vs 预测值散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想对角线
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# 残差图
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# # 特征重要性（增益）
# lgb.plot_importance(lgb_regressor, importance_type='gain', max_num_features=15, figsize=(10, 6))
# plt.title("Feature Importance (Gain)")
# plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_regressor.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
top_important_cols = list(feature_imp.head(15)['Feature'])

feature_imp.head(20)

#  对验证集进行预测
y_val_pred = lgb_regressor.predict(X_validation, num_iteration=lgb_regressor.best_iteration)

# 评估验证集效果
val_rmse = root_mean_squared_error(y_validation, y_val_pred)
val_mae = mean_absolute_error(y_validation, y_val_pred)
val_r2 = r2_score(y_validation, y_val_pred)

print("=== 跨期验证结果 ===")
print(f"Validation RMSE: {val_rmse:.4f} (Test RMSE: {root_mean_squared_error(y_test, y_pred):.4f})")
print(f"Validation MAE: {val_mae:.4f} (Test MAE: {mean_absolute_error(y_test, y_pred):.4f})")
print(f"Validation R2: {val_r2:.4f} (Test R2: {r2_score(y_test, y_pred):.4f})")



# 对比目标变量分布
plt.figure(figsize=(10, 4))
sns.kdeplot(y_train, label='Train')
sns.kdeplot(y_validation, label='Validation')
plt.title("MORTALITY_RATE Distribution Comparison")
plt.show()
