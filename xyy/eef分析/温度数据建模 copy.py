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
all_info_temdata=pd.read_csv('./data/data_cleaned/all_info_temdata0430.csv',encoding='gbk')
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')

all_info_temdata.columns.to_list()
col=[a.upper()for a in all_info_temdata.columns.to_list()]
all_info_temdata.columns=col

df=all_info_temdata.set_index('ID_NUM')


df=df.rename({'MORTALITY_RATE_X':'MORTALITY_RATE'},axis=1)
# df=df.fillna(0)
# df[['MORTALITY_RATE_X','MORTALITY_RATE_21']]
df.columns.to_list()
#筛除0占比超过阈值特征
drop_columns_eef=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','Mortality','livability_pct'
                        ,'yield_per_m2','Mortality_rate','Mortality_rate_21'
                        ]
drop_columns_eef=[a.upper()for a in drop_columns_eef]
drop_columns_eef=[i for i in df.columns.to_list() if i in drop_columns_eef]

# quantile_90 = np.quantile(df['EEF'], 0.9)
# df['eef_flg']=df['EEF'].apply(lambda x:1 if x>=quantile_90 else 0)


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
df2=df.drop(columns=drop_columns_marketingdata+drop_columns_eef,axis=1)

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
# df2['DOCDATE_month']
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


selected_cols=get_non_collinear_vars(0.8,df_keep[numeric_columns].drop(['EEF'],axis=1))

df_keep2=df_keep[selected_cols+object_columns+['EEF']].copy()
df_keep2.shape
for col in df_keep2.columns:
    if df_keep2[col].dtypes=='object':
        df_keep2[col]=df_keep2[col].astype('category')


# df_keep2['DOCDATE_month']

# X=df_keep2.drop(columns=['Mortality_flg','MORTALITY_RATE','MORTALITY_RATE_21'])

# y=df_keep2['Mortality_flg']
# 筛选出 2、3 月份的数据作为跨期验证数据
def assign_EEF_flg(group):
    quantile_85 = np.quantile(group['EEF'], 0.8)
    group['EEF_flg'] = group['EEF'].apply(lambda x: 1 if x >= quantile_85 else 0)
    return group

df_keep2 = df_keep2.groupby('DOCDATE_month').apply(assign_EEF_flg).drop('DOCDATE_month',axis=1)


# df_keep2.groupby('DOCDATE_month')['MORTALITY_RATE_21'].mean()

# print(result)

df_keep2=df_keep2.reset_index()
df_keep2=df_keep2.set_index('ID_NUM')

df_keep2['EEF_flg'].value_counts()
# 选取样本
validation_data = df_keep2[df_keep2['DOCDATE_month'].isin(['2', '3'])]

validation_data.shape
# 剩余的数据用于训练和测试集划分
training_test_data = df_keep2[~df_keep2['DOCDATE_month'].isin(['2', '3'])]
training_test_data.shape

# 对训练和测试数据进行特征和目标变量的划分

X = training_test_data.drop(columns=['EEF', 'EEF_flg'])
y = training_test_data['EEF_flg']

y.value_counts()
# 进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 对跨期验证数据进行特征和目标变量的划分
X_validation = validation_data.drop(columns=['EEF', 'EEF_flg'])
y_validation = validation_data['EEF_flg']

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
feature_imp.head(15)
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
for var in top_importantcol[:2]:
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
X = training_test_data.drop(columns=['EEF', 'EEF_flg'])
y = training_test_data['EEF']

# y.value_counts()
# 进行训练集和测试集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 对跨期验证数据进行特征和目标变量的划分
X_validation = validation_data.drop(columns=['EEF', 'EEF_flg'])
y_validation = validation_data['EEF']

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
plt.legend()  # 添加这行代码来显示图例
plt.show()