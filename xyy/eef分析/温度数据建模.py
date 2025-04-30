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

all_info_temdata=pd.read_csv('./data/data_cleaned/all_info_temdata.csv',encoding='gbk')
marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')


col=[a.upper()for a in all_info_temdata.columns.to_list()]
all_info_temdata.columns=col

df=all_info_temdata.set_index('ID_NUM')
# df=df.fillna(0)

#筛除0占比超过阈值特征
drop_columns_eef=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','Mortality','livability_pct'
                        ,'yield_per_m2','Mortality_rate'
                        ]
drop_columns_eef=[a.upper()for a in drop_columns_eef]


quantile_90 = np.quantile(df['EEF'], 0.9)
df['eef_flg']=df['EEF'].apply(lambda x:1 if x>=quantile_90 else 0)


marketingdata_columns=marketingdata.columns.to_list()
marketingdata_columns=[a.upper()for a in marketingdata_columns]

marketingdata_columns.remove('EEF')
drop_columns_marketingdata=[i for i in df.columns.to_list() if i in marketingdata_columns]

#把细分的死淘数据和出栏数据指标去掉
df2=df.drop(columns=drop_columns_marketingdata+drop_columns_eef,axis=1)

df2.shape
####缺失过高的去掉
# df2['EEF']
thres_zeros=0.8
# f=df[X].replace(0,np.nan)
missing_p=np.sum(df2.isnull(),axis=0)/df2.shape[0]
low_missing=missing_p[missing_p<thres_zeros].index.tolist()
df_keep=df[low_missing].copy()

df_keep.shape

# df_keep['eef_flg']
# 数值类别型变量
numeric_columns = []
object_columns = []
special_columns=['HOUSEID']

for column in df_keep.columns:
    if np.issubdtype(df_keep[column].dtype, np.number):
        numeric_columns.append(column)
    else:
        object_columns.append(column)
numeric_columns=[i for i in numeric_columns if i not in special_columns]
object_columns=object_columns+special_columns

len(numeric_columns)+len(object_columns)
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


df_num=df_keep[numeric_columns].drop(columns=['EEF','eef_flg'],axis=1)
selected_cols=get_non_collinear_vars(0.8,df_num)

df_keep2=df_keep[selected_cols+object_columns+['EEF','eef_flg']].copy()
df_keep2.shape

# df_keep2['eef_flg']

for col in df_keep2.columns:
    if df_keep2[col].dtypes=='object':
        df_keep2[col]=df_keep2[col].astype('category')
# df_keep2['HOUSENO']

X=df_keep2.drop(columns=['EEF','eef_flg','HOUSEID'],axis=1)

y=df_keep2['eef_flg']

y.value_counts()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


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
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# 评估指标
print("AUC Score:", roc_auc_score(y_test, y_pred_prob))
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))

# 绘制ROC曲线
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve")
plt.show()


# 特征重要性（增益）
lgb.plot_importance(lgb_baseline, importance_type='gain', max_num_features=15, figsize=(10, 6))
plt.title("Feature Importance (Gain)")
plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_baseline.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
top_importantcol=list(feature_imp.head(10)['Feature'])

print(feature_imp)
print(top_importantcol)


from optbinning import BinningProcess
import optbinning

cat_f=[col for col in top_importantcol if col in object_columns]

selection_criteria={"gini":{"min":0.15, "max":1}
                   }

binning_process=BinningProcess(top_importantcol,
                              categorical_variables=cat_f
                              ,selection_criteria=selection_criteria
                              )

binning_process.fit(X_train[top_importantcol],y_train)
binning_sum=binning_process.summary()
binning_sum=binning_sum.sort_values(by='gini', ascending=False)

# print("包含无限值的列：")
# print(X_train.columns[np.isinf(X_train).any()])

# # 替换无限值为NaN或特定值
# X_train = X_train.replace([np.inf, -np.inf], np.nan)

# # 或者直接删除包含无限值的行
# X_train = X_train[~np.isinf(X_train).any(axis=1)]





#分箱结果
bin_table=pd.DataFrame()
for i in top_importantcol:
    optb=binning_process.get_binned_variable(i)
    temp=optb.binning_table.build()
    temp['name']=i
    bin_table=pd.concat([bin_table,temp])


for var in X:
    optb=binning_process.get_binned_variable(var)
    bin_table_var=optb.binning_table.build()
    optb.binning_table.plot(metric='event_rate')
    plt.show()
    print(bin_table_var.iloc[:,:-3])
    print('')





#### 回归模型
# df_keep2['MORTALITY_RATE'].hist(bins=100)
# plt.show()
X=df_keep2.drop(columns=['eef_flg','EEF','HOUSEID'])

y=df_keep2['EEF']

# y.value_counts()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 假设X是特征矩阵，y是连续型目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

# 特征重要性（增益）
lgb.plot_importance(lgb_regressor, importance_type='gain', max_num_features=15, figsize=(10, 6))
plt.title("Feature Importance (Gain)")
plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_regressor.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
top_important_cols = list(feature_imp.head(10)['Feature'])


