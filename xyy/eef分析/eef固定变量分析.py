import pandas as pd
import numpy as np
import toad

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead.csv',encoding='gbk')
marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')
allinfo_dead.info()
allinfo_dead.describe()

allinfo_dead['DOCAmount'].dtypes

# 唯一值变量
data_detect = toad.detector.detect(allinfo_dead)
data_detect=data_detect.reset_index(drop=False)
unique_columns=list(data_detect[data_detect['unique']==1]['index'])

# data_detect.to_csv('./xyy/data_detect.csv', index=False,encoding='gbk')

allinfo_dead['eef'].hist(bins=30)
plt.show()


quantile_90 = np.quantile(allinfo_dead['eef'], 0.9)
allinfo_dead['eef_flg']=allinfo_dead['eef'].apply(lambda x:1 if x>=quantile_90 else 0)

# allinfo_dead['eef_flg'].value_counts()

eef_df=allinfo_dead.drop(columns=unique_columns,axis=1)

eef_df['id']=eef_df['id_no']+'_'+eef_df['HouseNo']

eef_df=eef_df.drop(columns=['id_no','HouseNo','HouseName','Houseid'],axis=1)
keep_columns=['DOCdate', 'DOCAmount', 'HouseArea', 'Density', 'BirdsVariety', 'HESource', 'HEAge', 'Age', 'Harveststatus', 'EstimatedSlaughterDate ',
  'FarmName', 'FarmSupervisor', 'HouseAmount', 'Batch', 'eef_flg','eef','id']
eef_df=eef_df[keep_columns]




#kruskal 变量分析


# 利用toad查看 iv值和分箱结果

quality_df=toad.quality(eef_df.drop(['id','eef'],axis=1),target='eef_flg',iv_only=False)



from toad.plot import bin_plot
c=toad.transform.Combiner()
c.fit(eef_df.drop(columns=['id','eef'],axis=1),y='eef_flg',min_samples=0.05)



transformed_df = c.transform(eef_df.drop(['id'], axis=1), labels=True)
transformed_df.to_csv('./xyy/eef_transformed_df.csv', index=False,encoding='gbk')
# 选取需要绘图的列
plot_df = transformed_df[['Density', 'eef_flg']]

# 绘制分箱图
bin_plot(plot_df, x='Density', target='eef_flg')
plt.show()




for col in eef_df.columns:
    if col not in ['eef_flg','id']:
        bin_plot(c.transform(eef_df[[col,'eef_flg']],label=True),x=col,target='eef_flg')
        plt.show()



####################建模

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

eef_df.columns.to_list()

print("目标变量分布:\n", eef_df['eef_flg'].value_counts())

# 移除无关列（如id）
data = eef_df.drop(columns=['id','eef'])

eef_df.info()
# 处理日期字段（转换为时间差或数值）
# data['DOCdate'] = pd.to_datetime(data['DOCdate'])
# data['EstimatedSlaughterDate '] = pd.to_datetime(data['EstimatedSlaughterDate '])
# data['Days_to_Slaughter'] = (data['EstimatedSlaughterDate '] - data['DOCdate']).dt.days
# data = data.drop(columns=['DOCdate', 'EstimatedSlaughterDate'])

for col in data.columns:
    if data[col].dtypes=='object':
        data[col]=data[col].astype('category')
data.info()

# 分离特征和目标
X = data.drop(columns=['eef_flg','Batch'])
y = data['eef_flg']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
model_eef = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=1000
)


# 预测概率
y_pred_prob = model_eef.predict(X_test, num_iteration=model_eef.best_iteration)

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
lgb.plot_importance(model_eef, importance_type='gain', max_num_features=15, figsize=(10, 6))
plt.title("Feature Importance (Gain)")
plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_eef.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
print(feature_imp)


# ------------------------- 新增KS指标计算 -------------------------
def calculate_ks(y_true, y_pred_prob):
    """
    计算KS指标
    :param y_true: 真实标签（0或1）
    :param y_pred_prob: 预测概率
    :return: KS值
    """
    # 将预测概率和真实标签组合
    df = pd.DataFrame({'prob': y_pred_prob, 'true': y_true})
    df = df.sort_values('prob', ascending=False)
    
    # 计算累积正负样本分布
    df['cum_true'] = df['true'].cumsum() / df['true'].sum()
    df['cum_false'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    
    # KS = max(TPR - FPR)
    ks = (df['cum_true'] - df['cum_false']).max()
    
    return ks

# 计算KS值
ks_score = calculate_ks(y_test, y_pred_prob)
print(f"KS Score: {ks_score:.4f}")

# 绘制KS曲线（可选）
def plot_ks_curve(y_true, y_pred_prob):
    df = pd.DataFrame({'prob': y_pred_prob, 'true': y_true})
    df = df.sort_values('prob', ascending=False)
    df['cum_true'] = df['true'].cumsum() / df['true'].sum()
    df['cum_false'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    df['ks'] = df['cum_true'] - df['cum_false']
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['prob'], df['cum_true'], label='Cumulative True Positive Rate')
    plt.plot(df['prob'], df['cum_false'], label='Cumulative False Positive Rate')
    plt.plot(df['prob'], df['ks'], label='KS Statistic', linestyle='--')
    
    ks_max = df['ks'].max()
    x_max = df['prob'][df['ks'].idxmax()]
    plt.axvline(x=x_max, color='r', linestyle='--', label=f'Max KS = {ks_max:.4f}')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Cumulative Rate')
    plt.title('KS Curve')
    plt.legend()
    plt.show()

plot_ks_curve(y_test, y_pred_prob)


#############单变量分析

num_cols = ['DOCAmount', 'HouseArea', 'Density', 'Age', 'HouseAmount', 'Days_to_Slaughter']
# 计算相关系数矩阵
correlation_matrix = eef_df.drop('eef_flg',axis=1).select_dtypes(include=['float64']).corr()
print(correlation_matrix['eef'].sort_values(ascending=False))

# 绘制散点图矩阵
import seaborn as sns
sns.pairplot(eef_df.select_dtypes(include=['float64']))
plt.show()

####分类变量
for col in eef_df.columns:
    if eef_df[col].dtypes=='object':
        eef_df[col]=eef_df[col].astype('category')
eef_df.info()

# 使用箱线图或小提琴图可视化
for col in eef_df.select_dtypes(include=['category']).columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=col, y='eef', data=eef_df)
    plt.xticks(rotation=45)
    plt.show()

# 或者使用ANOVA分析
from scipy.stats import f_oneway

for col in eef_df.select_dtypes(include=['category']).columns:
    groups = eef_df.groupby(col)['eef'].apply(list)
    f_val, p_val = f_oneway(*groups)
    print(f"{col}: F-value={f_val:.2f}, p-value={p_val:.4f}")


eef_df.select_dtypes(include=['category']).columns