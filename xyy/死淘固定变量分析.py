import pandas as pd
import numpy as np
import toad

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency
allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead.csv',encoding='gbk')
marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')
allinfo_dead.info()
allinfo_dead.describe()

# allinfo_dead['DOCAmount'].dtypes

# data_detect.to_csv('./xyy/data_detect.csv', index=False,encoding='gbk')

# 利用toad查看 iv值和分箱结果

drop_columns_Mortality=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','Mortality','livability_pct'
                        ,'yield_per_m2'
                        ]
allinfo_dead['Mortality_rate'].hist(bins=30)
plt.show()


quantile_10 = np.quantile(allinfo_dead['Mortality_rate'], 0.1)
allinfo_dead['Mortality_flg']=allinfo_dead['Mortality_rate'].apply(lambda x:1 if x<=quantile_10 else 0)

# allinfo_dead['Mortality_flg'].value_counts()
Mortality_df=allinfo_dead.drop(columns=drop_columns_Mortality,axis=1)

Mortality_df['id']=Mortality_df['id_no']+'_'+Mortality_df['HouseNo']

Mortality_df=Mortality_df.drop(columns=['id_no','HouseNo','HouseName','Houseid'],axis=1)
marketingdata_columns=marketingdata.columns.to_list()
drop_columns_marketingdata=[i for i in marketingdata_columns if i in Mortality_df.columns.to_list()]
Mortality_df=Mortality_df.drop(columns=drop_columns_marketingdata,axis=1)

# 去除唯一值变量
data_detect = toad.detector.detect(Mortality_df)
data_detect=data_detect.reset_index(drop=False)
unique_columns=list(data_detect[data_detect['unique']==1]['index'])
Mortality_df=Mortality_df.drop(unique_columns,axis=1)

quality_df=toad.quality(Mortality_df.drop('id',axis=1),target='Mortality_flg',iv_only=False)

from toad.plot import bin_plot
c=toad.transform.Combiner()
c.fit(Mortality_df.drop(columns=['id'],axis=1),y='Mortality_flg',min_samples=0.02)

transformed_df = c.transform(Mortality_df.drop(['id'], axis=1), labels=True)
transformed_df.to_csv('./xyy/transformed_df.csv', index=False,encoding='gbk')
# 选取需要绘图的列
plot_df = transformed_df[['Density', 'Mortality_flg']]

# 绘制分箱图
bin_plot(plot_df, x='Density', target='Mortality_flg')
plt.show()


bin_plot(c.transform(Mortality_df[['DOCdate','Mortality_flg']],labels=True),x='HEAge',target='Mortality_flg')




for col in Mortality_df.columns:
    if col not in ['Mortality_flg','id']:
        bin_plot(c.transform(Mortality_df[[col,'Mortality_flg']],label=True),x=col,target='Mortality_flg')
        plt.show()



####################建模

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

Mortality_df.columns.to_list()

print("目标变量分布:\n", Mortality_df['Mortality_flg'].value_counts())

# 移除无关列（如id）
data = Mortality_df.drop(columns=['id'])
data.columns.to_list()
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
X = data.drop(columns=['Mortality_flg','Batch'])
y = data['Mortality_flg']

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
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=1000
)


# 预测概率
y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)

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
lgb.plot_importance(model, importance_type='gain', max_num_features=15, figsize=(10, 6))
plt.title("Feature Importance (Gain)")
plt.show()

# 输出具体重要性值
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
print(feature_imp)



#############单变量分析

# num_cols = ['DOCAmount', 'HouseArea', 'Density', 'Age', 'HouseAmount', 'Days_to_Slaughter']


# 计算相关系数矩阵
correlation_matrix = Mortality_df.drop('Mortality_flg',axis=1).select_dtypes(include=['float64']).corr()
print(correlation_matrix['Mortality_rate'].sort_values(ascending=False))

# 绘制散点图矩阵
import seaborn as sns
sns.pairplot(Mortality_df.select_dtypes(include=['float64']))
plt.show()

####分类变量
for col in Mortality_df.columns:
    if Mortality_df[col].dtypes=='object':
        Mortality_df[col]=Mortality_df[col].astype('category')
Mortality_df.info()

# 使用箱线图或小提琴图可视化
for col in Mortality_df.select_dtypes(include=['category']).columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=col, y='Mortality_rate', data=Mortality_df)
    plt.xticks(rotation=45)
    plt.show()

# 或者使用ANOVA分析
from scipy.stats import f_oneway

for col in Mortality_df.select_dtypes(include=['category']).columns:
    groups = Mortality_df.groupby(col)['Mortality_rate'].apply(list)
    f_val, p_val = f_oneway(*groups)
    print(f"{col}: F-value={f_val:.2f}, p-value={p_val:.4f}")


Mortality_df.select_dtypes(include=['category']).columns


# 单独看分箱
def binary_binning_analysis(df, feature, target='eef_flg', n_bins=5, binning_method='equal_width'):
    """
    分箱并分析每个分箱与二分类目标变量的关系
    参数:
        df: 数据框
        feature: 要分箱的特征名
        target: 二分类目标变量名(默认'eef_flg')
        n_bins: 分箱数量(默认5)
        binning_method: 分箱方法('equal_width', 'equal_freq'或'kmeans')
    返回:
        分箱统计结果数据框
        并显示可视化图表
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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
    bin_stats = df_analysis.groupby(f'{feature}_bin',observed=False).agg({
        target: ['count', 'mean', 'sum'],
        feature: ['min', 'max', 'mean']
    }).round(4)
    
    bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns.values]
    bin_stats = bin_stats.rename(columns={
        f'{target}_count': '样本数',
        f'{target}_mean': '正样本比例',
        f'{target}_sum': '正样本数',
        f'{feature}_min': '分箱最小值',
        f'{feature}_max': '分箱最大值',
        f'{feature}_mean': '分箱均值'
    })
    
    # 计算WOE和IV值
    total_pos = df_analysis[target].sum()
    total_neg = len(df_analysis) - total_pos
    bin_stats['负样本数'] = bin_stats['样本数'] - bin_stats['正样本数']
    bin_stats['负样本比例'] = bin_stats['负样本数'] / total_neg
    bin_stats['正样本比例_g'] = bin_stats['正样本数'] / total_pos
    bin_stats['WOE'] = np.log(bin_stats['正样本比例_g'] / bin_stats['负样本比例'])
    bin_stats['IV'] = (bin_stats['正样本比例_g'] - bin_stats['负样本比例']) * bin_stats['WOE']
    total_iv = bin_stats['IV'].sum()
    
    # 可视化
    plt.figure(figsize=(18, 6))
    
    # 子图1: 正样本比例条形图
    plt.subplot(1, 3, 1)
    sns.barplot(x=bin_stats.index.astype(str), y='正样本比例', data=bin_stats.reset_index())
    plt.title(f'{feature}分箱的正样本比例\n(总IV值:{total_iv:.4f})')
    plt.xticks(rotation=45)
    
    # 子图2: WOE值条形图
    plt.subplot(1, 3, 2)
    sns.barplot(x=bin_stats.index.astype(str), y='WOE', data=bin_stats.reset_index())
    plt.title('各分箱WOE值')
    plt.xticks(rotation=45)
    plt.axhline(0, color='red', linestyle='--')
    
    # 子图3: 样本分布堆叠图
    plt.subplot(1, 3, 3)
    bin_stats[['正样本数','负样本数']].plot(kind='bar', stacked=True)
    plt.title('各分箱正负样本分布')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 卡方检验
    contingency_table = pd.crosstab(df_analysis[f'{feature}_bin'], df_analysis[target])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"\n卡方检验结果(χ²={chi2:.2f}, p={p_val:.4f})")
    if p_val < 0.05:
        print("各组间正样本比例存在显著差异")
    else:
        print("各组间正样本比例无显著差异")
    
    return bin_stats

# 等频分箱分析
Density_bin_stats = binary_binning_analysis(Mortality_df, 'Density', target='Mortality_flg', n_bins=5, binning_method='equal_freq')
print("密度分箱统计:")
print(Density_bin_stats)
