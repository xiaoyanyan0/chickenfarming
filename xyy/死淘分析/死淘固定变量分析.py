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


data_detect = toad.detector.detect(allinfo_dead)
data_detect=data_detect.reset_index(drop=False)
unique_columns=list(data_detect[data_detect['unique']==1]['index'])

# data_detect.to_csv('./xyy/data_detect.csv', index=False,encoding='gbk')

all_columns=allinfo_dead.columns.to_list()
keep_columns=[column for column in all_columns if column not in unique_columns]
len(keep_columns)
numeric_columns = []
object_columns = []
special_columns=['Houseid','Batch']

for column in keep_columns:
    if np.issubdtype(allinfo_dead[column].dtype, np.number):
        numeric_columns.append(column)
    else:
        object_columns.append(column)
numeric_columns=[i for i in numeric_columns if i not in special_columns]
object_columns=object_columns+special_columns

len(numeric_columns)+len(object_columns)

#相关性检验
data_dict = {
    'house': '栋舍 House',
    'birds_placed': '进雏只数\nBird placed No.',
    'gender': '公母\nGender',
    'house_area_m2': '鸡舍面积m2 \nHouse Area',
    'stocking_density': '出栏密度\nDensity',
    'birds_hung': '挂鸡只数\nHang No.',
    'total_hung_weight_kg': '挂鸡总重（kg）\nTotal hung weight',
    'avg_weight_kg': '均重（kg）\n Average weight',
    'small_broilers_count': '小毛鸡数量\nSmall broilers No.',
    'small_broilers_weight_kg': '小毛鸡总重（kg）\n Total weight of small broilers',
    'pp_dead_culled_count': 'PP死淘鸡只\nPP cull and dead No.',
    'dead_culled_weight_kg': '死淘总重（kg）\nDead and Cull Weight',
    'pp_rejects_count': 'PP不合格淘汰鸡\nPP Cull',
    'pp_rejects_weight_kg': 'PP淘汰鸡总重\nPP Cull bird Weight',
    'age_days': '日龄Age',
    'dead_during_catch_count': '出栏造成死亡只数 \nDead while catching',
    'birds_caught_count': '出鸡只数\nCatching No.',
    'livability_pct': '成活率\nLivability (%)',
    'total_caught_weight_kg': '出鸡总重（kg）\nTotal Catched weight',
    'yield_per_m2': '单位面积产肉率\nDensity, Yield(KG)/m2',
    'final_avg_weight_kg': '均重（kg）\nAverage weight',
    'total_feed_kg': '累计耗料（kg）\nFeed cons. Cum.',
    'fcr': '料肉比 FCR',
    'adjusted_fcr': 'Adjust FCR (base 2.45KG)',
    'eef': '欧洲指数 EEF',
    'feed_cost': '饲料成本（元）',
    'electricity_cost': '用电费用（元）',
    'gas_cost': '燃气费用（元）',
    'depreciation_cost': '折旧费（元）',
    'chick_cost': '雏鸡成本（元）',
    'total_cost': '总成本（元）',
    'cost_per_kg': '每公斤成本（元）',
    'revenue': '毛鸡销售收入（元）',
    'profit_per_house': '每栋纯利润（元）',
    'medicine_per_bird': '药品（元/只）',
    'vaccine_per_bird': '疫苗（元/只）',
    'mv_cost_per_bird': 'M&V费（元/只）',
    'disinfectant_per_bird': '消毒药费（元/只）',
    'feed_per_bird': '饲料（元/只）',
    'electricity_per_bird': '用电（元/只）',
    'gas_per_bird': '燃气（元/只）',
    'labor_per_bird': '人工（元/只）',
    'consumables_per_bird': '低值易耗品（元/只）',
    'depreciation_per_bird': '折旧费（元/只）',
    'chick_cost_per_bird': '雏鸡成本（元/元）',
    'cost_per_bird': '每只鸡成本（元）',
    'FarmName': '农场名称',
    'FarmSupervisor': '场长',
    'house_count': '鸡舍数量',
    'Batch': '批次号',
    'id_no': '日报文件名称（一般为农场名称+批次）',
    'HouseNo': '鸡舍号',
    'DOCdate': '入雏日期',
    'DOCAmount': '入雏数量',
    'HouseArea': '鸡舍面积',
    'Density': '饲养密度',
    'Gender': '公母',
    'BirdsVariety': '雏源',
    'HESource': '种蛋源',
    'HEAge': '种鸡周龄',
    'Age': '日龄',
    'Harveststatus': '出栏日期',
    'Houseid': '栋舍代码',
    'HouseName': '栋舍名称',
    'NumberofFloorEggs ': '地面蛋数量',
    'EstimatedSlaughterDate ': '预计出栏日期',
    'Dead': '死鸡',
    'Swollen_Head': '肿头',
    'Weak': '弱小鸡',
    'Navel_Disease': '脐炎',
    'Stick_Anus': '糊肛',
    'Lame_Paralysis': '腿病',
    'Mortality': '死淘',
    'Mortality_rate': '死淘率',
    'HouseAmount':'鸡舍数量',

}


# 数值变量
# 计算皮尔逊相关系数和 p 值
def calculate_pearson_correlation(df, target_variable):
    pearson_results = {}
    for column in df.columns:
        if column != target_variable:
            corr, p_value = pearsonr(df[column], df[target_variable])
            pearson_results[column] = (corr, p_value)
    return pearson_results


# 目标变量
target_variables = ['Mortality_rate', 'eef']
corr_data=allinfo_dead[numeric_columns]
# 计算皮尔逊相关系数

def corr_df_generate1(target):
    pearson_df=pd.DataFrame()
    variables=[]
    corrs=[]
    p_values=[]
    pearson_results = calculate_pearson_correlation(corr_data, target)
    print(f"皮尔逊相关系数（目标变量: {data_dict[target]}）:")
    for variable, (corr, p_value) in pearson_results.items():
        print(f"{data_dict[variable]}: 相关系数 = {corr:.4f}, p 值 = {p_value:.4f}")
        variables.append(data_dict[variable])
        corrs.append(corr)
        p_values.append(p_value)

    pearson_df['variables']=variables
    pearson_df['target']=target
    pearson_df['type']="皮尔逊相关系数"
    pearson_df['corrs']=corrs
    pearson_df['p_values']=p_values

    return pearson_df


pearson_df1=corr_df_generate1('Mortality_rate')
pearson_df2=corr_df_generate1('eef') 

corr_df = pd.concat([pearson_df1, pearson_df2], axis=0, ignore_index=True)

corr_df.to_csv('./xyy/corr_data.csv', index=False,encoding='gbk')

#变量分析

object_data=allinfo_dead[object_columns+['Mortality_rate', 'eef']]
# allinfo_dead['HEAge'].value_counts()
for column in object_columns:
    category_count = object_data[column].nunique()
    print(f"{data_dict[column]}: 类别数 = {category_count:.4f}")

print(object_data.describe())



def check_correlation(df, categorical_vars, target_vars):
    results = {}
    for target_var in target_vars:
        results[target_var] = {}
        for categorical_var in categorical_vars:
            groups = []
            for category in df[categorical_var].unique():
                group = df[df[categorical_var] == category][target_var]
                groups.append(group)
            try:
                h_statistic, p_value = stats.kruskal(*groups)
                results[target_var][categorical_var] = {
                    'H统计量': h_statistic,
                    'p值': p_value,
                    '是否显著相关': p_value < 0.05
                }
            except ValueError:
                results[target_var][categorical_var] = {
                    'H统计量': None,
                    'p值': None,
                    '是否显著相关': None
                }
    return results

kruskal_df=check_correlation(object_data, object_columns, target_variables)

for target_var, cat_results in kruskal_df.items():
    print(f"目标变量: {data_dict[target_var]}")
    for cat_var, result in cat_results.items():
        print(f"  字符型变量: {cat_var}")
        print(f"    H统计量: {result['H统计量']}")
        print(f"    p值: {result['p值']}")
        print(f"    是否显著相关: {result['是否显著相关']}")


# 利用toad查看 iv值和分箱结果

drop_columns_Mortality=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','Mortality','Mortality_rate','livability_pct'
                        ,'yield_per_m2'
                        ]

drop_columns_eef=['Deafinal_avg_weight_kgd','fcr','livability_pct','Age','total_caught_weight_kg', 'total_feed_kg','birds_caught_count']


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