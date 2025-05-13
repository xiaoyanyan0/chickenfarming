import pandas as pd
import toad
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import warnings
warnings.filterwarnings("ignore")

all_HumTem_data1=pd.read_csv('./data/data_cleaned/all_HumTem_data1.csv',encoding='gbk')
all_HumTem_data2=pd.read_csv('./data/data_cleaned/all_HumTem_data2.csv',encoding='gbk')

mixed_columns1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17]
for col in all_HumTem_data1.columns[mixed_columns1]:
    all_HumTem_data1[col] = pd.to_numeric(all_HumTem_data1[col], errors='coerce')  # 'coerce' 将无效值转NaN
mixed_columns2 = [4,5,6,7,8,9,10,11,12,13,14,17]
for col in all_HumTem_data2.columns[mixed_columns2]:
    all_HumTem_data2[col] = pd.to_numeric(all_HumTem_data2[col], errors='coerce')  # 'coerce' 将无效值转NaN


all_HumTem_data=pd.concat([all_HumTem_data1,all_HumTem_data2],ignore_index=True)
all_HumTem_data=all_HumTem_data.drop(columns=['house_no','id_no'],axis=1)
all_HumTem_data=all_HumTem_data.drop_duplicates()

###异常值处理
abs(all_HumTem_data['温度6-平均']-all_HumTem_data['外部-平均']).describe()

all_HumTem_data['温度6-平均'].notna().sum()
all_HumTem_data['外部-平均'].notna().sum()

all_HumTem_data['外部-平均'].describe()
all_HumTem_data['温度6-平均'].describe()

data=all_HumTem_data.drop(columns=['温度6-平均'],axis=1).copy()
###目标温度异常处理
# data['目标温度'].describe().round(2)
# data['目标温度'].nsmallest(100).iloc[-1]

# 99%是35度,1%是15.9度
lower_bound = 2  
upper_bound = 37  
data['目标温度'] = data['目标温度'].clip(lower=lower_bound, upper=upper_bound)



# 1. 计算盖帽后的温度差异
for i in range(1, 6):
    data[f'温度{i}与目标差异'] = data[f'温度{i}-平均'].astype(float) - data['目标温度'].astype(float)
    # print(data[f'温度{i}与目标差异'].describe().round(2))
    data[f'温度{i}与目标差异'] = data[f'温度{i}与目标差异'].clip(lower=-3, upper=data[f'温度{i}与目标差异'].quantile(0.95))

diff_cols=[f'温度{i}与目标差异' for i in range(1, 6)]

data[diff_cols].describe().round(2)
# 2. 更新温度{i}-平均
for i in range(1, 6):
    data[f'温度{i}-平均_new'] = data['目标温度'].astype(float) + data[f'温度{i}与目标差异']
    # 可选：覆盖原始列


# 3. 验证结果
print(data[['温度1-平均', '温度1-平均_new', '温度1与目标差异']].head())
for i in range(1, 6):
    data[f'温度{i}-平均'] = data[f'温度{i}-平均_new']
    data=data.drop(columns=[f'温度{i}-平均_new'],axis=1)

##############3#处理剩余外部温度，湿度等数据
data.columns.to_list()
# all_HumTem_data['温度1-平均'].describe().round(2)

data[['外部-平均', '湿度内部平均', '湿度外部平均','鸡舍温度-最低', '鸡舍温度-平均', '鸡舍温度-最高','水', '饲料', '水平']].describe().round(2)
data['鸡舍温度-最高'].quantile(1)
# 指定需要盖帽的列
cols_to_cap = ['外部-平均', '湿度内部平均', '湿度外部平均','鸡舍温度-最低', '鸡舍温度-平均', '鸡舍温度-最高', '水', '饲料', '水平']

# 计算1%和99%分位数，并应用盖帽
for col in cols_to_cap:
    lower_bound = data[col].quantile(0.01)  # 1%分位数
    upper_bound = data[col].quantile(0.99)  # 99%分位数
    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

# 验证结果（保留2位小数）
print(data[cols_to_cap].describe().round(2))


df_detect=toad.detect(data)
df_detect=df_detect.reset_index(drop=False)
df_detect.to_csv('./data/data_detected/HumTem_data_detect_修正后.csv',index=False,encoding='gbk')

#####################3##数据聚合
numeric_columns = [
     '目标温度', '鸡舍温度-最低', '鸡舍温度-平均',
    '鸡舍温度-最高', '温度1-平均', '温度2-平均', '温度3-平均',
    '温度4-平均', '温度5-平均',  '外部-平均',
    '湿度内部平均', '湿度外部平均', '水', '饲料', '水平',
]
# 将需要统计的字段转换为数值类型
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# 定义温度相关列
temp_cols = [f'温度{i}-平均' for i in range(1, 6)]
# 将日龄列转换为数值类型
data['日龄'] = pd.to_numeric(data['日龄'], errors='coerce').astype('Int64')
# 计算内外湿差和温差
# 计算每个日龄每个时间的最高温度（温度 1 - 平均到温度 5 - 平均的最高值）
# data['最高温度'] = data[temp_cols].max(axis=1)
# # 计算每个日龄每个时间的最低温度（温度 1 - 平均到温度 5 - 平均的最低值）
# data['最低温度'] = data[temp_cols].min(axis=1)
# # 计算每个日龄每个时间的平均温度（温度 1 - 平均到温度 5 - 平均的平均值）
# data['平均温度'] = data[temp_cols].mean(axis=1)

data['内外温差']=data['鸡舍温度-平均'].astype(float)-data['外部-平均'].astype(float)
data['内外湿差']=data['湿度内部平均'].astype(float)-data['湿度外部平均'].astype(float)


data.columns.to_list()
# 按 house_no、id_no 和日龄分组
grouped = data.groupby(['ID_NUM', '日龄'])

# 统计每个分组内的最高温度、最低温度、平均温度以及 Humidity In 1 Avg 的最值和均值
agg_result = grouped.agg({
    **{col: ['max', 'min', 'mean'] for col in temp_cols},
    '湿度内部平均': ['max', 'min', 'mean'],
    '内外温差': ['max', 'min', 'mean'],
    '内外湿差': ['max', 'min', 'mean'],
    '外部-平均': ['max', 'min', 'mean'],  
    '鸡舍温度-最低':['min'] ,
    '鸡舍温度-平均':['mean'] ,
    '鸡舍温度-最高':['max'] ,

})

# agg_result.head()

# # 计算每个日龄所有时间的最高温度（温度 1 - 平均到温度 6 - 平均的最高值）
# agg_result['最高温度'] = agg_result[[f'{col}_max' for col in temp_cols]].mean(axis=1)
# # 计算每个日龄所有时间的最低温度（温度 1 - 平均到温度 6 - 平均的最低值）
# agg_result['最低温度'] = agg_result[[f'{col}_min' for col in temp_cols]].mean(axis=1)
# # 计算每个日龄所有时间的平均温度（温度 1 - 平均到温度 6 - 平均的平均值）
# agg_result['平均温度'] = agg_result[[f'{col}_mean' for col in temp_cols]].mean(axis=1)

# 计算每日温差
agg_result[('鸡舍温度-最高', 'range')] = agg_result[('鸡舍温度-最高', 'max')] - agg_result[('鸡舍温度-最低', 'min')]


agg_result.columns = ['_'.join(col).strip() for col in agg_result.columns.values]

agg_result.columns.to_list()
# 重命名 Humidity In 1 Avg 的统计结果列
# agg_result = agg_result.rename(columns={
#     'Humidity In 1 Avg_max': 'Humidity In 1 Avg 最高值',
#     'Humidity In 1 Avg_min': 'Humidity In 1 Avg 最低值',
#     'Humidity In 1 Avg_mean': 'Humidity In 1 Avg 平均值'
# })

# 合并统计结果与原数据




grouped2 = data.sort_values(by=['ID_NUM', '日龄','时间']).reset_index(drop=True)

# 定义计算变化率的函数
def calculate_change_rate(series):
    """
    计算变化率：(当前值 - 前一个值) / |前一个值|
    处理异常值：忽略inf、-inf和NaN
    """
    # 计算原始变化率
    change_rate = series.diff() / series.shift(1).abs()  # 用绝对值避免符号影响
    
    # 处理异常值
    change_rate = change_rate.replace([np.inf, -np.inf], np.nan)  # inf → NaN
    return change_rate


# 计算平均温度、最高温度和最低温度的变化率
grouped2['平均温度变化率'] = grouped2.groupby(['ID_NUM', '日龄'])['鸡舍温度-平均'].transform(calculate_change_rate)
grouped2['最高温度变化率'] = grouped2.groupby(['ID_NUM', '日龄'])['鸡舍温度-最高'].transform(calculate_change_rate)
grouped2['最低温度变化率'] = grouped2.groupby(['ID_NUM', '日龄'])['鸡舍温度-最低'].transform(calculate_change_rate)

agg_result2=grouped2.groupby(['ID_NUM', '日龄'])[['平均温度变化率','最高温度变化率','最低温度变化率']].mean()

# 查看变化率中的inf和NaN数量
# print("Infinity counts:", grouped2[['平均温度变化率']].applymap(lambda x: np.isinf(x)).sum())
# print("NaN counts:", grouped2[['平均温度变化率']].isna().sum())

agg_result['平均温度变化率']=agg_result2['平均温度变化率']
agg_result['最高温度变化率']=agg_result2['最高温度变化率']
agg_result['最低温度变化率']=agg_result2['最低温度变化率']
agg_result=agg_result.reset_index()

agg_result=agg_result.rename({'鸡舍温度-最高_range':'每日温差'},axis=1)

agg_result.columns.to_list()


agg_result.to_csv('./data/data_cleaned/HumTem_data_agg0512.csv', index=False,encoding='gbk')


# 宽表加工

import pandas as pd
allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')
HumTem_data_agg=pd.read_csv('./data/data_cleaned/HumTem_data_agg0512.csv',encoding='gbk')

# HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]
# 'G28_25', 'G31_62'前后两个批次重复
# HumTem_data_agg = HumTem_data_agg[~HumTem_data_agg['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]


allinfo_dead['ID_NUM'].drop_duplicates().count()
HumTem_data_agg['ID_NUM'].drop_duplicates().count()
HumTem_data_agg[['ID_NUM','日龄']].drop_duplicates()

# allinfo_dead['Mortality_rate'].isna().sum()
#日报中的农场名字和文件名字中的对应不上
# HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith('G04')]['ID_NUM']

# allinfo_dead[allinfo_dead['ID_NUM'].str.startswith('G1A')]['ID_NUM']

allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 只有 G31_62匹不上
HumTem_data_t=pd.merge(HumTem_data_agg,allinfo_dead[['ID_NUM','age_days']],how='left',on=['ID_NUM'])

HumTem_data_t['age_days'].min()
HumTem_data_t['age_days'].notna().sum()

# HumTem_data_t[HumTem_data_t['age_days'].isna()]['ID_NUM'].drop_duplicates()

HumTem_data_normal=HumTem_data_t[HumTem_data_t['日龄']<HumTem_data_t['age_days']]
HumTem_data_abnormal=HumTem_data_t[HumTem_data_t['日龄']>=HumTem_data_t['age_days']]


HumTem_data_normal['日龄'].max()

# import matplotlib.pyplot as plt
# HumTem_data_normal.groupby(['Age'])['AvgTemperature'].mean().plot()
# plt.show()


HumTem_data_normal=HumTem_data_normal.drop('age_days',axis=1)
HumTem_data_normal=HumTem_data_normal.drop_duplicates()

# HumTem_data_normal[['ID_NUM','日龄']].drop_duplicates()
# HumTem_data_normal[['ID_NUM']].drop_duplicates()

# 计算每个组合出现的次数
# counts = HumTem_data_agg.groupby(['ID_NUM', '日龄']).size().reset_index(name='计数')

# 筛选出重复的组合
# repeated_pairs = counts[counts['计数'] > 1]
# repeated_pairs['ID_NUM'].drop_duplicates()
# print(repeated_pairs)
HumTem_data_normal.columns.to_list()

keep_cols=['ID_NUM', '日龄', '温度1-平均_mean', '温度2-平均_mean', '温度3-平均_mean','温度4-平均_mean',  
           '温度5-平均_mean', '内外温差_mean','内外湿差_mean','外部-平均_mean','湿度内部平均_mean', 
           '鸡舍温度-最低_min', '鸡舍温度-平均_mean', '鸡舍温度-最高_max', '每日温差', '平均温度变化率', '最高温度变化率',
             '最低温度变化率']


wide_df = HumTem_data_normal[keep_cols].pivot(index='ID_NUM', columns='日龄')

# 重置列名和索引
wide_df.columns = ['_'.join(map(str, (col[0], col[1]))) for col in wide_df.columns.values]
wide_df = wide_df.reset_index()

wide_df.columns.to_list()

wide_df.to_csv('./data/data_cleaned/wide_df_0512.csv', index=False,encoding='gbk')

##基本信息等拼接
# wide_df['ID_NUM']


allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 只有 G31_63,G28_22,匹不上，G27_24_H1，G27_25_H1为空

wide_df['ID_NUM_copy']=wide_df['ID_NUM']
all_info_temdata=pd.merge(allinfo_dead,wide_df,on='ID_NUM',how='left')

# all_info_temdata.info()
# all_info_temdata['ID_NUM_copy'].notna().sum()
# all_info_temdata[all_info_temdata['ID_NUM_copy'].isna()]['ID_NUM'].str[:6].unique()
# all_info_temdata[all_info_temdata['ID_NUM_copy'].isna()]['ID_NUM'].unique()
# wide_df['ID_NUM']
# all_info_temdata['ID_NUM_copy'].isna().sum()


all_info_temdata2=all_info_temdata.drop('ID_NUM_copy',axis=1)
import toad
data_detect = toad.detector.detect(all_info_temdata2)
data_detect=data_detect.reset_index(drop=False)
# all_info_temdata2.head()

all_info_temdata2.to_csv('./data/data_cleaned/all_info_temdata0512.csv',index=False,encoding='gbk')

all_info_temdata2['ID_NUM'].drop_duplicates()

all_HumTem_data1=pd.read_csv('./data/data_cleaned/all_HumTem_data1.csv',encoding='gbk')
all_HumTem_data2=pd.read_csv('./data/data_cleaned/all_HumTem_data2.csv', encoding='gbk')

allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')




# 检查是否完全读取环控数据
HumTem_data_normal[HumTem_data_normal['ID_NUM']=='G28_25_H1']['ID_NUM']
HumTem_data_abnormal[HumTem_data_abnormal['ID_NUM']=='G28_25_H1']['ID_NUM']
HumTem_data_agg[HumTem_data_agg['ID_NUM']=='G28_25_H1']['ID_NUM']
wide_df[wide_df['ID_NUM']=='G28_25_H1']['ID_NUM']



all_HumTem_data1[all_HumTem_data1['ID_NUM']=='G02_59_H1']['ID_NUM']
all_HumTem_data2[all_HumTem_data2['ID_NUM']=='G01_60_H1']['ID_NUM']




###############拼接按日龄的温度和死淘数据

HumTem_data_agg1=pd.read_csv('./data/data_cleaned/HumTem_data_agg1.csv',encoding='gbk')
HumTem_data_agg2=pd.read_csv('./data/data_cleaned/HumTem_data_agg2.csv',encoding='gbk')

# HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]

HumTem_data_agg2 = HumTem_data_agg2[~HumTem_data_agg2['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]
HumTem_data_agg=pd.concat([HumTem_data_agg1,HumTem_data_agg2],ignore_index=True)


all_dead_data1=pd.read_csv('./data/data_cleaned/all_dead_data.csv',encoding='gbk')

all_dead_data2=pd.read_csv('./data/data_cleaned/all_dead_data2.csv',encoding='gbk')

all_dead_data=pd.concat([all_dead_data1,all_dead_data2],ignore_index=True)

# HumTem_data_agg['ID_NUM'].drop_duplicates()


all_dead_data['ID_NUM'] = all_dead_data['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
all_dead_data['ID_NUM'] = all_dead_data['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)

all_dead_HumTem_byage=pd.merge(all_dead_data,HumTem_data_agg,left_on=['ID_NUM','Age'],right_on=['ID_NUM','日龄'],how='inner')



all_dead_HumTem_byage.to_csv('./data/data_cleaned/dead_HumTem_byage.csv',index=False,encoding='gbk')




