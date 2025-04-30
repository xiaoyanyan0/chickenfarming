import pandas as pd
allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')
HumTem_data_agg1=pd.read_csv('./data/data_cleaned/HumTem_data_agg1.csv',encoding='gbk')
HumTem_data_agg2=pd.read_csv('./data/data_cleaned/HumTem_data_agg2.csv',encoding='gbk')

# HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]

HumTem_data_agg2 = HumTem_data_agg2[~HumTem_data_agg2['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]
HumTem_data_agg=pd.concat([HumTem_data_agg1,HumTem_data_agg2],ignore_index=True)

allinfo_dead['ID_NUM'].drop_duplicates().count()
HumTem_data_agg['ID_NUM'].drop_duplicates().count()


#日报中的农场名字和文件名字中的对应不上
HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith('G04')]['ID_NUM']

allinfo_dead[allinfo_dead['ID_NUM'].str.startswith('G1A')]['ID_NUM']

allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 只有 G31_62匹不上
HumTem_data_t=pd.merge(HumTem_data_agg,allinfo_dead[['ID_NUM','age_days']],how='left',on=['ID_NUM'])

HumTem_data_t['age_days'].min()
HumTem_data_t['age_days'].notna().sum()

HumTem_data_t[HumTem_data_t['age_days'].isna()]['ID_NUM'].drop_duplicates()

HumTem_data_normal=HumTem_data_t[HumTem_data_t['日龄']<HumTem_data_t['age_days']]
HumTem_data_abnormal=HumTem_data_t[HumTem_data_t['日龄']>=HumTem_data_t['age_days']]


HumTem_data_normal['日龄'].max()

import matplotlib.pyplot as plt
HumTem_data_normal.groupby(['Age'])['AvgTemperature'].mean().plot()
plt.show()


HumTem_data_normal=HumTem_data_normal.drop('age_days',axis=1)
HumTem_data_normal=HumTem_data_normal.drop_duplicates()

HumTem_data_normal[['ID_NUM','日龄']].drop_duplicates()
HumTem_data_normal[['ID_NUM']].drop_duplicates()

# 计算每个组合出现的次数
# counts = HumTem_data_agg.groupby(['ID_NUM', '日龄']).size().reset_index(name='计数')

# 筛选出重复的组合
# repeated_pairs = counts[counts['计数'] > 1]
# repeated_pairs['ID_NUM'].drop_duplicates()
# print(repeated_pairs)
HumTem_data_normal.columns.to_list()

keep_cols=['ID_NUM', '日龄', '温度1-平均_mean', 
'温度2-平均_mean', '温度3-平均_mean',
'温度4-平均_mean',  '温度5-平均_mean', 
'温度6-平均_mean',  '湿度内部平均_mean', '最高温度', '最低温度',
 '平均温度', '每日温差', '平均温度变化率', '最高温度变化率', '最低温度变化率']


wide_df = HumTem_data_normal[keep_cols].pivot(index='ID_NUM', columns='日龄')

# 重置列名和索引
wide_df.columns = ['_'.join(map(str, (col[0], col[1]))) for col in wide_df.columns.values]
wide_df = wide_df.reset_index()

wide_df.columns.to_list()

wide_df.to_csv('./data/data_cleaned/wide_df_0430.csv', index=False,encoding='gbk')

##基本信息等拼接
# wide_df['ID_NUM']


allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 只有 G31_63,G28_22,匹不上，G27_24_H1，G27_25_H1为空

wide_df['ID_NUM_copy']=wide_df['ID_NUM']
all_info_temdata=pd.merge(allinfo_dead,wide_df,on='ID_NUM',how='left')

all_info_temdata.info()
all_info_temdata['ID_NUM_copy'].notna().sum()
all_info_temdata[all_info_temdata['ID_NUM_copy'].isna()]['ID_NUM'].str[:6].unique()
all_info_temdata[all_info_temdata['ID_NUM_copy'].isna()]['ID_NUM'].unique()
wide_df['ID_NUM']

all_info_temdata2=all_info_temdata.drop('ID_NUM_copy',axis=1)
import toad
data_detect = toad.detector.detect(all_info_temdata2)
data_detect=data_detect.reset_index(drop=False)


all_info_temdata2.to_csv('./data/data_cleaned/all_info_temdata0430.csv',index=False,encoding='gbk')

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

