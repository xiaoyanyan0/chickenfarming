import pandas as pd
allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead.csv',encoding='gbk')
HumTem_data_agg=pd.read_csv('./data/data_cleaned/HumTem_data_agg.csv',encoding='gbk')

allinfo_dead[allinfo_dead['ID_NUM'].str.startswith('G04')]['ID_NUM']

allinfo_dead['ID_NUM'].str.startswith('G1B')

allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 只有 G31_62匹不上
HumTem_data_t=pd.merge(HumTem_data_agg,allinfo_dead[['ID_NUM','age_days']],how='left',on=['ID_NUM'])

HumTem_data_t['age_days'].min()
HumTem_data_t['age_days'].notna().sum()

HumTem_data_t[HumTem_data_t['age_days'].isna()]['ID_NUM'].drop_duplicates()

HumTem_data_normal=HumTem_data_t[HumTem_data_t['Age']<HumTem_data_t['age_days']]
HumTem_data_abnormal=HumTem_data_t[HumTem_data_t['Age']>=HumTem_data_t['age_days']]


HumTem_data_normal['Age'].max()

import matplotlib.pyplot as plt
HumTem_data_normal.groupby(['Age'])['AvgTemperature'].mean().plot()
plt.show()


HumTem_data_normal=HumTem_data_normal.drop('age_days',axis=1)

HumTem_data_normal[['ID_NUM','Age']].drop_duplicates()

wide_df = HumTem_data_normal.pivot(index='ID_NUM', columns='Age')

HumTem_data_normal['ID_NUM'].drop_duplicates()


# 重置列名和索引
wide_df.columns = ['_'.join(map(str, (col[0], col[1]))) for col in wide_df.columns.values]
wide_df = wide_df.reset_index()

wide_df.columns

wide_df.to_csv('./data/data_cleaned/wide_df.csv', index=False,encoding='gbk')

##基本信息等拼接

allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead.csv',encoding='gbk')
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)


all_info_temdata=pd.merge(allinfo_dead,wide_df,on='ID_NUM',how='left')

all_info_temdata.info()
all_info_temdata['ID_NUM'].notna().sum()
wide_df['ID_NUM']
import toad
data_detect = toad.detector.detect(all_info_temdata)
data_detect=data_detect.reset_index(drop=False)


all_info_temdata.to_csv('./data/data_cleaned/all_info_temdata.csv', index=False,encoding='gbk')



HumTem_data_normal[HumTem_data_normal['ID_NUM']=='G1A_62_H1']['ID_NUM']
HumTem_data_abnormal[HumTem_data_abnormal['ID_NUM']=='G1A_62_H1']['ID_NUM']
HumTem_data_agg[HumTem_data_agg['ID_NUM']=='G1A_62_H1']['ID_NUM']






