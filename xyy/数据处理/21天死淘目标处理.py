import pandas as pd

all_dead_data1=pd.read_csv('./data/data_cleaned/all_dead_data.csv',encoding='gbk')

all_dead_data2=pd.read_csv('./data/data_cleaned/all_dead_data2.csv',encoding='gbk')

all_dead_data=pd.concat([all_dead_data1,all_dead_data2],ignore_index=True)

all_dead_data['ID_NUM'].drop_duplicates()

all_dead_data_21=all_dead_data[all_dead_data['Age']>=21][['ID_NUM','Mortality_rate']].groupby(by=['ID_NUM']).sum()


all_dead_data_21=all_dead_data_21.reset_index()
all_dead_data.columns
all_dead_data_21['ID_NUM'] = all_dead_data_21['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
all_dead_data_21['ID_NUM'] = all_dead_data_21['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 和其他信息拼起来

all_info_temdata=pd.read_csv('./data/data_cleaned/all_info_temdata0430.csv',encoding='gbk')

all_info_temdatanew=pd.merge(all_info_temdata,all_dead_data_21,on='ID_NUM',how='left')
all_info_temdatanew=all_info_temdatanew.rename({'Mortality_rate_y':'Mortality_rate_21'},axis=1)

# all_info_temdata['ID_NUM']

all_info_temdatanew.to_csv('./data/data_cleaned/all_info_temdata0430.csv',index=False,encoding='gbk')


