import pandas as pd

baseinfo=pd.read_csv('./data/data_cleaned/baseinfo.csv',encoding='gbk')
marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')


columns = [
    'ID_NUM',                    #  主键
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
    'feed_cost',                # 饲料成本（元）
    'electricity_cost',         # 用电费用（元）
    'gas_cost',                 # 燃气费用（元）
    'depreciation_cost',        # 折旧费（元）
    'chick_cost',               # 雏鸡成本（元）
    'total_cost',               # 总成本（元）
    'cost_per_kg',              # 每公斤成本（元）
    'revenue',                  # 毛鸡销售收入（元）
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
    # 'id_no'
]

marketingdata['ID_NUM']=marketingdata['farm_name']+'_'+marketingdata['batch'].astype(int).astype(str)+'_'+marketingdata['house']
marketingdata=marketingdata.drop(columns=['farm_name','batch','house'],axis=1)
baseinfo['ID_NUM']=baseinfo['FarmName']+'_'+baseinfo['Batch'].astype(int).astype(str)+'_'+baseinfo['HouseNo']
baseinfo=baseinfo.drop(columns=['FarmName','Batch','HouseNo','id_no'],axis=1)

all_info_df=pd.merge(baseinfo,marketingdata[columns],how='left',on='ID_NUM')

all_info_df.columns.to_list()
all_info_df['Mortality_rate']=1-all_info_df['livability_pct']

all_info_df.info()

# all_info_df.to_csv('./data/data_cleaned/allinfo_dead0430.csv', index=False,encoding='gbk')



# all_info_df2[all_info_df2['Mortality_rate'] == '']



# 数据清洗
cost_columns=[
'feed_cost',                # 饲料成本（元）
'electricity_cost',         # 用电费用（元）
'gas_cost',                 # 燃气费用（元）
'depreciation_cost',        # 折旧费（元）
'chick_cost',               # 雏鸡成本（元）
'total_cost',               # 总成本（元）
'cost_per_kg',              # 每公斤成本（元）

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
]
for col in cost_columns:
   all_info_df[col]=all_info_df[col].apply(lambda x:None if x<0 else x)

# allinfo_dead[allinfo_dead['electricity_cost']<0]['electricity_cost']

# allinfo_dead['electricity_cost'].notna().sum()


###############33 HEage处理
import numpy as np
if 'HEAge' in all_info_df.columns:
    def heage_to_int(x):
        # 去掉W，取最后两个数字
        if pd.isnull(x):
            return np.nan
        x = str(x).replace('W', '').replace('w', '')
        # 只取最后两位数字
        digits = ''.join([c for c in x if c.isdigit()])
        return int(digits[-2:]) if len(digits) >= 2 else (int(digits) if digits else np.nan)
    all_info_df['HEAge'] = all_info_df['HEAge'].apply(heage_to_int)

all_info_df['HEAge'].value_counts()

###去除唯一值变量

# data_detect=pd.read_csv('./xyy/data_detect.csv',encoding='gbk')
import toad
data_detect = toad.detector.detect(all_info_df)
data_detect=data_detect.reset_index(drop=False)
unique_columns=list(data_detect[data_detect['unique']==1]['index'])

all_info_df=all_info_df.drop(columns=unique_columns,axis=1)

all_info_df.to_csv('./data/data_cleaned/allinfo_dead0430.csv', index=False,encoding='gbk')

all_info_df.columns.to_list()