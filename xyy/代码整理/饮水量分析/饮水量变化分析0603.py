import pandas as pd
import toad
import numpy as np
df1=pd.read_csv('./data/data_cleaned/dead_HumTem_byage0603_2.csv',encoding='gbk')
df2=pd.read_csv('./data/data_cleaned/dead_HumTem_byage0604_1.csv',encoding='gbk')

df1[df1['ID_NUM']=='GTF_72_H3'][['ID_NUM','Age','Mortality_rate']]


# 提取饮水相关特征
cols = [i for i in df2.columns.to_list() 
        if ('water_per_diff' in i) or ('water_per_shift_diff' in i) ]
keep_cols=['ID_NUM','Age','Mortality_rate','season']+cols
df2.columns.to_list()
df=pd.concat([df1,df2],ignore_index=True)
df['Age_diff']=df.groupby('ID_NUM')['Age'].diff()

data=df[(df['Age'] >= 10) & (df['Age'] <= 35) & (df['Age_diff']==1) ]

# data['Age'].value_counts()
# 计算每组前20%
data['Mortality_flg'] = data.groupby('season')['Mortality_rate'].transform(
    lambda x: (x >= x.quantile(0.8)).astype(int)
)
# data.groupby(['season','Age'])['ID_NUM'].count()
data.groupby('season')['Mortality_flg'].value_counts()
data.columns.to_list()
data['water_per_shift_diff'].describe()
# data[data['ID_NUM']=='G30_66_H10'][['ID_NUM','Age','water_per','water_per_diff','water_per_shift','water_per_shift_diff','water_std_diff']]


# max_idx = data['water_per_shift_diff'].idxmin()
# max_id_first = data.loc[max_idx, ['ID_NUM','Age']]
def map_age_interval(age):
    if 10 <= age <= 12:
        return "10 - 12日龄"
    elif 13 <= age <= 15:
        return "13 - 15日龄"
    elif 16 <= age <= 18:
        return "16 - 18日龄"
    elif 19 <= age <= 21:
        return "19 - 21日龄"
    elif 22 <= age <= 24:
        return "22 - 24日龄"
    elif 25 <= age <= 27:
        return "25 - 27日龄"
    elif 28 <= age <= 30:
        return "28 - 30日龄"
    elif 31 <= age <= 33:
        return "31 - 33日龄"
    elif 34 <= age <= 35:
        return "34 - 35日龄"
data["Age_interval"] = data["Age"].apply(map_age_interval)
# quality_df=toad.quality(data[keep_cols],target='Mortality_flg',iv_only=True)
from optbinning import BinningProcess
import warnings
warnings.filterwarnings("ignore")
def feature_binning(top_importantcol, object_columns,Age_interval, X, y, max_n_bins=6):
    """
    对特征进行分箱分析，计算统计信息并可视化
    
    Parameters:
    -----------
    top_importantcol: list
        需要分箱的特征列表（不包含Mortality_rate列）
    object_columns: list
        分类特征列名列表
    X: DataFrame
        特征数据（包含Mortality_rate列）
    y: Series
        目标变量（如死亡标志0/1）
    file_prefix: str
        输出文件前缀
    max_n_bins: int, optional (default=5)
        最大分箱数量限制
    
    Returns:
    --------
    binning_sum: DataFrame
        分箱汇总统计信息
    bin_table: DataFrame
        所有特征的分箱明细表（含Mortality_rate均值）
    """
    # 0. 确保Mortality_rate列存在
    if 'Mortality_rate' not in X.columns:
        raise ValueError("X中必须包含'Mortality_rate'列")
    
    # 1. 识别分类特征
    cat_f = [col for col in top_importantcol if col in object_columns]
    
    # 2. 设置分箱选择标准
    selection_criteria = {
        "gini": {"min": 0.15, "max": 1}
    }
    
    # 3. 初始化分箱过程
    binning_process = BinningProcess(
        top_importantcol,
        categorical_variables=cat_f,
        selection_criteria=selection_criteria,
        max_n_bins=max_n_bins  # 在这里也设置最大分箱数
    )
    
    # 其余代码保持不变...
    # 4. 拟合分箱模型
    binning_process.fit(X[top_importantcol], y)
    
    # 5. 获取分箱汇总信息
    binning_sum = binning_process.summary()
    binning_sum = binning_sum.sort_values(by='gini', ascending=False)
    
    # 6. 构建完整分箱明细表
    bin_table = pd.DataFrame()
    for col in top_importantcol:
        try:
            optb = binning_process.get_binned_variable(col)
            temp = optb.binning_table.build()
            temp['feature'] = col  # 添加特征名列
            print(f"分箱阈值: {optb.splits}")
            X_binned = optb.transform(X[[col]],metric='bins').squeeze()  # 获取分箱结果
            bin_mortality = X.groupby(X_binned)[['Mortality_rate']].mean()
            bin_mortality=bin_mortality.reset_index(drop=False)
            temp2 = pd.merge(temp, bin_mortality, left_on='Bin', right_on='index', how='left').drop('index', axis=1)
            temp2['Age_interval']=Age_interval
            bin_table = pd.concat([bin_table, temp2])
            print(f"\n=== 死淘具体情况: {col} ===")
            print(bin_mortality)
             # 打印分箱详情
            print(f"\n=== 变量: {col} ===")
            display_cols = ['Bin', 'Count', 'Count (%)', 'Event rate', 'Mortality_rate']
            print(temp2[display_cols])
            print('-'*50)
            
        except Exception as e:
            print(f"处理变量 {col} 时出错: {str(e)}")
            continue
    
    return binning_sum, bin_table


def analyze_bins_by_season_and_age(data, output_dir='.\\xyy\\死淘分析\\output', max_n_bins=6):
    """
    按季节(season)和日龄段(Age_interval)对特征进行分箱分析
    
    Parameters:
    -----------
    data: DataFrame
        包含所有特征和目标变量的数据集
    output_dir: str
        输出文件目录
    max_n_bins: int, optional (default=6)
        最大分箱数量限制
    
    Returns:
    --------
    dict
        包含所有季节和日龄段的分箱结果和分布统计
    """
    # 确保必要的列存在
    required_cols = ['season', 'Age_interval', 'Mortality_flg']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"数据集中必须包含{col}列")
    
    # 识别数值型和分类型特征
    numeric_columns = []
    object_columns = []
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            object_columns.append(column)
    
    # 提取饮水相关特征
    cols = [i for i in data.columns.to_list() 
            if ('water_per_diff' in i) or ('water_per_shift_diff' in i) ]
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 结果存储字典
    results = {}
    
    # 获取所有季节和日龄段
    # seasons = data['season'].unique()
    seasons=['spring']
    age_intervals =  data[data['season'] == 'spring']['Age_interval'].unique()
    
    print(f"开始分析 {len(seasons)} 个季节和 {len(age_intervals)} 个日龄段的组合...")
    season_results = {}
    for season in seasons:
        season_data = data[data['season'] == season].copy()
       
        bin_table_all=pd.DataFrame()
        fx_all=pd.DataFrame()
        for age_interval in age_intervals:
            # 筛选日龄段数据
            age_data = season_data[season_data['Age_interval'] == age_interval].copy()
            
            if age_data.empty:
                print(f"警告: {season}季节-{age_interval}日龄段没有数据，跳过分析")
                continue
                
            # 准备特征和目标变量
            X = age_data
            y = age_data['Mortality_flg']
            top_importantcol = cols
            
            print(f"\n正在分析 {season} 季节 - {age_interval} 日龄段...")
            
            try:
                # 执行分箱分析（直接调用原有的feature_binning函数）
                binning_sum, bin_table = feature_binning(
                    top_importantcol=top_importantcol,
                    object_columns=object_columns,
                    Age_interval=age_interval,
                    X=X,
                    y=y,
                    max_n_bins=max_n_bins
                )
                
                # 重命名死亡率列并保存结果
                bin_table = bin_table.rename({'Mortality_rate': 'MORTALITY_RATE'}, axis=1)
                 # 计算并保存分布统计
                fx = X[top_importantcol].describe().round(2)
                fx = fx.reset_index(drop=False)
                fx['Age_interval']=age_interval

                bin_table_all=pd.concat([bin_table_all,bin_table],ignore_index=True)
                fx_all=pd.concat([fx_all,fx],ignore_index=True)
                print(f"✓ 完成{season}季节-{age_interval}日龄段的分箱分析和分布统计")
                
            except Exception as e:
                print(f"✗ 处理{season}季节-{age_interval}日龄段时出错: {str(e)}")
                continue
        
        bin_table_path = f"{output_dir}/{season}_饮水差值分箱.csv"
        bin_table_all.to_csv(bin_table_path, index=False, encoding='gbk')
                
               
        fx_path = f"{output_dir}/{season}_饮水差值分布.csv"
        fx_all.to_csv(fx_path, index=False, encoding='gbk')
        season_results[season] = {
                    'bin_table': bin_table_all,
                    'distribution': fx_all,
                } 
    print(f"\n分析完成！共处理 {len(seasons)} 个季节，每个季节平均 {len(age_intervals)} 个日龄段")
    print(f"结果已保存到目录: {output_dir}")
    
    return season_results

season_results=analyze_bins_by_season_and_age(data, output_dir='.\\xyy\\死淘分析\\output', max_n_bins=6)
data[(data['season']=='autumn') & (data['Age_interval']=='10 - 12日龄') & (data['water_per_diff']< -11.15)]['Mortality_rate'].count()
data[(data['season']=='spring')].groupby('Age_interval')['ID_NUM'].count()
data[(data['season']=='spring') & (data['Age_interval']=='10 - 12日龄')]
data[(data['season']=='spring')]['Age_interval'].unique()
# data.groupby(['Age','season'])['water_per'].mean()



age_intervals = data['Age_interval'].unique()


numeric_columns = []
object_columns = []
for column in data.columns:
    if np.issubdtype(data[column].dtype, np.number):
        numeric_columns.append(column)
    else:
        object_columns.append(column)
# data['Mortality_rate']
cols=[i for i in data.columns.to_list() if ('water_per_diff' in i) or ('water_per_shift_diff' in i) or ('water_per_rate_diff' in i) ]
keep_cols=['Mortality_flg']+cols
data1=data[data['season']=='winter'][keep_cols]
X=data1[(data1['Age']>0) & (data1['Age']<=35)]
y=data1[(data1['Age']>0) & (data1['Age']<=35)]['Mortality_flg']
# rate_cols=[ i for i in X.columns if '变化率' in i]
# X[rate_cols]=X[rate_cols]*100
X['Mortality_flg'].value_counts()
# seasonal_dfs['winter']['Mortality_flg'].value_counts()
# top_importantcol=['W3_15-17天_外部-平均_MEAN']
top_importantcol=cols

binning_sum, bin_table=feature_binning(top_importantcol, object_columns, X, y,file_prefix="winter")
bin_table=bin_table.rename({'Mortality_rate':'MORTALITY_RATE'},axis=1)
bin_table.to_csv('.\\xyy\\死淘分析\\output\\冬季饮水差值分箱.csv', index=False, encoding='gbk')

fx=X[top_importantcol].describe().round(2)
fx=fx.reset_index(drop=False)
print(fx.shape)
fx.to_csv('.\\xyy\\死淘分析\\output\\冬季饮水差值分布.csv', index=False, encoding='gbk')