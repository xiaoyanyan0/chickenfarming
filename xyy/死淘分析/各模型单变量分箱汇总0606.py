#########加上内外温差湿差等数据
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

from matplotlib import cm
import optbinning
from optbinning import BinningProcess
import warnings

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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
all_info_temdata1=pd.read_csv('./data/data_cleaned/all_info_temdata0602.csv',encoding='gbk')
all_info_temdata2=pd.read_csv('./data/data_cleaned/all_info_temdata0602_2.csv',encoding='gbk')

cols=[i for i in all_info_temdata2.columns.to_list() if '平均变化' in i]+['ID_NUM']
all_info_temdata=pd.merge(all_info_temdata1,all_info_temdata2[cols],on='ID_NUM',how='left')
all_info_temdata1.shape
all_info_temdata2.shape
all_info_temdata.shape
all_info_temdata1.columns.to_list()
# all_info_temdata['season']
# marketingdata=pd.read_csv('./data/data_cleaned/marketingdata.csv',encoding='gbk')
import warnings
warnings.filterwarnings("ignore")
# all_info_temdata
col=[a.upper()for a in all_info_temdata.columns.to_list()]
all_info_temdata.columns=col

df=all_info_temdata.set_index('ID_NUM')

# all_info_temdata['MORTALITY_RATE_X']
# df=df.rename({'MORTALITY_RATE_X':'MORTALITY_RATE'},axis=1)
# 查看各个月份的分布
# 将日期列转换为 datetime 类型

df.columns.to_list()
#筛除0占比超过阈值特征
drop_columns_Mortality=['Dead','Swollen_Head','Weak','Navel_Disease','Stick_Anus', 'Lame_Paralysis','livability_pct'
                        ,'yield_per_m2'
                        ]
drop_columns_Mortality=[a.upper()for a in drop_columns_Mortality]
drop_columns_Mortality=[i for i in df.columns.to_list() if i in drop_columns_Mortality]

marketingdata_columns = [
    'house',                    # 栋舍 House
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
    # 'eef',                      # 欧洲指数 EEF
    'revenue',                  # 毛鸡销售收入（元'）
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
    'chick_cost',               # 雏鸡成本（元）
    'total_cost',               # 总成本（元）
    'cost_per_kg',              # 每公斤成本（元）
    'feed_cost',                # 饲料成本（元）
]
marketingdata_columns=[a.upper()for a in marketingdata_columns]
drop_columns_marketingdata=[i for i in df.columns.to_list() if i in marketingdata_columns]

#把细分的死淘数据和出栏数据指标去掉
df2=df.drop(columns=drop_columns_marketingdata+drop_columns_Mortality,axis=1)

df2.shape



# df2.columns.to_list()
# date_columns = ['DOCDATE', 'ESTIMATEDSLAUGHTERDATE ', 'HARVESTSTATUS']
# for col in date_columns:
#     df2[col] = pd.to_datetime(df2[col])
#     df2[f'{col}_month'] = df2[col].dt.month
#     df2[f'{col}_month']=df2[f'{col}_month'].astype(str)


# df2=df2.drop(date_columns,axis=1)

# month_to_season = {
#     '12': 'winter', '1': 'winter', '2': 'winter',
#     '3': 'spring', '4': 'spring',
#     '5': 'summer', '6': 'summer', '7': 'summer', '8': 'summer', '9': 'summer',
#     '10': 'autumn', '11': 'autumn'
# }

# # 根据映射字典创建季节列
# df2['season'] = df2['HARVESTSTATUS_month'].map(month_to_season)
df2['Mortality_flg']= df2.groupby('SEASON')['MORTALITY_RATE'].transform(
    lambda x: (x >= x.quantile(0.8)).astype(int)
)
df2['EEF_flg']= df2.groupby('SEASON')['EEF'].transform(
    lambda x: (x >= x.quantile(0.8)).astype(int)
)


from optbinning import BinningProcess
import warnings
warnings.filterwarnings("ignore")
# 死淘
def feature_binning(top_importantcol, object_columns, X, y, target='MORTALITY_RATE', max_n_bins=6):
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
    if target not in X.columns:
        raise ValueError("X中必须包含{target}列")
    
    # 1. 识别分类特征
    cat_f = [col for col in top_importantcol if col in object_columns]
    
    # 2. 设置分箱选择标准
    selection_criteria = {
        "gini": {"min": 0.05, "max": 1},
    }
    
    # 3. 初始化分箱过程
    binning_process = BinningProcess(
        top_importantcol,
        categorical_variables=cat_f,
        selection_criteria=selection_criteria,
        max_n_bins=max_n_bins,  # 在这里也设置最大分箱数
        min_bin_size=0.1,
        min_n_bins=2
        # min_event_rate_diff=0.01
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
            bin_mortality = X.groupby(X_binned)[[target]].mean()
            bin_mortality=bin_mortality.reset_index(drop=False)
            temp2 = pd.merge(temp, bin_mortality, left_on='Bin', right_on='index', how='left').drop('index', axis=1)
            bin_table = pd.concat([bin_table, temp2])
            print(f"\n=== {target}具体情况: {col} ===")
            print(bin_mortality)
             # 打印分箱详情
            print(f"\n=== 变量: {col} ===")
            display_cols = ['Bin', 'Count', 'Count (%)', 'Event rate', target]
            print(temp2[display_cols])
            print('-'*50)
            
        except Exception as e:
            print(f"处理变量 {col} 时出错: {str(e)}")
            continue
    
    return binning_sum, bin_table

# df2[df2['season'] == 'winter']['EEF_flg']
def process_seasonal_data(
    df, 
    season, 
    target, 
    target_flg, 
    output_path='.\\xyy\\模型分析_new\\'
):
    """
    季节性数据处理及分箱分析函数
    
    参数:
    df (DataFrame): 原始数据集
    season (str): 季节标识（如'winter'）
    target (str): 目标指标列名（如'MORTALITY_RATE'）
    target_flg (str): 目标标签列名（如'Mortality_flg'）
    top_importantcol (list): 关键特征列名列表
    output_path (str): 输出文件路径，默认 './xyy/模型分析/output/'
    """
    # 1. 数据类型分类
    numeric_columns = []
    object_columns = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            object_columns.append(column)
    
    # 2. 按季节筛选数据
    try:
        data1 = df[df['season'] == season].copy()
        if data1.empty:
            raise ValueError(f"Season '{season}' not found in dataset")
    except KeyError:
        raise KeyError("Column 'season' not found in DataFrame")
    
    # 3. 特征与标签准备
    X = data1  # 排除目标列
    y = data1[target_flg]
    
    # 4. 数值列处理（可选：根据需求取消注释）
    # rate_cols = [i for i in X.columns if '变化率' in i]
    # if rate_cols:
    #     X[rate_cols] = X[rate_cols] * 100
    # top_importantcol = [col for col in df.columns if any(keyword in col for keyword in importance_dict[season][target])]
    top_importantcol=['探头温度标准差_24']
    X[top_importantcol]=X[top_importantcol].round(1)
    # 5. 特征分箱分析
    try:
        binning_sum, bin_table = feature_binning(
            top_importantcol, 
            object_columns, 
            X, 
            y, 
            target=target
        )
    except Exception as e:
        raise RuntimeError(f"Feature binning failed: {str(e)}")
    
    # 6. 保存分箱结果
    try:
        bin_table.to_csv(
            f"{output_path}{season}\\{season}_{target}_分箱1.csv", 
            index=False, 
            encoding='gbk'
        )
    except IOError as e:
        raise IOError(f"Failed to save bin table: {str(e)}")
    
    # 7. 特征分布统计
    try:
        fx = X[top_importantcol].describe().round(2).reset_index(drop=False)
        fx.to_csv(
            f"{output_path}{season}\\{season}_{target}_分布1.csv", 
            index=False, 
            encoding='gbk'
        )
        print(f"特征分布文件已保存，维度: {fx.shape}")
    except KeyError as e:
        raise KeyError(f"Column {e} not found in feature columns")
    
    return {
        'binning_summary': binning_sum,
        'bin_table': bin_table,
        'feature_stats': fx
    }


import pandas as pd
import numpy as np
import re

class AdvancedBinningOptimizer:
    def __init__(self, target="MORTALITY_RATE", max_eef_diff=0.08, min_bin_size=0.05, min_iv_loss=0.03):
        self.target = target  # 新增参数，可以是 "MORTALITY_RATE" 或 "EEF"
        self.max_eef_diff = max_eef_diff
        self.min_bin_size = min_bin_size
        self.min_iv_loss = min_iv_loss

    def _parse_bin_boundary(self, bin_str):
        # 边界解析函数不变
        if pd.isna(bin_str) or bin_str in ['Special', 'Missing']:
            return np.nan, np.nan
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", bin_str)
        if len(numbers) == 1:
            return (-np.inf, float(numbers[0])) if '(' in bin_str else (float(numbers[0]), np.inf)
        return (float(numbers[0]), float(numbers[1]))

    def _preprocess(self, df):
        # 数据预处理函数（将 target 列转为 float）
        df = df[df['Bin'].apply(lambda x: bool(re.search(r'\d', str(x))))].copy()
        boundaries = df['Bin'].apply(self._parse_bin_boundary)
        df['lower'] = [b[0] for b in boundaries]
        df['upper'] = [b[1] for b in boundaries]
        df = df.sort_values(by='lower').reset_index(drop=True)
        df['Count'] = df['Count'].astype(int)
        df[self.target] = df[self.target].astype(float)  # 动态使用 target 列
        df['Count_pct'] = df['Count (%)'].astype(float)
        df['Event_rate'] = df['Event'].astype(int) / df['Count']
        df['WoE'] = df['WoE'].astype(float)
        df['IV'] = df['IV'].astype(float)
        return df

    def _find_merge_candidates(self, df):
        merge_groups = []
        current_group = [0]
        for i in range(1, len(df)):
            prev_val = df.loc[current_group[-1], self.target]  # 动态使用 target 列
            curr_val = df.loc[i, self.target]
            val_diff = abs(curr_val - prev_val)
            size_condition = (df.loc[i, 'Count_pct'] < self.min_bin_size) | (
                    df.loc[current_group[-1], 'Count_pct'] < self.min_bin_size)
            trend_condition = val_diff < self.max_eef_diff
            if size_condition or trend_condition:
                current_group.append(i)
            else:
                merge_groups.append(current_group)
                current_group = [i]
        if current_group:
            merge_groups.append(current_group)
        return merge_groups

    def _merge_operation(self, df, merge_groups):
        total_events = df['Event'].sum()
        total_non_events = df['Non-event'].sum()
        merged_data = []
        for group in merge_groups:
            subset = df.iloc[group]
            new_lower = subset['lower'].min()
            new_upper = subset['upper'].max()
            bin_name = f"[{new_lower:.2f}, inf)" if np.isinf(new_upper) else f"[{new_lower:.2f}, {new_upper:.2f})"
            total_count = subset['Count'].sum()
            merged_events = subset['Event'].sum()
            merged_non_events = subset['Non-event'].sum()

            good_pct = merged_non_events / total_non_events if total_non_events != 0 else 0
            bad_pct = merged_events / total_events if total_events != 0 else 0
            woe = np.log(good_pct / bad_pct) if (good_pct != 0 and bad_pct != 0) else 0
            iv = (good_pct - bad_pct) * woe

            merged_record = {
                'Bin': bin_name,
                'Count': total_count,
                'Non-event': merged_non_events,
                'Event': merged_events,
                'Event rate': merged_events / total_count,
                'WoE': woe,
                'IV': iv,
                'feature': subset['feature'].iloc[0],
                self.target: np.average(subset[self.target], weights=subset['Count']),  # 动态使用 target 列
                'Count (%)': total_count / df['Count'].sum(),
                'JS': subset['JS'].sum() if 'JS' in subset.columns else 0
            }
            merged_data.append(merged_record)
        return pd.DataFrame(merged_data)

    def optimize(self, raw_data):
        optimized_results = []
        features = raw_data['feature'].unique()
        for feat in features:
            feat_data = raw_data[raw_data['feature'] == feat].copy()
            cleaned_data = self._preprocess(feat_data)
            if len(cleaned_data) < 3:
                optimized_results.append(cleaned_data)
                continue

            merge_candidates = self._find_merge_candidates(cleaned_data)
            merged_data = self._merge_operation(cleaned_data, merge_candidates)

            original_iv = cleaned_data['IV'].sum()
            new_iv = merged_data['IV'].sum()
            if (original_iv - new_iv) <= self.min_iv_loss:
                optimized_feat_data = merged_data
            else:
                optimized_feat_data = cleaned_data

            summary_row = self._generate_summary_row(optimized_feat_data)
            optimized_feat_data = pd.concat([optimized_feat_data, summary_row], ignore_index=True)
            optimized_results.append(optimized_feat_data)

        original_columns = raw_data.columns.tolist()
        optimized_data = pd.concat(optimized_results, ignore_index=True)[original_columns]
        return optimized_data

    def _generate_summary_row(self, df):
        """生成特征汇总行（动态使用 target 列）"""
        if df.empty:
            return pd.DataFrame()

        feat_name = df['feature'].iloc[0]
        total_count = df['Count'].sum()
        total_non_event = df['Non-event'].sum()
        total_event = df['Event'].sum()
        total_iv = df['IV'].sum()
        avg_target = np.average(df[self.target], weights=df['Count'])  # 动态计算 target 加权平均

        summary = pd.DataFrame({
            'Bin': [''],
            'Count': [total_count],
            'Non-event': [total_non_event],
            'Event': [total_event],
            'Event rate': [total_event / total_count if total_count != 0 else 0],
            'WoE': [np.nan],
            'IV': [total_iv],
            'feature': [feat_name],
            self.target: [avg_target],  # 动态使用 target 列
            'Count (%)': [1.0],
            'JS': [df['JS'].sum()] if 'JS' in df.columns else [0]
        })
        return summary
    
# 画图代码
def get_temp_range(stats_file, var_name):
    """从统计文件获取变量温度范围"""
    try:
        stats_df = pd.read_csv(stats_file, encoding='gbk')
        match_cols = [col for col in stats_df.columns if var_name in col]
        if not match_cols:
            print(f"Warning: No matching column for {var_name}")
            return None, None
            
        min_val = stats_df.loc[stats_df['index'] == 'min', match_cols[0]].values[0]
        max_val = stats_df.loc[stats_df['index'] == 'max', match_cols[0]].values[0]
        # padding = (max_val - min_val) * 0.1
        return min_val, max_val 
        
    except Exception as e:
        print(f"Error getting temp range: {e}")
        return None, None

def parse_bin_data(df):
    """解析分箱数据"""
    processed = df.copy()
    processed['Bin'] = processed['Bin'].astype(str)  # 新增这行
    bin_ranges = []
    for bin_str in processed['Bin']:
        if bin_str in ['Special', 'Missing', '']:
            bin_ranges.append((np.nan, np.nan))
            continue
            
        clean_str = bin_str.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        parts = clean_str.split(', ')
        
        if len(parts) != 2:
            bin_ranges.append((np.nan, np.nan))
            continue
            
        lower = float(parts[0]) if parts[0] != '-inf' else -np.inf
        upper = float(parts[1]) if parts[1] != 'inf' else np.inf
        bin_ranges.append((lower, upper))
    
    processed[['parsed_lower', 'parsed_upper']] = bin_ranges
    processed = processed[~processed['Bin'].isin(['Special', 'Missing', ''])]
    
    if processed.empty:
        return pd.DataFrame(), []
    
    return processed, []
def plot_temp_mortality(df, season, x_min=None, x_max=None, feature_name=""):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        processed, _ = parse_bin_data(df)
        if processed.empty:
            print("Empty data, skip plotting")
            return
    except Exception as e:
        print(f"Error parsing data: {e}")
        return

    # 计算总IV值
    total_iv =  processed['IV'].iloc[:-1].sum() if 'IV' in processed.columns else 0

    # 使用传入的x_min和x_max替换-inf和inf
    if x_min is not None:
        processed['parsed_lower'] = processed['parsed_lower'].replace(-np.inf, x_min)
    if x_max is not None:
        processed['parsed_upper'] = processed['parsed_upper'].replace(np.inf, x_max)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 添加主标题和副标题，包含IV值
    plt.suptitle(f"变量与死淘率关系 - {feature_name}-{season}", fontsize=16, y=0.98)
    plt.title(f"变量差值: {x_min:.1f} 至 {x_max:.1f} | 总IV值: {total_iv:.4f}", 
              fontsize=12, pad=10)

    # 计算柱状图的位置和宽度
    processed['bar_x'] = processed['parsed_lower']
    processed['bar_width'] = processed['parsed_upper'] - processed['parsed_lower']
    processed['line_x'] = (processed['parsed_lower'] + processed['parsed_upper']) / 2

    colors = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(processed)))
    bars = ax1.bar(processed['bar_x'], processed['Count'], 
                  width=processed['bar_width'], color=colors,
                  alpha=0.8, align='edge', label='数量')

    ax1.set_xlabel('变量范围', fontsize=12)
    ax1.set_ylabel('数量', color='dimgray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    
    # 关键修改：直接使用所有数据点的parsed_lower和parsed_upper作为刻度
    all_ticks = np.unique(processed[['parsed_lower', 'parsed_upper']].values.flatten())
    valid_ticks = [tick for tick in all_ticks if not np.isinf(tick)]  # 过滤无穷值
    
    if valid_ticks:
        ax1.set_xticks(valid_ticks)
        ax1.set_xticklabels([f"{tick:.1f}" for tick in valid_ticks], rotation=45, ha='right')
    else:
        ax1.set_xticks([])  # 无有效刻度时清空
    
    ax1.set_xlim(x_min, x_max)  # 保留原范围控制，确保刻度在范围内显示
    ax1.xaxis.set_tick_params(which='major', length=5, width=1)  # 可选：调整刻度线样式

    ax2 = ax1.twinx()
    line, = ax2.plot(processed['line_x'], processed['MORTALITY_RATE'], 
                    color='crimson', marker='o', linestyle='-',
                    linewidth=2, markersize=7, label='死淘率')
    
    # 动态计算y轴范围（解决标注超出问题）
    y_vals = processed['MORTALITY_RATE']
    # 先计算数据本身的最小、最大值
    y_min_data = y_vals.min()
    y_max_data = y_vals.max()
    # 计算标注文字需要的额外空间，这里简单按数据范围的 10% 预留（可根据实际情况调整）
    y_range = y_max_data - y_min_data
    extra_space = y_range * 0.1  
    # 最终y轴范围
    ax2.set_ylim(y_min_data - extra_space, y_max_data + extra_space)
    
    ax2.set_ylabel('死淘率', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    # 标注死淘率（保留四位小数）
    for i in range(len(processed)):
        y_coord = y_vals.iloc[i]
        # 这里基于动态计算的y轴范围，判断标注位置，简单处理直接用原始值标注（因为范围已预留空间）
        ax2.text(processed['line_x'].iloc[i],
                 y_coord,
                 f"{y_coord:.5f}",  # 保留四位小数
                 ha='center', va='bottom', fontsize=9, color='crimson')
    
    fig.legend([bars, line], ['数量', '死淘率'], loc='upper right')
    return fig

def plot_all_variables(binned_file, stats_file, output_dir='.\\xyy\\模型分析_new\\',season='winter'):
    """绘制所有变量的图表"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(binned_file, encoding='gbk')
    
    for feature in df['feature'].unique():
        if pd.isna(feature):
            continue
            
        feature_df = df[df['feature'] == feature]
        x_min, x_max = get_temp_range(stats_file, feature)
        
        if x_min is None or x_max is None:
            print(f"Skipping {feature} due to missing temp range")
            continue
            
        # 提取有意义的变量名称（去除特殊字符）
        clean_feature_name = ''.join(c for c in feature if c.isalnum() or c in ['_', '-', ' '])
        
        fig = plot_temp_mortality(feature_df,season, x_min, x_max, feature_name=clean_feature_name)
        
        if fig:
            safe_name = feature.replace(' ', '_').replace('/', '_')
            filename = f"{season}_mortality_{safe_name}.png"
            output_path = os.path.join(output_dir,f'{season}\\', filename)
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved {output_path}")


def plot_temp_eef(df, x_min=None, x_max=None, feature_name=""):
    """绘制温度-EEF图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        processed, _ = parse_bin_data(df)
        if processed.empty:
            print("Empty data, skip plotting")
            return
    except Exception as e:
        print(f"Error parsing data: {e}")
        return

    # 计算总IV值
    total_iv = processed['IV'].iloc[:-1].sum() if 'IV' in processed.columns else 0

    # 使用传入的x_min和x_max替换-inf和inf
    if x_min is not None:
        processed['parsed_lower'] = processed['parsed_lower'].replace(-np.inf, x_min)
    if x_max is not None:
        processed['parsed_upper'] = processed['parsed_upper'].replace(np.inf, x_max)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 添加主标题和副标题，包含IV值
    plt.suptitle(f"变量与EEF关系 - {feature_name}", fontsize=16, y=0.98)
    plt.title(f"变量范围: {x_min:.1f} 至 {x_max:.1f}℃ | 总IV值: {total_iv:.4f}", 
              fontsize=12, pad=10)

    # 计算柱状图的位置和宽度
    processed['bar_x'] = processed['parsed_lower']
    processed['bar_width'] = processed['parsed_upper'] - processed['parsed_lower']
    processed['line_x'] = (processed['parsed_lower'] + processed['parsed_upper']) / 2

    colors = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(processed)))
    bars = ax1.bar(processed['bar_x'], processed['Count'], 
                  width=processed['bar_width'], color=colors,
                  alpha=0.8, align='edge', label='数量')

    ax1.set_xlabel('变量范围)', fontsize=12)
    ax1.set_ylabel('数量', color='dimgray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    
    # 关键修改：直接使用所有数据点的parsed_lower和parsed_upper作为刻度
    all_ticks = np.unique(processed[['parsed_lower', 'parsed_upper']].values.flatten())
    valid_ticks = [tick for tick in all_ticks if not np.isinf(tick)]  # 过滤无穷值
    
    if valid_ticks:
        ax1.set_xticks(valid_ticks)
        ax1.set_xticklabels([f"{tick:.1f}" for tick in valid_ticks], rotation=45, ha='right')
    else:
        ax1.set_xticks([])  # 无有效刻度时清空
    
    ax1.set_xlim(x_min, x_max)  # 保留原范围控制，确保刻度在范围内显示
    ax1.xaxis.set_tick_params(which='major', length=5, width=1)  # 可选：调整刻度线样式

    ax2 = ax1.twinx()
    line, = ax2.plot(processed['line_x'], processed['EEF'], 
                    color='crimson', marker='o', linestyle='-',
                    linewidth=2, markersize=7, label='EEF')
    ax2.set_ylabel('EEF', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    for i in range(len(processed)):
        ax2.text(processed['line_x'].iloc[i],
                processed['EEF'].iloc[i] + 0.001,
                f"{processed['EEF'].iloc[i]:.3f}",
                ha='center', va='bottom', fontsize=9, color='crimson')

    fig.legend([bars, line], ['数量', 'EEF'], loc='upper right')
    return fig
def plot_all_variables_eef(binned_file, stats_file, output_dir='.\\xyy\\模型分析_new\\',season='winter'):
    """绘制所有变量的图表"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(binned_file, encoding='gbk')
    
    for feature in df['feature'].unique():
        if pd.isna(feature):
            continue
            
        feature_df = df[df['feature'] == feature]
        x_min, x_max = get_temp_range(stats_file, feature)
        
        if x_min is None or x_max is None:
            print(f"Skipping {feature} due to missing temp range")
            continue
            
        # 提取有意义的变量名称（去除特殊字符）
        clean_feature_name = ''.join(c for c in feature if c.isalnum() or c in ['_', '-', ' '])
        
        fig = plot_temp_eef(feature_df, x_min, x_max, feature_name=clean_feature_name)
        
        if fig:
            safe_name = feature.replace(' ', '_').replace('/', '_')
            filename = f"{season}_eef_{safe_name}.png"
            output_path = os.path.join(output_dir,f'{season}\\', filename)
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved {output_path}")
# 使用示例
# Mortality_flg
importance_dict={
    "winter": {
        "MORTALITY_RATE": ["探头温度标准差", "鸡舍温度-平均_MEAN", "温度4-平均_RANGE"],
        "EEF": ["鸡舍温度-平均_MEAN", "湿度内部平均_MEAN", "探头温度标准差"]
    },
    "autumn": {
        "MORTALITY_RATE": ["温度3-平均变化", "探头温度标准差", "温度2-平均_RANGE"],
        "EEF": ["鸡舍温度-平均_MEAN", "湿度内部平均_MEAN", "外部-平均_MEAN"]
    },
    "spring": {
        "MORTALITY_RATE": ["湿度内部平均_MEAN", "外部-平均_MEAN", "温度2-平均变化"],
        "EEF": ["温度3-平均变化","鸡舍温度-平均_MEAN", "湿度内部平均_MEAN"]
    },
    "summer": {
        "MORTALITY_RATE": ["温度5-平均变化", "湿度内部平均_MEAN", "鸡舍温度-平均_MEAN"],
        "EEF": ["温度3-平均_RANGE", "鸡舍温度-平均_MEAN", "湿度内部平均_MEAN"]
    }
}
df2=df2.rename({'SEASON':'season'},axis=1)
# cols=[i for i in df2.columns.to_list() if "温度2-平均变化" in i]
# print(cols)
# df2[df2['season']=='autumn']["温度3-平均变化_MEAN"]
seasons=['autumn','winter','summer','spring']
season='winter'
for season in seasons:
    target_flg='EEF_flg'
    target='EEF'
    # [i for i in df2.columns if '温度3-平均变化_MEAN'in i]
    process_seasonal_data(
        df=df2, 
        season=season, 
        target=target, 
        target_flg=target_flg, 
        output_path='.//xyy//模型分析_new//'
    )

    raw_data = pd.read_csv(f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱1.csv", encoding='gbk')
    if target=='MORTALITY_RATE':
        max_eef_diff=0.001
    elif target=='EEF':
        max_eef_diff=3
    optimizer = AdvancedBinningOptimizer(
        target=target,
        max_eef_diff=max_eef_diff,  # 差异阈值
        min_bin_size=0.05,
        min_iv_loss=0.1
    )
    optimized_data = optimizer.optimize(raw_data)
    optimized_data.to_csv(f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱_Optimized1.csv", index=False, encoding='gbk')

    if target=='MORTALITY_RATE':
        plot_all_variables(f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱_Optimized.csv",
                        f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分布.csv",
                        season=season)
    elif target=='EEF':
        plot_all_variables_eef(f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱_Optimized.csv",
                        f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分布.csv",
                        season=season)
    print(f"{season}_{target}分箱完毕")


df2[(df2['season']=='winter') & (df2['探头温度标准差_0'] < 0.15) ]['MORTALITY_RATE'].mean()
df2[(df2['season']=='winter') & (df2['探头温度标准差_0'] < 0.45) & (df2['探头温度标准差_0'] >= 0.35)]['MORTALITY_RATE'].mean()

df2[(df2['season']=='winter') & (df2['探头温度标准差_0'] < 0.15) ]['温度4-平均_RANGE_0'].mean()
df2[(df2['season']=='winter') & (df2['探头温度标准差_0'] < 0.45) & (df2['探头温度标准差_0'] >= 0.35)]['温度4-平均_RANGE_0'].mean()


df2[(df2['season']=='winter') & (df2['探头温度标准差_0'] < 0.15) ]['鸡舍温度-平均_MEAN_0'].mean()
df2[(df2['season']=='winter') & (df2['探头温度标准差_0'] < 0.45) & (df2['探头温度标准差_0'] >= 0.35)]['鸡舍温度-平均_MEAN_13'].mean()

#####
df2[(df2['season']=='winter') & (df2['探头温度标准差_13'] < 0.15) ]['MORTALITY_RATE'].mean()
df2[(df2['season']=='winter') & (df2['探头温度标准差_13'] < 0.35) & (df2['探头温度标准差_13'] >= 0.25)]['MORTALITY_RATE'].mean()


df2[(df2['season']=='winter') & (df2['探头温度标准差_13'] < 0.15) ]['温度4-平均_RANGE_13'].mean()
df2[(df2['season']=='winter') & (df2['探头温度标准差_13'] < 0.35) & (df2['探头温度标准差_13'] >= 0.25)]['温度4-平均_RANGE_13'].mean()


df2[(df2['season']=='winter') & (df2['探头温度标准差_13'] < 0.15) ]['鸡舍温度-平均_MEAN_13'].mean()
df2[(df2['season']=='winter') & (df2['探头温度标准差_13'] < 0.35) & (df2['探头温度标准差_13'] >= 0.25)]['鸡舍温度-平均_MEAN_13'].mean()

df2[(df2['season']=='winter')]['探头温度标准差_12'].drop_duplicates()

plot_all_variables(f".\\xyy\\模型分析_new\\{season}\\{season}_MORTALITY_RATE_分箱_Optimized1.csv",
                    f".\\xyy\\模型分析_new\\{season}\\{season}_MORTALITY_RATE_分布1.csv",
                    season=season)
plot_all_variables_eef(f".\\xyy\\模型分析_new\\{season}\\{season}_EEF_分箱_Optimized1.csv",
                    f".\\xyy\\模型分析_new\\{season}\\{season}_EEF_分布.csv",
                    season=season)
# 分箱阈值: [24.3465271  24.55416584 24.79087734 24.9416666  25.26875019]

# === MORTALITY_RATE具体情况: W3_21-23天_鸡舍温度-平均_MEAN ===
#             index  MORTALITY_RATE
# 0   (-inf, 24.35)        0.073582
# 1  [24.35, 24.55)        0.078739

df2[(df2['season']=='winter') & (df2['W3_21-23天_鸡舍温度-平均_MEAN']<24.3465271)]['鸡舍温度-平均_MEAN_0'].mean()
