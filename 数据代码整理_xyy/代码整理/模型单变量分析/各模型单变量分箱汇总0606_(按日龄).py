
###用于模型后各日龄变量的单变量分析
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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 读取原始数据文件（两个数据集）
all_info_temdata1 = pd.read_csv('./data/data_cleaned/all_info_temdata0602.csv', encoding='gbk')
all_info_temdata2 = pd.read_csv('./data/data_cleaned/all_info_temdata0602_2.csv', encoding='gbk')

# 提取第二个数据集中包含'平均变化'的列和ID_NUM列
cols = [i for i in all_info_temdata2.columns.to_list() if '平均变化' in i] + ['ID_NUM']
# 基于ID_NUM合并两个数据集（左连接）
all_info_temdata = pd.merge(all_info_temdata1, all_info_temdata2[cols], on='ID_NUM', how='left')

# 查看各数据集形状（调试用）
all_info_temdata1.shape
all_info_temdata2.shape
all_info_temdata.shape
# 查看列名（调试用）
all_info_temdata1.columns.to_list()

# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")

# 将列名转为大写（统一格式）
col = [a.upper() for a in all_info_temdata.columns.to_list()]
all_info_temdata.columns = col

# 设置ID_NUM为索引
df = all_info_temdata.set_index('ID_NUM')

# 定义需要删除的列（死淘率相关指标）
drop_columns_Mortality = ['Dead', 'Swollen_Head', 'Weak', 'Navel_Disease', 'Stick_Anus', 'Lame_Paralysis', 'livability_pct', 'yield_per_m2']
drop_columns_Mortality = [a.upper() for a in drop_columns_Mortality]  # 转为大写匹配列名
drop_columns_Mortality = [i for i in df.columns.to_list() if i in drop_columns_Mortality]  # 确保列存在

# 定义营销数据相关列（需删除的业务指标）
marketingdata_columns = [
    'house', 'birds_placed', 'gender', 'house_area_m2', 'stocking_density', 'birds_hung', 'total_hung_weight_kg',
    'avg_weight_kg', 'small_broilers_count', 'small_broilers_weight_kg', 'pp_dead_culled_count',
    'dead_culled_weight_kg', 'pp_rejects_count', 'pp_rejects_weight_kg', 'age_days', 'dead_during_catch_count',
    'birds_caught_count', 'livability_pct', 'total_caught_weight_kg', 'yield_per_m2', 'final_avg_weight_kg',
    'total_feed_kg', 'fcr', 'adjusted_fcr', 'revenue', 'profit_per_house', 'medicine_per_bird',
    'vaccine_per_bird', 'mv_cost_per_bird', 'disinfectant_per_bird', 'feed_per_bird', 'electricity_per_bird',
    'gas_per_bird', 'labor_per_bird', 'consumables_per_bird', 'depreciation_per_bird', 'chick_cost_per_bird',
    'cost_per_bird', 'chick_cost', 'total_cost', 'cost_per_kg', 'feed_cost'
]
marketingdata_columns = [a.upper() for a in marketingdata_columns]  # 转为大写
drop_columns_marketingdata = [i for i in df.columns.to_list() if i in marketingdata_columns]  # 筛选存在的列

# 删除冗余列，得到清洗后的数据集
df2 = df.drop(columns=drop_columns_marketingdata + drop_columns_Mortality, axis=1)
df2.shape  # 查看清洗后数据形状

# 生成死淘率标签：按季节分组，死淘率高于80%分位数标记为1，否则为0
df2['Mortality_flg'] = df2.groupby('SEASON')['MORTALITY_RATE'].transform(lambda x: (x >= x.quantile(0.8)).astype(int))
# 生成EEF标签：按季节分组，EEF高于80%分位数标记为1，否则为0
df2['EEF_flg'] = df2.groupby('SEASON')['EEF'].transform(lambda x: (x >= x.quantile(0.8)).astype(int))

# 重命名SEASON列为小写season（统一变量名）
df2 = df2.rename({'SEASON': 'season'}, axis=1)


### 分箱分析函数定义 ###
def feature_binning(top_importantcol, object_columns, X, y, target='MORTALITY_RATE', max_n_bins=6):
    """
    特征分箱分析函数
    参数：
    top_importantcol: 需要分箱的特征列表（不含目标列）
    object_columns: 分类特征列名列表
    X: 特征数据集（需包含target列）
    y: 目标标签（0/1二分类）
    target: 目标指标列名（如'MORTALITY_RATE'）
    max_n_bins: 最大分箱数限制
    返回：
    binning_sum: 分箱汇总统计（按Gini降序排列）
    bin_table: 分箱明细表（含各箱的目标指标均值）
    """
    if target not in X.columns:
        raise ValueError(f"X中必须包含{target}列")
    
    # 识别分类特征
    cat_f = [col for col in top_importantcol if col in object_columns]
    
    # 分箱选择标准（基于Gini系数，要求最小Gini>0.05）
    selection_criteria = {"gini": {"min": 0.05, "max": 1}}
    
    # 初始化分箱器（设置最大分箱数、最小箱占比等）
    binning_process = BinningProcess(
        top_importantcol,
        categorical_variables=cat_f,
        selection_criteria=selection_criteria,
        max_n_bins=max_n_bins,
        min_bin_size=0.1,  # 最小箱占比10%
        min_n_bins=3  # 最小分箱数2
    )
    
    # 拟合分箱模型
    binning_process.fit(X[top_importantcol], y)
    
    # 获取分箱汇总（按Gini排序）
    binning_sum = binning_process.summary().sort_values(by='gini', ascending=False)
    
    # 构建分箱明细表（包含各箱的目标指标均值）
    bin_table = pd.DataFrame()
    for col in top_importantcol:
        try:
            optb = binning_process.get_binned_variable(col)
            temp = optb.binning_table.build()  # 获取原始分箱表
            temp['feature'] = col  # 添加特征名列
            
            # 获取分箱结果并计算各箱的目标指标均值
            X_binned = optb.transform(X[[col]], metric='bins').squeeze()
            bin_mortality = X.groupby(X_binned)[[target]].mean().reset_index(drop=False)
            
            # 合并分箱表与目标均值
            temp2 = pd.merge(temp, bin_mortality, left_on='Bin', right_on='index', how='left').drop('index', axis=1)
            bin_table = pd.concat([bin_table, temp2])
            
            # 打印调试信息
            print(f"分箱阈值: {optb.splits}")
            print(f"\n=== {target}具体情况: {col} ===")
            print(bin_mortality)
            print(f"\n=== 变量: {col} ===")
            display_cols = ['Bin', 'Count', 'Count (%)', 'Event rate', target]
            print(temp2[display_cols])
            print('-'*50)
            
        except Exception as e:
            print(f"处理变量 {col} 时出错: {str(e)}")
            continue
    
    return binning_sum, bin_table


def process_seasonal_data(
    df, 
    season, 
    target, 
    target_flg, 
    output_path='.\\xyy\\模型分析_new\\'
):
    """
    季节性数据处理函数
    参数：
    df: 原始数据集
    season: 目标季节（如'winter'）
    target: 目标指标列名（如'MORTALITY_RATE'）
    target_flg: 目标标签列名（如'Mortality_flg'）
    output_path: 结果输出路径
    """
    # 分类特征与数值特征划分
    numeric_columns = []
    object_columns = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            object_columns.append(column)
    
    # 按季节筛选数据
    data1 = df[df['season'] == season].copy()
    if data1.empty:
        raise ValueError(f"季节'{season}'在数据中不存在")
    
    # 特征与标签准备
    X = data1  # 包含所有特征（含target）
    y = data1[target_flg]  # 目标标签（0/1）
    top_importantcol = [col for col in df.columns if any(keyword in col for keyword in importance_dict[season][target])]
    # 示例：关键特征列表（需根据实际业务调整，此处硬编码为['探头温度标准差_24']）
    # top_importantcol = ['探头温度标准差_24']
    X[top_importantcol] = X[top_importantcol].round(1)  # 保留一位小数
    
    # 执行特征分箱
    try:
        binning_sum, bin_table = feature_binning(
            top_importantcol, 
            object_columns, 
            X, 
            y, 
            target=target
        )
    except Exception as e:
        raise RuntimeError(f"分箱失败: {str(e)}")
    
    # 保存分箱结果到文件
    season_output_path = f"{output_path}{season}/"
    os.makedirs(season_output_path, exist_ok=True)  # 创建季节专属目录
    bin_table.to_csv(
        f"{season_output_path}{season}_{target}_分箱.csv", 
        index=False, 
        encoding='gbk'
    )
    
    # 保存特征分布统计（均值、标准差等）
    fx = X[top_importantcol].describe().round(2).reset_index(drop=False)
    fx.to_csv(
        f"{season_output_path}{season}_{target}_分布.csv", 
        index=False, 
        encoding='gbk'
    )
    print(f"特征分布文件已保存，维度: {fx.shape}")
    
    return {
        'binning_summary': binning_sum,
        'bin_table': bin_table,
        'feature_stats': fx
    }


### 分箱优化类定义 ###
class AdvancedBinningOptimizer:
    def __init__(self, target="MORTALITY_RATE", max_eef_diff=0.08, min_bin_size=0.05, min_iv_loss=0.03):
        """
        分箱优化器
        参数：
        target: 目标指标（'MORTALITY_RATE'或'EEF'）
        max_eef_diff: 相邻箱目标值差异阈值（用于合并条件）
        min_bin_size: 最小箱占比阈值
        min_iv_loss: 允许的IV值损失上限
        """
        self.target = target
        self.max_eef_diff = max_eef_diff
        self.min_bin_size = min_bin_size
        self.min_iv_loss = min_iv_loss

    def _parse_bin_boundary(self, bin_str):
        """解析分箱边界字符串为数值区间"""
        if pd.isna(bin_str) or bin_str in ['Special', 'Missing']:
            return np.nan, np.nan
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", bin_str)
        if len(numbers) == 1:
            return (-np.inf, float(numbers[0])) if '(' in bin_str else (float(numbers[0]), np.inf)
        return (float(numbers[0]), float(numbers[1]))

    def _preprocess(self, df):
        """预处理分箱数据（转换类型、计算衍生指标）"""
        df = df[df['Bin'].apply(lambda x: bool(re.search(r'\d', str(x))))].copy()  # 过滤无效箱
        boundaries = df['Bin'].apply(self._parse_bin_boundary)
        df['lower'] = [b[0] for b in boundaries]
        df['upper'] = [b[1] for b in boundaries]
        df = df.sort_values(by='lower').reset_index(drop=True)  # 按边界排序
        
        # 转换数据类型
        df['Count'] = df['Count'].astype(int)
        df[self.target] = df[self.target].astype(float)  # 目标指标转为浮点型
        df['Count_pct'] = df['Count (%)'].astype(float)
        df['Event_rate'] = df['Event'].astype(int) / df['Count']  # 事件率
        df['WoE'] = df['WoE'].astype(float)  # 证据权重
        df['IV'] = df['IV'].astype(float)  # 信息价值
        return df

    def _find_merge_candidates(self, df):
        """寻找可合并的箱组（基于目标值差异和箱大小）"""
        merge_groups = []
        current_group = [0]
        for i in range(1, len(df)):
            prev_val = df.loc[current_group[-1], self.target]  # 前一箱的目标值
            curr_val = df.loc[i, self.target]  # 当前箱的目标值
            val_diff = abs(curr_val - prev_val)  # 目标值差异
            
            # 合并条件：箱占比小于阈值，或目标值差异小于阈值
            size_condition = (df.loc[i, 'Count_pct'] < self.min_bin_size) | (df.loc[current_group[-1], 'Count_pct'] < self.min_bin_size)
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
        """执行箱合并操作"""
        total_events = df['Event'].sum()  # 总事件数
        total_non_events = df['Non-event'].sum()  # 总非事件数
        merged_data = []
        
        for group in merge_groups:
            subset = df.iloc[group]
            new_lower = subset['lower'].min()  # 合并后的下限
            new_upper = subset['upper'].max()  # 合并后的上限
            bin_name = f"[{new_lower:.2f}, inf)" if np.isinf(new_upper) else f"[{new_lower:.2f}, {new_upper:.2f})"
            
            # 计算合并后的统计量
            total_count = subset['Count'].sum()
            merged_events = subset['Event'].sum()
            merged_non_events = subset['Non-event'].sum()
            
            # 计算WoE和IV（基于整体样本比例）
            good_pct = merged_non_events / total_non_events if total_non_events != 0 else 0
            bad_pct = merged_events / total_events if total_events != 0 else 0
            woe = np.log(good_pct / bad_pct) if (good_pct != 0 and bad_pct != 0) else 0
            iv = (good_pct - bad_pct) * woe
            
            # 生成合并后的记录
            merged_record = {
                'Bin': bin_name,
                'Count': total_count,
                'Non-event': merged_non_events,
                'Event': merged_events,
                'Event rate': merged_events / total_count,
                'WoE': woe,
                'IV': iv,
                'feature': subset['feature'].iloc[0],
                self.target: np.average(subset[self.target], weights=subset['Count']),  # 加权平均目标值
                'Count (%)': total_count / df['Count'].sum(),
                'JS': subset['JS'].sum() if 'JS' in subset.columns else 0
            }
            merged_data.append(merged_record)
        return pd.DataFrame(merged_data)

    def optimize(self, raw_data):
        """分箱优化主流程"""
        optimized_results = []
        features = raw_data['feature'].unique()  # 所有特征列表
        
        for feat in features:
            feat_data = raw_data[raw_data['feature'] == feat].copy()
            cleaned_data = self._preprocess(feat_data)  # 预处理
            
            if len(cleaned_data) < 3:  # 箱数不足时不优化
                optimized_results.append(cleaned_data)
                continue
            
            merge_candidates = self._find_merge_candidates(cleaned_data)  # 寻找合并组
            merged_data = self._merge_operation(cleaned_data, merge_candidates)  # 执行合并
            
            # 计算IV值变化（原始VS合并后）
            original_iv = cleaned_data['IV'].sum()
            new_iv = merged_data['IV'].sum()
            
            # 判断是否接受合并（IV损失小于阈值）
            if (original_iv - new_iv) <= self.min_iv_loss:
                optimized_feat_data = merged_data
            else:
                optimized_feat_data = cleaned_data  # 保留原始分箱
            
            # 添加特征汇总行（总计数、总IV等）
            summary_row = self._generate_summary_row(optimized_feat_data)
            optimized_feat_data = pd.concat([optimized_feat_data, summary_row], ignore_index=True)
            optimized_results.append(optimized_feat_data)
        
        # 合并所有特征结果，保持原始列顺序
        original_columns = raw_data.columns.tolist()
        optimized_data = pd.concat(optimized_results, ignore_index=True)[original_columns]
        return optimized_data

    def _generate_summary_row(self, df):
        """生成特征汇总行（总计数、总IV、平均目标值等）"""
        if df.empty:
            return pd.DataFrame()
        
        feat_name = df['feature'].iloc[0]
        total_count = df['Count'].sum()
        total_non_event = df['Non-event'].sum()
        total_event = df['Event'].sum()
        total_iv = df['IV'].sum()
        avg_target = np.average(df[self.target], weights=df['Count'])  # 加权平均目标值
        
        return pd.DataFrame({
            'Bin': [''],
            'Count': [total_count],
            'Non-event': [total_non_event],
            'Event': [total_event],
            'Event rate': [total_event / total_count if total_count != 0 else 0],
            'WoE': [np.nan],
            'IV': [total_iv],
            'feature': [feat_name],
            self.target: [avg_target],
            'Count (%)': [1.0],
            'JS': [df['JS'].sum()] if 'JS' in df.columns else [0]
        })


### 绘图函数定义 ###
def get_temp_range(stats_file, var_name):
    """从统计文件中获取变量的数值范围（用于绘图坐标轴）"""
    try:
        stats_df = pd.read_csv(stats_file, encoding='gbk')
        match_cols = [col for col in stats_df.columns if var_name in col]
        if not match_cols:
            print(f"警告：未找到变量{var_name}的统计列")
            return None, None
            
        min_val = stats_df.loc[stats_df['index'] == 'min', match_cols[0]].values[0]
        max_val = stats_df.loc[stats_df['index'] == 'max', match_cols[0]].values[0]
        return min_val, max_val 
        
    except Exception as e:
        print(f"获取范围失败: {e}")
        return None, None

def parse_bin_data(df):
    """解析分箱数据中的边界字符串为数值区间"""
    processed = df.copy()
    processed['Bin'] = processed['Bin'].astype(str)  # 确保转为字符串
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
    processed = processed[~processed['Bin'].isin(['Special', 'Missing', ''])]  # 过滤无效箱
    return processed, [] if processed.empty else processed['feature'].unique()

def plot_temp_mortality(df, season, x_min=None, x_max=None, feature_name=""):
    """绘制变量与死淘率的关系图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 确保中文显示
    plt.rcParams['axes.unicode_minus'] = False

    processed, _ = parse_bin_data(df)
    if processed.empty:
        print("数据为空，跳过绘图")
        return
    
    total_iv = processed['IV'].iloc[:-1].sum()  # 排除汇总行的IV
    
    # 替换无穷值为指定范围
    if x_min is not None:
        processed['parsed_lower'] = processed['parsed_lower'].replace(-np.inf, x_min)
    if x_max is not None:
        processed['parsed_upper'] = processed['parsed_upper'].replace(np.inf, x_max)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 主标题和副标题（包含IV值和范围）
    plt.suptitle(f"变量与死淘率关系 - {feature_name}-{season}", fontsize=16, y=0.98)
    plt.title(f"变量范围: {x_min:.1f} 至 {x_max:.1f} | 总IV值: {total_iv:.4f}", fontsize=12, pad=10)

    # 绘制数量柱状图
    processed['bar_x'] = processed['parsed_lower']
    processed['bar_width'] = processed['parsed_upper'] - processed['parsed_lower']
    processed['line_x'] = (processed['parsed_lower'] + processed['parsed_upper']) / 2  # 中点坐标
    
    colors = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(processed)))
    bars = ax1.bar(processed['bar_x'], processed['Count'], width=processed['bar_width'], 
                  color=colors, alpha=0.8, align='edge', label='数量')

    ax1.set_xlabel('变量范围', fontsize=12)
    ax1.set_ylabel('数量', color='dimgray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    
    # 设置x轴刻度（基于实际边界值）
    all_ticks = np.unique(processed[['parsed_lower', 'parsed_upper']].values.flatten())
    valid_ticks = [tick for tick in all_ticks if not np.isinf(tick)]  # 过滤无穷值
    if valid_ticks:
        ax1.set_xticks(valid_ticks)
        ax1.set_xticklabels([f"{tick:.1f}" for tick in valid_ticks], rotation=45, ha='right')
    else:
        ax1.set_xticks([])
    
    ax1.set_xlim(x_min, x_max)  # 强制设置x轴范围
    ax1.xaxis.set_tick_params(which='major', length=5, width=1)  # 刻度线样式
    
    # 绘制死淘率折线图（次坐标轴）
    ax2 = ax1.twinx()
    line, = ax2.plot(processed['line_x'], processed['MORTALITY_RATE'], 
                    color='crimson', marker='o', linestyle='-', linewidth=2, markersize=7, label='死淘率')
    
    # 动态计算y轴范围（预留10%空间）
    y_vals = processed['MORTALITY_RATE']
    y_range = y_vals.max() - y_vals.min()
    ax2.set_ylim(y_vals.min() - y_range*0.1, y_vals.max() + y_range*0.1)
    
    ax2.set_ylabel('死淘率', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    # 标注死淘率数值
    for i in range(len(processed)):
        ax2.text(processed['line_x'].iloc[i], y_vals.iloc[i], 
                 f"{y_vals.iloc[i]:.5f}", ha='center', va='bottom', fontsize=9, color='crimson')
    
    fig.legend([bars, line], ['数量', '死淘率'], loc='upper right')
    return fig

def plot_temp_eef(df, x_min=None, x_max=None, feature_name=""):
    """绘制变量与EEF的关系图（逻辑与死淘率类似）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    processed, _ = parse_bin_data(df)
    if processed.empty:
        print("数据为空，跳过绘图")
        return
    
    total_iv = processed['IV'].iloc[:-1].sum()
    
    if x_min is not None:
        processed['parsed_lower'] = processed['parsed_lower'].replace(-np.inf, x_min)
    if x_max is not None:
        processed['parsed_upper'] = processed['parsed_upper'].replace(np.inf, x_max)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    plt.suptitle(f"变量与EEF关系 - {feature_name}", fontsize=16, y=0.98)
    plt.title(f"变量范围: {x_min:.1f} 至 {x_max:.1f}℃ | 总IV值: {total_iv:.4f}", fontsize=12, pad=10)

    processed['bar_x'] = processed['parsed_lower']
    processed['bar_width'] = processed['parsed_upper'] - processed['parsed_lower']
    processed['line_x'] = (processed['parsed_lower'] + processed['parsed_upper']) / 2

    colors = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(processed)))
    bars = ax1.bar(processed['bar_x'], processed['Count'], width=processed['bar_width'], 
                  color=colors, alpha=0.8, align='edge', label='数量')

    ax1.set_xlabel('变量范围', fontsize=12)
    ax1.set_ylabel('数量', color='dimgray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dimgray')
    
    all_ticks = np.unique(processed[['parsed_lower', 'parsed_upper']].values.flatten())
    valid_ticks = [tick for tick in all_ticks if not np.isinf(tick)]
    if valid_ticks:
        ax1.set_xticks(valid_ticks)
        ax1.set_xticklabels([f"{tick:.1f}" for tick in valid_ticks], rotation=45, ha='right')
    else:
        ax1.set_xticks([])
    
    ax1.set_xlim(x_min, x_max)
    ax1.xaxis.set_tick_params(which='major', length=5, width=1)
    
    ax2 = ax1.twinx()
    line, = ax2.plot(processed['line_x'], processed['EEF'], 
                    color='crimson', marker='o', linestyle='-', linewidth=2, markersize=7, label='EEF')
    ax2.set_ylabel('EEF', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    for i in range(len(processed)):
        ax2.text(processed['line_x'].iloc[i], processed['EEF'].iloc[i] + 0.001, 
                 f"{processed['EEF'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=9, color='crimson')

    fig.legend([bars, line], ['数量', 'EEF'], loc='upper right')
    return fig

def plot_all_variables(binned_file, stats_file, output_dir='.\\xyy\\模型分析_new\\', season='winter'):
    """批量绘制所有变量的死淘率图表"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(binned_file, encoding='gbk')
    
    for feature in df['feature'].unique():
        if pd.isna(feature):
            continue
            
        feature_df = df[df['feature'] == feature]
        x_min, x_max = get_temp_range(stats_file, feature)  # 获取变量范围
        
        if x_min is None or x_max is None:
            print(f"跳过{feature}：范围未找到")
            continue
            
        clean_feature_name = ''.join(c for c in feature if c.isalnum() or c in ['_', '-', ' '])  # 清洗变量名
        fig = plot_temp_mortality(feature_df, season, x_min, x_max, feature_name=clean_feature_name)
        
        if fig:
            safe_name = feature.replace(' ', '_').replace('/', '_')  # 生成安全文件名
            filename = f"{season}_mortality_{safe_name}.png"
            output_path = os.path.join(output_dir, f'{season}/', filename)
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"保存图片: {output_path}")

def plot_all_variables_eef(binned_file, stats_file, output_dir='.\\xyy\\模型分析_new\\', season='winter'):
    """批量绘制所有变量的EEF图表"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(binned_file, encoding='gbk')
    
    for feature in df['feature'].unique():
        if pd.isna(feature):
            continue
            
        feature_df = df[df['feature'] == feature]
        x_min, x_max = get_temp_range(stats_file, feature)
        
        if x_min is None or x_max is None:
            print(f"跳过{feature}：范围未找到")
            continue
            
        clean_feature_name = ''.join(c for c in feature if c.isalnum() or c in ['_', '-', ' '])
        fig = plot_temp_eef(feature_df, x_min, x_max, feature_name=clean_feature_name)
        
        if fig:
            safe_name = feature.replace(' ', '_').replace('/', '_')
            filename = f"{season}_eef_{safe_name}.png"
            output_path = os.path.join(output_dir, f'{season}/', filename)
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"保存图片: {output_path}")


###### 主程序执行 #######
# 定义各季节的关键特征（硬编码，需根据实际业务调整）
importance_dict = {
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

# 遍历四季进行处理

seasons = ['autumn', 'winter', 'summer', 'spring']
# target可为EEF或MORTALITY_RATE，分别对应死淘模型和EEF模型
# target_flg可为EEF_flg或Mortality_flg，分别对应死淘模型和EEF模型
for season in seasons:
    # 处理EEF相关分析
    target_flg = 'EEF_flg'
    target = 'EEF'
    
    # 执行季节性数据处理（分箱、保存原始结果）
    process_seasonal_data(
        df=df2, 
        season=season, 
        target=target, 
        target_flg=target_flg, 
        output_path='.\\xyy\\模型分析_new\\'
    )
    
    # 读取原始分箱结果
    raw_data_path = f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱1.csv"
    raw_data = pd.read_csv(raw_data_path, encoding='gbk')
    
    # 设置优化参数（根据目标类型调整差异阈值）
    if target == 'MORTALITY_RATE':
        max_eef_diff = 0.001  # 死淘率差异阈值
    elif target == 'EEF':
        max_eef_diff = 3  # EEF差异阈值
    
    # 初始化优化器并执行分箱优化
    optimizer = AdvancedBinningOptimizer(
        target=target,
        max_eef_diff=max_eef_diff,
        min_bin_size=0.05,
        min_iv_loss=0.1  # 允许的IV损失上限
    )
    optimized_data = optimizer.optimize(raw_data)
    optimized_data.to_csv(f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱_Optimized.csv", 
                          index=False, encoding='gbk')
    
    # 绘制图表（根据目标类型选择绘图函数）
    if target == 'MORTALITY_RATE':
        plot_all_variables(
            f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱_Optimized.csv",
            f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分布1.csv",
            season=season
        )
    elif target == 'EEF':
        plot_all_variables_eef(
            f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分箱_Optimized.csv",
            f".\\xyy\\模型分析_new\\{season}\\{season}_{target}_分布1.csv",
            season=season
        )
    
    print(f"{season}_{target}分箱优化及绘图完成")