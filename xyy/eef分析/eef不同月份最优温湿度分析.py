import numpy as np
# import math
import pandas as pd
# import pickle
import matplotlib.pyplot as plt
# from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
import seaborn as sns
import os
# import glob

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')
HumTem_data_agg=pd.read_csv('./data/data_cleaned/HumTem_data_agg0515.csv',encoding='gbk')

# HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]
# 'G28_25', 'G31_62'前后两个批次重复
# HumTem_data_agg = HumTem_data_agg[~HumTem_data_agg['ID_NUM'].str.startswith(tuple(['G28_25', 'G31_62']))]
allinfo_dead.columns.to_list()

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
HumTem_data_t=pd.merge(HumTem_data_agg,allinfo_dead[['ID_NUM','age_days','Harveststatus','eef']],how='left',on=['ID_NUM'])

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



date_columns = ['Harveststatus']
for col in date_columns:
    HumTem_data_normal[col] = pd.to_datetime(HumTem_data_normal[col])
    HumTem_data_normal[f'{col}_month'] = HumTem_data_normal[col].dt.month
    HumTem_data_normal[f'{col}_month']=HumTem_data_normal[f'{col}_month'].astype(str)

HumTem_data_normal=HumTem_data_normal.drop(date_columns,axis=1)


def assign_EEF_flg(group):
    # 计算80%和20%分位数
    quantile_80 = np.quantile(group['eef'], 0.8)
    quantile_20 = np.quantile(group['eef'], 0.2)
    
    # 根据分位数规则标记
    group['eef_high_flg'] = group['eef'].apply(
        lambda x: 1 if x >= quantile_80 else (-1 if x <= quantile_20 else 0)
    )
    return group

HumTem_data_agg_t20 = HumTem_data_normal.groupby('Harveststatus_month').apply(assign_EEF_flg).drop('Harveststatus_month',axis=1)

# HumTem_data_agg_t20['eef_high_flg'].value_counts()
HumTem_data_agg_t20.groupby(['Harveststatus_month','eef_high_flg'])['eef'].mean()
# HumTem_data_agg_t20=HumTem_data_agg_t20.reset_index()
# HumTem_data_agg_t20=HumTem_data_agg_t20.set_index('ID_NUM')
# HumTem_data_agg_t20=HumTem_data_agg_t20[HumTem_data_agg_t20['eef_high_flg']==1]

HumTem_data_agg_t20.columns.to_list()
# keep_col=['ID_NUM','eef','temperature','humidity',]
# HumTem_data_agg_t20_tttt=HumTem_data_agg_t20.reset_index(drop=False)

# unique_ids =list(HumTem_data_agg_t20_tttt[
#     (HumTem_data_agg_t20_tttt['Harveststatus_month'] == '3') & 
#     (HumTem_data_agg_t20_tttt['eef_high_flg'] == 1)
# ]['ID_NUM'].drop_duplicates())
# ouput=HumTem_data_agg_t20_tttt[(HumTem_data_agg_t20_tttt['Harveststatus_month']=='3')][['Harveststatus_month','ID_NUM','日龄','鸡舍温度-平均_mean','鸡舍温度-最低_mean','鸡舍温度-最高_mean','eef_high_flg','eef']].drop_duplicates()

# # monthly_stats[monthly_stats['Harveststatus_month']=='2'].to_csv('./xyy/eef分析/plt/2月温度数据聚合.csv',encoding='gbk')
# ouput.to_csv('./xyy/eef分析/plt/3月温度数据.csv',encoding='gbk')

# 按月份分组，计算温度和湿度的均值及允许的变动区间（标准差）
monthly_stats = HumTem_data_agg_t20.groupby(['Harveststatus_month','日龄','eef_high_flg']).agg(
    avg_temperature1=('温度1-平均_mean', 'mean'),
    avg_temperature2=('温度2-平均_mean', 'mean'),
    avg_temperature3=('温度3-平均_mean', 'mean'),
    avg_temperature4=('温度4-平均_mean', 'mean'),
    avg_temperature5=('温度5-平均_mean', 'mean'),
    avg_temperature=('鸡舍温度-平均_mean', 'mean'),

    min_temperature1=('温度1-平均_min', 'mean'),
    min_temperature2=('温度2-平均_min', 'mean'),
    min_temperature3=('温度3-平均_min', 'mean'),
    min_temperature4=('温度4-平均_min', 'mean'),
    min_temperature5=('温度5-平均_min', 'mean'),
    min_temperature=('鸡舍温度-最低_mean', 'mean'),

    max_temperature1=('温度1-平均_max', 'mean'),
    max_temperature2=('温度2-平均_max', 'mean'),
    max_temperature3=('温度3-平均_max', 'mean'),
    max_temperature4=('温度4-平均_max', 'mean'),
    max_temperature5=('温度5-平均_max', 'mean'),
    max_temperature=('鸡舍温度-最高_mean', 'mean'),
 
    avg_humidity=('湿度内部平均_mean', 'mean'),
    min_humidity=('湿度内部平均_min', 'mean'),
    max_humidity=('湿度内部平均_max', 'mean'),
).reset_index()

###################################################3
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取数据
months = monthly_stats['DOCdate_month'].unique()
sensors = range(1, 6)  # 1-5号传感器

for month in months:
    month_data = monthly_stats[monthly_stats['DOCdate_month'] == month]
    
    for sensor in sensors:
        # 准备数据
        plot_data = month_data.melt(
            id_vars=['日龄', 'eef_high_flg'],
            value_vars=[
                f'min_temperature{sensor}',
                f'avg_temperature{sensor}', 
                f'max_temperature{sensor}'
            ],
            var_name='温度类型',
            value_name='温度值'
        )
        
        # 映射中文标签
        type_map = {
            f'min_temperature{sensor}': '最低',
            f'avg_temperature{sensor}': '平均',
            f'max_temperature{sensor}': '最高'
        }
        plot_data['温度类型'] = plot_data['温度类型'].map(type_map)
        
        # 创建画布
        plt.figure(figsize=(12, 6))
        
        # 绘制核心折线图（仅用颜色区分温度类型）
        ax = sns.lineplot(
            data=plot_data,
            x='日龄',
            y='温度值',
            hue='温度类型',
            palette={'最低': '#377eb8', '平均': '#4daf4a', '最高': '#e41a1c'},
            linewidth=2,
            legend=False
        )
        
        # 添加eef_high_flg的区分（用标记样式）
        for line, (name, group) in zip(ax.lines, plot_data.groupby('温度类型')):
            # 为每组温度类型添加对应的标记
            for flag_val in [0, 1]:
                subset = group[group['eef_high_flg'] == flag_val]
                plt.scatter(
                    subset['日龄'],
                    subset['温度值'],
                    marker='o' if flag_val == 0 else 's',  # 圆形/方形区分0/1
                    color=line.get_color(),  # 继承线条颜色
                    s=60,  # 标记大小
                    edgecolor='w',  # 白色描边
                    linewidth=1,
                    label=f"{name}温度 (eef={flag_val})"
                )
        
        # 添加标题和标签
        plt.title(
            f'{month}月 - 传感器{sensor}温度趋势\n'
            '○: eef_high_flg=0 | □: eef_high_flg=1',
            fontsize=14,
            pad=20
        )
        plt.xlabel('日龄（天）', fontsize=12)
        plt.ylabel('温度（℃）', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 手动构建图例
        legend_elements = [
            Line2D([0], [0], color='#377eb8', lw=3, label='最低温度'),
            Line2D([0], [0], color='#4daf4a', lw=3, label='平均温度'), 
            Line2D([0], [0], color='#e41a1c', lw=3, label='最高温度'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
                  markersize=10, label='高效标志=0', linestyle='None'),
            Line2D([0], [0], marker='s', color='w', markeredgecolor='k',
                  markersize=10, label='高效标志=1', linestyle='None')
        ]
        
        plt.legend(
            handles=legend_elements,
            title='图例说明',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        plt.tight_layout()
        plt.show()

#################################################################33
# 筛选EEF前20%数据 (假设eef_high_flg=1表示前20%)
high_eef = HumTem_data_agg_t20[HumTem_data_agg_t20['eef_high_flg'] == 1].copy()
low_eef = HumTem_data_agg_t20[HumTem_data_agg_t20['eef_high_flg'] == 0].copy()

# 计算允许变动区间
def calculate_intervals(df):
    return df.groupby(['DOCdate_month', '日龄']).agg({
        '平均温度': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
        '湿度内部平均_mean': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)]
    })

# high_eef['avg_temperature']
interval_df = calculate_intervals(high_eef)
interval_df.columns = ['temp_mean', 'temp_lower', 'temp_upper', 
                      'humidity_mean', 'humidity_lower', 'humidity_upper']
interval_df = interval_df.reset_index()

# 生成标准化的允许变动区间表格
interval_table = interval_df.pivot_table(
    index='日龄',
    columns='DOCdate_month',
    values=['temp_lower', 'temp_upper', 'humidity_lower', 'humidity_upper'],
    aggfunc='first'
)
interval_table.to_csv('./xyy/eef分析/plt/interval_table.csv', encoding='gbk')

plt.figure(figsize=(15, 12))

# ================= 温度趋势 =================
plt.subplot(2, 1, 1)
# 绘制温度曲线（加粗线条）
for month in high_eef['DOCdate_month'].unique():
    month_data = high_eef[high_eef['DOCdate_month'] == month]
    plt.plot(month_data['日龄'], month_data['avg_temperature'], 
             linewidth=2, marker='o', markersize=4,
             label=f'{month}月平均温度')

plt.title('最优群温度趋势', fontsize=14, pad=15)
plt.ylabel('温度 (℃)', fontsize=12)
plt.grid(True, linestyle=':')
plt.legend(bbox_to_anchor=(1.15, 1))

# ================= 湿度趋势 =================
plt.subplot(2, 1, 2)
# 绘制湿度曲线
for month in high_eef['DOCdate_month'].unique():
    month_data = high_eef[high_eef['DOCdate_month'] == month]
    plt.plot(month_data['日龄'], month_data['avg_humidity'],
             linewidth=2, marker='s', markersize=4,
             label=f'{month}月平均湿度')

plt.title('最优群湿度趋势', fontsize=14, pad=15)
plt.xlabel('日龄（天）', fontsize=12)
plt.ylabel('湿度 (%)', fontsize=12)
plt.grid(True, linestyle=':')
plt.legend(bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.show()
    

##########################################################################
# 获取所有月份
# 创建输出目录（如果不存在）
output_dir = "./xyy/eef分析/plt/temperature_plots2"
os.makedirs(output_dir, exist_ok=True)

months = monthly_stats['DOCdate_month'].unique()

for month in months:
    # 筛选当前月份数据
    
    month_data = monthly_stats[monthly_stats['DOCdate_month'] == month]
    month_data = month_data[~month_data['eef_high_flg'] == 0]

    
    # 创建画布（增加顶部空间防止标题被截断）
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))
    
    # 设置主标题（缩小字体并确保显示完整）
    fig.suptitle(
        f'{month}月温度趋势 - 按高效标志分组', 
        fontsize=12,  # 进一步缩小主标题
        y=0.98,       # 向上调整位置
        va='bottom'
    )
    
    # 为每个温度传感器绘图
    for i in range(1, 6):
        ax = axes[i-1]
        lineplot = sns.lineplot(
            data=month_data, 
            x='日龄', 
            y=f'avg_temperature{i}', 
            hue='eef_high_flg',
            palette={-1: '#1f77b4', 1: '#ff7f0e'},
            style='eef_high_flg',
            markers={-1: 'o', 1: 's'},
            dashes={-1: (1,0), 1: (2,2)},
            ax=ax
        )
        
        ax.set_title(f'传感器 {i}', pad=6, fontsize=10)  # 进一步缩小子标题
        ax.set_xlabel('日龄（天）', fontsize=9)  
        ax.set_ylabel('温度（℃）', fontsize=9)  
        
        # 优化图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, 
            labels=['高效: 否 (-1)', '高效: 是 (1)'],  # 简化标签文本
            title=None,  # 移除图例标题
            loc='upper right',
            framealpha=0.9,
            fontsize=8   # 缩小图例字体
        )
        
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(labelsize=8)  # 缩小刻度标签
    
    # 调整整体布局（增加顶部空间）
    plt.subplots_adjust(top=0.94, hspace=0.3)  # 增加顶部空间
    
    # 保存图片（高质量PNG格式）
    # output_path = os.path.join(output_dir, f"{month}月温度趋势.png")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"已保存: {output_path}")
    
    plt.show()
    plt.close()

################################0512

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


HumTem_data_agg_t20.groupby(['eef_high_flg'])['eef'].mean()

# 筛选高效群和低效群数据
high_efficiency = monthly_stats[monthly_stats['eef_high_flg'] == 1]
low_efficiency = monthly_stats[monthly_stats['eef_high_flg'] == -1]
# monthly_stats[monthly_stats['DOCdate_month']=='2']
# 获取所有月份
months = monthly_stats['Harveststatus_month'].unique()

# 为每个月份创建温度对比图
for month in months:
    high_month = high_efficiency[high_efficiency['Harveststatus_month'] == month]
    low_month = low_efficiency[low_efficiency['Harveststatus_month'] == month]
    
    if high_month.empty or low_month.empty:
        print(f"月份 {month} 缺少高效群或低效群数据，跳过绘图")
        continue
    
    # 创建画布
    plt.figure(figsize=(14, 8))
    
    # 绘制高效群温度曲线（实线+标记点）
    plt.plot(high_month['日龄'], high_month['avg_temperature'], 
             label='高效群 - 平均温度', color='#1f77b4', linewidth=2.0, marker='o', markersize=6)
    plt.plot(high_month['日龄'], high_month['max_temperature'], 
             label='高效群 - 最高温度', color='#ff7f0e', linewidth=2.0, marker='^', markersize=6)
    plt.plot(high_month['日龄'], high_month['min_temperature'], 
             label='高效群 - 最低温度', color='#2ca02c', linewidth=2.0, marker='s', markersize=6)
    
    # 绘制低效群温度曲线（虚线+标记点）
    plt.plot(low_month['日龄'], low_month['avg_temperature'], 
             label='低效群 - 平均温度', color='#1f77b4', linestyle='--', linewidth=2.0, marker='o', markersize=6, fillstyle='none')
    plt.plot(low_month['日龄'], low_month['max_temperature'], 
             label='低效群 - 最高温度', color='#ff7f0e', linestyle='--', linewidth=2.0, marker='^', markersize=6, fillstyle='none')
    plt.plot(low_month['日龄'], low_month['min_temperature'], 
             label='低效群 - 最低温度', color='#2ca02c', linestyle='--', linewidth=2.0, marker='s', markersize=6, fillstyle='none')
    
    # 设置图表标题和坐标轴标签
    plt.title(f'{month}月 高效群与低效群温度对比图', fontsize=16, pad=15)
    plt.xlabel('日龄（天）', fontsize=14)
    plt.ylabel('温度（℃）', fontsize=14)
    
    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 添加图例并调整位置
    plt.legend(fontsize=12, loc='upper right')
    
    # 优化布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()

############0512-2

# 获取月份列表
months = monthly_stats['Harveststatus_month'].unique()

# 为每个月创建温度趋势图
for month in months:
    # 筛选当前月份数据
    month_data = monthly_stats[monthly_stats['Harveststatus_month'] == month]
    month_data=month_data[month_data['eef_high_flg'] != 0]  # 仅保留高效标志不为0的数据
    
    # 创建画布
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))
    
    # 设置主标题
    fig.suptitle(
        f'{month}月温度趋势 - 按高效标志分组', 
        fontsize=12,
        y=0.98,
        va='bottom'
    )
    
    # 为每个温度传感器绘图
    for i in range(1, 6):
        ax = axes[i-1]
        lineplot = sns.lineplot(
            data=month_data, 
            x='日龄', 
            y=f'avg_temperature{i}', 
            hue='eef_high_flg',
            palette={-1: '#1f77b4', 1: '#ff7f0e'},  # 修改为包含-1的颜色映射
            style='eef_high_flg',
            markers={-1: 'o',  1: 's'},  # 修改为包含-1的标记样式
            dashes={-1: (1,0), 1: (2,2)},  # 修改为包含-1的虚线样式
            ax=ax
        )
        
        ax.set_title(f'传感器 {i}', pad=6, fontsize=10)
        ax.set_xlabel('日龄（天）', fontsize=9)  
        ax.set_ylabel('温度（℃）', fontsize=9)  
        
        # 优化图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, 
            labels=['低效: 低 (-1)',  '高效: 高 (1)'],  # 更新图例标签
            title=None,
            loc='upper right',
            framealpha=0.9,
            fontsize=8
        )
        
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(labelsize=8)
    
    # 调整整体布局
    plt.subplots_adjust(top=0.94, hspace=0.3)
    
    # # 保存图片
    # output_path = os.path.join(output_dir, f"{month}月温度趋势.png")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"已保存: {output_path}")
    
    plt.show()
    plt.close()

############################0515温差图

# 假设你已经有了monthly_stats数据集
# 筛选高效群和低效群数据
high_efficiency = monthly_stats[monthly_stats['eef_high_flg'] == 1]
low_efficiency = monthly_stats[monthly_stats['eef_high_flg'] == -1]

# 获取所有月份
months = monthly_stats['Harveststatus_month'].unique()

# 为每个月份创建温差对比图
for month in months:
    high_month = high_efficiency[high_efficiency['Harveststatus_month'] == month]
    low_month = low_efficiency[low_efficiency['Harveststatus_month'] == month]
    
    if high_month.empty or low_month.empty:
        print(f"月份 {month} 缺少高效群或低效群数据，跳过绘图")
        continue
    
    # 确保日龄对齐（防止索引不一致导致的减法错误）
    aligned_data = pd.merge(high_month, low_month, on='日龄', suffixes=('_high', '_low'))
    
    # 计算温差
    aligned_data['delta_max'] = aligned_data['max_temperature_high'] - aligned_data['max_temperature_low']
    aligned_data['delta_min'] = aligned_data['min_temperature_high'] - aligned_data['min_temperature_low']
    aligned_data['delta_avg'] = aligned_data['avg_temperature_high'] - aligned_data['avg_temperature_low']
    
    # 创建画布
    plt.figure(figsize=(14, 8))
    
    # 绘制温差曲线（高效群 - 低效群）
    plt.plot(aligned_data['日龄'], aligned_data['delta_max'], 
             label='最高温度差（高效-低效）', color='#d62728', linewidth=2.0, marker='o', markersize=6)
    plt.plot(aligned_data['日龄'], aligned_data['delta_min'], 
             label='最低温度差（高效-低效）', color='#9467bd', linewidth=2.0, marker='^', markersize=6)
    plt.plot(aligned_data['日龄'], aligned_data['delta_avg'], 
             label='平均温度差（高效-低效）', color='#8c564b', linewidth=2.0, marker='s', markersize=6)
    
    # 添加零基准线（虚线）
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 设置图表标题和坐标轴标签
    plt.title(f'{month}月 高效群与低效群温差对比（高效群温度 - 低效群温度）', fontsize=16, pad=15)
    plt.xlabel('日龄（天）', fontsize=14)
    plt.ylabel('温度差异（℃）', fontsize=14)
    
    # 设置坐标轴范围和刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 添加图例并调整位置
    plt.legend(fontsize=12, loc='upper right')
    
    # 标记关键温差区域
    for day, delta_max, delta_avg in zip(aligned_data['日龄'], aligned_data['delta_max'], aligned_data['delta_avg']):
        if abs(delta_max) > 1:  # 标记显著差异点
            plt.annotate(f'{delta_max:.1f}℃', xy=(day, delta_max), xytext=(5, 5), 
                         textcoords='offset points', color='#d62728', fontsize=10)
        if abs(delta_avg) > 0.5:
            plt.annotate(f'{delta_avg:.1f}℃', xy=(day, delta_avg), xytext=(5, -15), 
                         textcoords='offset points', color='#8c564b', fontsize=10)
    
    # 优化布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()