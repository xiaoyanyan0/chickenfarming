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


all_info_temdata=pd.read_csv('./data/data_cleaned/all_info_temdata0430.csv',encoding='gbk')

HumTem_data_agg1=pd.read_csv('./data/data_cleaned/HumTem_data_agg1.csv',encoding='gbk')
HumTem_data_agg2=pd.read_csv('./data/data_cleaned/HumTem_data_agg2.csv',encoding='gbk')

allinfo_dead=pd.read_csv('./data/data_cleaned/allinfo_dead0430.csv',encoding='gbk')
HumTem_data_agg=pd.concat([HumTem_data_agg1,HumTem_data_agg2],ignore_index=True)


#日报中的农场名字和文件名字中的对应不上
# HumTem_data_agg[HumTem_data_agg['ID_NUM'].str.startswith('G04')]['ID_NUM']

# allinfo_dead[allinfo_dead['ID_NUM'].str.startswith('G1A')]['ID_NUM']

allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G01' + x[3:] if isinstance(x, str) and x.startswith('G1A') else x)
allinfo_dead['ID_NUM'] = allinfo_dead['ID_NUM'].apply(lambda x: 'G04' + x[3:] if isinstance(x, str) and x.startswith('G1B') else x)
# 只有 G31_62匹不上
HumTem_data_t=pd.merge(HumTem_data_agg,allinfo_dead[['ID_NUM','DOCdate','eef','age_days']],how='left',on=['ID_NUM'])


HumTem_data_t['age_days'].min()
HumTem_data_t['age_days'].notna().sum()

HumTem_data_t[HumTem_data_t['age_days'].isna()]['ID_NUM'].drop_duplicates()

HumTem_data_normal=HumTem_data_t[HumTem_data_t['日龄']<HumTem_data_t['age_days']]
HumTem_data_abnormal=HumTem_data_t[HumTem_data_t['日龄']>=HumTem_data_t['age_days']]


HumTem_data_normal=HumTem_data_normal.drop('age_days',axis=1)

HumTem_data_normal=HumTem_data_normal.drop_duplicates()

date_columns = ['DOCdate']
for col in date_columns:
    HumTem_data_normal[col] = pd.to_datetime(HumTem_data_normal[col])
    HumTem_data_normal[f'{col}_month'] = HumTem_data_normal[col].dt.month
    HumTem_data_normal[f'{col}_month']=HumTem_data_normal[f'{col}_month'].astype(str)

HumTem_data_normal=HumTem_data_normal.drop(date_columns,axis=1)


def assign_EEF_flg(group):
    quantile_80 = np.quantile(group['eef'], 0.8)
    group['eef_high_flg'] = group['eef'].apply(lambda x: 1 if x >= quantile_80 else 0)
    return group

HumTem_data_agg_t20 = HumTem_data_normal.groupby('DOCdate_month').apply(assign_EEF_flg).drop('DOCdate_month',axis=1)

HumTem_data_agg_t20=HumTem_data_agg_t20.reset_index()
# HumTem_data_agg_t20=HumTem_data_agg_t20.set_index('ID_NUM')
# HumTem_data_agg_t20=HumTem_data_agg_t20[HumTem_data_agg_t20['eef_high_flg']==1]

HumTem_data_agg_t20.columns.to_list()
# keep_col=['ID_NUM','eef','temperature','humidity',]




# 按月份分组，计算温度和湿度的均值及允许的变动区间（标准差）
monthly_stats = HumTem_data_agg_t20.groupby(['DOCdate_month','日龄','eef_high_flg']).agg(
    avg_temperature1=('温度1-平均_mean', 'mean'),
    avg_temperature2=('温度2-平均_mean', 'mean'),
    avg_temperature3=('温度3-平均_mean', 'mean'),
    avg_temperature4=('温度4-平均_mean', 'mean'),
    avg_temperature5=('温度5-平均_mean', 'mean'),
    avg_temperature=('平均温度', 'mean'),

    min_temperature1=('温度1-平均_min', 'mean'),
    min_temperature2=('温度2-平均_min', 'mean'),
    min_temperature3=('温度3-平均_min', 'mean'),
    min_temperature4=('温度4-平均_min', 'mean'),
    min_temperature5=('温度5-平均_min', 'mean'),
    min_temperature=('最低温度', 'mean'),

    max_temperature1=('温度1-平均_max', 'mean'),
    max_temperature2=('温度2-平均_max', 'mean'),
    max_temperature3=('温度3-平均_max', 'mean'),
    max_temperature4=('温度4-平均_max', 'mean'),
    max_temperature5=('温度5-平均_max', 'mean'),
    max_temperature=('最高温度', 'mean'),
 
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
high_eef = monthly_stats[monthly_stats['eef_high_flg'] == 1].copy()
low_eef = monthly_stats[monthly_stats['eef_high_flg'] == 0].copy()

# 计算允许变动区间
def calculate_intervals(df):
    return df.groupby(['DOCdate_month', '日龄']).agg({
        'avg_temperature': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
        'avg_humidity': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)]
    })

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
# interval_table.to_csv('./xyy/eef分析/plt/interval_table.csv', encoding='gbk')

plt.figure(figsize=(15, 12))


# ================= 温度趋势 =================
plt.subplot(2, 1, 1)
# 先绘制允许区间（提高透明度并使用对比色）
for month in high_eef['DOCdate_month'].unique():
    interval_data = interval_df[interval_df['DOCdate_month'] == month]
    plt.fill_between(interval_data['日龄'],
                    interval_data['temp_lower'],
                    interval_data['temp_upper'],
                    color='skyblue', alpha=0.3,  # 提高透明度到0.3
                    label=f'{month}月允许区间')

# 再绘制温度曲线（加粗线条）
for month in high_eef['DOCdate_month'].unique():
    month_data = high_eef[high_eef['DOCdate_month'] == month]
    plt.plot(month_data['日龄'], month_data['avg_temperature'], 
             linewidth=2.5, marker='o', markersize=8,
             label=f'{month}月平均温度')

plt.title('最优群温度趋势与允许区间（阴影区域）', fontsize=14, pad=15)
plt.ylabel('温度 (℃)', fontsize=12)
plt.grid(True, linestyle=':')
plt.legend(bbox_to_anchor=(1.15, 1))

# ================= 湿度趋势 =================
plt.subplot(2, 1, 2)
# 先绘制允许区间
for month in high_eef['DOCdate_month'].unique():
    interval_data = interval_df[interval_df['DOCdate_month'] == month]
    plt.fill_between(interval_data['日龄'],
                    interval_data['humidity_lower'],
                    interval_data['humidity_upper'],
                    color='lightgreen', alpha=0.3,
                    label=f'{month}月允许区间')

# 再绘制湿度曲线
for month in high_eef['DOCdate_month'].unique():
    month_data = high_eef[high_eef['DOCdate_month'] == month]
    plt.plot(month_data['日龄'], month_data['avg_humidity'],
             linewidth=2.5, marker='s', markersize=8,
             label=f'{month}月平均湿度')

plt.title('最优群湿度趋势与允许区间（阴影区域）', fontsize=14, pad=15)
plt.xlabel('日龄（天）', fontsize=12)
plt.ylabel('湿度 (%)', fontsize=12)
plt.grid(True, linestyle=':')
plt.legend(bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.show()


# 设置全局样式
plt.style.use('seaborn')
sns.set_palette("husl")  # 使用更美观的调色板

plt.figure(figsize=(14, 10), dpi=120)  # 适当调整画布大小和分辨率

# ================= 温度趋势 =================
ax1 = plt.subplot(2, 1, 1)

# 绘制允许区间（使用半透明渐变色）
for month in sorted(high_eef['DOCdate_month'].unique()):
    interval_data = interval_df[interval_df['DOCdate_month'] == month]
    ax1.fill_between(interval_data['日龄'],
                    interval_data['temp_lower'],
                    interval_data['temp_upper'],
                    color=plt.cm.Blues(0.3 + month*0.1), 
                    alpha=0.2,
                    label=f'{month}月允许区间')

# 绘制温度曲线（更美观的线条样式）
line_styles = ['-', '--', '-.', ':']  # 不同线型区分月份
for i, month in enumerate(sorted(high_eef['DOCdate_month'].unique())):
    month_data = high_eef[high_eef['DOCdate_month'] == month]
    ax1.plot(month_data['日龄'], month_data['avg_temperature'], 
             linewidth=2.5, 
             linestyle=line_styles[i % len(line_styles)],
             marker='o', markersize=6, markeredgewidth=1,
             label=f'{month}月平均温度')

# 图表装饰
ax1.set_title('最优群温度趋势与允许区间对比\n(阴影区域为允许范围)', 
             fontsize=14, pad=15, fontweight='bold')
ax1.set_ylabel('温度 (℃)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.9)

# 添加边框美化
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('#dddddd')
    spine.set_linewidth(1.5)

# ================= 湿度趋势 =================
ax2 = plt.subplot(2, 1, 2)

# 绘制允许区间
for month in sorted(high_eef['DOCdate_month'].unique()):
    interval_data = interval_df[interval_df['DOCdate_month'] == month]
    ax2.fill_between(interval_data['日龄'],
                    interval_data['humidity_lower'],
                    interval_data['humidity_upper'],
                    color=plt.cm.Greens(0.3 + month*0.1),
                    alpha=0.2,
                    label=f'{month}月允许区间')

# 绘制湿度曲线
for i, month in enumerate(sorted(high_eef['DOCdate_month'].unique())):
    month_data = high_eef[high_eef['DOCdate_month'] == month]
    ax2.plot(month_data['日龄'], month_data['avg_humidity'],
             linewidth=2.5,
             linestyle=line_styles[i % len(line_styles)],
             marker='s', markersize=6, markeredgewidth=1,
             label=f'{month}月平均湿度')

# 图表装饰
ax2.set_title('最优群湿度趋势与允许区间对比\n(阴影区域为允许范围)', 
             fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel('日龄（天）', fontsize=12)
ax2.set_ylabel('湿度 (%)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.9)

# 添加边框美化
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('#dddddd')
    spine.set_linewidth(1.5)

# 整体调整
plt.tight_layout(pad=3.0)  # 增加子图间距

# 添加整体标题（可选）
plt.suptitle('最优生产性能群环境参数监控分析', 
             y=1.02, fontsize=16, fontweight='bold')

plt.show()





##########################################################################
# 获取所有月份
months = monthly_stats['DOCdate_month'].unique()

for month in months:
    # 筛选当前月份数据
    month_data = monthly_stats[monthly_stats['DOCdate_month'] == month]
    
    # 创建画布（调整figsize和增加顶部空间）
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))  # 增加高度到25
    fig.suptitle(
        f'每月最高温度趋势 - {month}月\n（按高效标志分组）', 
        fontsize=16, 
        y=1.0,  # 将标题下移
        va='bottom'  # 垂直对齐方式
    )
    
    # 为每个温度传感器绘图
    for i in range(1, 6):
        ax = axes[i-1]
        lineplot = sns.lineplot(
            data=month_data, 
            x='日龄', 
            y=f'max_temperature{i}', 
            hue='eef_high_flg',
            palette={0: '#1f77b4', 1: '#ff7f0e'},  # 更标准的蓝色和橙色
            style='eef_high_flg',  # 新增：用线型区分
            markers={0: 'o', 1: 's'},  # 明确指定标记形状
            dashes={0: (1,0), 1: (2,2)},  # 0用实线，1用虚线
            ax=ax
        )
        
        ax.set_title(f'温度传感器 {i}', pad=10, fontsize=14)
        ax.set_xlabel('日龄（天）', fontsize=12)
        ax.set_ylabel('最高温度（℃）', fontsize=12)
        
        # 优化图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, 
            labels=['高效标志: 否 (0)', '高效标志: 是 (1)'],  # 完全匹配你的需求
            title='图例说明',
            loc='upper right',
            framealpha=0.9
        )
        
        ax.grid(True, linestyle=':', alpha=0.5)  # 更细的网格线
    
    # 调整整体布局（增加顶部间距）
    plt.subplots_adjust(top=0.95, hspace=0.4)  # 增加子图间距
    plt.show()

########################################################
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置输出文件夹路径（自动创建如果不存在）
output_dir = "./xyy/eef分析/plt/temperature_plots"
os.makedirs(output_dir, exist_ok=True)

# 获取不同的月份
months = monthly_stats['DOCdate_month'].unique()

# 设置图片参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 遍历每个月份
for month in months:
    # 筛选当前月份的数据
    month_data = monthly_stats[monthly_stats['DOCdate_month'] == month]
    
    # 创建画布（调整尺寸适应图例）
    fig = plt.figure(figsize=(14, 6))
    
    # 绘制 eef_high_flg = 0 的数据
    data_0 = month_data[month_data['eef_high_flg'] == 0]
    sns.lineplot(data=data_0, x='日龄', y='min_temperature', 
                label='最低温度 (eef=0)', marker='o', markersize=5)
    sns.lineplot(data=data_0, x='日龄', y='max_temperature', 
                label='最高温度 (eef=0)', marker='s', markersize=5)
    sns.lineplot(data=data_0, x='日龄', y='avg_temperature', 
                label='平均温度 (eef=0)', marker='^', markersize=5)
    
    # 绘制 eef_high_flg = 1 的数据
    data_1 = month_data[month_data['eef_high_flg'] == 1]
    sns.lineplot(data=data_1, x='日龄', y='min_temperature', 
                label='最低温度 (eef=1)', linestyle='--', marker='o', markersize=5)
    sns.lineplot(data=data_1, x='日龄', y='max_temperature', 
                label='最高温度 (eef=1)', linestyle='--', marker='s', markersize=5)
    sns.lineplot(data=data_1, x='日龄', y='avg_temperature', 
                label='平均温度 (eef=1)', linestyle='--', marker='^', markersize=5)
    
    # 设置图表标题和标签
    plt.title(f'{month}月 - 温度指标对比 (高效群 vs 普通群)', fontsize=12, pad=20)
    plt.xlabel('日龄（天）', fontsize=10)
    plt.ylabel('温度（℃）', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    # 设置图例（右侧外部）
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False,
        ncol=1
    )
    
    # 调整布局并保存图片
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True, linestyle=':', alpha=0.5, color='gray')
    
    # 构建保存路径（使用月份作为文件名）
    filename = f"{output_dir}/{month}月_温度对比.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"已保存: {filename}")
    
    # 关闭图形释放内存
    plt.close()

print(f"\n所有图片已保存至: {os.path.abspath(output_dir)}")