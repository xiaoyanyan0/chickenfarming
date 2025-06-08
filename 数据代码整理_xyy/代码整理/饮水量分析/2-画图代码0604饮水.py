###用于得到分箱结果后的画图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def get_temp_range(stats_file, var_name, age_interval):
    """从统计文件获取变量温度范围，增加 age_interval 参数"""
    try:
        stats_df = pd.read_csv(stats_file, encoding='gbk')
        # 增加按 Age_interval 过滤
        stats_df = stats_df[stats_df['Age_interval'] == age_interval]
        match_cols = [col for col in stats_df.columns if var_name in col]
        if not match_cols:
            print(f"Warning: No matching column for {var_name} in Age_interval {age_interval}")
            return None, None

        min_val = stats_df.loc[stats_df['index'] =='min', match_cols[0]].values[0]
        max_val = stats_df.loc[stats_df['index'] =='max', match_cols[0]].values[0]
        # padding = (max_val - min_val) * 0.1
        return min_val, max_val

    except Exception as e:
        print(f"Error getting temp range for Age_interval {age_interval}: {e}")
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


def plot_temp_mortality(df, season, age_interval, x_min=None, x_max=None, feature_name=""):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        processed, _ = parse_bin_data(df)
        if processed.empty:
            print("Empty data, skip plotting")
            return
    except Exception as e:
        print(f"Error parsing data for Age_interval {age_interval}: {e}")
        return

    # 计算总IV值
    total_iv = processed['IV'].iloc[:-1].sum() if 'IV' in processed.columns else 0

    # 使用传入的x_min和x_max替换-inf和inf
    if x_min is not None:
        processed['parsed_lower'] = processed['parsed_lower'].replace(-np.inf, x_min)
    if x_max is not None:
        processed['parsed_upper'] = processed['parsed_upper'].replace(np.inf, x_max)

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 添加主标题和副标题，包含IV值和 Age_interval
    plt.suptitle(f"饮水差值与死淘率关系 - {feature_name}-{season}-{age_interval}", fontsize=16, y=0.98)
    plt.title(f"饮水范围差值: {x_min:.1f}cm³/只 至 {x_max:.1f}cm³/只 | 总IV值: {total_iv:.4f}",
              fontsize=12, pad=10)

    # 计算柱状图的位置和宽度
    processed['bar_x'] = processed['parsed_lower']
    processed['bar_width'] = processed['parsed_upper'] - processed['parsed_lower']
    processed['line_x'] = (processed['parsed_lower'] + processed['parsed_upper']) / 2

    colors = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(processed)))
    bars = ax1.bar(processed['bar_x'], processed['Count'],
                   width=processed['bar_width'], color=colors,
                   alpha=0.8, align='edge', label='数量')

    ax1.set_xlabel('饮水范围差值 (cm³/只)', fontsize=12)
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


def plot_all_variables(binned_file, stats_file, output_dir='.\\xyy\\死淘分析\\output', season='winter'):
    """绘制所有变量的图表，增加按 Age_interval 处理逻辑"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(binned_file, encoding='gbk')
    stats_df = pd.read_csv(stats_file, encoding='gbk')
    age_intervals = stats_df['Age_interval'].unique()

    for age_interval in age_intervals:
        age_df = df[df['Age_interval'] == age_interval]
        for feature in age_df['feature'].unique():
            if pd.isna(feature):
                continue

            feature_df = age_df[age_df['feature'] == feature]
            x_min, x_max = get_temp_range(stats_file, feature, age_interval)

            if x_min is None or x_max is None:
                print(f"Skipping {feature} in Age_interval {age_interval} due to missing temp range")
                continue

            # 提取有意义的变量名称（去除特殊字符）
            clean_feature_name = ''.join(c for c in feature if c.isalnum() or c in ['_', '-',' '])

            fig = plot_temp_mortality(feature_df, season, age_interval, x_min, x_max, feature_name=clean_feature_name)

            if fig:
                safe_name = feature.replace(' ', '_').replace('/', '_')
                filename = f"{season}_{age_interval}_mortality_{safe_name}.png"
                output_path = os.path.join(output_dir, filename)
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"Saved {output_path}")

# 使用示例
plot_all_variables(".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv", ".\\xyy\\死淘分析\\output\\summer_饮水差值分布.csv",season='summer')