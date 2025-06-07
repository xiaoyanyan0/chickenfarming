import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
plt.rcParams["font.family"] = ["SimHei"]
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
    plt.suptitle(f"饮水差值与死淘率关系 - {feature_name}-{season}", fontsize=16, y=0.98)
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

def plot_all_variables(binned_file, stats_file, output_dir='.\\xyy\\死淘分析\\output',season='winter'):
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
            filename = f"{season}_temp_mortality_{safe_name}.png"
            output_path = os.path.join(output_dir, filename)
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
    plt.suptitle(f"湿度与EEF关系 - {feature_name}", fontsize=16, y=0.98)
    plt.title(f"湿度范围: {x_min:.1f}℃ 至 {x_max:.1f}℃ | 总IV值: {total_iv:.4f}", 
              fontsize=12, pad=10)

    # 计算柱状图的位置和宽度
    processed['bar_x'] = processed['parsed_lower']
    processed['bar_width'] = processed['parsed_upper'] - processed['parsed_lower']
    processed['line_x'] = (processed['parsed_lower'] + processed['parsed_upper']) / 2

    colors = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, len(processed)))
    bars = ax1.bar(processed['bar_x'], processed['Count'], 
                  width=processed['bar_width'], color=colors,
                  alpha=0.8, align='edge', label='数量')

    ax1.set_xlabel('湿度 %)', fontsize=12)
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
def plot_all_variables_eef(binned_file, stats_file, output_dir='.\\xyy\\eef分析\\output',season='winter'):
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
            filename = f"{season}_temp_eef_{safe_name}.png"
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved {output_path}")
# 使用示例


plot_all_variables(".\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv", ".\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN温度分布.csv",season='spring_O')
plot_all_variables(".\\xyy\\死淘分析\\output\\春天外部-平均_MEAN分箱_Optimized.csv", ".\\xyy\\死淘分析\\output\\春天外部-平均_MEAN温度分布.csv",season='spring_O')

plot_all_variables_eef(".\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv", ".\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN温度分布.csv",season='spring_O')
plot_all_variables_eef(".\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv", ".\\xyy\\死淘分析\\output\\春天外部-平均_MEAN温度分布.csv",season='spring_O')

plot_all_variables(".\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv", ".\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分布.csv",season='spring_O')

plot_all_variables_eef(".\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv", ".\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分布.csv",season='spring_O')



plot_all_variables(".\\xyy\\死淘分析\\output\\winter_饮水差值分箱.csv", ".\\xyy\\死淘分析\\output\\冬季饮水差值分布.csv",season='winter')

plot_all_variables(".\\xyy\\死淘分析\\output\\夏季饮水差值分箱.csv", ".\\xyy\\死淘分析\\output\\夏季饮水差值分布.csv",season='summer')
plot_all_variables(".\\xyy\\死淘分析\\output\\秋季饮水差值分箱.csv", ".\\xyy\\死淘分析\\output\\秋季饮水差值分布.csv",season='autumn')
plot_all_variables(".\\xyy\\死淘分析\\output\\春季饮水差值分箱.csv", ".\\xyy\\死淘分析\\output\\春季饮水差值分布.csv",season='spring')

# winter_饮水差值分箱
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置全局字体
plt.rcParams['font.family'] = 'Arial'

# 数据
age_groups = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23']
age_midpoints = [1, 4, 7, 10, 13, 16, 19, 22]
all_ages = range(0, 24)

# 温度数据
out_lower = [7.5, 7, 9.1, 11.4, 14.4, 8.6, 9.8, 9.2]  
out_upper = [8.2, 19.1, 16.3, 14.3, 17.4, 17.9, 17.3, 13.8]  
in_lower = [33.6, 30.8, 28.5, 28.6, 25.8, 27.1, 26.2, 26.4]  
in_upper = [33.8, 31.6, 29.4, 28.8, 26.4, 27.8, 26.9, 27.6]  
out_avg = [(l+u)/2 for l,u in zip(out_lower, out_upper)]
in_avg = [(l+u)/2 for l,u in zip(in_lower, in_upper)]

# 创建画布
plt.figure(figsize=(14, 7), dpi=100, facecolor='white')

# 定义颜色
colors = {
    'out': {'main': '#3498db', 'light': '#AED6F1'},
    'in': {'main': '#E74C3C', 'light': '#F5B7B1'},
    'bg': '#F9F9F9'
}

# 设置背景
ax = plt.gca()
ax.set_facecolor(colors['bg'])

# 绘制填充区域（先绘制确保在底层）
plt.fill_between(age_midpoints, out_lower, out_upper, color=colors['out']['light'], alpha=0.5, label='Out Range')
plt.fill_between(age_midpoints, in_lower, in_upper, color=colors['in']['light'], alpha=0.3, label='In Range')

# 绘制折线（后绘制确保在上层）
out_line, = plt.plot(age_midpoints, out_avg, color=colors['out']['main'], 
                     marker='o', markersize=8, linewidth=2.5, label='Out Average')
in_line, = plt.plot(age_midpoints, in_avg, color=colors['in']['main'], 
                    marker='s', markersize=8, linewidth=2.5, label='In Average')

# 添加数据标签（优化位置和样式）
def add_labels(x, y, lower, upper, color):
    offset = (max(in_upper + out_upper) - min(in_lower + out_lower)) * 0.03
    for xi, yi, l, u in zip(x, y, lower, upper):
        plt.text(xi, yi+offset, f'{yi:.1f}°C', ha='center', va='bottom', 
                color=color['main'], fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        plt.text(xi, l-offset/2, f'{l:.1f}', ha='center', va='top', 
                color=color['main'], fontsize=7, alpha=0.8)
        plt.text(xi, u+offset/2, f'{u:.1f}', ha='center', va='bottom', 
                color=color['main'], fontsize=7, alpha=0.8)

add_labels(age_midpoints, out_avg, out_lower, out_upper, colors['out'])
add_labels(age_midpoints, in_avg, in_lower, in_upper, colors['in'])

# 设置坐标轴
plt.title('Temperature Comparison: Outdoor vs Indoor by Age Group', 
          pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Age (days)', fontsize=12, labelpad=10)
plt.ylabel('Temperature (°C)', fontsize=12, labelpad=10)
plt.xticks(all_ages, fontsize=10)
plt.yticks(fontsize=10)

# 添加年龄组标签
for age, label in zip(age_midpoints, age_groups):
    plt.text(age, min(out_lower + in_lower) - 3, label, 
             ha='center', va='top', fontsize=10, color='#555555')

# 设置范围
plt.ylim(min(out_lower + in_lower) - 5, max(out_upper + in_upper) + 5)
plt.xlim(-0.5, 23.5)

# 网格线
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='#cccccc')

# 图例
legend = plt.legend(frameon=True, fontsize=10, 
                   facecolor='white', edgecolor='#dddddd',
                   bbox_to_anchor=(1, 1), loc='upper left')
legend.get_frame().set_linewidth(0.5)

# 调整布局
plt.tight_layout()
plt.show()