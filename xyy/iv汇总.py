import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def process_and_visualize_data(file_paths, output_image_path=None):
    """
    处理多个CSV文件并可视化结果
    
    参数:
    file_paths (list): CSV文件路径列表
    output_image_path (str, optional): 图表保存路径，如果为None则显示图表
    """
    # 使用字典来存储所有数据，确保年龄组一致
    data = {}
    
    # 先收集所有年龄组
    all_ages = set()
    
    # 第一遍：收集所有年龄组
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            continue
        
        # 提取年龄组
        for _, row in df[df['Bin'].isna()].iterrows():
            if 'feature' not in row or pd.isna(row['feature']):
                continue
            age = row['feature'].split('_')[1]
            if not age.endswith('天'):
                age += '天'
            all_ages.add(age)
    
    # 初始化数据结构
    for age in sorted(all_ages, key=lambda x: int(x.split('-')[0])):
        data[age] = {
            'EEF_IN_IV': None,
            'EEF_OUT_IV': None,
            'MORTALITY_RATE_IN_IV': None,
            'MORTALITY_RATE_OUT_IV': None
        }
    
    # 第二遍：填充数据
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except Exception as e:
            continue
        
        # 确定location和metric
        location = 'IN' if '鸡舍温度' in file_path else 'OUT' if '外部' in file_path else None
        metric = 'EEF' if 'EEF' in df.columns else 'MORTALITY_RATE' if 'MORTALITY_RATE' in df.columns else None
        
        if location is None or metric is None:
            continue
        
        # 填充数据
        for _, row in df[df['Bin'].isna()].iterrows():
            if 'feature' not in row or pd.isna(row['feature']):
                continue
            age = row['feature'].split('_')[1]
            if not age.endswith('天'):
                age += '天'
            
            col_name = f"{metric}_{location}_IV"
            if age in data:
                data[age][col_name] = row['IV']
    
    # 转换为DataFrame
    result_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    result_df.rename(columns={'index': 'age'}, inplace=True)
    
    # 按年龄排序
    result_df['age_num'] = result_df['age'].apply(lambda x: int(x.split('-')[0]))
    result_df = result_df.sort_values('age_num').drop('age_num', axis=1)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    plt.rcParams["font.family"] = ["SimHei"]
    
    # 绘制四条线，只绘制有数据的线
    for col in ['EEF_IN_IV', 'EEF_OUT_IV', 'MORTALITY_RATE_IN_IV', 'MORTALITY_RATE_OUT_IV']:
        if col in result_df.columns and not result_df[col].isnull().all():
            style = 'o-' if 'IN' in col else 's-'
            label = '鸡舍' if 'IN' in col else '外部'
            label += 'EEF' if 'EEF' in col else '死亡率'
            plt.plot(result_df['age'], result_df[col], style, label=label)
    
    plt.title('不同日龄阶段各指标IV值对比')
    plt.xlabel('日龄阶段')
    plt.ylabel('IV值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return result_df

# 示例用法
if __name__ == "__main__":
    file_paths = [
        '.\\xyy\\死淘分析\\output\\冬天外部-平均_MEAN分箱_Optimized.csv',
        '.\\xyy\\死淘分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
        '.\\xyy\\eef分析\\output\\冬天外部-平均_MEAN分箱_Optimized1.csv',
        '.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv'
    ]
    
    result = process_and_visualize_data(file_paths)
    print(result)    


def process_and_visualize_data(file_paths, output_image_path=None):
    """
    处理两个CSV文件并可视化结果
    
    参数:
    file_paths (list): 包含两个CSV文件路径的列表，第一个是EEF数据，第二个是MORTALITY_RATE数据
    output_image_path (str, optional): 图表保存路径，如果为None则显示图表
    """
    # 初始化存储数据的字典
    data = {
        'age': [],
        'EEF_IN_IV': [],
        'MORTALITY_RATE_IN_IV': []
    }
    
    # 检查文件数量
    if len(file_paths) != 2:
        raise ValueError("需要提供两个文件路径，第一个是EEF数据，第二个是MORTALITY_RATE数据")
    
    # 处理EEF文件
    try:
        eef_df = pd.read_csv(file_paths[0], encoding='gbk')  # 根据实际文件编码调整
    except Exception as e:
        print(f"无法读取EEF文件 {file_paths[0]}: {e}")
        return None
    
    # 处理MORTALITY_RATE文件
    try:
        mortality_df = pd.read_csv(file_paths[1], encoding='gbk')  # 根据实际文件编码调整
    except Exception as e:
        print(f"无法读取MORTALITY_RATE文件 {file_paths[1]}: {e}")
        return None
    
    # 创建年龄到IV值的映射字典
    eef_iv_map = {}
    mortality_iv_map = {}
    
    # 处理EEF数据
    for _, row in eef_df[eef_df['Bin'].isna()].iterrows():
        if 'feature' not in row or pd.isna(row['feature']):
            continue
        parts = row['feature'].split('_')
        if len(parts) < 2:
            continue
        age = parts[1]
        # 确保年龄格式统一，如果有"天"就保留，没有就加上
        if not age.endswith('天'):
            age += '天'
        eef_iv_map[age] = row['IV']
    
    # 处理MORTALITY_RATE数据
    for _, row in mortality_df[mortality_df['Bin'].isna()].iterrows():
        if 'feature' not in row or pd.isna(row['feature']):
            continue
        parts = row['feature'].split('_')
        if len(parts) < 2:
            continue
        age = parts[1]
        # 确保年龄格式统一，如果有"天"就保留，没有就加上
        if not age.endswith('天'):
            age += '天'
        mortality_iv_map[age] = row['IV']
    
    # 确保两个数据集有相同的年龄组
    all_ages = sorted(set(eef_iv_map.keys()).union(set(mortality_iv_map.keys())))
    
    # 填充数据
    for age in all_ages:
        data['age'].append(age)
        data['EEF_IN_IV'].append(eef_iv_map.get(age, None))
        data['MORTALITY_RATE_IN_IV'].append(mortality_iv_map.get(age, None))
    
    # 创建DataFrame
    result_df = pd.DataFrame(data)
    
    # 按年龄排序（按数字顺序而非字符串顺序）
    def age_sort_key(x):
        try:
            return int(x.split('-')[0].replace('天', ''))
        except:
            return float('inf')
    
    result_df['sort_key'] = result_df['age'].apply(age_sort_key)
    result_df = result_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    # 可视化展示 - 将两个变量放在一张图中
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei"]
    
    # 绘制两个变量的折线图
    plt.plot(result_df['age'], result_df['EEF_IN_IV'], 'o-', label='鸡舍EEF')
    plt.plot(result_df['age'], result_df['MORTALITY_RATE_IN_IV'], '^-', label='鸡舍死亡率')
    
    plt.title('不同日龄阶段鸡舍EEF和死亡率IV值对比')
    plt.xlabel('日龄阶段')
    plt.ylabel('IV值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存或显示图表
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_image_path}")
    else:
        plt.show()
    
    return result_df


result = process_and_visualize_data(['.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv', '.\\xyy\\死淘分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv'])


##0604计算不同饮水的IV

file_paths = {
    '冬季': ".\\xyy\\死淘分析\\output\\冬季饮水差值分箱.csv",
    '秋季': ".\\xyy\\死淘分析\\output\\秋季饮水差值分箱.csv",
    '夏季': ".\\xyy\\死淘分析\\output\\夏季饮水差值分箱.csv",
    '春季': ".\\xyy\\死淘分析\\output\\春季饮水差值分箱.csv"
}

# 创建一个空的DataFrame来存储结果
result_df = pd.DataFrame(columns=['季节', '特征', 'IV'])


for season, file_path in file_paths.items():
    # 读取CSV文件
    df = pd.read_csv(file_path, encoding='gbk')
    
    # 过滤掉不需要的行
    filtered_df = df[
        (~df['Bin'].isin(['Special', 'Missing'])) & 
        (~df['Bin'].isna()) & 
        (df['Bin'] != '') & 
        (~df['IV'].isna())
    ]
    
    # 按特征分组，获取每个特征的IV值
    for feature, group in filtered_df.groupby('feature'):
        # 取每个特征的IV总和（因为有些文件可能有多个分箱行）
        iv_sum = group['IV'].sum()
        
        # 创建新的DataFrame行并添加到结果中
        new_row = pd.DataFrame({
            '季节': [season],
            '特征': [feature],
            'IV': [iv_sum]
        })
        result_df = pd.concat([result_df, new_row], ignore_index=True)
wide_df = result_df.pivot_table(
    index='特征', 
    columns=''
).reset_index()


# 输出结果
print(wide_df)

# 如果需要保存到CSV文件
wide_df.to_csv('.\\xyy\\死淘分析\\output\\饮水分箱IV值汇总.csv', index=False, encoding='gbk')


#####分季节分阶段IV

df = pd.read_csv(".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv", encoding='gbk')
# 根据Age_interval和feature列分组，然后筛选出Bin列为空的数据
iv_summary = df.groupby(['Age_interval', 'feature']).apply(lambda group: group[group['Bin'].isnull()]['IV'].sum()).reset_index()

# 重命名列名
iv_summary.rename(columns={0: 'IV汇总'}, inplace=True)
wide_df = iv_summary.pivot_table(
    index='Age_interval', 
    columns='feature'
).reset_index()

# 如果需要保存到CSV文件
wide_df.to_csv('.\\xyy\\死淘分析\\output\\夏季饮水分箱IV值汇总.csv', index=False, encoding='gbk')

# 输出结果
iv_summary