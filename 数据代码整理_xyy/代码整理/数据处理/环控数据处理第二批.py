import os
import re
import pandas as pd
from collections import defaultdict

# 标准字段映射配置：定义业务标准字段与原始数据字段的映射关系
# 键为标准字段名，值为包含原始数据中可能的字段别名列表（支持中英文和英文缩写）
STANDARD_FIELDS = {
    '日龄': ['GROWTH_DAY', '日龄', 'Age', '生长日龄'],
    '时间': ['HISTORY_TIME', '时间', 'Time', '记录时间'],
    '目标温度': ['TARGET_TEMP', '目标温度', 'TargetTemp'],
    '鸡舍温度-最低': ['HOUSE_TEMP_MIN', '鸡舍温度-最低', 'MinTemp'],
    '鸡舍温度-平均': ['HOUSE_TEMP_AVG', '鸡舍温度-平均', 'AvgTemp'],
    '鸡舍温度-最高': ['HOUSE_TEMP_MAX', '鸡舍温度-最高', 'MaxTemp'],
    '温度1-平均':['TEMP_1_AVG','温度1-平均'],
    '温度2-平均':['TEMP_2_AVG','温度2-平均'],
    '温度3-平均':['TEMP_3_AVG','温度3-平均'],
    '温度4-平均':['TEMP_4_AVG','温度4-平均'],
    '温度5-平均':['TEMP_5_AVG','温度5-平均'],
    '温度6-平均':['TEMP_6_AVG','温度6-平均'],
    '外部-平均':['OUTSIDE_AVG','外部-平均'],
    '湿度内部平均':['HUMIDITY_IN_1_AVG','Humidity In 1 Avg'],
    '湿度外部平均':['HUMIDITY_OUT_AVG','湿度-外部-平均'],
    '水':['WATER_CON','水'],
    '饲料':['FEED_CON','饲料'],
    '水平':['LEVEL','水平']
}

# 最终标准列定义：指定数据清洗后必须包含的列（含业务标识列）
FINAL_COLUMNS = [
    '日龄', '时间', '目标温度', '鸡舍温度-最低', '鸡舍温度-平均',
    '鸡舍温度-最高', '温度1-平均', '温度2-平均', '温度3-平均',
    '温度4-平均', '温度5-平均', '温度6-平均', '外部-平均',
    '湿度内部平均', '湿度外部平均', '水', '饲料', '水平',
    'id_no', 'house_no'
]

def standardize_dataframe(df):
    """
    数据框列名标准化函数
    功能：将原始数据列名映射为标准字段名，并补全缺失的标准列
    参数：
    df (DataFrame): 原始数据框
    返回：
    DataFrame: 标准化后的数据框
    """
    # 创建列名映射字典：从原始字段名到标准字段名的映射（仅保留存在的原始字段）
    column_mapping = {}
    for standard_col, alt_names in STANDARD_FIELDS.items():
        for alt_name in alt_names:
            if alt_name in df.columns.to_list():
                column_mapping[alt_name] = standard_col
                break  # 优先匹配第一个存在的别名
    
    # 重命名列：将原始字段名替换为标准字段名
    df = df.rename(columns=column_mapping)
    
    # 补全缺失的标准列：确保最终列列表完整（除标识列外，缺失列填充None）
    for col in FINAL_COLUMNS:
        if col not in df.columns and col not in ['id_no', 'house_no']:
            df[col] = None  # 非标识列缺失时填充空值
    
    return df

def process_xls_files(root_folder):
    """
    处理xls格式环控数据文件
    功能：遍历指定文件夹下的农场目录，解析xls文件并标准化数据
    参数：
    root_folder (str): 根数据文件夹路径
    返回：
    DataFrame: 合并后的标准化数据
    """
    all_data = []  # 存储所有文件的处理结果
    
    # 遍历根文件夹下的子目录（农场编号目录，格式如G28-25、GTF-01）
    for farm_folder in os.listdir(root_folder):
        # 正则匹配农场编号格式：以G开头，后跟TF或两位数字，再跟两位数字（如G28-25）
        if re.match(r'^G(?:TF|\d{2})-\d{2}', farm_folder):
            farm_path = os.path.join(root_folder, farm_folder)
            id_no = re.search(r'G(?:TF|\d{2})-\d{2}', farm_folder).group()  # 提取农场ID
        
            # 遍历农场目录下的文件
            for file in os.listdir(farm_path):
                if file.endswith('.xls') and 'H' in file:  # 筛选xls文件且文件名包含'H'（鸡舍标识）
                    file_path = os.path.join(farm_path, file)
                    
                    # 从文件名中提取鸡舍编号（格式如H1、H23）
                    house_no_match = re.search(r'(H\d+)', file)
                    house_no = house_no_match.group(1) if house_no_match else "Unknown"  # 未匹配到则标记为Unknown
                    
                    try:
                        # 读取xls文件（默认第一个工作表）
                        df = pd.read_excel(file_path)
                        df = standardize_dataframe(df)  # 标准化列名
                        
                        # 添加农场ID和鸡舍编号（业务标识字段）
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        
                        # 保留标准列：过滤原始数据中不在FINAL_COLUMNS中的列
                        all_data.append(df[FINAL_COLUMNS])
                    
                    except Exception as e:
                        print(f"处理文件 {file_path} 出错: {str(e)}")
    
    # 合并所有文件数据（若有数据）
    return pd.concat(all_data, ignore_index=True) if all_data else None

def process_xlsx_files(root_folder):
    """
    处理xlsx格式环控数据文件（支持嵌套目录结构）
    功能：递归遍历文件夹，解析xlsx文件并标准化数据
    参数：
    root_folder (str): 根数据文件夹路径
    返回：
    DataFrame: 合并后的标准化数据
    """
    all_data = []  # 存储所有文件的处理结果
    
    # 递归遍历文件夹（root: 当前目录路径， dirs: 子目录列表， files: 文件列表）
    for root, dirs, files in os.walk(root_folder):
        # 提取农场编号：从目录路径中匹配G开头的编号（如G28-25）
        id_match = re.search(r'G(?:TF|\d{2})-\d{2}', root)
        if not id_match:
            continue  # 跳过无农场编号的目录
        id_no = id_match.group()  # 农场ID
        
        # 仅处理EXCEL_Files目录下的文件（假设数据存储在该子目录）
        if os.path.basename(root) == 'EXCEL_Files':
            for file in files:
                if file.endswith('.xlsx'):  # 筛选xlsx文件
                    file_path = os.path.join(root, file)
                    
                    # 优先从文件名中提取鸡舍编号（支持House_xxx或鸡群_xxxHouse_Hx格式）
                    house_match = re.search(r'(?:House_|鸡群_\d+House_)(H\d+)', file)
                    if not house_match:  # 若文件名中未匹配到，尝试从父目录名提取
                        parent_dir = os.path.basename(os.path.dirname(root))
                        house_match = re.search(r'(H\d+)', parent_dir)
                    house_no = house_match.group(1) if house_match else "Unknown"  # 未匹配到则标记为Unknown
                    
                    try:
                        # 读取指定工作表（假设数据在'History View'表）
                        df = pd.read_excel(file_path, sheet_name='History View')
                        df = standardize_dataframe(df)  # 标准化列名
                        
                        # 添加业务标识字段
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        
                        # 确保列顺序与标准一致（重排列，缺失列自动填充None）
                        final_df = df.reindex(columns=FINAL_COLUMNS)
                        all_data.append(final_df)
                        
                        print(f"成功处理: {file_path} | 提取的鸡舍号: {house_no}")
                    
                    except Exception as e:
                        print(f"文件 {file_path} 读取失败: {str(e)}")
    
    # 合并数据（过滤空DataFrame）
    if all_data:
        return pd.concat([df for df in all_data if not df.empty], ignore_index=True)
    return None

def read_data_from_files(xls_data, xlsx_data):
    """
    主合并函数：合并xls和xlsx处理结果
    参数：
    xls_data (DataFrame): xls文件处理结果
    xlsx_data (DataFrame): xlsx文件处理结果
    返回：
    DataFrame: 合并后的标准化数据集
    """
    # 合并两种格式数据（按行拼接，忽略原始索引）
    merged_df = pd.concat([xls_data, xlsx_data], ignore_index=True)
    
    # 最终标准化检查：补全可能缺失的列（处理空数据情况）
    for col in FINAL_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = None  # 确保列完整性
    
    return merged_df[FINAL_COLUMNS]  # 按标准列顺序返回数据


### 数据处理主流程 ###
# 处理24.10-环控数据
root_folder = 'D:\\太阳谷\\chickenfarming\\data\\24.10-环控数据'
merge_df1_1 = process_xls_files(root_folder)  # 处理xls文件
merge_df1_2 = process_xlsx_files(root_folder)  # 处理xlsx文件

# 检查列数（调试用）
len(merge_df1_1.columns.to_list())
len(merge_df1_2.columns.to_list())

merge_df1 = read_data_from_files(merge_df1_1, merge_df1_2)  # 合并数据
merge_df1 = merge_df1.drop_duplicates()  # 去重

# 处理24.09-环控数据
root_folder = 'D:\\太阳谷\\chickenfarming\\data\\24.09-环控数据'
merge_df2_1 = process_xls_files(root_folder)
merge_df2_2 = process_xlsx_files(root_folder)

len(merge_df2_1.columns.to_list())
len(merge_df2_2.columns.to_list())

merge_df2 = read_data_from_files(merge_df2_1, merge_df2_2)
merge_df2 = merge_df2.drop_duplicates()

# 处理24.11-环控数据（示例，原代码中被注释，此处保留逻辑）
root_folder = 'D:\\太阳谷\\chickenfarming\\data\\24.11-环控数据'
merge_df5_1 = process_xls_files(root_folder)
merge_df5_2 = process_xlsx_files(root_folder)

len(merge_df5_1.columns.to_list())
len(merge_df5_2.columns.to_list())

merge_df5 = read_data_from_files(merge_df5_1, merge_df5_2)
merge_df5 = merge_df5.drop_duplicates()

# 合并所有月份数据
all_HumTem_data = pd.concat([merge_df1, merge_df2, merge_df5]).reset_index(drop=True)

# 标准化ID格式：将农场编号中的'-'替换为'_'（便于后续处理）
all_HumTem_data['id_no'] = all_HumTem_data['id_no'].str.replace('-', '_')
# 生成唯一标识：农场ID_鸡舍编号
all_HumTem_data['ID_NUM'] = all_HumTem_data['id_no'] + '_' + all_HumTem_data['house_no']

# 保存中间结果
all_HumTem_data.to_csv('./data/data_cleaned/all_HumTem_data2.csv', index=False, encoding='gbk')


