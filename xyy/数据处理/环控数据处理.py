import os
import pandas as pd
import re
import matplotlib.pyplot as plt
def read_data_from_files(root_folder):
    # 定义需要提取的字段映射
    field_mapping = {
        'GROWTH_DAY': '日龄',
        'HISTORY_TIME': '时间',
        'TARGET_TEMP': '目标温度',
        'HOUSE_TEMP_MIN': '鸡舍温度-最低',
        'HOUSE_TEMP_AVG': '鸡舍温度-平均',
        'HOUSE_TEMP_MAX': '鸡舍温度-最高',
        'TEMP_1_AVG': '温度1-平均',
        'TEMP_2_AVG': '温度2-平均',
        'TEMP_3_AVG': '温度3-平均',
        'TEMP_4_AVG': '温度4-平均',
        'TEMP_5_AVG': '温度5-平均',
        'TEMP_6_AVG': '温度6-平均',
        'OUTSIDE_AVG': '外部-平均',
        'HUMIDITY_IN_1_AVG': 'Humidity In 1 Avg',
        'HUMIDITY_OUT_AVG': '湿度-外部-平均',
        'WATER_CON': '水',
        'FEED_CON': '饲料',
        'LEVEL': '水平'
    }

    # 定义最终需要的字段
    final_fields = [
        '日龄', '时间', '目标温度', '鸡舍温度-最低', '鸡舍温度-平均',
        '鸡舍温度-最高', '温度1-平均', '温度2-平均', '温度3-平均',
        '温度4-平均', '温度5-平均', '温度6-平均', '外部-平均',
        'Humidity In 1 Avg', '湿度-外部-平均', '水', '饲料', '水平',
        'id_no', 'house_no'
    ]

    # 用于存储所有符合条件的数据
    all_data = []

    # 第一部分：处理直接包含在G开头文件夹下的xls文件
    for farm_folder in os.listdir(root_folder):
        if re.match(r'^G(?:TF|\d{2})-\d{2}', farm_folder):
            farm_path = os.path.join(root_folder, farm_folder)
            id_no = re.search(r'G(?:TF|\d{2})-\d{2}', farm_folder).group()
            
            # 处理xls文件（保持原有逻辑不变）
            for file in os.listdir(farm_path):
                if file.endswith('.xls') and 'H' in file:
                    file_path = os.path.join(farm_path, file)
                    house_match = re.search(r'(H\d+)', file)
                    house_no = house_match.group(1) if house_match else "Unknown"
                    
                    try:
                        df = pd.read_excel(file_path)
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        all_data.append(df)
                        print(f"成功处理xls文件: {file_path}")
                    except Exception as e:
                        print(f"xls文件 {file_path} 读取失败: {e}")

    # 第二部分：处理嵌套在EXCEL_Files中的xlsx文件
    for root, dirs, files in os.walk(root_folder):
        # 提取id_no（从路径中匹配G开头文件夹）
        id_match = re.search(r'G(?:TF|\d{2})-\d{2}', root)
        if not id_match:
            continue
            
        id_no = id_match.group()
        
        # 检查是否是EXCEL_Files目录
        if os.path.basename(root) == 'EXCEL_Files':
            for file in files:
                if file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    
                    # 从父文件夹名提取house_no
                    parent_dir = os.path.basename(os.path.dirname(root))
                    house_match = re.search(r'(H\d+)', parent_dir)
                    house_no = house_match.group(1) if house_match else "Unknown"
                    
                    try:
                        df = pd.read_excel(file_path, sheet_name='History View')
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        all_data.append(df)
                        print(f"成功处理xlsx文件: {file_path}")
                    except Exception as e:
                        print(f"xlsx文件 {file_path} 读取失败: {e}")

    # 合并数据
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        # 确保包含所有需要的字段
        for col in final_fields:
            if col not in merged_df.columns:
                merged_df[col] = None
        merged_df = merged_df[final_fields]
        return merged_df
    else:
        print('未找到符合条件的数据')
        return None
    

# 公共配置
FIELD_MAPPING = {
        'GROWTH_DAY': '日龄',
        'HISTORY_TIME': '时间',
        'TARGET_TEMP': '目标温度',
        'HOUSE_TEMP_MIN': '鸡舍温度-最低',
        'HOUSE_TEMP_AVG': '鸡舍温度-平均',
        'HOUSE_TEMP_MAX': '鸡舍温度-最高',
        'TEMP_1_AVG': '温度1-平均',
        'TEMP_2_AVG': '温度2-平均',
        'TEMP_3_AVG': '温度3-平均',
        'TEMP_4_AVG': '温度4-平均',
        'TEMP_5_AVG': '温度5-平均',
        'TEMP_6_AVG': '温度6-平均',
        'OUTSIDE_AVG': '外部-平均',
        'HUMIDITY_IN_1_AVG': 'Humidity In 1 Avg',
        'HUMIDITY_OUT_AVG': '湿度-外部-平均',
        'WATER_CON': '水',
        'FEED_CON': '饲料',
        'LEVEL': '水平'
    }

FINAL_FIELDS = [
    '日龄', '时间', '目标温度', '鸡舍温度-最低', '鸡舍温度-平均',
        '鸡舍温度-最高', '温度1-平均', '温度2-平均', '温度3-平均',
        '温度4-平均', '温度5-平均', '温度6-平均', '外部-平均',
        'Humidity In 1 Avg', '湿度-外部-平均', '水', '饲料', '水平',
        'id_no', 'house_no'
]

def process_xls_files(root_folder):
    """处理直接包含在G开头文件夹下的xls文件"""
    all_data = []
    
    for farm_folder in os.listdir(root_folder):
        if re.match(r'^G(?:TF|\d{2})-\d{2}', farm_folder):
            farm_path = os.path.join(root_folder, farm_folder)
            id_no = re.search(r'G(?:TF|\d{2})-\d{2}', farm_folder).group()
            
            for file in os.listdir(farm_path):
                if file.endswith('.xls') and 'H' in file:
                    file_path = os.path.join(farm_path, file)
                    house_match = re.search(r'(H\d+)', file)
                    house_no = house_match.group(1) if house_match else "Unknown"
                    
                    try:
                        df = pd.read_excel(file_path)
                        df = df.rename(columns=FIELD_MAPPING)
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        all_data.append(df)
                        print(f"成功处理xls文件: {file_path}")
                    except Exception as e:
                        print(f"xls文件 {file_path} 读取失败: {e}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def process_xlsx_files(root_folder):
    """处理嵌套在EXCEL_Files中的xlsx文件"""
    all_data = []
    
    for root, dirs, files in os.walk(root_folder):
        id_match = re.search(r'G(?:TF|\d{2})-\d{2}', root)
        if not id_match:
            continue
            
        id_no = id_match.group()
        
        if os.path.basename(root) == 'EXCEL_Files':
            for file in files:
                if file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    parent_dir = os.path.basename(os.path.dirname(root))
                    house_match = re.search(r'(H\d+)', parent_dir)
                    house_no = house_match.group(1) if house_match else "Unknown"
                    
                    try:
                        df = pd.read_excel(file_path, sheet_name='History View')
                        df = df.rename(columns=FIELD_MAPPING)
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        all_data.append(df)
                        print(f"成功处理xlsx文件: {file_path}")
                    except Exception as e:
                        print(f"xlsx文件 {file_path} 读取失败: {e}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def read_data_from_files(xls_df,xlsx_df):
    """合并两种文件类型的处理结果"""
    # 合并数据
    if xls_df is not None and xlsx_df is not None and len(xls_df.columns)==len(xlsx_df.columns):
        merged_df = pd.concat([xls_df, xlsx_df], ignore_index=True)
        print("直接合并")
    elif len(xls_df.columns)!=len(xlsx_df.columns):
        xlsx_df=xlsx_df.drop(columns=[])
    elif xls_df is not None:
        merged_df = xls_df
    elif xlsx_df is not None:
        merged_df = xlsx_df
    else:
        print('未找到符合条件的数据')
        return None
    
    # 确保包含所有需要的字段
    for col in FINAL_FIELDS:
        if col not in merged_df.columns:
            merged_df[col] = None
    
    return merged_df[FINAL_FIELDS]


root_folder='D:\\太阳谷\\chickenfarming\\data\\24.10-环控数据'
merge_df1_1=process_xls_files(root_folder)
merge_df1_2=process_xlsx_files(root_folder)





merge_df1[merge_df1['house_no']=='H4历史数据收集'][['id_no','house_no']]
merge_df1['id_no']
merge_df1[merge_df1['id_no']=='Unknown']['id_no']
merge_df1['house_no'].value_counts()
# df = pd.read_excel('D:\\太阳谷\\chickenfarming\\data\\24.09-环控数据\\G03-64\\G03-64-H1.xls')
# farm_path='D:\\太阳谷\\chickenfarming\\data\\24.09-环控数据'

merge_df1=merge_df1.drop_duplicates()
# 159243
# merge_df1[['id_no','house_no','日龄','时间']].drop_duplicates()
159224

# 找出重复的组合
dup_combinations = merge_df1[['id_no','house_no','日龄','时间']].value_counts().reset_index()
dup_combinations = dup_combinations[dup_combinations['count'] > 1]

print("重复的组合及出现次数:")
print(dup_combinations)

merge_df1['id_no'].drop_duplicates()

root_folder='D:\\太阳谷\\chickenfarming\\data\\24.09-环控数据'
merge_df2=read_data_from_files(root_folder)

merge_df2[merge_df2['id_no']=='G01-60']

df = pd.read_excel('D:\\太阳谷\\chickenfarming\\data\\24.09-环控数据\\G28-25\\G28\\鸡群_25\\House_H9-L6-26700\EXCEL_Files\\鸡群_25House_H9-L6-26700.xlsx', sheet_name='History View')
df.columns.to_list()
df2 = pd.read_excel('D:\\太阳谷\\chickenfarming\\data\\24.09-环控数据\\G28-25\\G28\\鸡群_25\\House_H9-L6-26700\EXCEL_Files\\鸡群_25House_H9-L6-26700.xlsx', sheet_name='History View')