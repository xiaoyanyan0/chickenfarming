import os
import re
import pandas as pd
from collections import defaultdict

# 标准字段映射配置
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



FINAL_COLUMNS = [
    '日龄', '时间', '目标温度', '鸡舍温度-最低', '鸡舍温度-平均',
    '鸡舍温度-最高', '温度1-平均', '温度2-平均', '温度3-平均',
    '温度4-平均', '温度5-平均', '温度6-平均', '外部-平均',
    '湿度内部平均', '湿度外部平均', '水', '饲料', '水平',
    'id_no', 'house_no'
]

def standardize_dataframe(df):
    """标准化数据框列名和结构"""
    # 创建列名映射字典
    column_mapping = {}
    for standard_col, alt_names in STANDARD_FIELDS.items():
        for alt_name in alt_names:
            if alt_name in df.columns.to_list():
                column_mapping[alt_name] = standard_col
                break
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 添加缺失的标准列
    for col in FINAL_COLUMNS:
        if col not in df.columns and col not in ['id_no', 'house_no']:
            df[col] = None
    
    return df

def process_xls_files(root_folder):
    """处理xls文件并标准化（修复索引错误）"""
    all_data = []
    
    # 确保 FINAL_COLUMNS 无重复
    # FINAL_COLUMNS = list(dict.fromkeys(FINAL_COLUMNS))
    
    for root, dirs, files in os.walk(root_folder):
        farm_match = re.search(r'G(?:TF|\d{2})[_-]\d{2}', os.path.basename(root), re.IGNORECASE)
        if not farm_match:
            continue
            
        id_no = farm_match.group()
        
        for file in files:
            if file.lower().endswith('.xls'):
                file_path = os.path.join(root, file)
                house_match = re.search(r'([Hh]\d+)', file, re.IGNORECASE) or re.search(r'\d+', file)
                house_no = house_match.group() if house_match else "Unknown"
                
                try:
                    # 读取数据
                    df = pd.read_excel(file_path)
                    
                    # 标准化数据（确保返回的列名唯一）
                    df = standardize_dataframe(df)
                    df = df.loc[:, ~df.columns.duplicated()]  # 去重列名
                    
                    # 添加标识列
                    df['id_no'] = id_no
                    df['house_no'] = house_no
                    
                    # 筛选最终列（仅保留存在的列）
                    final_columns = [col for col in FINAL_COLUMNS if col in df.columns]
                    final_df = df[final_columns]
                    all_data.append(final_df)
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {str(e)}")
    
    # 合并时忽略索引
    return pd.concat(all_data, ignore_index=True) if all_data else None

def process_xlsx_files(root_folder):
    """处理xlsx文件并标准化"""
    all_data = []
    
    for root, dirs, files in os.walk(root_folder):
        # 提取农场编号（G开头格式）
        id_match = re.search(r'G(?:TF|\d{2})-\d{2}', root)
        if not id_match:
            continue
            
        id_no = id_match.group()
        
        # 检查是否是EXCEL_Files目录
        if os.path.basename(root) == 'EXCEL_Files':
            for file in files:
                if file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    
                    # 改进的鸡舍编号提取逻辑（兼容多种文件名格式）
                    house_match = re.search(r'(?:House_|鸡群_\d+House_)(H\d+)', file)
                    if not house_match:  # 如果文件名中没有，再从父目录名尝试提取
                        parent_dir = os.path.basename(os.path.dirname(root))
                        house_match = re.search(r'(H\d+)', parent_dir)
                    
                    house_no = house_match.group(1) if house_match else "Unknown"
                    
                    try:
                        # 读取Excel文件
                        df = pd.read_excel(file_path, sheet_name='History View')
                        
                        # 标准化数据
                        df = standardize_dataframe(df)
                        
                        # 添加标识字段
                        df['id_no'] = id_no
                        df['house_no'] = house_no
                        
                        # 确保只保留标准列
                        final_df = df.reindex(columns=FINAL_COLUMNS)
                        all_data.append(final_df)
                        
                        print(f"成功处理: {file_path} | 提取的鸡舍号: {house_no}")
                        
                    except Exception as e:
                        print(f"文件 {file_path} 读取失败: {str(e)}")
    
    # 合并数据（添加空DataFrame检查）
    if all_data:
        return pd.concat([df for df in all_data if not df.empty], ignore_index=True)
    return None

def read_data_from_files(xls_data,xlsx_data):
    """主函数，合并处理结果"""
    
    # 合并所有数据
    merged_df = pd.concat([xls_data,xlsx_data],ignore_index=True)
    
    # 最终标准化检查
    for col in FINAL_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = None
    
    return merged_df[FINAL_COLUMNS]



root_folder='D:\\太阳谷\\chickenfarming\\data\\24.12 2\\环控数据'
merge_df1_1=process_xls_files(root_folder)
merge_df1_2=process_xlsx_files(root_folder)

len(merge_df1_1.columns.to_list())
len(merge_df1_2.columns.to_list())

merge_df1=read_data_from_files(merge_df1_1,merge_df1_2)
merge_df1=merge_df1.drop_duplicates()
# merge_df1_2['湿度外部平均'].value_counts()
merge_df1['id_no'].value_counts()
merge_df1['house_no'].value_counts()
merge_df1.shape



root_folder='D:\\太阳谷\\chickenfarming\\data\\25.01\\环控数据'
merge_df2_1=process_xls_files(root_folder)
merge_df2_2=process_xlsx_files(root_folder)

len(merge_df2_1.columns.to_list())
len(merge_df2_2.columns.to_list())

merge_df2=read_data_from_files(merge_df2_1,merge_df2_2)
merge_df2=merge_df2.drop_duplicates()
# merge_df1_2['湿度外部平均'].value_counts()
merge_df2['id_no'].value_counts()
merge_df2['house_no'].value_counts()

merge_df2[merge_df2['house_no']=='Unknown']['id_no']
merge_df2.shape
merge_df2[merge_df2['id_no']=='G28-24']



root_folder='D:\\太阳谷\\chickenfarming\\data\\25.02\\环控数据-2502\\环控数据-2502'
merge_df3_1=process_xls_files(root_folder)
merge_df3_2=process_xlsx_files(root_folder)

len(merge_df3_1.columns.to_list())
len(merge_df3_2.columns.to_list())

merge_df3=read_data_from_files(merge_df3_1,merge_df3_2)
merge_df3=merge_df3.drop_duplicates()
# merge_df1_2['湿度外部平均'].value_counts()
merge_df3['id_no'].value_counts()
merge_df3['house_no'].value_counts()
merge_df3.shape


# root_folder='D:\\太阳谷\\chickenfarming\\data\\24.10-环控数据'
# merge_df4_1=process_xls_files(root_folder)
# merge_df4_2=process_xlsx_files(root_folder)

# len(merge_df4_1.columns.to_list())
# len(merge_df4_2.columns.to_list())

# merge_df4=read_data_from_files(merge_df4_1,merge_df4_2)
# merge_df4=merge_df4.drop_duplicates()
# # merge_df4_2['湿度外部平均'].value_counts()
# merge_df4['id_no'].value_counts()
# merge_df4['house_no'].value_counts()
# merge_df4.shape



root_folder='D:\\太阳谷\\chickenfarming\\data\\25.03\\2503-环控'
merge_df5_1=process_xls_files(root_folder)
merge_df5_2=process_xlsx_files(root_folder)

len(merge_df5_1.columns.to_list())
len(merge_df5_2.columns.to_list())

merge_df5=read_data_from_files(merge_df5_1,merge_df5_2)
merge_df5=merge_df5.drop_duplicates()
# merge_df4_2['湿度外部平均'].value_counts()
merge_df5['id_no'].value_counts()
merge_df5['house_no'].value_counts()
merge_df5.shape
merge_df5[merge_df5['id_no']=='G28-25']



all_HumTem_data=pd.concat([merge_df1,merge_df2,merge_df3,merge_df5]).reset_index(drop=True)
# all_HumTem_data=all_HumTem_data.reset_index(drop=True)

all_HumTem_data['id_no'].value_counts()
all_HumTem_data['house_no'].value_counts()

all_HumTem_data=all_HumTem_data[all_HumTem_data['house_no']!='30']

# all_HumTem_data[all_HumTem_data['id_no']=='G30_66'][['id_no','house_no']].drop_duplicates()



all_HumTem_data['id_no']=all_HumTem_data['id_no'].str.replace('-','_')
all_HumTem_data['ID_NUM']=all_HumTem_data['id_no']+'_'+all_HumTem_data['house_no']

all_HumTem_data.to_csv('./data/data_cleaned/all_HumTem_data1.csv', index=False,encoding='gbk')
all_HumTem_data.head()

data=all_HumTem_data.drop(columns=['id_no','house_no'],axis=1).copy()
# 需要转换为数值类型的字段

numeric_columns = [
     '目标温度', '鸡舍温度-最低', '鸡舍温度-平均',
    '鸡舍温度-最高', '温度1-平均', '温度2-平均', '温度3-平均',
    '温度4-平均', '温度5-平均', '温度6-平均', '外部-平均',
    '湿度内部平均', '湿度外部平均', '水', '饲料', '水平',
]
# 将需要统计的字段转换为数值类型
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
# 将日龄列转换为数值类型
data['日龄'] = pd.to_numeric(data['日龄'], errors='coerce')


# 定义温度相关列
temp_cols = [f'温度{i}-平均' for i in range(1, 7)]

# 按 house_no、id_no 和日龄分组
grouped = data.groupby(['ID_NUM', '日龄'])

# 统计每个分组内的最高温度、最低温度、平均温度以及 Humidity In 1 Avg 的最值和均值
agg_result = grouped.agg({
    **{col: ['max', 'min', 'mean'] for col in temp_cols},
    '湿度内部平均': ['max', 'min', 'mean']
})

# 重新设置列名
agg_result.columns = ['_'.join(col).strip() for col in agg_result.columns.values]
# agg_result.head()

# 计算每个日龄所有时间的最高温度（温度 1 - 平均到温度 6 - 平均的最高值）
agg_result['最高温度'] = agg_result[[f'{col}_max' for col in temp_cols]].max(axis=1)
# 计算每个日龄所有时间的最低温度（温度 1 - 平均到温度 6 - 平均的最低值）
agg_result['最低温度'] = agg_result[[f'{col}_min' for col in temp_cols]].min(axis=1)
# 计算每个日龄所有时间的平均温度（温度 1 - 平均到温度 6 - 平均的平均值）
agg_result['平均温度'] = agg_result[[f'{col}_mean' for col in temp_cols]].mean(axis=1)

# 计算每日温差
agg_result['每日温差'] = agg_result['最高温度'] - agg_result['最低温度']


agg_result.columns.to_list()
# 重命名 Humidity In 1 Avg 的统计结果列
# agg_result = agg_result.rename(columns={
#     'Humidity In 1 Avg_max': 'Humidity In 1 Avg 最高值',
#     'Humidity In 1 Avg_min': 'Humidity In 1 Avg 最低值',
#     'Humidity In 1 Avg_mean': 'Humidity In 1 Avg 平均值'
# })

# 合并统计结果与原数据



# 计算每个日龄每个时间的最高温度（温度 1 - 平均到温度 6 - 平均的最高值）
data['最高温度'] = data[temp_cols].max(axis=1)
# 计算每个日龄每个时间的最低温度（温度 1 - 平均到温度 6 - 平均的最低值）
data['最低温度'] = data[temp_cols].min(axis=1)
# 计算每个日龄每个时间的平均温度（温度 1 - 平均到温度 6 - 平均的平均值）
data['平均温度'] = data[temp_cols].mean(axis=1)


grouped2 = data.sort_values(by=['ID_NUM', '日龄','时间']).reset_index(drop=True)

# 定义计算变化率的函数
def calculate_change_rate(series):
    return series.pct_change()


# 计算平均温度、最高温度和最低温度的变化率
grouped2['平均温度变化率'] = grouped2.groupby(['ID_NUM', '日龄'])['平均温度'].transform(calculate_change_rate)
grouped2['最高温度变化率'] = grouped2.groupby(['ID_NUM', '日龄'])['最高温度'].transform(calculate_change_rate)
grouped2['最低温度变化率'] = grouped2.groupby(['ID_NUM', '日龄'])['最低温度'].transform(calculate_change_rate)

agg_result2=grouped2.groupby(['ID_NUM', '日龄'])[['平均温度变化率','最高温度变化率','最低温度变化率']].mean()

agg_result['平均温度变化率']=agg_result2['平均温度变化率']
agg_result['最高温度变化率']=agg_result2['最高温度变化率']
agg_result['最低温度变化率']=agg_result2['最低温度变化率']

agg_result=agg_result.reset_index()

agg_result.columns.to_list()

agg_result.head()
agg_result[agg_result['ID_NUM']=='G28_25_H1']

agg_result.to_csv('./data/data_cleaned/HumTem_data_agg1.csv', index=False,encoding='gbk')