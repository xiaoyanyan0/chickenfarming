

import os
import pandas as pd
import xlwings as xw

def read_excel_data(file_path, header_row=6, data_start_row=7, farm_name_cell='C3', farm_manager_cell='C4',
                    chicken_house_count_cell='F3', breeding_batch_cell='F4'):
    try:
        # 启动 Excel 应用
        app = xw.App(visible=False)
        # 打开 Excel 文件
        workbook = app.books.open(file_path)
        # 获取指定工作表
        sheet = workbook.sheets['基本信息']

        # 读取农场相关信息
        farm_name = sheet.range(farm_name_cell).value
        farm_manager = sheet.range(farm_manager_cell).value
        chicken_house_count = sheet.range(chicken_house_count_cell).value
        breeding_batch = sheet.range(breeding_batch_cell).value

        # 获取表头
        headers = sheet.range(f'A{header_row}:Z{header_row}').value
        headers = [h for h in headers if h is not None]
        # print('####headers', headers)
        # 获取数据范围
        last_row = sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row
        data_range = sheet.range(f'A{data_start_row}:{chr(64 + len(headers))}{last_row}')

        # 读取数据
        data = data_range.value

        # 创建 DataFrame
        df = pd.DataFrame(data, columns=headers)

        # 添加农场相关信息列
        df['农场名称'] = farm_name
        df['场长'] = farm_manager
        df['鸡舍数量'] = chicken_house_count
        df['饲养批次'] = breeding_batch

        df = df[pd.notna(df['鸡舍号\nHouse No'])]
        keys = ['鸡舍号\nHouse No', '入雏日期\nDOC date', '入雏数量\nDOC Amount', '鸡舍面积\nHouse Area', '饲养密度\nDensity', '公母\nGender', '雏源\nBirds Variety', '种蛋源\nHE Source', '种鸡周龄\nHE Age(W)', '日龄\nAge', '出栏日期\nHarvest status', '栋舍代码', '栋舍名称', '地面蛋数量', '预计出栏日期',
                '农场名称', '场长', '鸡舍数量', '饲养批次']
        values = ['HouseNo', 'DOCdate', 'DOCAmount', 'HouseArea', 'Density', 'Gender', 'BirdsVariety', 'HESource', 'HEAge', 'Age', 'Harveststatus', 'Houseid', 'HouseName', 'NumberofFloorEggs ', 'EstimatedSlaughterDate ',
                  'FarmName', 'FarmSupervisor', 'HouseAmount', 'Batch']

        columns_dict = dict(zip(keys, values))
        df = df.rename(columns=columns_dict)
        # 关闭工作簿
        workbook.close()
        # 退出 Excel 应用
        app.quit()

        return df

    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        if 'app' in locals() and app:
            app.quit()
        return None

def process_all_files(root_folders):

    # 遍历根文件夹及其子文件夹
    all_dfs = []
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(('.xlsm', '.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    df = read_excel_data(file_path)
                    if df is not None:
                        all_dfs.append(df)
                    print(f"当前处理的文件路径: {file_path}")  # 打印文件路径
                    print(f"当前处理的文件名: {file}")  # 打印文件名
        # 合并所有 DataFrame
    processed_dfs = []
    for df in all_dfs:
        # 移除全是NA的列
        df = df.dropna(axis=1, how='all')
        processed_dfs.append(df)

    # 合并所有 DataFrame
    if processed_dfs:
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        combined_df=combined_df[combined_df['HouseNo']!='Total']
        return combined_df
    else:
        return None


# 示例调用
root_folders = ['D:\\太阳谷\\chickenfarming\\data\\24.12\\日报','D:\\太阳谷\\chickenfarming\\data\\25.01\\日报','D:\\太阳谷\\chickenfarming\\data\\25.02\\日报-2502\\日报-2502','D:\\太阳谷\\chickenfarming\\data\\25.03\\2503-日报']
# excel_path = r"C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE"
result_df = process_all_files(root_folders)
result_df=result_df[result_df['HouseNo']!='Total']
result_df.to_csv('baseinfo.csv', index=False,encoding='gbk')

baseinfo=pd.read_csv('baseinfo.csv',encoding='gbk')
##出栏数据


