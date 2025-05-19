import os
import pandas as pd
import xlwings as xw


def get_dead_data(sheet, house_nos):
    # 假设数据从第3行开始（索引2），这里根据实际情况调整
    n_house = len(house_nos)
    # 假设每列数据宽度，这里根据实际列数调整
    n_cols = n_house * 17 + 3
    last_row = 60
    data_range = sheet.range(f'A6:{sheet.cells(last_row, n_cols).address}')
    data = data_range.value
    dead_data = pd.DataFrame(data)

    # 设置列名，假设第3行为列名（索引2）
    dead_data.columns = dead_data.iloc[0]
    dead_data = dead_data[4:].reset_index(drop=True)

    # 这里假设没有“合计”行等特殊处理，若有可按原逻辑添加
    result = []
    for i, house in enumerate(house_nos):
        start_col = 3 + i * 17
        end_col = start_col + 17
        # 提取日期和对应鸡舍的相关数据列
        sub_data = dead_data.iloc[:, [0, 1, 2]].join(dead_data.iloc[:, start_col:end_col])
        sub_data['House_No'] = house
        result.append(sub_data)

    new_dead_data = pd.concat(result, ignore_index=True)
    new_dead_data.columns = [
        'Date', 'Highest_Temp_Outside', 'Lowest_Temp_Outside', 'Age',
        'Dead', 'Cull', 'Water', 'Feed', 'Fuel', 'Power',
        'Highest_humidity', 'Lowest_Humidity', 'Highest_Temn',
        'Lowest_Temn', 'Ventilation_Coefficient_Cold',
        'Ventilation_Coefficient_Warm', 'Fan_Num_Less36',
        'Fan_Num_More36','Fan_Num_Less50',
        'Fan_Num_More50','House_No'
    ]

    return new_dead_data



def read_transpose_excel(file_path):
    try:
        
        app = xw.App(visible=False)
        workbook = app.books.open(file_path)
        sheet = workbook.sheets['日报']
        sheet2 = workbook.sheets['基本信息']

        # 读取 HouseNo 列数据作为 house_nos
        house_no_range = sheet2.range(f'A7:A{sheet2.api.UsedRange.Rows.Count}').value
        house_nos = [no for no in house_no_range if no]  # 去除空值
        house_nos=[x for x in house_nos if x.startswith('H')]

        dead_data = get_dead_data(sheet, house_nos)

        dead_data=dead_data[dead_data['Highest_Temp_Outside'].notna()]
        # dead_data[dead_data['House_No']=='H1']
        # dead_data['Age']
        # 读取农场相关信息
        farm_name_cell = 'C3'
        breeding_batch_cell = 'F4'
        farm_name = sheet2.range(farm_name_cell).value
        breeding_batch = sheet2.range(breeding_batch_cell).value


        dead_data['farm_name'] = farm_name
        dead_data['Batch'] = breeding_batch

        dead_data['ID_NUM'] = dead_data['farm_name']+'_'+dead_data['Batch'].astype(int).astype(str)+'_'+dead_data['House_No']

        dead_data=dead_data.drop(columns=['farm_name','Batch','House_No'],axis=1)

        workbook.close()
        app.quit()


        return dead_data

    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        if 'app' in locals() and app:
            app.quit()
        return None


def process_all_files(root_folders):
    all_dfs = []
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(('.xlsm', '.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    df = read_transpose_excel(file_path)
                    if df is not None:
                        all_dfs.append(df)
                    print(f"当前处理的文件路径: {file_path}")
                    print(f"当前处理的文件名: {file}")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return None


root_folders = ['D:\\太阳谷\\chickenfarming\\data\\24.09-日报',
                'D:\\太阳谷\\chickenfarming\\data\\24.10-日报',
                'D:\\太阳谷\\chickenfarming\\data\\24.11-日报',
                'D:\\太阳谷\\chickenfarming\\data\\24.12 2\\日报',
                'D:\\太阳谷\\chickenfarming\\data\\25.01\\日报',
                'D:\\太阳谷\\chickenfarming\\data\\25.02\\日报-2502\\\日报-2502',
                'D:\\太阳谷\\chickenfarming\\data\\25.03\\2503-日报'
                ]
# root_folders = ['D:\\太阳谷\\chickenfarming\\data\\24.09-日报']
# excel_path = r"C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE"
result_df_dead = process_all_files(root_folders)

result_df_dead.shape
# result_df=result_df[result_df['HouseNo']!='Total']
# result_df_dead[result_df_dead['ID_NUM']=='G30_64_H20']

result_df_dead.to_csv('./data/data_cleaned/daily_report_data.csv', index=False, encoding='gbk')