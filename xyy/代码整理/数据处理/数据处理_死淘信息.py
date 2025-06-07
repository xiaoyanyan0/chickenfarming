import os
import pandas as pd
import xlwings as xw


def get_dead_data(sheet, house_nos):
    # 提取死淘数据（假设从第6行开始，索引5）
    n_house = len(house_nos)
    n_cols = n_house * 10 + 1
    last_row = 56
    data_range = sheet.range(f'A6:{sheet.cells(last_row, n_cols).address}')
    data = data_range.value
    dead_data = pd.DataFrame(data)

    # 设置死淘数据列名（假设第6行为列名，索引5）
    dead_data.columns = dead_data.iloc[0]
    dead_data = dead_data[1:].reset_index(drop=True)

    # 查找“合计”所在行的索引并删除该行及之后的行
    try:
        drop_index = dead_data[dead_data['Date'] == '合计'].index[0]
        dead_data = dead_data.iloc[:drop_index]
    except IndexError:
        # 如果没有找到“合计”行，不做处理
        pass

    # 再次设置列名并删除第2行
    dead_data.columns = dead_data.iloc[0]
    dead_data = dead_data[2:]

    result = []
    for i, house in enumerate(house_nos):
        start_col = 1 + i * 10
        end_col = start_col + 10
        # 提取日期和对应鸡舍的 10 列数据
        sub_data = dead_data.iloc[:, [0]].join(dead_data.iloc[:, start_col:end_col])
        sub_data['House_No'] = house
        result.append(sub_data)

    new_dead_data = pd.concat(result, ignore_index=True)
    new_dead_data.columns = [
        'Date', 'Age', 'Dead', 'Swollen_Head', 'Weak',
        'Navel_Disease', 'Stick_Anus', 'Lame_Paralysis',
        'Mortality', 'Mortality_rate', 'Remark', 'House_No'
    ]

    return new_dead_data

# app = xw.App(visible=False)
# # 打开 Excel 文件
# workbook = app.books.open("D:\\太阳谷\\chickenfarming\\data\\24.09-日报\\01A_60.xlsm")

# # 获取指定工作表
# sheet = workbook.sheets['死淘分类']
# sheet2 = workbook.sheets['基本信息']

# # 读取 HouseNo 列数据作为 house_nos
# house_no_range = sheet2.range(f'A7:A{sheet2.api.UsedRange.Rows.Count}').value
# house_nos = [no for no in house_no_range if no]  # 去除空值
# house_nos=[x for x in house_nos if x.startswith('H')]

# # 获取死淘数据
# dead_data = get_dead_data(sheet, house_nos)

# # 读取农场相关信息
# farm_name_cell = 'C3'
# breeding_batch_cell = 'F4'
# farm_name = sheet2.range(farm_name_cell).value
# breeding_batch = sheet2.range(breeding_batch_cell).value


# dead_data['farm_name'] = farm_name
# dead_data['Batch'] = breeding_batch

# dead_data['ID_NUM'] = dead_data['farm_name']+'_'+dead_data['Batch'].astype(int).astype(str)+'_'+dead_data['House_No']

# dead_data=dead_data.drop(columns=['farm_name','Batch','House_No'],axis=1)
# workbook.close()
# # 退出 Excel 应用
# app.quit()


def read_transpose_excel(file_path):
    try:
        # 启动 Excel 应用
        app = xw.App(visible=False)
        # 打开 Excel 文件
        workbook = app.books.open(file_path)
        # 获取指定工作表
        sheet = workbook.sheets['死淘分类']
        sheet2 = workbook.sheets['基本信息']

        # 读取 HouseNo 列数据作为 house_nos
        house_no_range = sheet2.range(f'A7:A{sheet2.api.UsedRange.Rows.Count}').value
        house_nos = [no for no in house_no_range if no]  # 去除空值
        house_nos=[x for x in house_nos if x.startswith('H')]

        # 获取死淘数据
        dead_data = get_dead_data(sheet, house_nos)

        # 读取农场相关信息
        farm_name_cell = 'C3'
        breeding_batch_cell = 'F4'
        farm_name = sheet2.range(farm_name_cell).value
        breeding_batch = sheet2.range(breeding_batch_cell).value

        
        dead_data['farm_name'] = farm_name
        dead_data['Batch'] = breeding_batch

        dead_data['ID_NUM'] = dead_data['farm_name']+'_'+dead_data['Batch'].astype(int).astype(str)+'_'+dead_data['House_No']

        dead_data=dead_data.drop(columns=['farm_name','Batch','House_No'],axis=1)

        # 关闭工作簿
        workbook.close()
        # 退出 Excel 应用
        app.quit()

        return dead_data

    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        if 'app' in locals() and app:
            app.quit()
        return None

# df=read_transpose_excel("D:\\太阳谷\\chickenfarming\\data\\24.09-日报\\01A_60.xlsm")



def process_all_files(root_folders):
    # 遍历根文件夹及其子文件夹
    all_dfs = []
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(('.xlsm', '.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    df = read_transpose_excel(file_path)
                    if df is not None:
                        all_dfs.append(df)
                    print(f"当前处理的文件路径: {file_path}")  # 打印文件路径
                    print(f"当前处理的文件名: {file}")  # 打印文件名
    # 合并所有 DataFrame
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return None
    
root_folders = ['D:\\太阳谷\\chickenfarming\\data\\24.09-日报','D:\\太阳谷\\chickenfarming\\data\\24.10-日报','D:\\太阳谷\\chickenfarming\\data\\24.11-日报']
# root_folders = ['D:\\太阳谷\\chickenfarming\\data\\24.09-日报']
# excel_path = r"C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE"
result_df_dead = process_all_files(root_folders)
# result_df=result_df[result_df['HouseNo']!='Total']
result_df_dead[result_df_dead['ID_NUM']=='G30_64_H20']

result_df_dead.to_csv('./data/data_cleaned/all_dead_data2.csv', index=False,encoding='gbk')