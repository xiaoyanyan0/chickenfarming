import os
import pandas as pd

def readBasedata(folder_path):
    all_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为 .xlsx 或 .xlsm
            if file.endswith(('.xlsx', '.xlsm')):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                try:
                    # 读取文件的所有工作表
                    df_dict = pd.read_excel(file_path, sheet_name=None)
                    # 检查是否存在“基本信息”工作表
                    if '基本信息' in df_dict:
                        # 获取“基本信息”工作表的数据
                        basic_info_sheet = df_dict['基本信息']
                        print(f"成功读取 的基本信息工作表")
                        # 提取农场名称、场长、鸡舍数量、饲养批次
                        farm_name = basic_info_sheet.iloc[2, 2]
                        farm_supervisor = basic_info_sheet.iloc[3, 2]
                        house_amount = basic_info_sheet.iloc[2, 4]
                        batch = basic_info_sheet.iloc[3, 4]

                        # 提取表格数据部分
                        data_start_row = 6
                        try:
                            data_end_row = basic_info_sheet[basic_info_sheet.iloc[:, 0].isnull()].index[0]
                        except IndexError:
                            print(f"在 {file_path} 中未找到数据结束行，跳过该文件。")
                            continue
                        data_df = basic_info_sheet.iloc[data_start_row:data_end_row].reset_index(drop=True)
                        data_df.columns = ['鸡舍号', '入雏日期', '入雏数量', '鸡舍面积', '饲养密度', '公母', '雏源', '种蛋源', '种鸡周龄', '日龄', '出栏日期', '栋舍代码', '栋舍名称', '地面蛋数量', '预计出栏日期']

                        # 添加农场名称、场长、鸡舍数量、饲养批次到表格数据中
                        data_df['农场名称'] = farm_name
                        data_df['场长'] = farm_supervisor
                        data_df['鸡舍数量'] = house_amount
                        data_df['饲养批次'] = batch
                        all_data.append(data_df)
                    else:
                        print(f"{file_path} 中不存在 '基本信息' 工作表")
                except Exception as e:
                    print(f"读取 {file_path} 时出现错误: {e}")

    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)
    else:
        all_data = pd.DataFrame()
    return all_data



