

import os
import pandas as pd
import xlwings as xw
def read_transpose_excel(file_path, header_row=1, data_start_row=1):
    try:
        # 启动 Excel 应用
        app = xw.App(visible=False)
        # 打开 Excel 文件
        workbook = app.books.open(file_path)
        # 获取指定工作表
        sheet = workbook.sheets['出栏数据']

        # 获取表头（排除第一列）
        # headers = sheet.range(f'B{header_row}:{sheet.cells(header_row, sheet.api.UsedRange.Columns.Count).address}').value

        # 获取数据范围（排除第一列）
        last_row = 56
        print('####last_row',last_row)
        data_range = sheet.range(f'B{data_start_row}:{sheet.cells(last_row, sheet.api.UsedRange.Columns.Count).address}')
        data_range
        # 读取数据
        data = data_range.value
        # 创建 DataFrame
        df = pd.DataFrame(data)
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        # 创建 DataFrame
        df = df.reset_index(drop=True)

        # 转置 DataFrame
        transposed_df = df.T
        transposed_df=transposed_df.reset_index(drop=True)
        new_header = transposed_df.iloc[0]
        transposed_df = transposed_df[1:]
        transposed_df.columns = new_header

        transposed_df=transposed_df.drop(columns=[None],axis=1)
        transposed_df=transposed_df.drop(labels=[1,2,3,4],axis=0)

        sheet2 = workbook.sheets['基本信息']

        # 读取农场相关信息
        farm_name_cell='C3'
        farm_manager_cell='C4'
        chicken_house_count_cell='F3'
        breeding_batch_cell='F4'
        farm_name = sheet2.range(farm_name_cell).value
        farm_manager = sheet2.range(farm_manager_cell).value
        chicken_house_count = sheet2.range(chicken_house_count_cell).value
        breeding_batch = sheet2.range(breeding_batch_cell).value

        transposed_df['FarmName']=farm_name
        transposed_df['FarmSupervisor']=farm_manager
        transposed_df['HouseAmount']=chicken_house_count
        transposed_df['Batch']=breeding_batch

        # keepcolumns=['栋舍 House', '进雏只数\nBird placed No.', '公母\nGender', '鸡舍面积m2 \nHouse Area', '出栏密度\nDensity', '挂鸡只数\nHang No.',
        #               '挂鸡总重（kg）\nTotal hung weight', '均重（kg）\n Average weight', '小毛鸡数量\nSmall broilers No.', '小毛鸡总重（kg）\n Total weight of small broilers', 'PP死淘鸡只\nPP cull and dead No.',
        #                 '死淘总重（kg）\nDead and Cull Weight', 'PP不合格淘汰鸡\nPP Cull', 'PP淘汰鸡总重\nPP Cull bird Weight', '日龄Age', '出栏造成死亡只数 \nDead while catching', '出鸡只数\nCatching No.',
        #                   '成活率\nLivability (%)', '出鸡总重（kg）\nTotal Catched weight', '单位面积 产率\nensity, Yield(KG)/m2', '均重（kg）\nAverage weight', '累计耗料（kg）\nFeed cons. Cum.', '料肉比 FCR',
        #                     'Adjust FCR (base 2.45KG)', '欧洲指数 EEF', '药品费用（元）', '疫苗费用（元）', '消毒药费用（元）', '饲料成本（元）', '用电费用（元）', '燃气费用（元）',
        #                       '人工费用（元）', '低值易耗品（元）', '折旧费（元）', '雏鸡成本（元）', '总成本（元）', '每公斤成本（元）', '毛鸡销售收入（元）', '每栋纯利润（元）', '药品 （元/只）', '疫苗（元/只）', 
        #                       'M&V费（元/只）', '消毒药费（元/只）', '饲料（元/只）', '用电（元/只）', '燃气（元/只）', '人工（元/只）', '低值易耗品（元/只）', '折旧费（元/只）', '雏鸡成本（元/元）', '每只鸡成本（元）']
        # transposed_df=transposed_df[keepcolumns]

        # 关闭工作簿
        workbook.close()
        # 退出 Excel 应用
        app.quit()


        return transposed_df

    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        if 'app' in locals() and app:
            app.quit()
        return None

# file_path = 'D:\\太阳谷\\chickenfarming\\data\\24.12\\日报\\00_71.xlsm'
# result_df = read_transpose_excel(file_path)




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
    processed_dfs = []
    for df in all_dfs:
        # 移除全是NA的列
        df = df.dropna(axis=1, how='all')
        processed_dfs.append(df)

    # 合并所有 DataFrame
    if processed_dfs:
        combined_df = pd.concat(processed_dfs, ignore_index=True)
    
        return combined_df
    else:
        return None


# 示例调用
root_folders = ['D:\\太阳谷\\chickenfarming\\data\\24.09-日报','D:\\太阳谷\\chickenfarming\\data\\24.10-日报','D:\\太阳谷\\chickenfarming\\data\\24.11-日报','D:\\太阳谷\\chickenfarming\\data\\24.12 2\\日报','D:\\太阳谷\\chickenfarming\\data\\25.01\\日报','D:\\太阳谷\\chickenfarming\\data\\25.02\\日报-2502\\日报-2502','D:\\太阳谷\\chickenfarming\\data\\25.03\\2503-日报']
# excel_path = r"C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE"
result_df = process_all_files(root_folders)
# result_df=result_df[result_df['HouseNo']!='Total']


result_df.columns.to_list()
##出栏数据
columns = [
    'house',                    # 栋舍 House
    'birds_placed',             # 进雏只数\nBird placed No.
    'gender',                   # 公母\nGender
    'house_area_m2',            # 鸡舍面积m2 \nHouse Area
    'stocking_density',         # 出栏密度\nDensity
    'birds_hung',               # 挂鸡只数\nHang No.
    'total_hung_weight_kg',     # 挂鸡总重（kg）\nTotal hung weight
    'avg_weight_kg',            # 均重（kg）\n Average weight
    'small_broilers_count',     # 小毛鸡数量\nSmall broilers No.
    'small_broilers_weight_kg', # 小毛鸡总重（kg）\n Total weight of small broilers
    'pp_dead_culled_count',     # PP死淘鸡只\nPP cull and dead No.
    'dead_culled_weight_kg',    # 死淘总重（kg）\nDead and Cull Weight
    'pp_rejects_count',         # PP不合格淘汰鸡\nPP Cull
    'pp_rejects_weight_kg',     # PP淘汰鸡总重\nPP Cull bird Weight
    'age_days',                 # 日龄Age
    'dead_during_catch_count',  # 出栏造成死亡只数 \nDead while catching
    'birds_caught_count',       # 出鸡只数\nCatching No.
    'livability_pct',           # 成活率\nLivability (%)
    'total_caught_weight_kg',   # 出鸡总重（kg）\nTotal Catched weight
    'yield_per_m2',             # 单位面积产肉率\nDensity, Yield(KG)/m2
    'final_avg_weight_kg',      # 均重（kg）\nAverage weight
    'total_feed_kg',            # 累计耗料（kg）\nFeed cons. Cum.
    'fcr',                      # 料肉比 FCR
    'adjusted_fcr',             # Adjust FCR (base 2.45KG)
    'eef',                      # 欧洲指数 EEF
    'feed_cost',                # 饲料成本（元）
    'electricity_cost',         # 用电费用（元）
    'gas_cost',                 # 燃气费用（元）
    'depreciation_cost',        # 折旧费（元）
    'chick_cost',               # 雏鸡成本（元）
    'total_cost',               # 总成本（元）
    'cost_per_kg',              # 每公斤成本（元）
    'revenue',                  # 毛鸡销售收入（元）
    'profit_per_house',         # 每栋纯利润（元）
    'medicine_per_bird',        # 药品（元/只）
    'vaccine_per_bird',         # 疫苗（元/只）
    'mv_cost_per_bird',         # M&V费（元/只）
    'disinfectant_per_bird',    # 消毒药费（元/只）
    'feed_per_bird',            # 饲料（元/只）
    'electricity_per_bird',     # 用电（元/只）
    'gas_per_bird',             # 燃气（元/只）
    'labor_per_bird',           # 人工（元/只）
    'consumables_per_bird',     # 低值易耗品（元/只）
    'depreciation_per_bird',    # 折旧费（元/只）
    'chick_cost_per_bird',      # 雏鸡成本（元/元）
    'cost_per_bird',            # 每只鸡成本（元）
    'farm_name',                # FarmName
    'farm_supervisor',          # FarmSupervisor
    'house_count',              # HouseAmount
    'batch'                     # Batch
]
result_df.columns=columns

result_df.to_csv('./data/data_cleaned/marketingdata.csv', index=False,encoding='gbk')