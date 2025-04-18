import xlwings as xw
import pandas as pd
import re

def extract_en(col):
    if col is None:
        return None
    s = str(col)
    m = re.findall(r'\n([A-Za-z][A-Za-z0-9_ ()]*)$', s)
    if m:
        return m[-1].strip()
    m2 = re.findall(r'([A-Za-z][A-Za-z0-9_ ()]+)$', s)
    if m2:
        return m2[-1].strip()
    return None

def extract_basic_info_xlsm(file_path, sheet_name='基本信息', header_row=6, max_col=20, max_row=100):
    app = xw.App(visible=False)
    try:
        wb = app.books.open(file_path)
        sheet = wb.sheets[sheet_name]
        # 读取元数据
        farm_name_value = sheet.range('C3').value
        house_amount_value = sheet.range('F3').value
        farm_supervisor_value = sheet.range('C4').value
        batch_value = sheet.range('F4').value
        # 读取数据区，扩展到A:O列（含L、M、N、O）
        rng = f'A{header_row}:O{header_row+max_row-1}'
        data = sheet.range(rng).value
        header = data[0]
        # 提取英文名并清洗字段名，L-O列强制指定
        def clean_en(col, idx):
            if idx == 11:
                return 'Building_Code'
            if idx == 12:
                return 'Building_Name'
            if idx == 13:
                return 'Eggs_on_the_Ground'
            if idx == 14:
                return 'Estimated_Harvest_Date'
            en = extract_en(col)
            if en:
                return en.replace(' ', '_')
            return None
        eng_header = [clean_en(col, i) for i, col in enumerate(header)]
        # 过滤掉包含'Total'的列
        valid_idx = [i for i, h in enumerate(eng_header) if h and 'Total' not in str(h).lower()]
        eng_header = [eng_header[i] for i in valid_idx]
        # 只读取A列有值的行
        data_rows = [
            [row[i] for i in valid_idx]
            for row in data[1:]
            if row and row[0] not in [None, '', ' ']
        ]
        df = pd.DataFrame(data_rows, columns=eng_header)

        # 排除任何一列中包含'Total'的行
        mask_total = df.apply(lambda row: not any('total' in str(cell).lower() for cell in row), axis=1)
        df = df[mask_total]
        if 'House_No' in df.columns:
            df = df[~df['House_No'].astype(str).str.lower().isin(['growout'])]
        elif 'House No' in df.columns:
            df = df[~df['House No'].astype(str).str.lower().isin(['growout'])]
        # 把元数据字段加到df前四列
        df.insert(0, 'Batch', batch_value)
        df.insert(0, 'Farm_Supervisor', farm_supervisor_value)
        df.insert(0, 'House_Amount', house_amount_value)
        df.insert(0, 'Farm_Name', farm_name_value)
        wb.close()
    finally:
        app.quit()
    return df

if __name__ == '__main__':
    import glob
    import os
    base_dir = r'C:/Users/sbjalz/OneDrive - SAS/Desktop/开发代码/太阳谷/chickenfarming/data/24.12 2/日报'
    all_files = glob.glob(os.path.join(base_dir, '*.xlsm'))
    dfs = []
    for f in all_files:
        try:
            df = extract_basic_info_xlsm(f)
            dfs.append(df)
            print(f"读取成功: {os.path.basename(f)} 共{len(df)}行")
        except Exception as e:
            print(f"读取失败: {os.path.basename(f)} 错误: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv('basic_info.csv', index=False, encoding='utf-8-sig')
        print(f'已保存所有文件数据为 basic_info.csv, 共{len(df_all)}行')
    else:
        print('没有成功读取任何数据，未生成csv文件。')
