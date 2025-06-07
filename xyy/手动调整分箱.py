import pandas as pd
import numpy as np
import re
from io import StringIO
def update_binning_with_woe_iv(variable_name, new_bins, input_file, output_file,target='EEF'):
    # Read the original data
    df = pd.read_csv(input_file, encoding='gbk')
    
    # Filter rows for the specified variable (excluding the total row)
    variable_rows = df[df['feature'] == variable_name]
    variable_rows = variable_rows[variable_rows['Bin'].notna()]
    
    if len(variable_rows) == 0:
        raise ValueError(f"Variable '{variable_name}' not found in the input file")
    
    # Process the original bins to extract boundaries
    def parse_bin(bin_str):
        # 处理特殊情况
        if pd.isna(bin_str) or not isinstance(bin_str, str):
            raise ValueError(f"Invalid bin string: {bin_str}")
        
        # 标准化分箱字符串格式
        bin_str = bin_str.strip()
        bin_str = bin_str.replace('(-inf', '[-inf').replace('inf)', 'inf]')  # 统一无穷边界格式
        
        # 尝试标准区间格式 [a,b) 或 (a,b] 等
        match = re.match(r'^([\[\(])(.*?),(.*?)([\]\)])$', bin_str)
        if match:
            left_bracket, left_num_str, right_num_str, right_bracket = match.groups()
            
            # 转换数值部分
            try:
                left_num = float(left_num_str) if not 'inf' in left_num_str else -np.inf
                right_num = float(right_num_str) if not 'inf' in right_num_str else np.inf
            except ValueError:
                # 处理非数字的特殊值
                left_num = -np.inf if 'inf' in left_num_str else np.nan
                right_num = np.inf if 'inf' in right_num_str else np.nan
            
            # 确定边界是否包含在内
            left_inclusive = left_bracket == '['
            right_inclusive = right_bracket == ']'
            
            return (left_num, left_inclusive, right_num, right_inclusive)
        
        # 处理单值分箱（例如 "Missing" 或 "999"）
        if bin_str in ['Missing', 'missing', 'MISSING', 'nan', 'NaN', 'NAN']:
            return (np.nan, False, np.nan, False)
        
        # 尝试解析单个数值
        try:
            num = float(bin_str)
            return (num, True, num, True)
        except ValueError:
            pass
        
        # 如果无法解析，抛出明确的错误
        raise ValueError(f"Cannot parse bin string: '{bin_str}'. Expected format like '[a,b)' or '(-inf,a]'.")
    
    # Create new bin structure
    for bin_str in new_bins:
        try:
            left_num, left_inc, right_num, right_inc = parse_bin(bin_str)
        except ValueError as e:
            print(f"Error parsing bin: {bin_str} - {str(e)}")
            raise
    new_bin_boundaries = []
    for bin_str in new_bins:
        left_num, left_inc, right_num, right_inc = parse_bin(bin_str)
        new_bin = {
            'bin_str': bin_str,
            'left_num': left_num,
            'left_inclusive': left_inc,
            'right_num': right_num,
            'right_inclusive': right_inc
        }
        new_bin_boundaries.append(new_bin)
    
    # Sort new bins by their left boundary
    new_bin_boundaries.sort(key=lambda x: x['left_num'])
    
    # Function to check if a value falls in a bin
    def in_bin(value, bin_def):
        left_cond = (value >= bin_def['left_num']) if bin_def['left_inclusive'] else (value > bin_def['left_num'])
        right_cond = (value <= bin_def['right_num']) if bin_def['right_inclusive'] else (value < bin_def['right_num'])
        return left_cond and right_cond
    
    # Improved function to find which new bin an original bin falls into
    def find_new_bin(original_bin):
        orig_left, orig_left_inc, orig_right, orig_right_inc = parse_bin(original_bin)
        
        # Check all points that might need to be mapped
        check_points = []
        if orig_left != -np.inf:
            check_points.append(orig_left + (0.0001 if orig_left_inc else -0.0001))
        if orig_right != np.inf:
            check_points.append(orig_right + (-0.0001 if orig_right_inc else 0.0001))
        check_points.append((orig_left + orig_right)/2)  # midpoint
        
        found_bins = set()
        for point in check_points:
            for new_bin in new_bin_boundaries:
                if in_bin(point, new_bin):
                    found_bins.add(new_bin['bin_str'])
                    break
        
        if len(found_bins) == 1:
            return found_bins.pop()
        elif len(found_bins) > 1:
            # If multiple bins found, choose the one that contains the midpoint
            midpoint = (orig_left + orig_right)/2
            for new_bin in new_bin_boundaries:
                if in_bin(midpoint, new_bin):
                    return new_bin['bin_str']
        
        # If still not found, try to find the closest bin
        if orig_right != np.inf:
            for new_bin in new_bin_boundaries:
                if orig_right <= new_bin['right_num']:
                    return new_bin['bin_str']
        elif orig_left != -np.inf:
            for new_bin in reversed(new_bin_boundaries):
                if orig_left >= new_bin['left_num']:
                    return new_bin['bin_str']
        
        raise ValueError(f"Cannot map original bin {original_bin} to any new bin")
    
    # Create mapping from original bins to new bins
    bin_mapping = {}
    for orig_bin in variable_rows['Bin'].unique():
        try:
            new_bin = find_new_bin(orig_bin)
            bin_mapping[orig_bin] = new_bin
        except ValueError as e:
            print(f"Warning: {e}. Attempting to find closest bin...")
            # Try to find the closest bin by numeric distance
            orig_left, _, orig_right, _ = parse_bin(orig_bin)
            min_dist = float('inf')
            best_bin = None
            
            for new_bin in new_bin_boundaries:
                new_left, _, new_right, _ = parse_bin(new_bin['bin_str'])
                
                # Calculate distance between intervals
                dist = min(abs(orig_left - new_left), abs(orig_left - new_right),
                          abs(orig_right - new_left), abs(orig_right - new_right))
                
                if dist < min_dist:
                    min_dist = dist
                    best_bin = new_bin['bin_str']
            
            if best_bin:
                print(f"Mapping {orig_bin} to closest bin {best_bin}")
                bin_mapping[orig_bin] = best_bin
            else:
                raise ValueError(f"Failed to map original bin {orig_bin} to any new bin")
    
    # Aggregate data for new bins
    new_bin_data = []
    for new_bin in new_bins:
        # Get all original bins that map to this new bin
        original_bins_in_new = [k for k, v in bin_mapping.items() if v == new_bin]
        
        # Filter rows for these original bins
        mask = variable_rows['Bin'].isin(original_bins_in_new)
        rows_for_new_bin = variable_rows[mask]
        
        if len(rows_for_new_bin) == 0:
            # If no original bins found for this new bin, skip it
            continue
        
        # Sum counts
        total_count = rows_for_new_bin['Count'].sum()
        non_event = rows_for_new_bin['Non-event'].sum()
        event = rows_for_new_bin['Event'].sum()
        
        # Calculate percentages and rates

        count_pct = total_count / variable_rows['Count'].sum()  # 修正：Count占总行数(404)的比例
        event_rate = event / total_count if total_count > 0 else 0
        
        new_bin_data.append({
            'Bin': new_bin,
            'Count': total_count,
            'Count (%)': count_pct,
            'Non-event': non_event,
            'Event': event,
            'Event rate': event_rate,
            'feature': variable_name
            
        })
    
    # Calculate WoE and IV for new bins
    total_non_event = variable_rows['Non-event'].sum()
    total_event = variable_rows['Event'].sum()
    overall_event_rate = total_event / (total_non_event + total_event)
    
    for bin_data in new_bin_data:
        # Calculate WoE
        non_event_pct = bin_data['Non-event'] / total_non_event if total_non_event > 0 else 0
        event_pct = bin_data['Event'] / total_event if total_event > 0 else 0
        
        if non_event_pct > 0 and event_pct > 0:
            woe = np.log(non_event_pct / event_pct)
        else:
            woe = 0  # Handle cases where there are no events or non-events
        
        # Calculate IV component
        iv_component = (non_event_pct - event_pct) * woe
        
        bin_data['WoE'] = woe
        bin_data['IV'] = iv_component
    
    # Calculate total IV
    total_iv = sum(bin['IV'] for bin in new_bin_data)
    
    # Calculate JS (Jensen-Shannon divergence)
    for bin_data in new_bin_data:
        p = bin_data['Non-event'] / total_non_event if total_non_event > 0 else 0
        q = bin_data['Event'] / total_event if total_event > 0 else 0
        m = 0.5 * (p + q)
        
        if p > 0 and m > 0:
            js_p = p * np.log(p / m)
        else:
            js_p = 0
            
        if q > 0 and m > 0:
            js_q = q * np.log(q / m)
        else:
            js_q = 0
            
        js = 0.5 * (js_p + js_q)
        bin_data['JS'] = js
    
    # Calculate total JS
    total_js = sum(bin['JS'] for bin in new_bin_data)
    
    # Calculate target (use weighted average of original target values)
    for bin_data in new_bin_data:
        original_bins_in_new = [k for k, v in bin_mapping.items() if v == bin_data['Bin']]
        mask = variable_rows['Bin'].isin(original_bins_in_new)
        rows_for_bin = variable_rows[mask]
        
        if len(rows_for_bin) > 0:
            # Calculate weighted average of target based on counts
            total_weight = rows_for_bin['Count'].sum()
            if total_weight > 0:
                weighted_eef = (rows_for_bin[target] * rows_for_bin['Count']).sum() / total_weight
                bin_data[target] = weighted_eef
            else:
                bin_data[target] = variable_rows[target].mean()
        else:
            bin_data[target] = variable_rows[target].mean()
    
    # Create new DataFrame for the variable
    new_variable_df = pd.DataFrame(new_bin_data)
    
    # 计算汇总行的加权平均EEF
    total_eef = 0
    total_count = sum(bin['Count'] for bin in new_bin_data)
    
    if total_count > 0:
        for bin_data in new_bin_data:
            total_eef += bin_data[target] * bin_data['Count']
        weighted_avg_eef = total_eef / total_count
    else:
        weighted_avg_eef = variable_rows[target].mean()
    
    # Add the total row
    total_row = {
        'Bin': '',
        'Count': total_non_event + total_event,
        'Count (%)': 1.0,
        'Non-event': total_non_event,
        'Event': total_event,
        'Event rate': overall_event_rate,
        'WoE': np.nan,
        'IV': total_iv,
        'JS': total_js,
        'feature': variable_name,
        target: weighted_avg_eef  # 修正：使用加权平均EEF
    }
    new_variable_df = pd.concat([new_variable_df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Reorder columns to match original
    columns_order = ['Bin', 'Count', 'Count (%)', 'Non-event', 'Event', 'Event rate', 
                     'WoE', 'IV', 'JS', 'feature', target]
    new_variable_df = new_variable_df[columns_order]
    
    # Replace the variable's rows in the original DataFrame
    # First remove all rows for this variable
    df = df[df['feature'] != variable_name]
    # Then add the new rows
    df = pd.concat([df, new_variable_df], ignore_index=True)
    
    # Save to output file
    df.to_csv(output_file, index=False, encoding='gbk')
    
    return df
# 使用示例
####冬季eef鸡舍温度
update_binning_with_woe_iv(
    variable_name='W3_3-5天_鸡舍温度-平均_MEAN',
    new_bins=['(-inf, 31.25)','[31.25, 31.65)','[31.65, 31.95)' ,'[31.95, 32.05)', '[32.05, 32.25)','[32.25, inf)'],
    input_file='.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv'
)
update_binning_with_woe_iv(
    variable_name='W3_15-17天_鸡舍温度-平均_MEAN',
    new_bins=['[-inf, 26.45)','[26.45, 26.65)','[26.65, inf)' ],
    input_file='.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv'
)

update_binning_with_woe_iv(
    variable_name='W3_18-20天_鸡舍温度-平均_MEAN',
    new_bins=[
    '(-inf, 24.45)',
    '[24.45, 24.65)',
    '[24.65, 25.45)',
    '[25.45, 25.65)',
    '[25.65, inf)'
],
    input_file='.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv'
)
#######冬季eef外部平均
update_binning_with_woe_iv(
    variable_name='W3_6-8天_外部-平均_MEAN',
    new_bins=[
    '[-inf, 1.15)',
    '[1.15, 5.05)',
    '[5.05, 6.75)',
    '[6.75, inf)'
],
    input_file='.\\xyy\\eef分析\\output\\冬天外部-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天外部-平均_MEAN分箱_Optimized1.csv'
)
update_binning_with_woe_iv(
    variable_name='W3_21-23天_外部-平均_MEAN',
    new_bins=[
    '[-inf, 2.75)',
    '[2.75, 3.85)',
    '[3.85, 6.30)',
    '[6.30, 8.85)',
    '[8.85, inf)'
],
    input_file='.\\xyy\\eef分析\\output\\冬天外部-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天外部-平均_MEAN分箱_Optimized1.csv'
)
####冬季死淘鸡舍温度


update_binning_with_woe_iv(
    variable_name='W3_3-5天_鸡舍温度-平均_MEAN',
    new_bins=[
    '(-inf, 30.95)',
    '[30.95, 31.25)',
    '[31.25, 31.95)',
    '[31.95, 32.05)',
    '[32.05, 32.25)',
    '[32.25, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)
update_binning_with_woe_iv(
    variable_name='W3_21-23天_鸡舍温度-平均_MEAN',
    new_bins=[
    '[-inf, 23.15)',
    '[23.15, 23.95)',
    '[23.95, 24.55)',
    '[24.55, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\冬天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)


###秋季eef外部温度

update_binning_with_woe_iv(
    variable_name='W3_0-2天_外部-平均_MEAN',
    new_bins=[
    '(-inf, 13.75)',
    '[13.75, 18.95)',
    '[18.95, 27.05)',
    '[27.05, 29.65)',
    '[29.65, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\秋天外部-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\秋天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)
###秋季eef鸡舍温度


update_binning_with_woe_iv(
    variable_name='W3_3-5天_鸡舍温度-平均_MEAN',
    new_bins=[
        '[-inf, 31.25)',
    '[31.25, 31.55)',
    '[31.55, 31.75)',
    '[31.75, 31.95)',
    '[31.95, inf)',
    ],
    input_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)


update_binning_with_woe_iv(
    variable_name='W3_9-11天_鸡舍温度-平均_MEAN',
    new_bins=[

    '[-inf, 28.65)',
    '[28.65, 29.05)',
    '[29.05, 29.65)',
    '[29.65, inf)'

    ],
    input_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)



update_binning_with_woe_iv(
    variable_name='W3_18-20天_鸡舍温度-平均_MEAN',
    new_bins=[
   '(-inf, 25.35)',
    '[25.35, 25.65)',
    '[25.65, 25.95)',
    '[25.95, 26.05)',
    '[26.05, 26.35)',
    '[26.35, 28.75)',
    '[28.75, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)


update_binning_with_woe_iv(
    variable_name='W3_21-23天_鸡舍温度-平均_MEAN',
    new_bins=[
   '(-inf, 23.75)',
    '[23.75, 24.25)',
    '[24.25, 24.75)',
    '[24.75, 24.95)',
    '[24.95, 25.05)',
    '[25.05, 25.35)',
    '[25.35, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\秋天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

###夏天鸡舍温度eef
update_binning_with_woe_iv(
    variable_name='W3_18-20天_鸡舍温度-平均_MEAN',
    new_bins=[
  '[-inf, 29.45)',
    '[29.45, 30.55)',
    '[30.55, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)
update_binning_with_woe_iv(
    variable_name='W3_21-23天_鸡舍温度-平均_MEAN',
    new_bins=[
   '(-inf, 28.55)',
    '[28.55, 29.15)',
    '[29.15, 30.25)',
    '[30.25, inf)'
    ],
      input_file='.\\xyy\\eef分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_3-5天_鸡舍温度-平均_MEAN',
    new_bins=[
   '(-inf, 31.85)',
    '[31.85, 32.25)',
    '[32.25, 32.65)',
    '[32.65, 32.95)',
    '[32.95, inf)'
    ],
      input_file='.\\xyy\\eef分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)


###夏天eef外部温度

update_binning_with_woe_iv(
    variable_name='W3_9-11天_外部-平均_MEAN',
    new_bins=[
     '(-inf, 12.05)',
    '[12.05, 24.55)',
    '[24.55, 27.20)',
    '[27.20, 29.85)',
    '[29.85, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\夏天外部-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)
update_binning_with_woe_iv(
    variable_name='W3_18-20天_外部-平均_MEAN',
    new_bins=[
      '(-inf, 27.55)',
    '[27.55, 28.85)',
    '[28.85, inf)',
        ],
    input_file='.\\xyy\\eef分析\\output\\夏天外部-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_15-17天_外部-平均_MEAN',
    new_bins=[
     '(-inf, 10.30)',
    '[10.30, 19.55)',
    '[19.55, 27.25)',
    '[27.25, 29.25)',
    '[29.25, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\夏天外部-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)
###夏季死淘鸡舍温度



update_binning_with_woe_iv(
    variable_name='W3_18-20天_鸡舍温度-平均_MEAN',
    new_bins=[
    '[-inf, 29.15)',
    '[29.15, 29.35)',
    '[29.35, 30.55)',
    '[30.55, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_15-17天_鸡舍温度-平均_MEAN',
    new_bins=[
   '[-inf, 29.45)',
    '[29.45, 30.15)',
    '[30.15, 30.45)',
    '[30.45, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)


update_binning_with_woe_iv(
    variable_name='W3_18-20天_鸡舍温度-平均_MEAN',
    new_bins=[
    '[-inf, 29.15)',
    '[29.15, 29.35)',
    '[29.35, 30.35)',
    '[30.35, 30.55)',
    '[30.55, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\夏天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)
###夏季死淘外部温度

update_binning_with_woe_iv(
    variable_name='W3_15-17天_外部-平均_MEAN',
    new_bins=[
    '[-inf, 19.55)',
    '[19.55, 28.15)',
    '[28.15, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\夏天外部-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\夏天外部-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

####春季eef外部温度

update_binning_with_woe_iv(
    variable_name='W3_0-2天_外部-平均_MEAN',
    new_bins=[
     '[-inf, 4.75)',
    '[4.75, 5.95)',
    '[5.95, 7.55)',
    '[7.55, 8.25)',
    '[8.25, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_6-8天_外部-平均_MEAN',
    new_bins=[
    '[-inf, 5.35)',
    '[5.35, 7.65)',
    '[7.65, 9.10)',
    '[9.10, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)


update_binning_with_woe_iv(
    variable_name='W3_15-17天_外部-平均_MEAN',
    new_bins=[
   '(-inf, 6.45)',
    '[6.45, 8.05)',
    '[8.05, inf)',
        ],
    input_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_18-20天_外部-平均_MEAN',
    new_bins=[
    '[-inf, 4.55)',
    '[4.55, 6.15)',
    '[6.15, 9.85)',
    '[9.85, inf)',
        ],
    input_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天外部-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)


###春季eef鸡舍温度
update_binning_with_woe_iv(
    variable_name='W3_0-2天_鸡舍温度-平均_MEAN',
    new_bins=[
    '[-inf, 33.65)',
    '[33.65, 33.75)',
    '[33.75, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_9-11天_鸡舍温度-平均_MEAN',
    new_bins=[
     '[-inf, 28.55)',
    '[28.55, 28.75)',
    '[28.75, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_21-23天_鸡舍温度-平均_MEAN',
    new_bins=[
    '(-inf, 25.35)',
    '[25.35, 26.35)',
    '[26.35, inf)'
        ],
    input_file='.\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

####春季死淘鸡舍温度
update_binning_with_woe_iv(
    variable_name='W3_0-2天_鸡舍温度-平均_MEAN',
    new_bins=[
       '[-inf, 33.65)',
    '[33.65, 33.75)',
    '[33.75, 34.15)',
    '[34.15, inf)',
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)


update_binning_with_woe_iv(
    variable_name='W3_9-11天_鸡舍温度-平均_MEAN',
    new_bins=[
     '[-inf, 28.55)',
    '[28.55, 28.75)',
    '[28.75, 29.45)',
    '[29.45, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_12-14天_鸡舍温度-平均_MEAN',
    new_bins=[
    '[-inf, 27.15)',
    '[27.15, 27.65)',
    '[27.65, 27.75)',
    '[27.75, 27.85)',
    '[27.85, inf)',
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_15-17天_鸡舍温度-平均_MEAN',
    new_bins=[
    '[-inf, 26.45)',
    '[26.45, 26.75)',
    '[26.75, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_21-23天_鸡舍温度-平均_MEAN',
    new_bins=[
   '[-inf, 24.85)',
    '[24.85, 25.35)',
    '[25.35, 25.85)',
    '[25.85, 26.15)',
    '[26.15, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天鸡舍温度-平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)
####冬季死淘内部湿度


update_binning_with_woe_iv(
    variable_name='W3_3-5天_湿度内部平均_MEAN',
    new_bins=[
   '[-inf, 60.15)',
    '[60.15, 68.70)',
    '[68.70, 70.35)',
    '[70.35, 78.10)',
    '[78.10, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)
####冬季eef内部湿度
update_binning_with_woe_iv(
    variable_name='W3_0-2天_湿度内部平均_MEAN',
    new_bins=[
   '[-inf, 39.35)',
    '[39.35, 48.05)',
    '[48.05, 57.60)',
    '[57.60, 65.55)',
    '[65.55, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_9-11天_湿度内部平均_MEAN',
    new_bins=[
      '[-inf, 62.65)',
    '[62.65, 72.05)',
    '[72.05, 75.75)',
    '[75.75, 84.65)',
    '[84.65, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_12-14天_湿度内部平均_MEAN',
    new_bins=[
   '(-inf, 67.95)',
    '[67.95, 84.55)',
    '[84.55, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_18-20天_湿度内部平均_MEAN',
    new_bins=[
    '[-inf, 62.50)',
    '[62.50, 69.55)',
    '[69.55, 70.95)',
    '[70.95, inf)',
    ],
    input_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\冬天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

####秋季死淘内部湿度
update_binning_with_woe_iv(
    variable_name='W3_6-8天_湿度内部平均_MEAN',
    new_bins=[
      '(-inf, 66.95)',
    '[66.95, 72.05)',
    '[72.05, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\秋天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\秋天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

###夏季eef内部湿度

update_binning_with_woe_iv(
    variable_name='W3_0-2天_湿度内部平均_MEAN',
    new_bins=[
    '(-inf, 50.25)',
    '[50.25, 67.40)',
    '[67.40, 74.45)',
    '[74.45, 79.05)',
    '[79.05, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_3-5天_湿度内部平均_MEAN',
    new_bins=[
    '[-inf, 71.75)',
    '[71.75, 80.80)',
    '[80.80, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_6-8天_湿度内部平均_MEAN',
    new_bins=[
     '(-inf, 76.45)',
    '[76.45, 80.95)',
    '[80.95, 85.35)',
    '[85.35, inf)',
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_12-14天_湿度内部平均_MEAN',
    new_bins=[
   '[-inf, 65.45)',
    '[65.45, 79.45)',
    '[79.45, 81.75)',
    '[81.75, inf)',
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_18-20天_湿度内部平均_MEAN',
    new_bins=[
   '(-inf, 64.05)',
    '[64.05, 76.10)',
    '[76.10, 81.05)',
    '[81.05, 84.65)',
    '[84.65, inf)',
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_18-20天_湿度内部平均_MEAN',
    new_bins=[
   '(-inf, 64.05)',
    '[64.05, 76.10)',
    '[76.10, 81.05)',
    '[81.05, 84.65)',
    '[84.65, inf)',
    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_21-23天_湿度内部平均_MEAN',
    new_bins=[
   '(-inf, 63.55)',
    '[63.55, 81.95)',
    '[81.95, inf)',

    ],
    input_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

###夏季死淘内部湿度
update_binning_with_woe_iv(
    variable_name='W3_9-11天_湿度内部平均_MEAN',
    new_bins=[
      '(-inf, 66.65)',
    '[66.65, 77.95)',
    '[77.95, 79.30)',
    '[79.30, 82.40)',
    '[82.40, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

##春季死淘内部湿度
  

update_binning_with_woe_iv(
    variable_name='W3_3-5天_湿度内部平均_MEAN',
    new_bins=[
     '(-inf, 55.00)',
    '[55.00, 62.55)',
    '[62.55, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_6-8天_湿度内部平均_MEAN',
    new_bins=[
     '[-inf, 47.55)',
    '[47.55, 60.40)',
    '[60.40, 63.05)',
    '[63.05, inf)',
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_12-14天_湿度内部平均_MEAN',
    new_bins=[
   '[-inf,63.00)',
    '[63.00, 70.25)',
    '[70.25, 72.15)',
    '[72.15, inf)'
    ],
    input_file='.\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\死淘分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)


#####春季eef内部湿度

update_binning_with_woe_iv(
    variable_name='W3_3-5天_湿度内部平均_MEAN',
    new_bins=[
    '[-inf, 54.40)',
    '[54.40, 71.35)',
    '[71.35, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized.csv',
    output_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_6-8天_湿度内部平均_MEAN',
    new_bins=[
       '[-inf, 59.95)',
    '[59.95,69.50)',
    '[69.50, 74.15)',
    '[74.15, 83.35)',
    '[83.35, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_9-11天_湿度内部平均_MEAN',
    new_bins=[
      '[-inf, 65.95)',
    '[65.95, 72.35)',
    '[72.35, 76.35)',
    '[76.35, 79.35)',
    '[79.35, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_15-17天_湿度内部平均_MEAN',
    new_bins=[
     '[-inf, 63.50)',
    '[63.50, 75.60)',
    '[75.60, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)


update_binning_with_woe_iv(
    variable_name='W3_18-20天_湿度内部平均_MEAN',
    new_bins=[
   '[-inf, 63.60)',
    '[63.60, 70.60)',
    '[70.60, 77.65)',
    '[77.65, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_21-23天_湿度内部平均_MEAN',
    new_bins=[
    '[-inf, 65.15)',
    '[65.15, 77.25)',
    '[77.25, 81.70)',
    '[81.70, inf)'
    ],
    input_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    output_file='.\\xyy\\eef分析\\output\\春天湿度内部平均_MEAN分箱_Optimized1.csv',
    target='EEF'
)

####0605
update_binning_with_woe_iv(
    variable_name='W3_9-11天_探头温度标准差_MEAN',
    new_bins=[
    '[-inf, 0.24)',
    '[0.24, 0.26)',
    '[0.26, inf)'
    ],
    input_file='.\\xyy\\模型分析\\winter\\winter_MORTALITY_RATE_分箱_Optimized.csv',
    output_file='.\\xyy\\模型分析\\winter\\winter_MORTALITY_RATE_分箱_Optimized1.csv',
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='W3_3-5天_探头温度标准差_MEAN',
    new_bins=[
     '[-inf, 0.15)',
    '[0.15, 0.35)',
    '[0.35, inf)'
    ],
    input_file='.\\xyy\\模型分析\\winter\\winter_EEF_分箱_Optimized.csv',
    output_file='.\\xyy\\模型分析\\winter\\winter_EEF_分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_3-5天_探头温度标准差_MEAN',
    new_bins=[
     '[-inf, 0.15)',
    '[0.15, 0.35)',
    '[0.35, inf)'
    ],
    input_file='.\\xyy\\模型分析\\winter\\winter_EEF_分箱_Optimized.csv',
    output_file='.\\xyy\\模型分析\\winter\\winter_EEF_分箱_Optimized1.csv',
    target='EEF'
)

update_binning_with_woe_iv(
    variable_name='W3_21-23天_探头温度标准差_MEAN',
    new_bins=[
     '[-inf, 0.45)',
    '[0.45, 0.55)',
    '[0.55, inf)'
    ],
    input_file='.\\xyy\\模型分析\\winter\\winter_EEF_分箱_Optimized1.csv',
    output_file='.\\xyy\\模型分析\\winter\\winter_EEF_分箱_Optimized1.csv',
    target='EEF'
)

