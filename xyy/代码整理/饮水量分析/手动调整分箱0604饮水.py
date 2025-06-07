import pandas as pd
import re
import numpy as np


def update_binning_with_woe_iv(variable_name, Age_interval, new_bins, input_file, output_file, target='MORTALITY_RATE'):
    # Read the original data
    df = pd.read_csv(input_file, encoding='gbk')

    # Filter rows for the specified variable and Age_interval (excluding the total row)
    variable_rows = df[(df['feature'] == variable_name) & (df['Age_interval'] == Age_interval)]
    variable_rows = variable_rows[variable_rows['Bin'].notna()]

    if len(variable_rows) == 0:
        raise ValueError(f"Variable '{variable_name}' or Age_interval '{Age_interval}' not found in the input file")

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
        check_points.append((orig_left + orig_right) / 2)  # midpoint

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
            midpoint = (orig_left + orig_right) / 2
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
            'feature': variable_name,
            'Age_interval': Age_interval
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

    # Create new DataFrame for the variable and Age_interval
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

        target: weighted_avg_eef , # 修正：使用加权平均EEF
        'Age_interval': Age_interval,
    }
    new_variable_df = pd.concat([new_variable_df, pd.DataFrame([total_row])], ignore_index=True)

    # Reorder columns to match original
    columns_order = ['Bin', 'Count', 'Count (%)', 'Non-event', 'Event', 'Event rate',
                     'WoE', 'IV', 'JS', 'feature', target, 'Age_interval']
    new_variable_df = new_variable_df[columns_order]

    # Replace the variable's and Age_interval's rows in the original DataFrame
    # First remove all rows for this variable and Age_interval
    df = df[~((df['feature'] == variable_name) & (df['Age_interval'] == Age_interval))]
    # Then add the new rows
    df = pd.concat([df, new_variable_df], ignore_index=True)

    # Save to output file
    df.to_csv(output_file, index=False, encoding='gbk')

    return df

# 调用示例
update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前2天',
    Age_interval='16 - 18日龄',
    new_bins=[
         '[-inf, -5.76)',
    '[-5.76, 8.45)',
    '[8.45, 14.47)',
    '[14.47, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized.csv",
    output_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)
update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前5天',
    Age_interval='16 - 18日龄',
    new_bins=[
        '[-inf, -1.53)',
    '[-1.53, 0.40)',
    '[0.40, 1.35)',
    '[1.35, 7.03)',
    '[7.03, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前1天',
    Age_interval='16 - 18日龄',
    new_bins=[
       '[-inf, -8.21)',
    '[-8.21, 0.27)',
    '[0.27, 6.52)',
    '[6.52, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前7天',
    Age_interval='19 - 21日龄',
    new_bins=[
          '[-inf, -1.67)',
    '[-1.67, 12.78)',
    '[12.78, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\winter_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)
##秋季
update_binning_with_woe_iv(
    variable_name='water_per_diff_前3天',
    Age_interval='16 - 18日龄',
    new_bins=[
         '[-inf, -15.51)',
    '[-15.51, -0.40)',
    '[-0.40, 2.34)',
    '[2.34, 20.21)',
    '[20.21, inf)',
    ],
    input_file=".\\xyy\\死淘分析\\output\\autumn_饮水差值分箱_Optimized.csv",
    output_file=".\\xyy\\死淘分析\\output\\autumn_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_diff_前7天',
    Age_interval='22 - 24日龄',
    new_bins=[
    '[-inf, -0.41)',
    '[-0.41, 3.02)',
    '[3.02, 7.40)',
    '[7.40, 27.70)',
    '[27.70, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\autumn_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\autumn_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_diff_前2天',
    Age_interval='25 - 27日龄',
    new_bins=[
     '[-inf, -6.64)',
    '[-6.64,  6.62)',
    '[6.62, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\autumn_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\autumn_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)
##夏季

update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前7天',
    Age_interval='10 - 12日龄',
    new_bins=[
        '[-inf, 0.25)',
    '[0.25, 1.74)',
    '[1.74, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_diff_前7天',
    Age_interval='10 - 12日龄',
    new_bins=[
       '[-inf, -2.71)',
    '[-2.71, -0.47)',
    '[-0.47, 2.18)',
    '[2.18, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前7天',
    Age_interval='19 - 21日龄',
    new_bins=[
       '[-inf, 0.83)',
    '[0.83, 4.31)',
    '[4.31, 12.07)',
    '[12.07, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_diff_前7天',
    Age_interval='25 - 27日龄',
    new_bins=[
       '[-inf, -22.24)',
    '[-22.24, -6.53)',
    '[-6.53,  5.08)',
    '[5.08, 13.61)',
    '[13.61, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_diff_前2天',
    Age_interval='25 - 27日龄',
    new_bins=[
    '[-inf, -8.40)',
    '[-8.40, 2.24)',
    '[2.24, 8.08)',
    '[8.08, 19.98)',
    '[19.98, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)


update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前3天',
    Age_interval='28 - 30日龄',
    new_bins=[
     '[-inf, -17.73)',
    '[-17.73, -12.78)',
    '[-12.78, -7.55)',
    '[-7.55, -3.98)',
    '[-3.98, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前3天',
    Age_interval='28 - 30日龄',
    new_bins=[
     '[-inf, -17.73)',
    '[-17.73, -12.78)',
    '[-12.78, -7.55)',
    '[-7.55, -3.98)',
    '[-3.98, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)

update_binning_with_woe_iv(
    variable_name='water_per_diff_前2天',
    Age_interval='28 - 30日龄',
    new_bins=[
   '[-inf, -17.72)',
    '[-17.72, -10.26)',
    '[-10.26, 14.45)',
    '[14.45, 19.94)',
    '[19.94, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)


update_binning_with_woe_iv(
    variable_name='water_per_diff_前7天',
    Age_interval='28 - 30日龄',
    new_bins=[
   '[-inf, 1.50)',
    '[1.50, 6.92)',
    '[6.92, 23.64)',
    '[23.64, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)


update_binning_with_woe_iv(
    variable_name='water_per_shift_diff_前5天',
    Age_interval='34 - 35日龄',
    new_bins=[
    '[-inf, -19.04)',
    '[-19.04, -6.26)',
    '[-6.26, 0.04)',
    '[0.04,  3.00)',
    '[3.00, inf)'
    ],
    input_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    output_file=".\\xyy\\死淘分析\\output\\summer_饮水差值分箱_Optimized1.csv",
    target='MORTALITY_RATE'
)