import pandas as pd
import re
import numpy as np


class AdvancedBinningOptimizer:
    def __init__(self, max_eef_diff=0.08, min_bin_size=0.05, min_iv_loss=0.03):
        self.max_eef_diff = max_eef_diff
        self.min_bin_size = min_bin_size
        self.min_iv_loss = min_iv_loss

    def _parse_bin_boundary(self, bin_str):
        # 边界解析函数不变
        if pd.isna(bin_str) or bin_str in ['Special', 'Missing']:
            return np.nan, np.nan
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", bin_str)
        if len(numbers) == 1:
            return (-np.inf, float(numbers[0])) if '(' in bin_str else (float(numbers[0]), np.inf)
        return (float(numbers[0]), float(numbers[1]))

    def _preprocess(self, df):
        # 数据预处理函数不变
        df = df[df['Bin'].apply(lambda x: bool(re.search(r'\d', str(x))))].copy()
        boundaries = df['Bin'].apply(self._parse_bin_boundary)
        df['lower'] = [b[0] for b in boundaries]
        df['upper'] = [b[1] for b in boundaries]
        df = df.sort_values(by='lower').reset_index(drop=True)
        df['Count'] = df['Count'].astype(int)
        df['EEF'] = df['EEF'].astype(float)
        df['Count_pct'] = df['Count (%)'].astype(float)
        df['Event_rate'] = df['Event'].astype(int) / df['Count']
        df['WoE'] = df['WoE'].astype(float)
        df['IV'] = df['IV'].astype(float)
        return df

    def _find_merge_candidates(self, df):
        # 合并候选函数不变
        merge_groups = []
        current_group = [0]
        for i in range(1, len(df)):
            prev_eef = df.loc[current_group[-1], 'EEF']
            curr_eef = df.loc[i, 'EEF']
            eef_diff = abs(curr_eef - prev_eef)
            size_condition = (df.loc[i, 'Count_pct'] < self.min_bin_size) | (df.loc[current_group[-1], 'Count_pct'] < self.min_bin_size)
            trend_condition = eef_diff < self.max_eef_diff
            if size_condition or trend_condition:
                current_group.append(i)
            else:
                merge_groups.append(current_group)
                current_group = [i]
        if current_group:
            merge_groups.append(current_group)
        return merge_groups

    def _merge_operation(self, df, merge_groups):
        # 合并操作函数不变
        total_events = df['Event'].sum()
        total_non_events = df['Non-event'].sum()
        merged_data = []
        for group in merge_groups:
            subset = df.iloc[group]
            new_lower = subset['lower'].min()
            new_upper = subset['upper'].max()
            bin_name = f"[{new_lower:.2f}, inf)" if np.isinf(new_upper) else f"[{new_lower:.2f}, {new_upper:.2f})"
            total_count = subset['Count'].sum()
            merged_events = subset['Event'].sum()
            merged_non_events = subset['Non-event'].sum()
            
            good_pct = merged_non_events / total_non_events if total_non_events != 0 else 0
            bad_pct = merged_events / total_events if total_events != 0 else 0
            woe = np.log(good_pct / bad_pct) if (good_pct != 0 and bad_pct != 0) else 0
            iv = (good_pct - bad_pct) * woe
            
            merged_record = {
                'Bin': bin_name,
                'Count': total_count,
                'Non-event': merged_non_events,
                'Event': merged_events,
                'Event rate': merged_events / total_count,
                'WoE': woe,
                'IV': iv,
                'feature': subset['feature'].iloc[0],
                'EEF': np.average(subset['EEF'], weights=subset['Count']),
                'Count (%)': total_count / df['Count'].sum(),
                'JS': subset['JS'].sum() if 'JS' in subset.columns else 0
            }
            merged_data.append(merged_record)
        return pd.DataFrame(merged_data)

    def optimize(self, raw_data):
        optimized_results = []
        features = raw_data['feature'].unique()
        for feat in features:
            feat_data = raw_data[raw_data['feature'] == feat].copy()
            cleaned_data = self._preprocess(feat_data)
            if len(cleaned_data) < 3:
                optimized_results.append(cleaned_data)
                continue
            
            # 原逻辑：执行合并
            merge_candidates = self._find_merge_candidates(cleaned_data)
            merged_data = self._merge_operation(cleaned_data, merge_candidates)
            
            # IV损失校验
            original_iv = cleaned_data['IV'].sum()
            new_iv = merged_data['IV'].sum()
            if (original_iv - new_iv) <= self.min_iv_loss:
                optimized_feat_data = merged_data
            else:
                optimized_feat_data = cleaned_data
            
            # 新增：生成特征汇总行
            summary_row = self._generate_summary_row(optimized_feat_data)
            optimized_feat_data = pd.concat([optimized_feat_data, summary_row], ignore_index=True)
            optimized_results.append(optimized_feat_data)
        
        # 合并所有特征数据并保留原始列序
        original_columns = raw_data.columns.tolist()
        optimized_data = pd.concat(optimized_results, ignore_index=True)[original_columns]
        return optimized_data

    def _generate_summary_row(self, df):
        """生成特征汇总行"""
        if df.empty:
            return pd.DataFrame()
        
        feat_name = df['feature'].iloc[0]
        total_count = df['Count'].sum()
        total_non_event = df['Non-event'].sum()
        total_event = df['Event'].sum()
        total_iv = df['IV'].sum()
        avg_eef = np.average(df['EEF'], weights=df['Count'])  # 按Count加权平均EEF
        
        # 构造汇总行（注意：Bin列标记为"[汇总]"）
        summary = pd.DataFrame({
            'Bin': [''],
            'Count': [total_count],
            'Non-event': [total_non_event],
            'Event': [total_event],
            'Event rate': [total_event / total_count if total_count != 0 else 0],
            'WoE': [np.nan],  # 汇总行WoE无意义，设为NaN
            'IV': [total_iv],
            'feature': [feat_name],
            'EEF': [avg_eef],
            'Count (%)': [1.0],  # 汇总行占比为100%
            'JS': [df['JS'].sum()]  # 汇总JS值
        })
        return summary

# 使用示例
if __name__ == "__main__":
    # 读取原始数据（注意文件编码，通常为gbk或utf-8）
    raw_data = pd.read_csv(
        ".\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱.csv", 
        encoding='gbk'  # 根据实际文件编码调整
    )
    
    # 初始化优化器（可调整参数）
    optimizer = AdvancedBinningOptimizer(
        max_eef_diff=5,    # EEF差异阈值（可根据业务调整）
        min_bin_size=0.05,     # 最小分箱占比5%
        min_iv_loss=0.05     # 允许IV损失5%
    )
    
    # 执行分箱优化
    optimized_data = optimizer.optimize(raw_data)
    
    # 保存结果
    optimized_data.to_csv(
        ".\\xyy\\eef分析\\output\\夏天湿度内部平均_MEAN分箱_Optimized.csv", 
        index=False, 
        encoding='gbk'
    )
    print("EEF分箱优化完成，结果已保存至Optimized.csv文件")