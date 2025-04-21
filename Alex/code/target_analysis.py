import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os

# 设置matplotlib支持中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 数据读取
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/data_cleaned/allinfo_dead.csv')
# 自动尝试不同编码读取 CSV 文件
try:
    df = pd.read_csv(DATA_PATH, encoding='gbk')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except Exception as e:
    print(f'读取CSV文件失败: {e}')
    exit(1)

# 1.1 衍生DOCdate和Harveststatus的月份字段
for col, new_col in [('DOCdate', 'DOC_month'), ('Harveststatus', 'Harveststatus_month')]:
    if col in df.columns:
        df[new_col] = pd.to_datetime(df[col], errors='coerce').dt.month

# 2. 简单预处理
# 将DOC_month和Harveststatus_month转为字符串类型
for col in ['DOC_month', 'Harveststatus_month']:
    if col in df.columns:
        df[col] = df[col].astype('Int64').astype(str)

print('数据维度:', df.shape)
print('字段名:', df.columns.tolist())
print('缺失值统计:')
print(df.isnull().sum())

# 2.1 eef和Mortality_rate直方图
plt.figure(figsize=(8,4))
sns.histplot(df['eef'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('eef 直方图')
plt.xlabel('eef')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'eef_hist.png'))
plt.close()

plt.figure(figsize=(8,4))
sns.histplot(df['Mortality_rate'].dropna(), bins=30, kde=True, color='salmon')
plt.title('Mortality_rate 直方图')
plt.xlabel('Mortality_rate')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'Mortality_rate_hist.png'))
plt.close()

# 3. 描述性统计
print('\n描述性统计:')
print(df.describe())

# 3.1 按月份分析eef和Mortality_rate的显著性
from scipy.stats import f_oneway
for month_col in ['DOC_month', 'Harveststatus_month']:
    for target in ['eef', 'Mortality_rate']:
        groups = [g.dropna().values for _, g in df.groupby(month_col)[target] if g.notna().sum() > 1]
        if len(groups) > 1:
            stat, pval = f_oneway(*groups)
            print(f'【单因素方差分析】{month_col} 不同月份对 {target} 的p值: {pval:.4g}')
        else:
            print(f'【单因素方差分析】{month_col} 不同月份对 {target} 样本不足，无法检验')
        # 箱线图
        plt.figure(figsize=(8,4))
        sns.boxplot(x=month_col, y=target, data=df)
        plt.title(f'{month_col} 不同月份对 {target} 的分布')
        plt.xlabel(f'{month_col} (月份)')
        plt.ylabel(target)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), f'{month_col}_{target}_boxplot.png'))
        plt.close()

# 4. 相关性分析（只对数值型列做）
df_numeric = df.select_dtypes(include=[np.number])
corr = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('变量相关性热力图（仅数值型特征）')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'correlation_heatmap.png'))
plt.close()

# 5. 针对目标变量的相关性
targets = ['eef', 'Mortality_rate']
for target in targets:
    print(f'\n与{target}相关性:')
    for col in df.columns:
        if col != target and df[col].dtype in [np.float64, np.int64, float, int]:
            corr_val, p_val = pearsonr(df[target], df[col])
            print(f'  {col}: 相关系数={corr_val:.3f}, p值={p_val:.3g}')

# 6. 多元回归分析
# 尝试将所有非目标变量列整体转为数值型，无法转换的自动变为NaN
candidate_cols = [col for col in df.columns if col not in targets]
df_num = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
# 只保留缺失比例低于80%的列
valid_cols = df_num.columns[df_num.isnull().mean() < 0.8].tolist()
feature_cols = valid_cols
# 用转换后的数值型数据替换原数据
df[feature_cols] = df_num[feature_cols]

for target in targets:
    print(f'\n----- {target} 多元线性回归分析 -----')
    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print('R2分数:', r2_score(y, y_pred))
    print('各变量回归系数:')
    for col, coef in zip(feature_cols, model.coef_):
        print(f'  {col}: {coef:.4f}')

    # 随机森林重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    print('随机森林特征重要性:')
    for i in idx:
        print(f'  {feature_cols[i]}: {importances[i]:.4f}')

    # 可视化
    plt.figure(figsize=(10, 4))
    sns.barplot(x=np.array(feature_cols)[idx], y=importances[idx])
    plt.title(f'{target} 随机森林特征重要性')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{target}_feature_importance.png'))
    plt.close()

print('\n分析完成，相关可视化图片已保存。')
