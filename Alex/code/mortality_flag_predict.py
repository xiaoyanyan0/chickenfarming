import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# 每次执行前自动删除code目录下所有图片文件
img_types = ('*.png', '*.jpg', '*.jpeg')
code_dir = os.path.dirname(__file__)
for ext in img_types:
    for f in glob.glob(os.path.join(code_dir, ext)):
        try:
            os.remove(f)
        except Exception:
            pass

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 检查并导入lightgbm/xgboost
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    print('未安装 lightgbm，相关模型将跳过')
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print('未安装 xgboost，相关模型将跳过')

# 1. 读取数据
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/data_cleaned/allinfo_dead.csv')
try:
    df = pd.read_csv(DATA_PATH, encoding='gbk')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

# 2. 新建Mortality_flag
flag_col = 'Mortality_flag'
df[flag_col] = np.where(df['Mortality_rate'] <= 0.05, 1, 0).astype(str)

# 3. 删除指定字段
remove_cols = [
    'DOCdate','Harveststatus','Dead','Swollen_Head','Weak','Navel_Disease',
    'Stick_Anus','Lame_Paralysis','Mortality','livability_pct',
    'pp_dead_culled_coun','dead_culled_weight_kg','pp_rejects_weight_kg',
    'id_no','Batch','revenue','birds_hung',
    'yield_per_m2','stocking_density','profit_per_house','dead_during_catch_count',
    'cost_per_kg',
    'total_hung_weight_kg','fcr','pp_dead_culled_count',
    'HouseName','EstimatedSlaughterDate','EstimatedSlaughterDate ','birds_caught_count',
    'total_caught_weight_kg',
    'adjusted_fcr','feed_per_bird','electricity_per_bird','gas_per_bird','labor_per_bird','consumables_per_bird','depreciation_per_bird','cost_per_bird','chick_cost_per_bird',
    'HouseNo','total_feed_kg','total_cost',
    'Houseid','feed_cost'
]
# 去除所有字段名首尾空格，防止导入时因空格导致无法删除
if hasattr(df, 'columns'):
    df.columns = df.columns.str.strip()

remove_cols = [c for c in remove_cols if c in df.columns]
df = df.drop(columns=remove_cols)

# 4. 衍生日期字段为月份
if 'DOCdate' in df.columns:
    df['DOC_month'] = pd.to_datetime(df['DOCdate'], errors='coerce').dt.month.astype('Int64')
if 'Harveststatus' in df.columns:
    df['Harveststatus_month'] = pd.to_datetime(df['Harveststatus'], errors='coerce').dt.month.astype('Int64')

# 4. 处理特征
if 'eef' in df.columns:
    df = df.drop(columns=['eef'])

# 4.1 处理HEAge字段
if 'HEAge' in df.columns:
    def heage_to_int(x):
        # 去掉W，取最后两个数字
        if pd.isnull(x):
            return np.nan
        x = str(x).replace('W', '').replace('w', '')
        # 只取最后两位数字
        digits = ''.join([c for c in x if c.isdigit()])
        return int(digits[-2:]) if len(digits) >= 2 else (int(digits) if digits else np.nan)
    df['HEAge'] = df['HEAge'].apply(heage_to_int)

# Houseid转换为字符型变量
if 'Houseid' in df.columns:
    df['Houseid'] = df['Houseid'].astype(str)

X = df.drop(columns=['Mortality_rate', flag_col])
y = df[flag_col]
import pandas.api.types as ptypes
for col in X.columns:
    if ptypes.is_numeric_dtype(X[col]):
        continue  # 保持原样
    else:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def fit_and_report(model, model_name, roc_dict=None):
    if model_name == 'XGBoost':
        model.fit(X_train, y_train.astype(int))
        y_pred_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test.astype(int), y_pred_prob)
    else:
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test.astype(int), y_pred_prob)
    auc_val = auc(fpr, tpr)
    if roc_dict is not None:
        roc_dict[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val}
    print(f'\n【{model_name}】AUC: {auc_val:.4f}')
    top10_vars = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print(f'前十特征:')
        for i in indices:
            print(f'  {X.columns[i]}: {importances[i]:.4f}')
        top10_vars = [X.columns[i] for i in indices]
        plt.figure(figsize=(8,4))
        plt.barh([X.columns[i] for i in indices[::-1]], importances[indices[::-1]])
        plt.title(f'{model_name} 前十特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), f'{model_name}_top10_importance.png'))
        plt.close()
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC={auc_val:.3f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{model_name}_roc.png'))
    plt.close()
    return auc_val, top10_vars

global_roc_dict = {}
rf = RandomForestClassifier(n_estimators=100, random_state=42)
auc_rf, top10_rf = fit_and_report(rf, 'RandomForest', global_roc_dict)
auc_lgb, top10_lgb = None, []
if LGBMClassifier is not None:
    lgb = LGBMClassifier(n_estimators=100, random_state=42)
    auc_lgb, top10_lgb = fit_and_report(lgb, 'LightGBM', global_roc_dict)
auc_xgb, top10_xgb = None, []
if XGBClassifier is not None:
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    auc_xgb, top10_xgb = fit_and_report(xgb, 'XGBoost', global_roc_dict)
auc_dict = {'RandomForest': auc_rf, 'LightGBM': auc_lgb, 'XGBoost': auc_xgb}
top10_vars_dict = {'RandomForest': top10_rf, 'LightGBM': top10_lgb, 'XGBoost': top10_xgb}
best_model = max([(k, v) for k, v in auc_dict.items() if v is not None], key=lambda x: x[1])

# 画所有模型的ROC曲线到一张图
plt.figure(figsize=(7,7))
for model_name, roc_data in global_roc_dict.items():
    plt.plot(roc_data['fpr'], roc_data['tpr'], label=f"{model_name} (AUC={roc_data['auc']:.3f})")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('三模型ROC曲线对比')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'all_models_roc.png'))
plt.close()

def calc_woe_iv(df, feature, target):
    # 分箱不超过6个
    max_bins = 6
    df = df[[feature, target]].copy()
    import pandas.api.types as ptypes
    if ptypes.is_numeric_dtype(df[feature]):
        if df[feature].nunique() > max_bins:
            df[feature] = pd.qcut(df[feature], max_bins, duplicates="drop")
    grouped = df.groupby(feature)[target]
    total_good = (df[target] == '1').sum()
    total_bad = (df[target] == '0').sum()
    woe_list, iv = [], 0
    for cat, group in grouped:
        good = (group == '1').sum()
        bad = (group == '0').sum()
        good_pct = good / total_good if total_good else 0.001
        bad_pct = bad / total_bad if total_bad else 0.001
        woe = np.log(good_pct / bad_pct) if bad_pct > 0 and good_pct > 0 else 0
        woe_list.append((cat, woe, good_pct, bad_pct))
        iv += (good_pct - bad_pct) * woe
    return woe_list, iv

def plot_woe(woe_list, feature, model_name):
    cats = [str(x[0]) for x in woe_list]
    woe_vals = [x[1] for x in woe_list]
    good_rates = [x[2] for x in woe_list]
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.bar(cats, woe_vals, color='skyblue', label='WoE')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('WoE')
    ax1.set_title(f'{model_name} 变量 {feature} 的WoE分布及正样本比例')
    ax2 = ax1.twinx()
    ax2.plot(cats, good_rates, color='green', marker='o', label='正样本比例')
    ax2.set_ylabel('正样本比例')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{model_name}_{feature}_woe.png'))
    plt.close()

print(f'\n【IV值与WoE分布（最佳模型{best_model[0]}前十变量）】')
# WoE分析时用的是经过所有上述预处理和衍生的新变量（如HEAge、DOC_month等）
for feat in top10_vars_dict[best_model[0]]:
    if feat in df.columns:
        woe_list, iv = calc_woe_iv(df, feat, flag_col)
        print(f'  {feat}: IV={iv:.4f}')
        plot_woe(woe_list, feat, best_model[0])

print('\n分析完成，重要性图片、ROC曲线、IV值和WoE图已保存。')
