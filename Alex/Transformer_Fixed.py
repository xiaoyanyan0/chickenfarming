import os
import sys
import math
import logging
import warnings
import pickle
import datetime
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import logging
import datetime
import os
import sys
import io
import math
import joblib
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese labels
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs

def load_model_safely(model_path, device, model, optimizer=None):
    """Safely load model checkpoint with fallback mechanism."""
    try:
        # First try with weights_only=True and safe globals
        import numpy as np
        import torch.serialization
        torch.serialization.add_safe_globals([np.number, np.ndarray, np.dtype, 
                                            np.float32, np.float64, np.int64,
                                            np.bool_, np.void, np.record])
        
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Model loaded successfully with secure weights_only=True")
        result = {
            'model': model,
            'optimizer': optimizer,
            'loss': checkpoint.get('loss', float('inf')),
            'r2': checkpoint.get('r2', 0),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'val_preds': checkpoint.get('val_preds', None),
            'val_targets': checkpoint.get('val_targets', None),
            'epoch': checkpoint.get('epoch', 0)
        }
        return result
        
    except (TypeError, pickle.UnpicklingError) as e:
        logger.warning(f"Secure loading failed with error: {str(e)}")
        logger.warning("Attempting fallback to weights_only=False (less secure)")
        
        # Fallback to less secure loading
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        logger.warning("Model loaded with weights_only=False")
        result = {
            'model': model,
            'optimizer': optimizer,
            'loss': checkpoint.get('loss', float('inf')),
            'r2': checkpoint.get('r2', 0),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'val_preds': checkpoint.get('val_preds', None),
            'val_targets': checkpoint.get('val_targets', None),
            'epoch': checkpoint.get('epoch', 0)
        }
        return result

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# 确保目录存在
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 导入可视化函数
try:
    from visualization_utils import (
        plot_loss_curve,
        plot_true_vs_pred,
        plot_attention_weights
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入可视化工具: {e}，可视化功能将被禁用")
    VISUALIZATION_AVAILABLE = False



# Clear visualizations directory before each run
visualizations_dir = 'visualizations'
if os.path.exists(visualizations_dir):
    shutil.rmtree(visualizations_dir)
os.makedirs(visualizations_dir, exist_ok=True)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import lime
import lime.lime_tabular

# Import visualization functions
from visualization_utils import (
    plot_loss_curve,
    plot_true_vs_pred,
    plot_attention_weights,
    plot_shap_summary,
    plot_shap_force,
    plot_lime_explanation
)

# Fix Windows console encoding for logging
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def setup_logging():
    """Configure logging with UTF-8 encoding support"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Define log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create a custom stream handler that handles UTF-8 encoding
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                if sys.platform == 'win32':
                    msg = msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    # Create handlers
    file_handler = logging.FileHandler(
        os.path.join('logs', f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    console_handler = SafeStreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add the handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 读取数据
try:
    df = pd.read_csv(r'C:\Users\sbjalz\OneDrive - SAS\Desktop\开发代码\太阳谷\chickenfarming\Alex\winter_valid_samples0530.csv', encoding='GB2312')
    logger.info("数据加载成功")
except Exception as e:
    logger.error(f"数据加载失败: {e}")
    raise

# 序列特征（0-25天）
temp_cols = [f'鸡舍温度-平均_mean_{i}' for i in range(26)]
humidity_cols = [f'湿度内部平均_mean_{i}' for i in range(26)]
out_temp_cols = [f'外部-平均_mean_{i}' for i in range(26)]
# water_cols = [f'Water_{i}' for i in range(26)]
# feed_cols = [f'Feed_{i}' for i in range(26)]
# max_temp_cols = [f'鸡舍温度-最高_mean_{i}' for i in range(26)]
# min_temp_cols = [f'鸡舍温度-最低_mean_{i}' for i in range(26)]
all_seq_cols = temp_cols + humidity_cols + out_temp_cols

# 固定变量
fixed_vars = ['Density', 'BirdsVariety', 'FarmSupervisor']
target_col = 'Mortality_rate'  # 目标变量

# 将分类变量转换为数值编码
from sklearn.preprocessing import LabelEncoder

# 标准化 BirdsVariety 变量
def standardize_birds_variety(variety):
    variety = str(variety).lower().strip()
    
    # 定义 Cobb 相关的关键词
    cobb_keywords = ['cobb', 'cob', 'cob500', 'cobb500', 'cobb 500','CObb']
    
    # 如果包含 cobb 相关关键词，则归类为 'Cobb'
    if any(keyword in variety for keyword in cobb_keywords):
        return 'Cobb'
    # 其他情况都归类为 'SZ901 Plus'
    else:
        return 'SZ901 Plus'

# 应用标准化
df['BirdsVariety_standardized'] = df['BirdsVariety'].apply(standardize_birds_variety)

# 将标准化后的变量转换为数值编码
variety_encoder = LabelEncoder()
supervisor_encoder = LabelEncoder()

# 对分类变量进行编码
df['BirdsVariety_encoded'] = variety_encoder.fit_transform(df['BirdsVariety_standardized'])
df['FarmSupervisor_encoded'] = supervisor_encoder.fit_transform(df['FarmSupervisor'])

# 更新固定变量列为编码后的列
fixed_vars = ['Density', 'BirdsVariety_encoded', 'FarmSupervisor_encoded']

# 记录编码信息
logger.info(f"BirdsVariety 标准化前: {df['BirdsVariety'].unique().tolist()}")
logger.info(f"BirdsVariety 标准化后: {df['BirdsVariety_standardized'].unique().tolist()}")
logger.info(f"BirdsVariety 编码: {dict(zip(variety_encoder.classes_, range(len(variety_encoder.classes_))))}")
logger.info(f"FarmSupervisor 编码: {dict(zip(supervisor_encoder.classes_, range(len(supervisor_encoder.classes_))))}")

# 检查所有列是否存在
missing_cols = [col for col in all_seq_cols + [target_col] if col not in df.columns]
if missing_cols:
    error_msg = f"列名缺失: {missing_cols}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# 准备序列数据
# === 先生成 X_seq，然后划分训练/验证集，再分别标准化 ===
from sklearn.preprocessing import StandardScaler

# 1. 构建原始序列数据，shape = [samples, seq_len, 2]
X_seq = np.stack([
    df[temp_cols].values,
    df[humidity_cols].values,
    df[out_temp_cols].values,
    # df[water_cols].values,
    # df[feed_cols].values
], axis=-1)
y = df[target_col].values.astype(np.float32)

# 2. 提取固定变量
X_fixed = df[fixed_vars].values.astype(np.float32)

# 3. 划分训练集和验证集
from sklearn.model_selection import train_test_split
if X_fixed is not None:
    X_train_seq, X_val_seq, X_train_fixed, X_val_fixed, y_train, y_val = train_test_split(
        X_seq, X_fixed, y, test_size=0.2, random_state=42
    )
else:
    X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_seq, y, test_size=0.2, random_state=42
    )
    X_train_fixed = X_val_fixed = None

# 4. 分别对温度和湿度标准化
# 注意：X_train_seq[..., 0] 是温度，X_train_seq[..., 1] 是湿度,X_train_seq[..., 2]是外部温度
N_train, seq_len, _ = X_train_seq.shape
N_val = X_val_seq.shape[0]
temp_train = X_train_seq[..., 0].reshape(N_train, seq_len)
humidity_train = X_train_seq[..., 1].reshape(N_train, seq_len)
temp_val = X_val_seq[..., 0].reshape(N_val, seq_len)
humidity_val = X_val_seq[..., 1].reshape(N_val, seq_len)
out_temp_train = X_train_seq[..., 2].reshape(N_train, seq_len)
out_temp_val = X_val_seq[..., 2].reshape(N_val, seq_len)
# water_train = X_train_seq[..., 3].reshape(N_train, seq_len)
# water_val = X_val_seq[..., 3].reshape(N_val, seq_len)
# feed_train = X_train_seq[..., 4].reshape(N_train, seq_len)
# feed_val = X_val_seq[..., 4].reshape(N_val, seq_len)

temp_scaler = StandardScaler().fit(temp_train)
humidity_scaler = StandardScaler().fit(humidity_train)
out_temp_scaler = StandardScaler().fit(out_temp_train)
# water_scaler = StandardScaler().fit(water_train)
# feed_scaler = StandardScaler().fit(feed_train)

X_train_temp_scaled = temp_scaler.transform(temp_train)
X_val_temp_scaled = temp_scaler.transform(temp_val)
X_train_humidity_scaled = humidity_scaler.transform(humidity_train)
X_val_humidity_scaled = humidity_scaler.transform(humidity_val)
X_train_out_temp_scaled = out_temp_scaler.transform(out_temp_train)
X_val_out_temp_scaled = out_temp_scaler.transform(out_temp_val)
# X_train_water_scaled = water_scaler.transform(water_train)
# X_val_water_scaled = water_scaler.transform(water_val)
# X_train_feed_scaled = feed_scaler.transform(feed_train)
# X_val_feed_scaled = feed_scaler.transform(feed_val)

# 重新拼接为 [N, seq_len, 2]
X_train_seq_scaled = np.stack([X_train_temp_scaled, X_train_humidity_scaled, X_train_out_temp_scaled], axis=-1)
X_val_seq_scaled = np.stack([   X_val_temp_scaled, X_val_humidity_scaled, X_val_out_temp_scaled], axis=-1)

X_train_seq_scaled = np.nan_to_num(X_train_seq_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_val_seq_scaled = np.nan_to_num(X_val_seq_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# 2. 固定特征（密度）
if X_train_fixed is not None and X_train_fixed.shape[1] > 0:
    density_scaler = StandardScaler().fit(X_train_fixed[:, 0:1])
    X_train_fixed[:, 0:1] = density_scaler.transform(X_train_fixed[:, 0:1])
    X_val_fixed[:, 0:1] = density_scaler.transform(X_val_fixed[:, 0:1])
else:
    density_scaler = None

# 3. 目标变量
# 只用训练集fit，验证集transform
# 确保没有NaN/inf
y_train_clean = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
y_val_clean = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
target_scaler = StandardScaler().fit(y_train_clean.reshape(-1, 1))
y_train_scaled = target_scaler.transform(y_train_clean.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val_clean.reshape(-1, 1)).flatten()

# 保存scaler到models目录
import joblib
import os
os.makedirs('models', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)
joblib.dump(temp_scaler, 'models/scalers/temp_scaler.pkl')
joblib.dump(humidity_scaler, 'models/scalers/humidity_scaler.pkl')
joblib.dump(out_temp_scaler, 'models/scalers/out_temp_scaler.pkl')
# joblib.dump(water_scaler, 'models/scalers/water_scaler.pkl')
# joblib.dump(feed_scaler, 'models/scalers/feed_scaler.pkl')
# joblib.dump(max_temp_scaler, 'models/scalers/max_temp_scaler.pkl')
# joblib.dump(min_temp_scaler, 'models/scalers/min_temp_scaler.pkl')
if density_scaler is not None:
    joblib.dump(density_scaler, 'models/scalers/density_scaler.pkl')
joblib.dump(target_scaler, 'models/scalers/target_scaler.pkl')

# 对特征数据进行四舍五入到4位小数
X_train_seq_scaled = np.round(X_train_seq_scaled, 4)
X_val_seq_scaled = np.round(X_val_seq_scaled, 4)

# ===================
# 后续训练流程全部使用 *_scaled 变量
# ===================

# 对目标变量也进行四舍五入
y_train = np.round(y_train, 4)
y_val = np.round(y_val, 4)

# 转换为tensor
X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# 保存标准化器
os.makedirs('models/scalers', exist_ok=True)
joblib.dump(temp_scaler, 'models/scalers/temp_scaler.pkl')
joblib.dump(humidity_scaler, 'models/scalers/humidity_scaler.pkl')
joblib.dump(out_temp_scaler, 'models/scalers/out_temp_scaler.pkl')
# joblib.dump(water_scaler, 'models/scalers/water_scaler.pkl')
# joblib.dump(feed_scaler, 'models/scalers/feed_scaler.pkl')
# joblib.dump(max_temp_scaler, 'models/scalers/max_temp_scaler.pkl')
# joblib.dump(min_temp_scaler, 'models/scalers/min_temp_scaler.pkl')
if density_scaler is not None:
    joblib.dump(density_scaler, 'models/scalers/density_scaler.pkl')
joblib.dump(target_scaler, 'models/scalers/target_scaler.pkl')

# 创建数据集类
class PoultryDataset(Dataset):
    def __init__(self, sequences, fixed_features, targets, seq_len=26):
        """
        参数:
            sequences: 序列特征，形状为 [num_samples, seq_len, num_features]
            fixed_features: 固定特征，形状为 [num_samples, num_fixed_features]
            targets: 目标值，形状为 [num_samples]
            seq_len: 序列长度
        """
        self.sequences = sequences
        self.fixed_features = fixed_features
        self.targets = targets
        self.seq_len = seq_len
        
        # 预处理数据
        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            fixed = fixed_features[i] if fixed_features is not None else np.array([])
            target = targets[i]
            
            # 确保序列是二维的 [seq_len, num_features]
            if len(seq.shape) == 1:
                seq = seq.reshape(-1, 1)
                
            # 确保序列长度一致
            if seq.shape[0] < self.seq_len:
                # 如果序列太短，进行填充
                padding = np.zeros((self.seq_len - seq.shape[0], seq.shape[1]))
                seq = np.vstack([seq, padding])
            elif seq.shape[0] > self.seq_len:
                # 如果序列太长，截断
                seq = seq[-self.seq_len:]
            
            # 确保固定特征是1D数组
            if len(fixed.shape) > 1:
                fixed = fixed.squeeze()
                
            self.processed_data.append((seq, fixed, target))
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        seq_data, fixed_data, target = self.processed_data[idx]
        
        # 转换为PyTorch张量
        seq_tensor = torch.FloatTensor(seq_data)
        if isinstance(fixed_data, np.ndarray) and fixed_data.size > 0:
            fixed_tensor = torch.FloatTensor(fixed_data)
        else:
            fixed_tensor = None
            
        target_tensor = torch.FloatTensor([target])
        
        return seq_tensor, fixed_tensor, target_tensor

# 创建数据集
train_dataset = PoultryDataset(
    sequences=X_train_seq_scaled,
    fixed_features=X_train_fixed,
    targets=y_train_scaled,
    seq_len=26
)

val_dataset = PoultryDataset(
    sequences=X_val_seq_scaled,
    fixed_features=X_val_fixed,
    targets=y_val_scaled,
    seq_len=26
)

# 自定义批处理函数
def collate_fn(batch):
    # 解压批次数据
    seq_data, fixed_data, targets = zip(*batch)
    
    # 确保所有序列长度一致
    max_seq_len = max([s.size(0) for s in seq_data])
    
    # 处理序列数据
    padded_seqs = []
    for seq in seq_data:
        if seq.size(0) < max_seq_len:
            # 填充序列
            padding = torch.zeros(max_seq_len - seq.size(0), seq.size(1))
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    
    seq_data_padded = torch.stack(padded_seqs)
    
    # 处理固定特征
    if fixed_data[0] is not None:
        fixed_data = torch.stack(fixed_data)
    else:
        fixed_data = None
    
    # 处理目标值
    targets = torch.stack(targets)
    
    return seq_data_padded, fixed_data, targets

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, pin_memory=True, collate_fn=collate_fn)

logger.info(f"训练集大小: {len(train_dataset)}, 批次数量: {len(train_loader)}")
logger.info(f"验证集大小: {len(val_dataset)}, 批次数量: {len(val_loader)}")

# ================== 固定位置编码 ==================

class TransformerWithFixedToken(nn.Module):
    def __init__(self, seq_input_dim, fixed_input_dim, hidden_dim=128, output_dim=1, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerWithFixedToken, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 序列特征嵌入
        self.seq_embedding = nn.Linear(seq_input_dim, hidden_dim)
        
        # 固定变量处理
        self.density_embedding = nn.Linear(1, hidden_dim // 4)
        
        # 动态计算类别数量
        self.num_variety = len(variety_encoder.classes_)
        self.num_supervisor = len(supervisor_encoder.classes_)
        
        # 类别变量嵌入
        self.variety_embedding = nn.Embedding(self.num_variety, hidden_dim // 4)
        self.supervisor_embedding = nn.Embedding(self.num_supervisor, hidden_dim // 4)
        
        # 固定变量投影到隐藏维度
        self.fixed_proj = nn.Linear(hidden_dim // 4 * 3, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer 编码器（支持注意力权重记录）
        from attention_patch import AttentionRecorderTransformerEncoderLayer
        self.encoder_layers = nn.ModuleList([
            AttentionRecorderTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 特征重要性层
        self.feature_importance = FeatureImportanceLayer(seq_input_dim)
        
        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        logger.info(f"使用 TransformerWithFixedToken 模型, hidden_dim={hidden_dim}, nhead={nhead}, num_layers={num_layers}")
        
    def forward(self, x_seq, x_fixed=None, register_hook=False):
        # x_seq: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x_seq.size()
        
        # 应用特征重要性
        x_seq = self.feature_importance(x_seq)
        
        # 序列特征嵌入
        seq_emb = self.seq_embedding(x_seq)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        seq_emb = self.pos_encoder(seq_emb)  # [batch_size, seq_len, d_model]
        
        # 处理固定特征
        if x_fixed is not None:
            # 分离固定特征
            density = x_fixed[:, 0:1]  # [batch_size, 1]
            variety = torch.clamp(x_fixed[:, 1].long(), 0, self.num_variety - 1)  # [batch_size]
            supervisor = torch.clamp(x_fixed[:, 2].long(), 0, self.num_supervisor - 1)  # [batch_size]
            
            # 嵌入固定特征
            density_emb = self.density_embedding(density)  # [batch_size, d_model//4]
            variety_emb = self.variety_embedding(variety)  # [batch_size, d_model//4]
            supervisor_emb = self.supervisor_embedding(supervisor)  # [batch_size, d_model//4]
            
            # 合并固定特征
            fixed_combined = torch.cat([density_emb, variety_emb, supervisor_emb], dim=1)  # [batch_size, 3*(d_model//4)]
            fixed_token = self.fixed_proj(fixed_combined).unsqueeze(1)  # [batch_size, 1, d_model]
            
            # 将固定token添加到序列开头
            x = torch.cat([fixed_token, seq_emb], dim=1)  # [batch_size, seq_len+1, d_model]
        else:
            x = seq_emb
            
        # 添加残差连接和层归一化
        residual = x
        x = self.norm1(x + self.dropout1(x))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        x = x + residual
        
        # Transformer编码（逐层，记录注意力）
        self.attention_maps = []
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
            if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
                self.attention_maps.append(layer.attn_weights)  # [batch, num_heads, seq_len, seq_len]
        transformer_out = out  # [batch_size, seq_len(+1), hidden_dim]
        
        # ------ Grad-CAM支持 ------
        self.gradcam_activations = transformer_out
        self.gradcam_gradients = None
        if register_hook:
            def save_gradients(grad):
                self.gradcam_gradients = grad
            transformer_out.register_hook(save_gradients)
        # ------ End Grad-CAM支持 ------

        # 取第一个token（固定变量）的输出作为表示
        context = transformer_out[:, 0, :]  # [batch_size, hidden_dim]
        
        # 输出层
        output = self.output_fc(context)  # [batch_size, output_dim]
        return output

    def get_gradcam(self):
        """返回Grad-CAM重要性分数 [batch, seq_len+1]"""
        if self.gradcam_activations is not None and self.gradcam_gradients is not None:
            grads = self.gradcam_gradients  # [batch, seq_len+1, hidden_dim]
            activations = self.gradcam_activations  # [batch, seq_len+1, hidden_dim]
            # 取hidden_dim维度均值作为权重
            weights = grads.mean(dim=-1, keepdim=True)  # [batch, seq_len+1, 1]
            gradcam = (activations * weights).sum(dim=-1)  # [batch, seq_len+1]
            return gradcam
        else:
            return None

    def get_attention_weights(self):
        """返回所有层的注意力权重: List[Tensor], 每个shape=[batch, num_heads, seq_len, seq_len]"""
        return self.attention_maps if hasattr(self, 'attention_maps') else None


class ImprovedTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=1, fixed_input_dim=3, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 序列特征嵌入
        self.seq_embedding = nn.Linear(input_dim, d_model)
        
        # 固定变量处理
        self.density_embedding = nn.Linear(1, d_model // 4)
        
        # 动态计算类别数量
        self.num_variety = len(variety_encoder.classes_)
        self.num_supervisor = len(supervisor_encoder.classes_)
        
        # 类别变量嵌入
        self.variety_embedding = nn.Embedding(self.num_variety, d_model // 4)
        self.supervisor_embedding = nn.Embedding(self.num_supervisor, d_model // 4)
        
        # 固定变量投影到隐藏维度
        self.fixed_proj = nn.Linear(d_model // 4 * 3, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len=27)  # 26天 + 1个固定token
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        logger.info(f"使用 ImprovedTransformer 模型, hidden_dim={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    def forward(self, x_seq, x_fixed=None):
        # x_seq: [batch, seq_len, input_dim]
        # x_fixed: [batch, 3]  # [density, variety_idx, supervisor_idx]
        
        # 1. 序列特征嵌入
        seq_emb = self.seq_embedding(x_seq)  # [batch, seq_len, d_model]
        
        if x_fixed is not None:
            # 2. 处理固定特征
            density = x_fixed[:, 0:1]  # [batch, 1]
            variety = torch.clamp(x_fixed[:, 1].long(), 0, self.num_variety - 1)  # [batch]
            supervisor = torch.clamp(x_fixed[:, 2].long(), 0, self.num_supervisor - 1)  # [batch]
            
            # 嵌入固定特征
            density_emb = self.density_embedding(density)  # [batch, d_model//4]
            variety_emb = self.variety_embedding(variety)  # [batch, d_model//4]
            supervisor_emb = self.supervisor_embedding(supervisor)  # [batch, d_model//4]
            
            # 合并固定特征并投影到d_model维度
            fixed_combined = torch.cat([density_emb, variety_emb, supervisor_emb], dim=1)  # [batch, 3*(d_model//4)]
            fixed_token = self.fixed_proj(fixed_combined).unsqueeze(1)  # [batch, 1, d_model]
            
            # 将固定token添加到序列开头
            x = torch.cat([fixed_token, seq_emb], dim=1)  # [batch, seq_len+1, d_model]
        else:
            x = seq_emb
        
        # 3. 添加位置编码
        x = self.positional_encoding(x)  # [batch, seq_len(+1), d_model]
        
        # 4. Transformer编码
        x = self.encoder(x)  # [batch, seq_len(+1), d_model]
        
        # 5. 取第一个token（固定变量）的输出作为表示
        x = x[:, 0, :]  # [batch, d_model]
        
        # 6. 输出层
        output = self.output(x)  # [batch, output_dim]
        
        return output.squeeze(-1)

class FeatureImportanceLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.importance = nn.Parameter(torch.ones(input_dim) / input_dim)
        
    def forward(self, x):
        return x * self.importance.unsqueeze(0).unsqueeze(0).expand_as(x)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models with support for variable sequence lengths.
    Handles both [batch_size, seq_len, d_model] and [seq_len, batch_size, d_model] inputs.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings for max_len positions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Register div_term as buffer for device compatibility
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
        Returns:
            Tensor: Output tensor with positional encodings added
        """
        # Handle different input formats
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3D, got {x.dim()}D")
            
        seq_len = x.size(1) if x.size(2) == self.d_model else x.size(0)
        
        # Generate positional encodings for the current sequence length
        if seq_len > self.max_len:
            # Dynamically generate positional encodings for longer sequences
            position = torch.arange(seq_len, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * 
                               (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_len, 1, self.d_model, device=x.device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe = self.pe[:seq_len]
        
        # Add positional encodings to input
        if x.size(2) == self.d_model:  # [batch_size, seq_len, d_model]
            pe = pe.permute(1, 0, 2)  # [1, seq_len, d_model]
            x = x + pe.to(x.device)
        else:  # [seq_len, batch_size, d_model]
            x = x + pe.to(x.device)
            
        return self.dropout(x)


class SelfAttentionWrapper(nn.Module):
    """
    Wrapper around MultiheadAttention to capture attention weights.
    """
    def __init__(self, attn_layer, nhead, d_model, dropout=0.1):
        super().__init__()
        self.self_attn = attn_layer
        self.nhead = nhead
        self.d_model = d_model
        self.dropout = dropout
        self.attention_weights = None
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, is_causal=False):
        # Forward pass through the attention layer
        attn_output, attn_weights = self.self_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal
        )
        
        # Store attention weights for later analysis
        if need_weights:
            self.attention_weights = attn_weights.detach()
            
        return attn_output, attn_weights


class AttentionAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None
        
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # Save a reference to self.self_attn for easier access
        self_attn = self.self_attn
        
        # Call the attention with need_weights=True to get attention weights
        x, attn_weights = self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal
        )
        self.attention_weights = attn_weights.detach()
        return self.dropout1(x)

class TransformerWithPostFusion(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=8, dim_feedforward=1024, dropout=0.2, num_fixed_features=3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 特征重要性层
        self.feature_importance = FeatureImportanceLayer(input_dim)
        
        # 序列特征处理
        self.seq_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=26)  # 26天
        
        # 保存注意力权重
        self.attention_weights = []
        
        # 自定义Transformer编码器层
        encoder_layer = AttentionAwareTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 创建编码器层列表
        self.encoder_layers = nn.ModuleList([
            AttentionAwareTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)]
        )
        
        # 创建编码器
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        # 固定特征处理
        self.density_embedding = nn.Linear(1, d_model // 4)
        
        # 动态计算类别数量
        self.num_variety = len(variety_encoder.classes_)
        self.num_supervisor = len(supervisor_encoder.classes_)
        
        # 类别变量嵌入
        self.variety_embedding = nn.Embedding(self.num_variety, d_model // 4)
        self.supervisor_embedding = nn.Embedding(self.num_supervisor, d_model // 4)
        
        # 固定特征投影
        self.fixed_proj = nn.Sequential(
            nn.Linear(d_model // 4 * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 序列特征投影
        self.seq_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 后融合层
        self.post_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"使用 TransformerWithPostFusion 模型, hidden_dim={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'bias' in name:
                    nn.init.constant_(p, 0.0)
                elif 'weight' in name and 'bn' not in name:
                    if 'post_fusion' in name and len(p.shape) > 1:
                        # 最后一层初始化
                        nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
                    else:
                        nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x_seq, x_fixed=None):
        # x_seq: [batch, seq_len, input_dim]
        # x_fixed: [batch, 3]  # [density, variety_idx, supervisor_idx]
        
        # 清空之前的注意力权重
        self.attention_weights = []
        
        # 1. 序列特征处理
        seq_emb = self.seq_embedding(x_seq)  # [batch, seq_len, d_model]
        seq_emb = self.pos_encoder(seq_emb)
        
        # 2. Transformer编码
        seq_features = seq_emb
        for layer in self.encoder_layers:
            seq_features = layer(seq_features)
            # 保存当前层的注意力权重
            if hasattr(layer, 'attention_weights') and layer.attention_weights is not None:
                self.attention_weights.append(layer.attention_weights.detach().cpu())
        
        # 如果没有捕获到注意力权重，尝试从最后一个编码器层获取
        if not self.attention_weights and hasattr(self.encoder_layers[-1], 'attention_weights'):
            self.attention_weights = [self.encoder_layers[-1].attention_weights.detach().cpu()]
        
        # 3. 序列特征聚合 (取平均)
        seq_output = seq_features.mean(dim=1)  # [batch, d_model]
        
        # 4. 处理固定特征
        if x_fixed is not None and x_fixed.dim() > 1 and x_fixed.size(1) >= 3:
            # 提取固定特征
            density = x_fixed[:, 0:1]  # [batch, 1]
            variety = torch.clamp(x_fixed[:, 1].long(), 0, self.num_variety - 1)  # [batch]
            supervisor = torch.clamp(x_fixed[:, 2].long(), 0, self.num_supervisor - 1)  # [batch]
            
            # 嵌入固定特征
            density_emb = self.density_embedding(density)  # [batch, d_model//4]
            variety_emb = self.variety_embedding(variety)  # [batch, d_model//4]
            supervisor_emb = self.supervisor_embedding(supervisor)  # [batch, d_model//4]
            
            # 拼接固定特征
            fixed_combined = torch.cat([density_emb, variety_emb, supervisor_emb], dim=-1)  # [batch, d_model//4 * 3]
            fixed_output = self.fixed_proj(fixed_combined)  # [batch, d_model]
            
            # 序列特征投影
            seq_output = self.seq_proj(seq_output)  # [batch, d_model//2]
            
            # 后融合: 拼接序列特征和固定特征
            combined = torch.cat([seq_output, fixed_output], dim=-1)  # [batch, d_model//2 + d_model]
            output = self.post_fusion(combined)  # [batch, 1]
        else:
            # 如果没有固定特征，只使用序列特征
            seq_output = self.seq_proj(seq_output)  # [batch, d_model//2]
            output = self.post_fusion(seq_output)  # [batch, 1]
        
        return output.squeeze(-1)  # [batch]
    
    def get_attention_weights(self, layer_idx=-1, head_idx=None, average_heads=True):
        """
        获取注意力权重
        
        参数:
            layer_idx (int): 层索引，-1表示最后一层
            head_idx (int, optional): 头索引，None表示返回所有头
            average_heads (bool): 如果为True且head_idx为None，则对所有头取平均
            
        返回:
            torch.Tensor: 注意力权重 
                - 如果head_idx不是None: [batch_size, seq_len, seq_len]
                - 如果head_idx是None且average_heads=True: [batch_size, seq_len, seq_len]
                - 如果head_idx是None且average_heads=False: [batch_size, num_heads, seq_len, seq_len]
        """
        if not hasattr(self, 'attention_weights') or not self.attention_weights:
            logger.warning("No attention weights available. Make sure to run forward pass first.")
            return None
            
        # 确保layer_idx是Python整数
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.item()
            
        if isinstance(layer_idx, (int, float)) and layer_idx < 0:
            layer_idx = len(self.attention_weights) + layer_idx
            
        if layer_idx < 0 or layer_idx >= len(self.attention_weights):
            logger.warning(f"Layer index {layer_idx} out of range [0, {len(self.attention_weights)-1}]")
            return None
            
        # 获取指定层的注意力权重 [batch_size, seq_len, seq_len]
        layer_weights = self.attention_weights[layer_idx]
        
        # 如果是一般的注意力权重（没有头维度）
        if len(layer_weights.shape) == 3:
            return layer_weights
            
        # 处理多头注意力权重 [batch_size, num_heads, seq_len, seq_len]
        if head_idx is not None:
            # 返回特定头 [batch_size, seq_len, seq_len]
            if head_idx < 0 or head_idx >= layer_weights.size(1):
                logger.warning(f"Head index {head_idx} out of range [0, {layer_weights.size(1)-1}]")
                return None
            return layer_weights[:, head_idx]
        elif average_heads:
            # 对所有头取平均 [batch_size, seq_len, seq_len]
            return layer_weights.mean(dim=1)
        else:
            # 返回所有头 [batch_size, num_heads, seq_len, seq_len]
            return layer_weights

def visualize_gradcam(model, val_loader, device, num_sequences=5):
    """
    可视化Grad-CAM
    
    参数:
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 设备 (cuda/cpu)
        num_sequences: 要可视化的序列数量
    """
    model.eval()
    
    # 创建保存目录
    save_dir = os.path.join('visualizations', 'gradcam')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"开始Grad-CAM可视化，结果将保存到: {os.path.abspath(save_dir)}")
    
    try:
        # 获取一个批次的数据
        data_iter = iter(val_loader)
        X_seq_all, X_fixed_all, _ = next(data_iter)
        X_seq_all = X_seq_all.to(device)
        X_fixed_all = X_fixed_all.to(device) if X_fixed_all is not None else None
        
        batch_size, seq_len, num_features = X_seq_all.shape
        logger.info(f"输入数据形状 - 序列: {X_seq_all.shape}, 固定特征: {X_fixed_all.shape if X_fixed_all is not None else 'None'}")
        
        # 限制处理的序列数量
        num_sequences = min(num_sequences, batch_size)
        
        # 为每个特征维度创建存储Grad-CAM的列表
        feature_gradcams = [[] for _ in range(num_features)]
        
        # 对每个样本单独处理
        for sample_idx in range(num_sequences):
            logger.info(f"处理序列 {sample_idx+1}/{num_sequences}")
            
            # 获取当前样本
            X_seq = X_seq_all[sample_idx:sample_idx+1]  # [1, seq_len, num_features]
            X_fixed = X_fixed_all[sample_idx:sample_idx+1] if X_fixed_all is not None else None
            
            # 确保需要梯度
            X_seq.requires_grad = True
            
            # 前向传播
            with torch.set_grad_enabled(True):
                # 注册hook来捕获梯度
                gradients = []
                activations = []
                
                def backward_hook(module, grad_input, grad_output):
                    gradients.append(grad_output[0].detach())
                    
                def forward_hook(module, input, output):
                    activations.append(output.detach())
                
                # 注册hook到最后一个transformer层
                if hasattr(model, 'encoder_layers') and len(model.encoder_layers) > 0:
                    last_layer = model.encoder_layers[-1]
                    handle_forward = last_layer.register_forward_hook(forward_hook)
                    handle_backward = last_layer.register_backward_hook(backward_hook)
                
                try:
                    # 前向传播
                    output = model(X_seq, X_fixed)
                    logger.info(f"样本 {sample_idx+1} 输出形状: {output.shape}")
                    
                    # 反向传播
                    model.zero_grad()
                    output.mean().backward()
                                        # 获取梯度和激活值
                    if gradients and activations:
                        gradients = gradients[0]  # [1, seq_len, d_model]
                        activations = activations[0]  # [1, seq_len, d_model]
                        
                        # 对每个特征维度计算重要性
                        for feat_idx in range(num_features):
                            # 1. 计算特征特定的梯度
                            # 获取输入梯度 [batch_size=1, seq_len, num_features]
                            X_seq_grad = X_seq.grad[0]  # [seq_len, num_features]
                            
                            # 2. 计算特征特定的激活值
                            # 使用全局平均池化获取通道重要性
                            alpha = torch.mean(gradients, dim=1, keepdim=True)  # [1, seq_len, 1]
                            
                            # 3. 计算加权激活图
                            # 使用特征特定的梯度作为权重
                            feat_importance = torch.abs(X_seq_grad[:, feat_idx])  # [seq_len]
                            
                            # 4. 计算Grad-CAM
                            # 使用全局平均池化的梯度作为权重
                            weights = torch.softmax(alpha[0, :, 0], dim=0)  # [seq_len]
                            feat_gradcam = (weights * feat_importance).detach().cpu().numpy()
                            
                            # 5. 应用ReLU并归一化
                            feat_gradcam = np.maximum(feat_gradcam, 0)  # ReLU
                            if np.max(feat_gradcam) > 0:
                                feat_gradcam = (feat_gradcam - feat_gradcam.min()) / (feat_gradcam.max() - feat_gradcam.min() + 1e-10)
                            
                            feature_gradcams[feat_idx].append(feat_gradcam)
                            
                finally:
                    # 移除hook
                    if 'handle_forward' in locals():
                        handle_forward.remove()
                    if 'handle_backward' in locals():
                        handle_backward.remove()
        
        # 为每个特征维度创建可视化
        feature_names = ['鸡舍温度-平均', '湿度内部平均', '外部-平均']
        
        for feat_idx in range(num_features):
            if len(feature_gradcams[feat_idx]) == 0:
                logger.warning(f"特征 {feat_idx} 没有有效的Grad-CAM数据")
                continue
                
            # 计算平均Grad-CAM
            avg_gradcam = np.mean(feature_gradcams[feat_idx], axis=0)
            
            # 创建图形
            plt.figure(figsize=(14, 8))
            
            # 创建子图
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            
            # 绘制Grad-CAM重要性分数
            ax1.plot(avg_gradcam, 'b-', linewidth=2, marker='o')
            ax1.set_title(f'Grad-CAM 重要性分数 ({feature_names[feat_idx]})', fontsize=14, pad=20)
            ax1.set_xlabel('时间步', fontsize=12)
            ax1.set_ylabel('重要性分数', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制热力图
            im = ax2.imshow(avg_gradcam.reshape(1, -1),
                         cmap='viridis',
                         aspect='auto',
                         interpolation='nearest')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2)
            cbar.ax.tick_params(labelsize=10)
            
            ax2.set_title(f'Grad-CAM 热力图 ({feature_names[feat_idx]})', fontsize=14, pad=20)
            ax2.set_xlabel('时间步', fontsize=12)
            ax2.set_yticks([])
            
            plt.tight_layout()
            
            # 保存图像
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_filename = f'gradcam_{feature_names[feat_idx]}_{timestamp}.png'.replace(' ', '_')
            save_path = os.path.join(save_dir, save_filename)
            
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            
            logger.info(f"{feature_names[feat_idx]} 的Grad-CAM可视化已保存: {os.path.abspath(save_path)}")
            
            # 同时保存原始数据
            data_save_path = os.path.join(save_dir, f'gradcam_{feature_names[feat_idx]}_{timestamp}.npy')
            np.save(data_save_path, avg_gradcam)
            logger.info(f"{feature_names[feat_idx]} 的Grad-CAM数据已保存: {os.path.abspath(data_save_path)}")
        
        logger.info("Grad-CAM可视化完成")
            
    except Exception as e:
        logger.error(f"执行Grad-CAM可视化时发生错误: {str(e)}", exc_info=True)
    finally:
        if 'plt' in locals() and plt.fignum_exists(plt.gcf().number):
            plt.close()

def visualize_attention_weights(model, X_seq_sample, X_fixed_sample, device, num_heads=4, num_layers=8, save_dir='visualizations'):
    """
    可视化注意力权重
    
    参数:
        model: 训练好的模型
        X_seq_sample: 序列输入样本 [batch_size, seq_len, input_dim]
        X_fixed_sample: 固定特征样本 [batch_size, num_fixed_features]
        device: 设备 (cuda/cpu)
        num_heads: 注意力头数
        num_layers: 模型层数
        save_dir: 保存目录
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 前向传播获取注意力权重
        with torch.no_grad():
            _ = model(X_seq_sample, X_fixed_sample)
        
        # 获取注意力权重
        attention_weights = []
        for layer_idx in range(num_layers):
            try:
                # 获取当前层的注意力权重 [batch_size, num_heads, seq_len, seq_len]
                weights = model.get_attention_weights(layer_idx=layer_idx, average_heads=False)
                attention_weights.append(weights.cpu().numpy())
            except Exception as e:
                logger.warning(f"Failed to get attention weights for layer {layer_idx}: {str(e)}")
                continue
        
        if not attention_weights:
            logger.warning("No attention weights were retrieved")
            return
            
        # 可视化每个样本的注意力权重
        batch_size = min(3, X_seq_sample.size(0))  # 最多可视化3个样本
        for sample_idx in range(batch_size):
            # 创建图形
            fig, axes = plt.subplots(num_layers, num_heads, figsize=(4*num_heads, 4*num_layers))
            if num_layers == 1:
                axes = axes.reshape(1, -1)
            
            for layer_idx in range(num_layers):
                if layer_idx >= len(attention_weights):
                    continue
                    
                for head_idx in range(num_heads):
                    try:
                        # 获取当前头和层的注意力权重
                        attn = attention_weights[layer_idx][sample_idx, head_idx]
                        
                        # 绘制热力图
                        ax = axes[layer_idx, head_idx]
                        im = ax.imshow(attn, cmap='viridis', vmin=0, vmax=1)
                        
                        # 设置标题和标签
                        if layer_idx == 0:
                            ax.set_title(f'Head {head_idx+1}')
                        if head_idx == 0:
                            ax.set_ylabel(f'Layer {layer_idx+1}')
                            
                        # 添加颜色条
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        
                    except Exception as e:
                        logger.warning(f"Error visualizing attention for layer {layer_idx}, head {head_idx}: {str(e)}")
                        continue
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(save_dir, f'attention_weights_sample_{sample_idx}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved attention weights visualization to {save_path}")
            
    except Exception as e:
        logger.error(f"Error in visualize_attention_weights: {str(e)}", exc_info=True)
    finally:
        plt.close('all')


def visualize_attention_per_sequence(model, val_loader, device, num_sequences=3):
    """
    可视化每个序列的注意力权重
    """
    model.eval()
    os.makedirs('visualizations', exist_ok=True)
    
    try:
        # 获取一个批次的数据
        X_seq_sample, X_fixed_sample, _ = next(iter(val_loader))
        X_seq_sample = X_seq_sample.to(device)
        X_fixed_sample = X_fixed_sample.to(device) if X_fixed_sample is not None else None
        
        # 限制序列数量
        num_sequences = min(num_sequences, X_seq_sample.size(0))
        
        # 对每个序列进行可视化
        for seq_idx in range(num_sequences):
            try:
                # 获取单个序列
                seq = X_seq_sample[seq_idx:seq_idx+1]  # Keep batch dimension
                fixed = X_fixed_sample[seq_idx:seq_idx+1] if X_fixed_sample is not None else None
                
                # 获取注意力权重
                with torch.no_grad():
                    _ = model(seq, fixed)
                    # 获取最后一层的注意力权重 [batch_size, num_heads, seq_len, seq_len]
                    attn_weights = model.get_attention_weights(layer_idx=-1, average_heads=False)
                    attn_weights = attn_weights.squeeze(0)  # Remove batch dimension
                
                # 创建热力图
                num_heads = attn_weights.size(0)
                fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))
                
                if num_heads == 1:
                    axes = [axes]
                
                for head_idx in range(num_heads):
                    ax = axes[head_idx]
                    im = ax.imshow(attn_weights[head_idx].cpu().numpy(), cmap='viridis')
                    ax.set_title(f'Head {head_idx+1}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                
                # 保存图像
                save_path = f'visualizations/attention_sequence_{seq_idx}.png'
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Saved attention visualization for sequence {seq_idx} to {save_path}")
                
            except Exception as e:
                logger.error(f"Error visualizing attention for sequence {seq_idx}: {str(e)}", exc_info=True)
                if 'plt' in locals() and plt.fignum_exists(plt.gcf().number):
                    plt.close()
    
    except Exception as e:
        logger.error(f"Error in visualize_attention_per_sequence: {str(e)}", exc_info=True)

def attention_rollout(model, X_seq_sample, X_fixed_sample, device, num_heads=8, num_layers=8):
    """
    实现Attention Rollout可视化，仅分析时间序列部分（前26个时间步）
    
    参数:
        model: 训练好的模型
        X_seq_sample: 输入序列 [batch_size, seq_len, input_dim]
        X_fixed_sample: 固定特征 [batch_size, num_fixed_features]
        device: 设备 (cuda/cpu)
        num_heads: 注意力头数
        num_layers: 模型层数
        
    返回:
        list: 每个样本的attention rollout矩阵列表
    """
    model.eval()
    
    # 创建保存目录
    save_dir = os.path.join('visualizations', 'attention_rollout')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"开始Attention Rollout分析，结果将保存到: {os.path.abspath(save_dir)}")
    
    # 确保输入是torch.Tensor类型
    if not isinstance(X_seq_sample, torch.Tensor):
        X_seq_sample = torch.FloatTensor(X_seq_sample)
    
    if X_fixed_sample is not None and not isinstance(X_fixed_sample, torch.Tensor):
        X_fixed_sample = torch.FloatTensor(X_fixed_sample)
    
    try:
        # 确保输入在正确的设备上
        X_seq_sample = X_seq_sample.to(device)
        if X_fixed_sample is not None:
            X_fixed_sample = X_fixed_sample.to(device)
        
        batch_size = X_seq_sample.size(0)
        seq_len = 26  # 固定为26个时间步
        
        # 获取注意力权重
        attention_weights = []
        
        # 注册hook来捕获注意力权重
        def get_attention_weights(module, input, output):
            # 只取前26个时间步的注意力权重
            attn = output[1].detach().cpu()  # [batch_size, num_heads, seq_len, seq_len]
            attn = attn[..., :seq_len, :seq_len]  # 只取前26个时间步
            attention_weights.append(attn)
        
        # 注册hook到每个注意力层
        handles = []
        for layer in model.encoder_layers:
            handles.append(layer.self_attn.register_forward_hook(get_attention_weights))
        
        # 前向传播
        with torch.no_grad():
            _ = model(X_seq_sample, X_fixed_sample)
        
        # 移除hook
        for handle in handles:
            handle.remove()
        
        if not attention_weights:
            logger.error("未捕获到注意力权重")
            return None
            
        # 初始化注意力矩阵为单位矩阵 [batch_size, seq_len, seq_len]
        attention_rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        # 逐层计算Attention Rollout
        for layer_weights in attention_weights:
            # 平均所有注意力头 [batch_size, num_heads, seq_len, seq_len] -> [batch_size, seq_len, seq_len]
            layer_weights = layer_weights.mean(dim=1).to(device)
            
            # 确保注意力权重维度正确
            if layer_weights.size(-1) > seq_len:
                layer_weights = layer_weights[..., :seq_len, :seq_len]
            
            # 添加残差连接
            identity = torch.eye(seq_len).unsqueeze(0).to(device)
            layer_weights = 0.5 * layer_weights + 0.5 * identity
            
            # 确保数值稳定性
            layer_weights = layer_weights / layer_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            
            # 更新Attention Rollout
            attention_rollout = torch.bmm(layer_weights, attention_rollout)
        
        # 对batch中的每个样本单独处理
        all_attention_rollouts = []
        for i in range(batch_size):
            # 获取单个样本的attention rollout
            sample_attention = attention_rollout[i].cpu().numpy()
            all_attention_rollouts.append(sample_attention)
            
            # 保存为npy文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_id = f'sample_{i:03d}_{timestamp}'
            
            # 保存npy文件
            npy_path = os.path.join(save_dir, f'attention_rollout_{sample_id}.npy')
            np.save(npy_path, sample_attention)
            
            # 可视化
            plt.figure(figsize=(12, 10))
            plt.imshow(sample_attention, cmap='viridis', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title(f'Attention Rollout - {sample_id}')
            plt.xlabel('Target Time Step (0-25)')
            plt.ylabel('Source Time Step (0-25)')
            plt.xticks(range(0, 26, 5))
            plt.yticks(range(0, 26, 5))
            
            # 保存图片
            img_path = os.path.join(save_dir, f'attention_rollout_{sample_id}.png')
            plt.savefig(img_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f'已保存样本 {sample_id} 的Attention Rollout可视化: {img_path}')
        
        # 计算平均attention rollout
        if all_attention_rollouts:
            avg_attention = np.mean(all_attention_rollouts, axis=0)
            
            # 保存平均attention rollout
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            avg_npy_path = os.path.join(save_dir, f'attention_rollout_avg_{timestamp}.npy')
            np.save(avg_npy_path, avg_attention)
            
            # 可视化平均attention rollout
            plt.figure(figsize=(12, 10))
            plt.imshow(avg_attention, cmap='viridis', aspect='auto')
            plt.colorbar(label='Average Attention Weight')
            plt.title(f'Average Attention Rollout ({batch_size} samples)')
            plt.xlabel('Target Time Step (0-25)')
            plt.ylabel('Source Time Step (0-25)')
            plt.xticks(range(0, 26, 5))
            plt.yticks(range(0, 26, 5))
            
            # 保存平均attention rollout图片
            avg_img_path = os.path.join(save_dir, f'attention_rollout_avg_{timestamp}.png')
            plt.savefig(avg_img_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f'已保存平均Attention Rollout可视化: {avg_img_path}')
        
        return all_attention_rollouts
        
    except Exception as e:
        logger.error(f'执行Attention Rollout时出错: {str(e)}')
        logger.error(traceback.format_exc())
        return None

def visualize_model_interpretability(model, val_loader, device):
    """主函数：执行所有模型可解释性可视化"""
    logger.info("开始模型可解释性分析...")
    
    try:
        # 获取一个批次的数据
        data_iter = iter(val_loader)
        X_seq_sample, X_fixed_sample, _ = next(data_iter)
        X_seq_sample = X_seq_sample.to(device)
        X_fixed_sample = X_fixed_sample.to(device) if X_fixed_sample is not None else None
        
        # 1. 可视化Grad-CAM
        logger.info("开始Grad-CAM可视化...")
        visualize_gradcam(model, val_loader, device)
        
        # 2. 可视化注意力权重
        logger.info("开始注意力权重可视化...")
        visualize_attention_weights(model, X_seq_sample, X_fixed_sample, device)
        
        # 3. 可视化每个序列的注意力
        logger.info("开始序列级注意力可视化...")
        visualize_attention_per_sequence(model, val_loader, device)
        
        # 4. 执行Attention Rollout分析
        logger.info("开始Attention Rollout分析...")
        attention_rollout(model, X_seq_sample, X_fixed_sample, device)
        
        logger.info("模型可解释性分析完成")
    except Exception as e:
        logger.error(f"模型可解释性分析时出错: {str(e)}", exc_info=True)

def combined_loss(preds, targets, alpha=0.6):
    """
    组合损失函数，结合MSE和R2损失
    
    参数:
        preds: 模型预测值 [batch_size]
        targets: 真实值 [batch_size]
        alpha: MSE损失的初始权重，R2损失的初始权重为(1-alpha)
        
    返回:
        tuple: (总损失, MSE损失值, R2值)
    """
    # 计算MSE损失
    mse_loss = F.mse_loss(preds, targets)
    
    # 计算R2分数
    target_mean = targets.mean()
    ss_tot = ((targets - target_mean) ** 2).sum()
    ss_res = ((targets - preds) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # 动态调整权重
    # 当R2为负时，增加MSE的权重
    if r2 < 0:
        alpha = 0.8
    # 当R2接近1时，增加R2的权重
    elif r2 > 0.8:
        alpha = 0.4
    
    # 计算R2损失 (1 - R²)
    r2_loss = 1 - r2
    
    # 组合损失
    loss = alpha * mse_loss + (1 - alpha) * r2_loss
    
    # 记录中间值用于调试
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Invalid loss value - MSE: {mse_loss.item():.4f}, R2: {r2.item():.4f}, Alpha: {alpha:.2f}")
    
    return loss, mse_loss.item(), r2.item()

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

def evaluate_regression(y_true, y_pred, scaler=None):
    """
    还原标准化，计算R2、MAE、MAPE
    """
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'R2': r2, 'MAE': mae, 'MAPE': mape}

def train_model(model, train_loader, val_loader, device, num_epochs=200, patience=30, lr=1e-3):
    grad_accum_steps = 4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        eps=1e-8,
        amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    best_val_r2 = -np.inf
    best_model_state = None
    best_epoch = 0
    best_val_preds = None
    best_val_targets = None
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
    no_improve_count = 0
    mse_loss = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        optimizer.zero_grad()
        for batch_idx, (X_seq, X_fixed, y) in enumerate(train_loader):
            X_seq = X_seq.to(device)
            if X_fixed is not None:
                X_fixed = X_fixed.to(device)
            y = y.to(device)
            outputs = model(X_seq, X_fixed)
            loss = mse_loss(outputs, y)
            loss = loss / grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(epoch + batch_idx / len(train_loader))
            train_losses.append(loss.item() * X_seq.size(0))
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())
        train_loss = np.sum(train_losses) / len(train_loader.dataset)
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_r2 = r2_score(train_targets, train_preds)
        # 验证
        model.eval()
        val_losses = []
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for X_seq, X_fixed, y in val_loader:
                X_seq = X_seq.to(device)
                if X_fixed is not None:
                    X_fixed = X_fixed.to(device)
                y = y.to(device)
                outputs = model(X_seq, X_fixed)
                loss = mse_loss(outputs, y)
                val_losses.append(loss.item())
                val_targets.append(y.cpu().numpy())
                val_preds.append(outputs.cpu().numpy())
        val_loss = np.mean(val_losses)
        val_targets = np.concatenate(val_targets)
        val_preds = np.concatenate(val_preds)
        val_metrics = evaluate_regression(val_targets, val_preds, scaler=target_scaler)
        val_r2 = val_metrics['R2']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f} | Best Val R2: {best_val_r2:.4f}")
        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save({'model_state_dict': model.state_dict()}, 'models/best_model.pth')
    return model, history, best_val_targets, best_val_preds

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # 获取序列特征维度
    seq_input_dim = X_train_seq.shape[-1]

    results = []

    # 1. TransformerWithPostFusion
    model1 = TransformerWithPostFusion(
        input_dim=seq_input_dim,
        d_model=256,
        nhead=16,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        num_fixed_features=3
    ).to(device)
    model1, hist1, val_targets1, val_preds1 = train_model(model1, train_loader, val_loader, device, num_epochs=200, patience=30, lr=1e-3)
    metrics1 = evaluate_regression(val_targets1, val_preds1, scaler=target_scaler)
    print("TransformerWithPostFusion 验证集还原后指标:", metrics1)
    logger.info(f"TransformerWithPostFusion 验证集还原后指标: {metrics1}")
    results.append(('TransformerWithPostFusion', model1, metrics1))

    # 2. TransformerWithFixedToken
    model2 = TransformerWithFixedToken(
        seq_input_dim=seq_input_dim,
        fixed_input_dim=X_train_fixed.shape[-1],
        hidden_dim=128,
        output_dim=1,
        nhead=8,
        num_layers=3,
        dropout=0.1
    ).to(device)
    model2, hist2, val_targets2, val_preds2 = train_model(model2, train_loader, val_loader, device, num_epochs=200, patience=30, lr=1e-3)
    metrics2 = evaluate_regression(val_targets2, val_preds2, scaler=target_scaler)
    print("TransformerWithFixedToken 验证集还原后指标:", metrics2)
    logger.info(f"TransformerWithFixedToken 验证集还原后指标: {metrics2}")
    results.append(('TransformerWithFixedToken', model2, metrics2))

    # 3. ImprovedTransformer
    model3 = ImprovedTransformer(
        input_dim=seq_input_dim,
        output_dim=1,
        fixed_input_dim=X_train_fixed.shape[-1],
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1
    ).to(device)
    model3, hist3, val_targets3, val_preds3 = train_model(model3, train_loader, val_loader, device, num_epochs=200, patience=30, lr=1e-3)
    metrics3 = evaluate_regression(val_targets3, val_preds3, scaler=target_scaler)
    print("ImprovedTransformer 验证集还原后指标:", metrics3)
    logger.info(f"ImprovedTransformer 验证集还原后指标: {metrics3}")
    results.append(('ImprovedTransformer', model3, metrics3))

    # 选择R2最高的模型
    best_model_name, best_model, best_metrics = max(results, key=lambda x: x[2]['R2'])
    print(f"最优模型: {best_model_name}, 验证集指标: {best_metrics}")
    logger.info(f"最优模型: {best_model_name}, 验证集指标: {best_metrics}")
    logger.info("程序执行完成")

    # ===== SHAP可解释性分析 =====
    try:
        from visualization_utils import plot_shap_fixed_and_sequence
        # 假定 X_val_seq, X_val_fixed, seq_feature_names, fixed_feature_names 已定义
        # seq_feature_names: 序列变量名列表，如 ['feature1', 'feature2']
        # fixed_feature_names: 固定变量名列表，如 ['Density', 'BirdsVariety', 'FarmSupervisor']
        seq_feature_names = [f"seq_{i+1}" for i in range(X_val_seq.shape[-1])] if 'X_val_seq' in locals() else None
        fixed_feature_names = [f"fixed_{i+1}" for i in range(X_val_fixed.shape[-1])] if 'X_val_fixed' in locals() else None
        plot_shap_fixed_and_sequence(
            best_model,
            X_val_seq[:100] if X_val_seq.shape[0] > 100 else X_val_seq,
            X_val_fixed[:100] if X_val_fixed is not None and X_val_fixed.shape[0] > 100 else X_val_fixed,
            seq_feature_names=seq_feature_names,
            fixed_feature_names=fixed_feature_names,
            device=device,
            save_dir='visualizations/shap'
        )
        logger.info("SHAP特征重要性分析已输出到visualizations/shap目录")
    except Exception as e:
        logger.error(f"SHAP可解释性分析失败: {e}")

    # ===== Grad-CAM 分析 =====
    try:
        from torch.utils.data import DataLoader, TensorDataset
        import os
        
        # 创建测试集DataLoader
        X_test_seq_tensor = torch.tensor(X_val_seq, dtype=torch.float32) if not torch.is_tensor(X_val_seq) else X_val_seq.clone().detach()
        X_test_fixed_tensor = torch.tensor(X_val_fixed, dtype=torch.float32) if not torch.is_tensor(X_val_fixed) else X_val_fixed.clone().detach()
        y_test_tensor = torch.tensor(y_val, dtype=torch.float32) if not torch.is_tensor(y_val) else y_val.clone().detach()
        
        test_dataset = TensorDataset(X_test_seq_tensor, X_test_fixed_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 确保输出目录存在
        os.makedirs('visualizations/gradcam', exist_ok=True)
        
        # 调用visualize_gradcam函数
        logger.info("开始执行Grad-CAM分析...")
        visualize_gradcam(best_model, test_loader, device)
        
        logger.info("Grad-CAM分析已完成，结果已保存到visualizations/gradcam目录")
    except Exception as e:
        logger.error(f"Grad-CAM分析失败: {e}")

    # ===== Attention Rollout 分析 =====
    try:
        # 准备输入数据
        batch_size = 32  # 批处理大小，根据GPU内存调整
        total_samples = len(X_test_seq_tensor)
        
        logger.info(f"开始执行Attention Rollout分析，总样本数: {total_samples}，批处理大小: {batch_size}...")
        
        # 确保输出目录存在
        rollout_dir = 'visualizations/attention_rollout'
        os.makedirs(rollout_dir, exist_ok=True)
        
        # 存储所有样本的attention rollout结果
        all_rollouts = []
        
        # 分批处理数据
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_size_actual = batch_end - i
            
            logger.info(f"正在处理样本 {i+1}-{batch_end}/{total_samples}...")
            
            # 准备当前批次数据
            X_seq_batch = X_test_seq_tensor[i:batch_end].to(device)
            X_fixed_batch = X_test_fixed_tensor[i:batch_end].to(device) if X_test_fixed_tensor is not None else None
            
            # 执行attention rollout
            rollout = attention_rollout(
                model=best_model,
                X_seq_sample=X_seq_batch,
                X_fixed_sample=X_fixed_batch,
                device=device,
                num_heads=8,  # 根据模型的实际头数调整
                num_layers=3   # 根据模型的层数调整
            )
            
            if rollout is not None:
                all_rollouts.append(rollout)
        
        # 保存所有样本的attention rollout结果
        if all_rollouts:
            # 合并所有批次的attention rollout
            all_rollouts = np.concatenate(all_rollouts, axis=0)
            
            # 计算平均attention rollout
            mean_rollout = np.mean(all_rollouts, axis=0)
            
            # 保存平均attention rollout
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(rollout_dir, f'attention_rollout_mean_{timestamp}.npy')
            np.save(save_path, mean_rollout)
            
            # 可视化平均attention rollout
            plt.figure(figsize=(12, 10))
            plt.imshow(mean_rollout, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.title('Mean Attention Rollout (All Samples)')
            plt.grid(False)
            
            # 保存图像
            img_save_path = os.path.join(rollout_dir, f'attention_rollout_mean_{timestamp}.png')
            plt.savefig(img_save_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            
            logger.info(f"平均Attention Rollout已保存: {os.path.abspath(img_save_path)}")
        
        logger.info(f"Attention Rollout分析已完成，共处理 {total_samples} 个样本，结果已保存到 {os.path.abspath(rollout_dir)} 目录")
        
    except Exception as e:
        logger.error(f"Attention Rollout分析失败: {e}")
        logger.error(traceback.format_exc())