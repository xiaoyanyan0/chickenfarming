import os
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader

# 导入自定义模块
from model import TransformerWithPostFusion
from train import PoultryDataset, collate_fn, R2Loss, train_model, evaluate
from visualize import visualize_model, plot_training_curves, plot_predictions

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """加载和预处理数据"""
    logger.info(f"Loading data from {file_path}")
    
    # 读取数据
    df = pd.read_excel(file_path)
    
    # 数据预处理
    # 1. 处理日期列
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. 处理类别变量
    variety_encoder = LabelEncoder()
    supervisor_encoder = LabelEncoder()
    
    if 'BirdsVariety' in df.columns:
        df['variety_encoded'] = variety_encoder.fit_transform(df['BirdsVariety'])
    if 'Supervisor' in df.columns:
        df['supervisor_encoded'] = supervisor_encoder.fit_transform(df['Supervisor'])
    
    # 3. 选择特征和目标列
    # 假设目标列是'Target'
    target_col = 'Target'  # 请替换为实际的目标列名
    
    # 序列特征 (时间序列数据)
    seq_cols = ['Feature1', 'Feature2']  # 请替换为实际的序列特征列名
    
    # 固定特征
    fixed_cols = ['variety_encoded', 'supervisor_encoded', 'Density']  # 请替换为实际的固定特征列名
    
    # 4. 处理缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 5. 标准化数值特征
    scaler = StandardScaler()
    df[seq_cols] = scaler.fit_transform(df[seq_cols])
    
    # 6. 准备序列数据
    def create_sequences(data, seq_length=25):
        sequences = []
        targets = []
        fixed_features = []
        
        # 按组处理数据
        group_cols = ['GroupID']  # 请替换为实际的分组列名
        
        for _, group in data.groupby(group_cols):
            group = group.sort_values('Date')
            
            # 提取固定特征 (取第一个样本的固定特征)
            fixed_feature = group[fixed_cols].iloc[0].values if fixed_cols else None
            
            # 创建序列
            for i in range(len(group) - seq_length + 1):
                seq = group[seq_cols].iloc[i:i+seq_length].values
                target = group[target_col].iloc[i+seq_length-1]
                
                sequences.append(seq)
                targets.append(target)
                fixed_features.append(fixed_feature)
        
        return np.array(sequences), np.array(fixed_features), np.array(targets)
    
    # 创建序列数据
    sequences, fixed_features, targets = create_sequences(df)
    
    # 划分训练集和验证集
    X_train_seq, X_val_seq, X_train_fixed, X_val_fixed, y_train, y_val = train_test_split(
        sequences, fixed_features, targets, test_size=0.2, random_state=42
    )
    
    return (X_train_seq, X_val_seq, X_train_fixed, X_val_fixed, y_train, y_val,
            variety_encoder, supervisor_encoder, scaler)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    try:
        # 1. 加载数据
        data_path = 'path_to_your_data.xlsx'  # 请替换为实际的数据文件路径
        (X_train_seq, X_val_seq, X_train_fixed, X_val_fixed, y_train, y_val,
         variety_encoder, supervisor_encoder, scaler) = load_data(data_path)
        
        # 2. 创建数据集和数据加载器
        train_dataset = PoultryDataset(X_train_seq, X_train_fixed, y_train)
        val_dataset = PoultryDataset(X_val_seq, X_val_fixed, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
        
        # 3. 初始化模型
        input_dim = X_train_seq.shape[2]  # 序列特征的维度
        num_variety = len(variety_encoder.classes_) if hasattr(variety_encoder, 'classes_') else 1
        num_supervisor = len(supervisor_encoder.classes_) if hasattr(supervisor_encoder, 'classes_') else 1
        
        model = TransformerWithPostFusion(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=6,
            num_variety=num_variety,
            num_supervisor=num_supervisor
        ).to(device)
        
        # 4. 定义损失函数和优化器
        criterion = R2Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 5. 训练模型
        logger.info("Starting model training...")
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=100,
            patience=20
        )
        
        # 6. 绘制训练曲线
        plot_training_curves(train_losses, val_losses)
        
        # 7. 在验证集上评估模型
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        
        # 8. 绘制预测结果
        plot_predictions(val_targets, val_preds)
        
        # 9. 可视化模型注意力
        logger.info("Generating model visualizations...")
        visualize_model(model, val_loader, device)
        
        logger.info("Training and evaluation completed!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
