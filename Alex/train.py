import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from model import TransformerWithPostFusion

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

# 确保目录存在
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

class PoultryDataset(Dataset):
    def __init__(self, sequences, fixed_features, targets, seq_len=25):
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
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        fixed = self.fixed_features[idx] if self.fixed_features is not None else np.array([])
        target = self.targets[idx]
        
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
            
        return torch.FloatTensor(seq), torch.FloatTensor(fixed), torch.FloatTensor([target])

def collate_fn(batch):
    """将批次数据整理成模型需要的格式"""
    sequences = [item[0] for item in batch]
    fixed_features = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    
    # 转换为张量
    sequences = torch.stack(sequences)
    fixed_features = torch.stack(fixed_features) if fixed_features[0].numel() > 0 else None
    targets = torch.stack(targets)
    
    return sequences, fixed_features, targets

class R2Loss(nn.Module):
    """自定义 R2 损失函数，最小化 1 - R²"""
    def __init__(self):
        super(R2Loss, self).__init__()
        
    def forward(self, y_pred, y_true):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-7)
        return 1 - r2

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        x_seq, x_fixed, y = batch
        x_seq, y = x_seq.to(device), y.to(device)
        x_fixed = x_fixed.to(device) if x_fixed is not None else None
        
        # 前向传播
        outputs = model(x_seq, x_fixed)
        loss = criterion(outputs, y.squeeze())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_seq, x_fixed, y = batch
            x_seq, y = x_seq.to(device), y.to(device)
            x_fixed = x_fixed.to(device) if x_fixed is not None else None
            
            outputs = model(x_seq, x_fixed)
            loss = criterion(outputs, y.squeeze())
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return avg_loss, all_preds, all_targets

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=20):
    """训练模型"""
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, 'models/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        # 打印日志
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载数据
    # TODO: 添加数据加载代码
    
    # 创建模型
    model = TransformerWithPostFusion(
        input_dim=1,  # 根据实际特征维度调整
        d_model=128,
        nhead=8,
        num_layers=6,
        num_variety=len(variety_encoder.classes_) if 'variety_encoder' in globals() else 1,
        num_supervisor=len(supervisor_encoder.classes_) if 'supervisor_encoder' in globals() else 1
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = R2Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 训练模型
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=20
    )
    
    logger.info('Training completed')

if __name__ == '__main__':
    main()
