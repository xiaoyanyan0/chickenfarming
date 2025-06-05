import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from torch.utils.data import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visualization.log')
    ]
)
logger = logging.getLogger(__name__)

def plot_attention_weights(model, dataloader, device, num_layers=6):
    """可视化注意力权重"""
    model.eval()
    
    # 获取一个批次的数据
    X_seq_sample, X_fixed_sample, _ = next(iter(dataloader))
    X_seq_sample = X_seq_sample.to(device)
    X_fixed_sample = X_fixed_sample.to(device) if X_fixed_sample is not None else None
    
    with torch.no_grad():
        # 前向传播以获取注意力权重
        _ = model(X_seq_sample, X_fixed_sample)
        
        # 为每一层创建可视化
        for layer_idx in range(num_layers):
            # 获取该层的注意力权重 [batch_size, num_heads, seq_len, seq_len]
            layer_weights = model.get_attention_weights(layer_idx=layer_idx, average_heads=False)
            
            if layer_weights is not None:
                # 取第一个样本的注意力权重 [num_heads, seq_len, seq_len]
                sample_weights = layer_weights[0].cpu().numpy()
                
                # 计算所有头的平均注意力权重 [seq_len, seq_len]
                avg_weights = sample_weights.mean(axis=0)
                
                # 绘制平均注意力权重热力图
                plt.figure(figsize=(10, 8))
                sns.heatmap(avg_weights, cmap='viridis', 
                            xticklabels=range(avg_weights.shape[1]),
                            yticklabels=range(avg_weights.shape[0]),
                            vmin=0, vmax=1)
                plt.title(f'Layer {layer_idx+1} - Average Attention Weights')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                
                # 保存图像
                save_dir = 'visualizations/attention_layers'
                os.makedirs(save_dir, exist_ok=True)
                save_path = f'{save_dir}/layer_{layer_idx+1}_avg_attention.png'
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"第 {layer_idx+1} 层平均注意力权重热力图已保存到 {save_path}")
                
                # 可视化每个注意力头的权重
                num_heads = min(sample_weights.shape[0], 4)  # 最多显示4个头
                if num_heads > 1:
                    plt.figure(figsize=(15, 10))
                    for h in range(num_heads):
                        plt.subplot(2, 2, h+1)
                        sns.heatmap(sample_weights[h], 
                                   cmap='viridis',
                                   vmin=0, vmax=1,
                                   cbar=True if h == 0 else False)
                        plt.title(f'Head {h+1} Attention Weights')
                        plt.xlabel('Key Position')
                        plt.ylabel('Query Position')
                    
                    plt.tight_layout()
                    heads_save_path = f'{save_dir}/layer_{layer_idx+1}_attention_heads.png'
                    plt.savefig(heads_save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    logger.info(f"第 {layer_idx+1} 层各注意力头权重热力图已保存到 {heads_save_path}")

def plot_gradcam(model, dataloader, device):
    """可视化Grad-CAM（每个样本单独输出）"""
    model.eval()
    
    try:
        save_dir = 'visualizations/gradcam'
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        for X_seq_batch, X_fixed_batch, _ in dataloader:
            X_seq_batch = X_seq_batch.to(device).requires_grad_(True)
            X_fixed_batch = X_fixed_batch.to(device) if X_fixed_batch is not None else None
            batch_size = X_seq_batch.shape[0]
            for i in range(batch_size):
                model.zero_grad()
                X_seq_sample = X_seq_batch[i:i+1]
                X_fixed_sample = X_fixed_batch[i:i+1] if X_fixed_batch is not None else None
                output = model(X_seq_sample, X_fixed_sample, register_hook=True)
                output = output.sum()
                output.backward()
                gradcam = model.get_gradcam()
                if gradcam is not None:
                    gradcam_sample = gradcam[0].detach().cpu().numpy()
                    plt.figure(figsize=(12, 5))
                    plt.subplot(2, 1, 1)
                    seq_data = X_seq_sample[0].detach().cpu().numpy()
                    plt.plot(seq_data[:, 0], 'b-', label='Input Sequence (1st feature)')
                    plt.title(f'Sample {idx} Input Sequence (First Feature)')
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.grid(True)
                    plt.subplot(2, 1, 2)
                    plt.bar(range(len(gradcam_sample)), gradcam_sample, 
                            color='r', alpha=0.5, label='Importance Score')
                    plt.title(f'Sample {idx} Grad-CAM Importance Scores')
                    plt.xlabel('Time Step')
                    plt.ylabel('Score')
                    plt.grid(True)
                    plt.tight_layout()
                    save_path = f'{save_dir}/gradcam_seq{idx}.png'
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    logger.info(f"Grad-CAM分析图已保存到 {save_path}")
                idx += 1
    except Exception as e:
        logger.error(f"Grad-CAM可视化时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def plot_training_curves(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    save_path = 'visualizations/training_curves.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"训练和验证损失曲线已保存到 {save_path}")

def plot_predictions(y_true, y_pred):
    """绘制真实值与预测值对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.grid(True)
    
    # 保存图像
    save_path = 'visualizations/true_vs_predicted.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"真实值与预测值对比图已保存到 {save_path}")

def visualize_model(model, dataloader, device):
    """执行所有可视化"""
    logger.info("开始模型可视化...")
    
    # 确保可视化目录存在
    os.makedirs('visualizations', exist_ok=True)
    
    # 可视化注意力权重
    plot_attention_weights(model, dataloader, device)
    
    # 可视化Grad-CAM
    plot_gradcam(model, dataloader, device)
    
    logger.info("模型可视化完成")
