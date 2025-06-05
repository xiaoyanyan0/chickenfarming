import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SelfAttentionWrapper(nn.Module):
    """包装自注意力层以捕获注意力权重"""
    def __init__(self, self_attn, nhead, d_model, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.nhead = nhead
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, is_causal=False):
        # 调用原始的自注意力计算
        attn_output, attn_weights = self.self_attn(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal
        )
        self.attn_weights = attn_weights
        return attn_output, attn_weights

class TransformerWithPostFusion(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, 
                 dim_feedforward=512, dropout=0.1, num_variety=1, num_supervisor=1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 序列特征处理
        self.seq_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=26)
        
        # 保存注意力权重
        self.attention_weights = None
        
        # 自定义的Transformer编码器层
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            
            # 包装自注意力层以获取注意力权重
            encoder_layer.self_attn = SelfAttentionWrapper(
                encoder_layer.self_attn, 
                nhead=nhead,
                d_model=d_model,
                dropout=dropout
            )
            
            encoder_layers.append(encoder_layer)
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # 固定变量处理
        self.density_embedding = nn.Linear(1, d_model // 4)
        self.variety_embedding = nn.Embedding(num_variety, d_model // 4)
        self.supervisor_embedding = nn.Embedding(num_supervisor, d_model // 4)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 梯度捕获
        self.gradients = None
        self.activations = None
        
        logger.info(f"初始化 TransformerWithPostFusion: hidden_dim={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    def forward(self, x_seq, x_fixed=None):
        # 序列特征嵌入
        x = self.seq_embedding(x_seq)
        x = self.pos_encoder(x)
        
        # 编码器前向传播
        attn_weights_all_layers = []
        for i, layer in enumerate(self.encoder):
            # 保存梯度钩子
            x.requires_grad_(True)
            x.register_hook(self._save_grad)
            
            # 前向传播
            x = layer(x)
            
            # 获取注意力权重
            if hasattr(layer.self_attn, 'attn_weights') and layer.self_attn.attn_weights is not None:
                attn_weights_all_layers.append(layer.self_attn.attn_weights.detach())
        
        # 保存激活和注意力权重
        self.activations = x
        if attn_weights_all_layers:
            self.attention_weights = torch.stack(attn_weights_all_layers)
        
        # 池化
        x = x.mean(dim=1)  # 平均池化
        
        # 输出层
        output = self.output(x)
        return output.squeeze(-1)
    
    def _save_grad(self, grad):
        self.gradients = grad
    
    def get_attention_weights(self, layer_idx=-1, head_idx=None, average_heads=True):
        """
        获取注意力权重
        
        参数:
            layer_idx: 层索引，-1表示最后一层
            head_idx: 头索引，None表示返回所有头
            average_heads: 如果为True且head_idx为None，则对所有头取平均
            
        返回:
            torch.Tensor: 注意力权重 
        """
        if self.attention_weights is None:
            return None
            
        # 确保层索引在有效范围内
        if layer_idx < 0:
            layer_idx = len(self.attention_weights) + layer_idx
        if layer_idx >= len(self.attention_weights) or layer_idx < 0:
            return None
            
        # 获取指定层的注意力权重 [batch_size, num_heads, seq_len, seq_len]
        layer_weights = self.attention_weights[layer_idx]
        
        # 如果指定了头索引，返回该头的注意力权重
        if head_idx is not None:
            if head_idx < 0 or head_idx >= layer_weights.size(1):
                return None
            return layer_weights[:, head_idx]  # [batch_size, seq_len, seq_len]
        
        # 如果不需要平均所有头，直接返回所有头的权重
        if not average_heads:
            return layer_weights  # [batch_size, num_heads, seq_len, seq_len]
            
        # 返回所有头的平均注意力权重
        return layer_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
    
    def get_gradcam(self):
        """计算Grad-CAM热力图"""
        if self.gradients is None or self.activations is None:
            return None
            
        # 获取梯度的全局平均
        alpha = self.gradients.mean(dim=(1, 2), keepdim=True)  # [batch_size, 1, 1]
        
        # 计算加权激活
        gradcam = (self.activations * alpha).sum(dim=-1)  # [batch_size, seq_len]
        
        # ReLU操作
        gradcam = F.relu(gradcam)
        
        # 归一化
        gradcam = gradcam / (gradcam.sum(dim=-1, keepdim=True) + 1e-10)
        
        return gradcam.detach()
