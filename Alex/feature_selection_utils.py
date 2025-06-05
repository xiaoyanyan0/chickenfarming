import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from copy import deepcopy

class ChickenDataset(Dataset):
    def __init__(self, X_seq, X_fixed, y):
        self.X_seq = X_seq
        self.X_fixed = X_fixed
        self.y = y
    def __len__(self):
        return len(self.X_seq)
    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_fixed[idx], self.y[idx]

def quick_train_eval(model_class, train_X_seq, val_X_seq, train_X_fixed, val_X_fixed, train_y, val_y, input_dim, fixed_dim, device, epochs=20, batch_size=64):
    train_X_seq = torch.tensor(train_X_seq, dtype=torch.float32).to(device)
    val_X_seq = torch.tensor(val_X_seq, dtype=torch.float32).to(device)
    train_X_fixed = torch.tensor(train_X_fixed, dtype=torch.float32).to(device)
    val_X_fixed = torch.tensor(val_X_fixed, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).to(device)
    train_dataset = ChickenDataset(train_X_seq, train_X_fixed, train_y)
    val_dataset = ChickenDataset(val_X_seq, val_X_fixed, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = model_class(input_dim=input_dim, fixed_dim=fixed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    best_r2 = -np.inf
    for _ in range(epochs):
        model.train()
        for X_seq, X_fixed, y in train_loader:
            optimizer.zero_grad()
            output = model(X_seq, X_fixed)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for X_seq, X_fixed, y in val_loader:
                output = model(X_seq, X_fixed)
                preds.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        ss_res = np.sum((preds - targets) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best_r2:
            best_r2 = r2
    return best_r2

from sklearn.preprocessing import OneHotEncoder

def feature_selection_search(seq_candidates, train_df, val_df, fixed_vars, target_col, model_class, fixed_dim, device):
    results = []
    # 对 fixed_vars 做 OneHot 编码，保证全为 float32
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_X_fixed = enc.fit_transform(train_df[fixed_vars])
    val_X_fixed = enc.transform(val_df[fixed_vars])
    train_X_fixed = train_X_fixed.astype(np.float32)
    val_X_fixed = val_X_fixed.astype(np.float32)
    train_y = train_df[[target_col]].values.astype(np.float32)
    val_y = val_df[[target_col]].values.astype(np.float32)
    for seq_group in seq_candidates:
        train_seqs = []
        val_seqs = []
        for cols in seq_group:
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_df[cols].values)
            val_scaled = scaler.transform(val_df[cols].values)
            train_seqs.append(train_scaled)
            val_seqs.append(val_scaled)
        train_X_seq = np.stack(train_seqs, axis=-1)  # (N, 25, n_group)
        val_X_seq = np.stack(val_seqs, axis=-1)
        input_dim = len(seq_group)
        r2 = quick_train_eval(model_class, train_X_seq, val_X_seq, train_X_fixed, val_X_fixed, train_y, val_y, input_dim, train_X_fixed.shape[1], device)
        results.append((seq_group, r2))
    return results
