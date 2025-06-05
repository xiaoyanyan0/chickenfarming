import matplotlib.pyplot as plt
import numpy as np
import torch
import shap
from matplotlib import cm

def plot_loss_curve(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training/Validation Loss Curve')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_true_vs_pred(y_true, y_pred, save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(y_true, label='True', marker='o')
    plt.plot(y_pred, label='Pred', marker='x')
    plt.xlabel('Sample')
    plt.ylabel('Target')
    plt.legend()
    plt.title('True vs Predicted')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_attention_weights(attentions, seq_labels=None, save_path=None):
    # attentions: (num_layers, num_heads, seq_len, seq_len)
    num_layers, num_heads, seq_len, _ = attentions.shape
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(3*num_heads, 3*num_layers))
    for l in range(num_layers):
        for h in range(num_heads):
            ax = axes[l, h] if num_layers > 1 else axes[h]
            im = ax.imshow(attentions[l, h], cmap=cm.viridis)
            ax.set_title(f'Layer {l+1} Head {h+1}')
            if seq_labels:
                ax.set_xticks(range(seq_len))
                ax.set_yticks(range(seq_len))
                ax.set_xticklabels(seq_labels, rotation=90)
                ax.set_yticklabels(seq_labels)
            fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_summary(model, X, feature_names, device='cpu', save_path=None):
    # X: (N, 25, 1) -> (N, 25) for SHAP
    X_flat = X.squeeze(-1)
    explainer = shap.DeepExplainer(model, torch.tensor(X_flat, dtype=torch.float32).to(device))
    shap_values = explainer.shap_values(torch.tensor(X_flat, dtype=torch.float32).to(device))
    plt.figure(figsize=(10,5))
    shap.summary_plot(shap_values, X_flat, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_force(model, X, idx, feature_names, device='cpu', save_path=None):
    X_flat = X.squeeze(-1)
    explainer = shap.DeepExplainer(model, torch.tensor(X_flat, dtype=torch.float32).to(device))
    shap_values = explainer.shap_values(torch.tensor(X_flat, dtype=torch.float32).to(device))
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][idx], X_flat[idx], feature_names=feature_names, matplotlib=True, show=False)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_fixed_and_sequence(model, X_seq, X_fixed, seq_feature_names, fixed_feature_names, device='cpu', save_dir=None):
    """
    分别计算固定特征和每个序列特征的SHAP贡献，保存summary图。
    X_seq: (N, seq_len, seq_dim)
    X_fixed: (N, fixed_dim)
    """
    import os
    import torch
    import numpy as np
    import shap
    import matplotlib.pyplot as plt
    os.makedirs(save_dir or 'visualizations/shap', exist_ok=True)
    model.eval()
    N = X_fixed.shape[0] if X_fixed is not None else X_seq.shape[0]
    seq_len, seq_dim = X_seq.shape[1:3] if X_seq is not None else (None, None)
    fixed_dim = X_fixed.shape[1] if X_fixed is not None else None

    import logging
    logger = logging.getLogger(__name__)
    # 固定特征 SHAP
    if X_fixed is not None and X_seq is not None:
        seq_zeros = np.zeros((N, seq_len, seq_dim), dtype=np.float32)
        try:
            background = [torch.tensor(seq_zeros, dtype=torch.float32).to(device), torch.tensor(X_fixed, dtype=torch.float32).to(device)]
            explainer_fixed = shap.DeepExplainer(model, background)
            shap_values = explainer_fixed.shap_values([torch.tensor(seq_zeros, dtype=torch.float32).to(device), torch.tensor(X_fixed, dtype=torch.float32).to(device)])
            plt.figure(figsize=(10,5))
            shap.summary_plot(shap_values[1], X_fixed, feature_names=fixed_feature_names, show=False)
            plt.title('Fixed Feature SHAP Summary')
            plt.savefig(os.path.join(save_dir or 'visualizations/shap', 'shap_fixed_summary.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"DeepExplainer failed for fixed features, switching to KernelExplainer: {e}")
            # KernelExplainer for fixed features
            def model_fixed(X):
                X = np.array(X).astype(np.float32)
                seq_zeros_kernel = np.zeros((X.shape[0], seq_len, seq_dim), dtype=np.float32)
                X_seq_tensor = torch.tensor(seq_zeros_kernel, dtype=torch.float32).to(device)
                X_fixed_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                with torch.no_grad():
                    return model(X_seq_tensor, X_fixed_tensor).cpu().numpy().reshape(-1)
            X_fixed_np = X_fixed[:20]
            if isinstance(X_fixed_np, torch.Tensor):
                X_fixed_np = X_fixed_np.cpu().numpy()
            X_fixed_np = np.array(X_fixed_np).astype(np.float32)
            explainer_fixed = shap.KernelExplainer(model_fixed, X_fixed_np)
            X_fixed_test = X_fixed[:5]
            if isinstance(X_fixed_test, torch.Tensor):
                X_fixed_test = X_fixed_test.cpu().numpy()
            shap_values = explainer_fixed.shap_values(np.array(X_fixed_test).astype(np.float32))
            plt.figure(figsize=(10,5))
            shap.summary_plot(shap_values, np.array(X_fixed_test), feature_names=fixed_feature_names, show=False)
            plt.title('Fixed Feature SHAP Summary (KernelExplainer)')
            plt.savefig(os.path.join(save_dir or 'visualizations/shap', 'shap_fixed_summary_kernel.png'))
            plt.close()

    # 序列特征 SHAP
    if X_seq is not None and X_fixed is not None:
        fixed_zeros = np.zeros((N, fixed_dim), dtype=np.float32)
        try:
            background = [torch.tensor(X_seq, dtype=torch.float32).to(device), torch.tensor(fixed_zeros, dtype=torch.float32).to(device)]
            explainer_seq = shap.DeepExplainer(model, background)
            shap_values = explainer_seq.shap_values([torch.tensor(X_seq, dtype=torch.float32).to(device), torch.tensor(fixed_zeros, dtype=torch.float32).to(device)])
            X_seq_reshaped = X_seq.reshape(N, -1)
            seq_names = [f"{name}_t{t+1}" for t in range(seq_len) for name in seq_feature_names]
            plt.figure(figsize=(12,6))
            shap.summary_plot(shap_values[0], X_seq_reshaped, feature_names=seq_names, show=False)
            plt.title('Sequence Feature SHAP Summary')
            plt.savefig(os.path.join(save_dir or 'visualizations/shap', 'shap_sequence_summary.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"DeepExplainer failed for sequence features, switching to KernelExplainer: {e}")
            # KernelExplainer for sequence features
            def model_seq(X):
                X = np.array(X).astype(np.float32)
                X = X.reshape(-1, seq_len, seq_dim)
                X_seq_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                fixed_zeros_kernel = np.zeros((X.shape[0], fixed_dim), dtype=np.float32)
                X_fixed_tensor = torch.tensor(fixed_zeros_kernel, dtype=torch.float32).to(device)
                with torch.no_grad():
                    return model(X_seq_tensor, X_fixed_tensor).cpu().numpy().reshape(-1)
            X_seq_reshaped = X_seq.reshape(N, -1)
            if isinstance(X_seq_reshaped, torch.Tensor):
                X_seq_reshaped = X_seq_reshaped.cpu().numpy()
            X_seq_reshaped = np.array(X_seq_reshaped).astype(np.float32)
            seq_names = [f"{name}_t{t+1}" for t in range(seq_len) for name in seq_feature_names]
            explainer_seq = shap.KernelExplainer(model_seq, X_seq_reshaped[:20])
            X_seq_test = X_seq_reshaped[:5]
            shap_values = explainer_seq.shap_values(X_seq_test)
            plt.figure(figsize=(12,6))
            shap.summary_plot(shap_values, X_seq_test, feature_names=seq_names, show=False)
            plt.title('Sequence Feature SHAP Summary (KernelExplainer)')
            plt.savefig(os.path.join(save_dir or 'visualizations/shap', 'shap_sequence_summary_kernel.png'))
            plt.close()

def plot_lime_explanation(model, X, y, idx, feature_names, save_path=None):
    from lime.lime_tabular import LimeTabularExplainer
    X_flat = X.squeeze(-1)
    explainer = LimeTabularExplainer(X_flat, feature_names=feature_names, class_names=['target'], discretize_continuous=True)
    exp = explainer.explain_instance(X_flat[idx], model.predict, num_features=10)
    fig = exp.as_pyplot_figure()
    if save_path:
        plt.savefig(save_path)
    plt.show()
