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
