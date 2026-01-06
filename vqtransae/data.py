"""
Data processing for VQTransAE: preprocessing, dataset, dataloaders.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .utils import save_json


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(
    original_dir: str = Config.ORIGINAL_DATA_DIR,
    output_dir: str = Config.PROCESSED_DATA_DIR,
    val_ratio: float = Config.VAL_RATIO,
    feature_columns: List[str] = None
) -> Tuple[Path, Dict]:
    """Preprocess train/test CSVs: split val, fit scaler, standardize, save."""

    if feature_columns is None:
        feature_columns = Config.FEATURE_COLUMNS

    print("=" * 70)
    print("Step 1: Preprocess data")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = Path(original_dir) / 'train.csv'
    test_path = Path(original_dir) / 'test.csv'

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if 'anomaly' not in df_train_full.columns:
        df_train_full['anomaly'] = 0
        print("Warning: train.csv missing 'anomaly' column; filled with zeros.")

    n_total = len(df_train_full)
    n_val = int(n_total * val_ratio)
    df_train = df_train_full.iloc[:-n_val].copy()
    df_val = df_train_full.iloc[-n_val:].copy()

    scaler = StandardScaler()
    scaler.fit(df_train[feature_columns].values)

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[feature_columns] = scaler.transform(df_train[feature_columns].values)
    df_val_scaled[feature_columns] = scaler.transform(df_val[feature_columns].values)
    df_test_scaled[feature_columns] = scaler.transform(df_test[feature_columns].values)

    df_train_scaled.to_csv(output_path / 'train.csv', index=False)
    df_val_scaled.to_csv(output_path / 'val.csv', index=False)
    df_test_scaled.to_csv(output_path / 'test.csv', index=False)

    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'features': feature_columns
    }
    save_json(scaler_params, output_path / 'scaler_params.json')

    stats = {
        'original_train_size': len(df_train_full),
        'new_train_size': len(df_train),
        'new_val_size': len(df_val),
        'test_size': len(df_test),
        'val_ratio': val_ratio,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'train_anomaly_dist': df_train_scaled['anomaly'].value_counts().to_dict() if 'anomaly' in df_train_scaled.columns else {},
        'val_anomaly_dist': df_val_scaled['anomaly'].value_counts().to_dict() if 'anomaly' in df_val_scaled.columns else {},
        'test_anomaly_dist': df_test_scaled['anomaly'].value_counts().to_dict() if 'anomaly' in df_test_scaled.columns else {}
    }
    save_json(stats, output_path / 'preprocessing_stats.json')

    print(f"Preprocessing done. Saved to: {output_path}")
    return output_path, stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Sliding-window dataset for time series."""

    def __init__(self, csv_path: str, win_size: int, step: int, features: List[str] = None):
        if features is None:
            features = Config.FEATURE_COLUMNS

        df = pd.read_csv(csv_path)
        self.data = df[features].values.astype(np.float32)
        self.labels = df['anomaly'].values if 'anomaly' in df.columns else np.zeros(len(df))
        self.speeds = df['Speed'].values if 'Speed' in df.columns else np.zeros(len(df))
        self.win_size = win_size
        self.step = step
        self.n_windows = (len(self.data) - win_size) // step + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.win_size
        center = start + self.win_size // 2

        window = self.data[start:end]
        label = float(np.max(self.labels[start:end]))  # window is anomalous if any point is
        speed = float(self.speeds[center]) if not np.isnan(self.speeds[center]) else 0.0

        return torch.from_numpy(window), torch.tensor(label), torch.tensor(speed)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str,
    win_size: int = Config.WIN_SIZE,
    step: int = Config.STEP,
    batch_size: int = Config.BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from preprocessed CSVs."""

    data_path = Path(data_dir)

    train_dataset = WindowDataset(data_path / 'train.csv', win_size, step)
    val_dataset = WindowDataset(data_path / 'val.csv', win_size, step)
    test_dataset = WindowDataset(data_path / 'test.csv', win_size, step)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
