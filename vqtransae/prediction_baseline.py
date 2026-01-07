"""
Prediction-reference anomaly detection baseline.

Train a single predictor on clean data, generate predicted routes for a clean
reference and the test route, then score anomalies by disagreement between the
two predicted routes (prediction vs prediction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import Config


def _resolve_device(device: torch.device | None = None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@dataclass
class MotherSegment:
    macro_id: int
    start: int
    end: int
    pre: np.ndarray
    post: np.ndarray


def load_route(csv_path: str, feature_columns: List[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if feature_columns is None:
        feature_columns = Config.FEATURE_COLUMNS
    df = pd.read_csv(csv_path)
    route = df[feature_columns].values.astype(np.float32)
    labels = df['anomaly'].values.astype(np.int32) if 'anomaly' in df.columns else np.zeros(len(df), dtype=np.int32)
    return route, labels


def segment_route(route: np.ndarray, length: int, stride: int) -> List[MotherSegment]:
    """Split route into 3 macro segments, then sliding mother segments."""
    total_len = len(route)
    macro_len = total_len // 3
    segments: List[MotherSegment] = []

    for macro_id in range(3):
        macro_start = macro_id * macro_len
        macro_end = (macro_id + 1) * macro_len if macro_id < 2 else total_len
        if macro_end - macro_start < length:
            continue
        for start in range(macro_start, macro_end - length + 1, stride):
            end = start + length
            mid = start + length // 2
            pre = route[start:mid]
            post = route[mid:end]
            segments.append(MotherSegment(macro_id, start, end, pre, post))
    return segments


class MotherSegmentDataset(Dataset):
    def __init__(self, segments: Iterable[MotherSegment]):
        self.segments = list(segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = self.segments[idx]
        pre = torch.from_numpy(seg.pre)
        post = torch.from_numpy(seg.post)
        return pre, post


class LSTMPredictor(nn.Module):
    """Simple sequence-to-sequence LSTM predictor: X_pre -> X_post_hat."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out)


def build_dataset(segments: Iterable[MotherSegment]) -> MotherSegmentDataset:
    return MotherSegmentDataset(segments)


def train_predictor(
    model: LSTMPredictor,
    train_segments: Iterable[MotherSegment],
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
) -> Dict[str, List[float]]:
    dataset = build_dataset(train_segments)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    device = _resolve_device(device)
    model.to(device)

    history: Dict[str, List[float]] = {'loss': []}
    for _ in range(epochs):
        model.train()
        losses = []
        for pre, post in loader:
            pre = pre.to(device)
            post = post.to(device)
            pred = model(pre)
            loss = criterion(pred, post)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history['loss'].append(float(np.mean(losses)) if losses else 0.0)
    return history


def _predict_segments(
    model: LSTMPredictor,
    segments: Iterable[MotherSegment],
    device: torch.device | None = None,
    batch_size: int = 128,
) -> List[np.ndarray]:
    dataset = build_dataset(segments)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    device = _resolve_device(device)
    model.to(device)
    model.eval()

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for pre, _ in loader:
            pre = pre.to(device)
            pred = model(pre).cpu().numpy()
            outputs.extend(list(pred))
    return outputs


def generate_predicted_route(
    model: LSTMPredictor,
    route: np.ndarray,
    length: int,
    stride: int,
    device: torch.device | None = None,
) -> Tuple[List[MotherSegment], List[np.ndarray]]:
    segments = segment_route(route, length, stride)
    preds = _predict_segments(model, segments, device=device)
    return segments, preds


def compute_disagreement_score(
    base_preds: List[np.ndarray],
    test_preds: List[np.ndarray],
) -> List[float]:
    scores = []
    for base, test in zip(base_preds, test_preds):
        diffs = base - test
        l2 = np.linalg.norm(diffs, axis=1)
        scores.append(float(l2.max()))
    return scores


def _align_by_macro(
    base_segments: List[MotherSegment],
    base_preds: List[np.ndarray],
    test_segments: List[MotherSegment],
    test_preds: List[np.ndarray],
) -> Tuple[List[MotherSegment], List[np.ndarray], List[MotherSegment], List[np.ndarray]]:
    aligned_base_segments: List[MotherSegment] = []
    aligned_base_preds: List[np.ndarray] = []
    aligned_test_segments: List[MotherSegment] = []
    aligned_test_preds: List[np.ndarray] = []

    for macro_id in range(3):
        base_idx = [i for i, seg in enumerate(base_segments) if seg.macro_id == macro_id]
        test_idx = [i for i, seg in enumerate(test_segments) if seg.macro_id == macro_id]
        count = min(len(base_idx), len(test_idx))
        for i in range(count):
            b_i = base_idx[i]
            t_i = test_idx[i]
            aligned_base_segments.append(base_segments[b_i])
            aligned_base_preds.append(base_preds[b_i])
            aligned_test_segments.append(test_segments[t_i])
            aligned_test_preds.append(test_preds[t_i])

    return aligned_base_segments, aligned_base_preds, aligned_test_segments, aligned_test_preds


def _assign_scores_to_timepoints(
    route_len: int,
    segments: List[MotherSegment],
    scores: List[float],
) -> np.ndarray:
    per_time = np.full(route_len, -np.inf, dtype=np.float32)
    for seg, score in zip(segments, scores):
        per_time[seg.start:seg.end] = np.maximum(per_time[seg.start:seg.end], score)
    per_time[np.isneginf(per_time)] = 0.0
    return per_time


def threshold_from_validation(scores: np.ndarray, percentile: float = 99.5) -> float:
    valid = scores[np.isfinite(scores)]
    return float(np.percentile(valid, percentile)) if len(valid) > 0 else 0.0


def evaluate_test(
    scores: List[float],
    test_segments: List[MotherSegment],
    test_labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = np.array(scores) > threshold
    seg_labels = np.array([
        int(test_labels[seg.start:seg.end].max()) for seg in test_segments
    ])

    tp = int(((preds == 1) & (seg_labels == 1)).sum())
    fp = int(((preds == 1) & (seg_labels == 0)).sum())
    tn = int(((preds == 0) & (seg_labels == 0)).sum())
    fn = int(((preds == 0) & (seg_labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def run_prediction_baseline(
    data_dir: str = Config.PROCESSED_DATA_DIR,
    length: int = 128,
    stride: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 99.5,
) -> Dict[str, float]:
    train_route, _ = load_route(f"{data_dir}/train.csv")
    val_route, _ = load_route(f"{data_dir}/val.csv")
    test_route, test_labels = load_route(f"{data_dir}/test.csv")

    train_segments = segment_route(train_route, length, stride)
    model = LSTMPredictor(train_route.shape[1])
    train_predictor(
        model,
        train_segments,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    reference_route = np.concatenate([train_route, val_route], axis=0)
    base_segments, base_preds = generate_predicted_route(
        model, reference_route, length, stride
    )
    val_segments, val_preds = generate_predicted_route(
        model, val_route, length, stride
    )
    test_segments, test_preds = generate_predicted_route(
        model, test_route, length, stride
    )

    _, base_val_preds, val_segments, val_preds = _align_by_macro(
        base_segments, base_preds, val_segments, val_preds
    )
    _, base_test_preds, test_segments, test_preds = _align_by_macro(
        base_segments, base_preds, test_segments, test_preds
    )

    val_scores = compute_disagreement_score(base_val_preds, val_preds)
    test_scores = compute_disagreement_score(base_test_preds, test_preds)

    threshold = threshold_from_validation(np.array(val_scores), threshold_percentile)
    metrics = evaluate_test(test_scores, test_segments, test_labels, threshold)

    per_time_scores = _assign_scores_to_timepoints(len(test_route), test_segments, test_scores)
    metrics['threshold'] = float(threshold)
    metrics['per_time_scores'] = per_time_scores
    return metrics
