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
import textwrap
from sklearn.metrics import average_precision_score
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


def _smoothness_score(window: np.ndarray, second_weight: float = 1.0) -> float:
    if window.shape[0] < 2:
        return 0.0
    first_diff = np.diff(window, axis=0)
    first_score = float(np.mean(np.linalg.norm(first_diff, axis=1)))
    if window.shape[0] < 3:
        return first_score
    second_diff = np.diff(window, n=2, axis=0)
    second_score = float(np.mean(np.linalg.norm(second_diff, axis=1)))
    return first_score + second_weight * second_score


def build_smooth_route(
    route: np.ndarray,
    length: int,
    stride: int,
    keep_percent: float,
    second_weight: float = 1.0,
    select_stride: int | None = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if keep_percent >= 100.0:
        return route, {
            'selection_windows': 0,
            'total_windows': 0,
        }
    if keep_percent <= 0.0 or length <= 0:
        return route[:0], {
            'selection_windows': 0,
            'total_windows': 0,
        }
    step = select_stride if select_stride is not None else length
    windows = []
    scores = []
    for start in range(0, len(route) - length + 1, step):
        window = route[start:start + length]
        windows.append(window)
        scores.append(_smoothness_score(window, second_weight=second_weight))
    if not windows:
        return route[:0], {
            'selection_windows': 0,
            'total_windows': 0,
        }
    keep_count = max(1, int(len(windows) * keep_percent / 100.0))
    ranked_idx = np.argsort(scores)[:keep_count]
    ranked_idx = np.sort(ranked_idx)
    selected = [windows[i] for i in ranked_idx]
    return (
        np.concatenate(selected, axis=0) if selected else route[:0],
        {
            'selection_windows': int(len(selected)),
            'total_windows': int(len(windows)),
        },
    )


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


def compute_prediction_error_scores(
    model: LSTMPredictor,
    segments: Iterable[MotherSegment],
    device: torch.device | None = None,
) -> List[float]:
    preds = _predict_segments(model, segments, device=device)
    scores = []
    for pred, seg in zip(preds, segments):
        diffs = pred - seg.post
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


def summarize_window_coverage(route_len: int, segments: List[MotherSegment]) -> Dict[str, object]:
    if route_len <= 0:
        return {
            'total_points': 0,
            'total_windows': 0,
            'min_cover': 0,
            'max_cover': 0,
            'mean_cover': 0.0,
            'cover_histogram': {},
        }
    coverage = np.zeros(route_len, dtype=np.int32)
    for seg in segments:
        coverage[seg.start:seg.end] += 1
    unique, counts = np.unique(coverage, return_counts=True)
    histogram = {int(k): int(v) for k, v in zip(unique, counts)}
    return {
        'total_points': int(route_len),
        'total_windows': int(len(segments)),
        'min_cover': int(coverage.min()),
        'max_cover': int(coverage.max()),
        'mean_cover': float(coverage.mean()),
        'cover_histogram': histogram,
    }


def threshold_from_validation(scores: np.ndarray, percentile: float = 99.5) -> float:
    valid = scores[np.isfinite(scores)]
    return float(np.percentile(valid, percentile)) if len(valid) > 0 else 0.0


def summarize_scores(
    scores: List[float],
    quantiles: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'quantiles': {},
        }
    quantile_values = np.quantile(values, list(quantiles))
    quantile_stats = {
        f"q{int(q * 100):02d}": float(v) for q, v in zip(quantiles, quantile_values)
    }
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'quantiles': quantile_stats,
    }


def plot_score_histogram(scores: List[float], title: str = 'Score distribution') -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=30, color='steelblue', alpha=0.8)
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def format_baseline_summary(results: Dict[str, object]) -> str:
    train_stats = results.get('train_score_stats', {})
    val_stats = results.get('val_score_stats', {})
    quantiles = val_stats.get('quantiles', {})
    percentile_results = results.get('percentile_results', [])
    train_thresholds = results.get('train_percentile_thresholds', [])
    best = results.get('best_percentile_result', {})
    train_threshold_metrics = results.get('train_threshold_metrics', {})
    route_stats = results.get('route_stats', {})
    window_params = results.get('window_params', {})
    smooth_stats = results.get('smooth_stats', {})

    lines = [
        "Prediction-baseline summary",
        "-" * 30,
        f"F1: {results.get('f1')}",
        f"Precision: {results.get('precision')}",
        f"Recall: {results.get('recall')}",
        f"PR_AUC: {results.get('pr_auc')}",
        f"Threshold: {results.get('threshold')}",
    ]

    if train_stats:
        lines.extend([
            "",
            "Train score stats:",
            f"  mean: {train_stats.get('mean')}",
            f"  std: {train_stats.get('std')}",
        ])
        train_quantiles = train_stats.get('quantiles', {})
        if train_quantiles:
            lines.append("  quantiles:")
            for key in sorted(train_quantiles.keys()):
                lines.append(f"    {key}: {train_quantiles[key]}")

    if val_stats:
        lines.extend([
            "",
            "Validation score stats:",
            f"  mean: {val_stats.get('mean')}",
            f"  std: {val_stats.get('std')}",
        ])
        if quantiles:
            lines.append("  quantiles:")
            for key in sorted(quantiles.keys()):
                lines.append(f"    {key}: {quantiles[key]}")

    if route_stats:
        lines.extend([
            "",
            "Route/window stats:",
        ])
        if window_params:
            lines.extend([
                f"  window_length: {window_params.get('length')}",
                f"  window_stride: {window_params.get('stride')}",
            ])
        if smooth_stats:
            lines.extend([
                "  smooth_selection:",
                f"    keep_percent: {smooth_stats.get('keep_percent')}",
                f"    second_weight: {smooth_stats.get('second_weight')}",
                f"    select_stride: {smooth_stats.get('select_stride')}",
                f"    selection_windows: {smooth_stats.get('selection_windows')}",
                f"    total_windows: {smooth_stats.get('total_windows')}",
            ])
        for name in ('train', 'val', 'test'):
            stats = route_stats.get(name, {})
            if not stats:
                continue
            lines.extend([
                f"  {name}:",
                f"    total_points: {stats.get('total_points')}",
                f"    total_windows: {stats.get('total_windows')}",
                f"    min_cover: {stats.get('min_cover')}",
                f"    max_cover: {stats.get('max_cover')}",
                f"    mean_cover: {stats.get('mean_cover')}",
            ])

    if best:
        lines.extend([
            "",
            "Best percentile result:",
            f"  percentile: {best.get('percentile')}",
            f"  threshold: {best.get('threshold')}",
            f"  f1: {best.get('f1')}",
            f"  precision: {best.get('precision')}",
            f"  recall: {best.get('recall')}",
        ])

    if train_threshold_metrics:
        lines.extend([
            "",
            "Train-threshold metrics (test set):",
            f"  threshold: {results.get('train_threshold')}",
            f"  precision: {train_threshold_metrics.get('precision')}",
            f"  recall: {train_threshold_metrics.get('recall')}",
            f"  f1: {train_threshold_metrics.get('f1')}",
            f"  pr_auc: {train_threshold_metrics.get('pr_auc')}",
        ])

    if percentile_results:
        lines.append("")
        lines.append("Percentile sweep:")
        header = f"{'pct':>6}  {'thr':>8}  {'prec':>8}  {'rec':>8}  {'f1':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in percentile_results:
            lines.append(
                f"{row.get('percentile', 0):>6.1f}  "
                f"{row.get('threshold', 0):>8.3f}  "
                f"{row.get('precision', 0):>8.3f}  "
                f"{row.get('recall', 0):>8.3f}  "
                f"{row.get('f1', 0):>8.3f}"
            )

    if train_thresholds:
        lines.append("")
        lines.append("Train percentile thresholds:")
        header = f"{'pct':>6}  {'thr':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in train_thresholds:
            lines.append(
                f"{row.get('percentile', 0):>6.1f}  "
                f"{row.get('threshold', 0):>8.3f}"
            )

    return textwrap.dedent("\n".join(lines)).strip()


def print_baseline_summary(results: Dict[str, object]) -> None:
    print(format_baseline_summary(results))


def evaluate_pointwise(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int32)
    valid_mask = np.isfinite(score_array)
    score_array = score_array[valid_mask]
    label_array = label_array[valid_mask]
    if score_array.size == 0 or label_array.size == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pr_auc': 0.0,
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
        }

    preds = score_array > threshold
    pr_auc = average_precision_score(label_array, score_array)

    tp = int(((preds == 1) & (label_array == 1)).sum())
    fp = int(((preds == 1) & (label_array == 0)).sum())
    tn = int(((preds == 0) & (label_array == 0)).sum())
    fn = int(((preds == 0) & (label_array == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'pr_auc': float(pr_auc),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def sweep_percentiles(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    percentiles: Iterable[float] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.5),
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for pct in percentiles:
        threshold = threshold_from_validation(val_scores, pct)
        metrics = evaluate_pointwise(test_scores, test_labels, threshold)
        results.append({
            'percentile': float(pct),
            'threshold': float(threshold),
            **metrics
        })
    best = max(results, key=lambda x: x['f1']) if results else {}
    return results, best


def summarize_thresholds(
    scores: np.ndarray,
    percentiles: Iterable[float],
) -> List[Dict[str, float]]:
    summary = []
    for pct in percentiles:
        threshold = threshold_from_validation(scores, pct)
        summary.append({
            'percentile': float(pct),
            'threshold': float(threshold),
        })
    return summary


def run_prediction_baseline(
    data_dir: str = Config.PROCESSED_DATA_DIR,
    length: int = 128,
    stride: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 99.5,
    smooth_keep_percent: float = 100.0,
    smooth_second_weight: float = 1.0,
    smooth_select_stride: int | None = None,
) -> Dict[str, object]:
    percentiles = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.5)
    train_route, _ = load_route(f"{data_dir}/train.csv")
    val_route, _ = load_route(f"{data_dir}/val.csv")
    test_route, test_labels = load_route(f"{data_dir}/test.csv")

    smooth_train_route, smooth_stats = build_smooth_route(
        train_route,
        length,
        stride,
        keep_percent=smooth_keep_percent,
        second_weight=smooth_second_weight,
        select_stride=smooth_select_stride,
    )
    train_segments = segment_route(smooth_train_route, length, stride)
    train_score_segments = segment_route(train_route, length, stride)
    model = LSTMPredictor(train_route.shape[1])
    train_history = train_predictor(
        model,
        train_segments,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    reference_route = train_route
    base_segments, base_preds = generate_predicted_route(
        model, reference_route, length, stride
    )
    train_segments, train_preds = generate_predicted_route(
        model, train_route, length, stride
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
    _, base_train_preds, train_segments, train_preds = _align_by_macro(
        base_segments, base_preds, train_segments, train_preds
    )
    _, base_test_preds, test_segments, test_preds = _align_by_macro(
        base_segments, base_preds, test_segments, test_preds
    )

    train_scores = compute_prediction_error_scores(model, train_score_segments)
    val_scores = compute_disagreement_score(base_val_preds, val_preds)
    test_scores = compute_disagreement_score(base_test_preds, test_preds)
    train_score_stats = summarize_scores(train_scores)
    val_score_stats = summarize_scores(val_scores)
    route_stats = {
        'train': summarize_window_coverage(len(train_route), train_segments),
        'val': summarize_window_coverage(len(val_route), val_segments),
        'test': summarize_window_coverage(len(test_route), test_segments),
    }
    window_params = {
        'length': int(length),
        'stride': int(stride),
    }
    smooth_stats = {
        **smooth_stats,
        'keep_percent': float(smooth_keep_percent),
        'second_weight': float(smooth_second_weight),
        'select_stride': int(smooth_select_stride) if smooth_select_stride is not None else None,
    }

    train_per_time_scores = _assign_scores_to_timepoints(
        len(train_route),
        train_score_segments,
        train_scores,
    )
    val_per_time_scores = _assign_scores_to_timepoints(len(val_route), val_segments, val_scores)
    test_per_time_scores = _assign_scores_to_timepoints(len(test_route), test_segments, test_scores)

    threshold = threshold_from_validation(val_per_time_scores, threshold_percentile)
    train_threshold = threshold_from_validation(train_per_time_scores, threshold_percentile)
    metrics = evaluate_pointwise(test_per_time_scores, test_labels, threshold)
    train_threshold_metrics = evaluate_pointwise(test_per_time_scores, test_labels, train_threshold)
    percentile_results, best_percentile_result = sweep_percentiles(
        val_per_time_scores,
        test_per_time_scores,
        test_labels,
        percentiles=percentiles,
    )
    train_thresholds = summarize_thresholds(train_per_time_scores, percentiles)

    metrics['threshold'] = float(threshold)
    metrics['train_threshold'] = float(train_threshold)
    metrics['train_threshold_metrics'] = train_threshold_metrics
    metrics['percentile_results'] = percentile_results
    metrics['best_percentile_result'] = best_percentile_result
    metrics['train_percentile_thresholds'] = train_thresholds
    metrics['per_time_scores'] = test_per_time_scores
    metrics['train_history'] = train_history
    metrics['train_score_stats'] = train_score_stats
    metrics['val_score_stats'] = val_score_stats
    metrics['route_stats'] = route_stats
    metrics['window_params'] = window_params
    metrics['smooth_stats'] = smooth_stats
    return metrics
