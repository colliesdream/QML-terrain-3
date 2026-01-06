"""
Utility helpers for VQTransAE.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def to_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def save_json(data, path: Path, indent: int = 2):
    """Persist JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(data), f, indent=indent, ensure_ascii=False)


def speed_bins(speeds: np.ndarray, bounds: Tuple = (3.0, 6.0, 9.0)) -> np.ndarray:
    """Bucket speeds into ranges for stratified scaling."""
    if speeds is None or len(speeds) == 0:
        return np.zeros(0, dtype=int)
    return np.digitize(speeds, bounds, right=False)


def robust_scale_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Median-MAD scaling per group to remove speed-dependent baselines."""
    if values.size == 0:
        return values

    scaled = np.zeros_like(values, dtype=np.float64)
    unique_groups = np.unique(groups) if groups.size else np.array([0])

    for gid in unique_groups:
        mask = groups == gid if groups.size else np.ones_like(values, dtype=bool)
        if not np.any(mask):
            continue
        subset = values[mask]
        median = np.median(subset)
        mad = np.median(np.abs(subset - median)) + 1e-6  # avoid divide-by-zero
        scaled[mask] = (subset - median) / mad

    return scaled


def token_stats(counts: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Compute active token count, perplexity, and usage probabilities."""
    total = counts.sum()
    if total == 0:
        return 0, 0.0, np.zeros_like(counts, dtype=np.float64)

    probs = counts.astype(np.float64) / total
    non_zero = probs > 0
    entropy = -(probs[non_zero] * np.log(probs[non_zero])).sum()
    perplexity = float(np.exp(entropy))
    active = int((counts > 0).sum())

    return active, perplexity, probs
