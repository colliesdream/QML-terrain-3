"""
Visualization helpers for training and evaluation outputs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .config import Config


def plot_training_curves(history: List[Dict], save_path: Path):
    """Plot training/validation losses, active tokens, perplexity, recon vs VQ."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    active_tokens = [h['active_tokens'] for h in history]
    perplexity = [h['perplexity'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train')
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, active_tokens, 'g-', linewidth=2)
    axes[0, 1].axhline(Config.ACTIVE_TOKEN_TARGET, color='r', linestyle='--', label=f'Target={Config.ACTIVE_TOKEN_TARGET}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Active Tokens')
    axes[0, 1].set_title('Codebook Utilization')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, perplexity, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].set_title('Token Distribution Perplexity')
    axes[1, 0].grid(True, alpha=0.3)

    recon_loss = [h['train_recon'] for h in history]
    vq_loss = [h['train_vq'] for h in history]
    axes[1, 1].plot(epochs, recon_loss, 'b-', label='Recon')
    axes[1, 1].plot(epochs, vq_loss, 'orange', label='VQ')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Reconstruction vs VQ Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")


def plot_evaluation_results(val_scores, test_scores, test_labels,
                            threshold, metrics, pr_auc, output_dir):
    """Plot score distributions, PR curve, confusion matrix."""
    output_path = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    test_normal = test_scores[test_labels == 0]
    test_anomaly = test_scores[test_labels == 1]

    axes[0].hist(val_scores, bins=50, alpha=0.5, label=f'Val (n={len(val_scores)})', color='green', density=True)
    axes[0].hist(test_normal, bins=50, alpha=0.5, label=f'Test Normal (n={len(test_normal)})', color='steelblue', density=True)
    axes[0].hist(test_anomaly, bins=50, alpha=0.5, label=f'Test Anomaly (n={len(test_anomaly)})', color='coral', density=True)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
    axes[0].set_xlabel('Composite Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    metric_names = ['Precision', 'Recall', 'F1', 'Specificity']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1'], metrics['specificity']]
    colors = ['steelblue', 'coral', 'green', 'purple']

    bars = axes[1].bar(metric_names, metric_values, color=colors)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Performance Metrics (F1 = {metrics["f1"]:.4f})')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, metric_values):
        axes[1].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords='offset points', ha='center')

    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    valid_mask = np.isfinite(test_scores) & np.isfinite(test_labels)
    valid_scores = test_scores[valid_mask]
    valid_labels = test_labels[valid_mask]

    if len(valid_labels) > 0 and len(np.unique(valid_labels)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(valid_labels, valid_scores)
        ax.plot(recall_curve, precision_curve, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
        ax.scatter([metrics['recall']], [metrics['precision']], color='red', s=100, zorder=5, label='Operating Point')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'pr_curve.png', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Normal', 'Pred Anomaly'])
    ax.set_yticklabels(['True Normal', 'True Anomaly'])
    ax.set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=150)
    plt.close()

    print(f"Evaluation plots saved to: {output_path}")
