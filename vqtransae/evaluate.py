"""
Evaluation utilities: score computation, composite scoring, metrics.
"""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from .config import Config
from .data import create_dataloaders
from .utils import save_json, speed_bins, robust_scale_by_group
from .visualize import plot_evaluation_results


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(model, data_loader, device) -> Dict[str, np.ndarray]:
    """Compute reconstruction error (E), quantization distance (D), attention dispersion (A)."""
    recon_errors = []
    quant_dists = []
    attention_scores = []
    all_labels = []
    all_speeds = []

    model.eval()
    with torch.no_grad():
        for x, labels, speeds in data_loader:
            x = x.to(device)

            recon, attn_list, _, indices = model(x)

            rec_err = nn.functional.mse_loss(recon, x, reduction='none').mean(dim=(1, 2))
            recon_errors.append(rec_err.cpu().numpy())

            B, T, _ = x.shape
            z_e = model.encoder(x)
            quantized = model.quant.embed(indices.view(-1)).view_as(z_e)
            q_dist = (z_e - quantized).pow(2).sum(dim=2).mean(dim=1)
            quant_dists.append(q_dist.cpu().numpy())

            attn = attn_list[-1].mean(dim=1).clamp_min(1e-9)
            t_dim = attn.shape[-1]
            log_t = math.log(max(t_dim, 2))
            entropy = -(attn * torch.log(attn)).sum(dim=-1) / (log_t + 1e-9)
            gini_like = 1.0 - attn.max(dim=-1).values
            composite_attn = 0.5 * (entropy.mean(dim=-1) + gini_like.mean(dim=-1))
            attention_scores.append(composite_attn.cpu().numpy())

            all_labels.append(labels.numpy())
            all_speeds.append(speeds.numpy())

    return {
        'E': np.nan_to_num(np.concatenate(recon_errors), nan=0.0, posinf=0.0, neginf=0.0),
        'D': np.nan_to_num(np.concatenate(quant_dists), nan=0.0, posinf=0.0, neginf=0.0),
        'A': np.nan_to_num(np.concatenate(attention_scores), nan=0.0, posinf=0.0, neginf=0.0),
        'labels': np.concatenate(all_labels),
        'speeds': np.concatenate(all_speeds)
    }


def compute_composite_score(scores: Dict, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Compute composite score S = alpha*E_norm + beta*D_norm + gamma*A_norm (gamma usually negative)."""
    groups = speed_bins(scores['speeds'])

    E_norm = np.nan_to_num(robust_scale_by_group(scores['E'], groups), nan=0.0, posinf=0.0, neginf=0.0)
    D_norm = np.nan_to_num(robust_scale_by_group(scores['D'], groups), nan=0.0, posinf=0.0, neginf=0.0)
    A_norm = np.nan_to_num(robust_scale_by_group(scores['A'], groups), nan=0.0, posinf=0.0, neginf=0.0)

    composite = alpha * E_norm + beta * D_norm + gamma * A_norm
    composite = np.nan_to_num(composite, nan=0.0, posinf=0.0, neginf=0.0)

    return composite, E_norm, D_norm, A_norm


# ---------------------------------------------------------------------------
# Thresholded metrics
# ---------------------------------------------------------------------------

def evaluate_with_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    """Evaluate precision/recall/F1/accuracy/specificity at a fixed threshold."""
    preds = (scores > threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    data_dir: str = Config.PROCESSED_DATA_DIR,
    output_dir: str = None,
    alpha: float = Config.ALPHA,
    beta: float = Config.BETA,
    gamma: float = Config.GAMMA,
    percentile: int = Config.THRESHOLD_PERCENTILE
) -> Dict:
    """Evaluate a trained checkpoint: pick threshold on val, report metrics/plots."""

    print("\n" + "=" * 70)
    print("Step 3: Evaluate model")
    print("=" * 70)
    print(f"Composite score: S = {alpha}*E_norm + {beta}*D_norm + ({gamma})*A_norm")
    print(f"Threshold percentile: {percentile}th")
    print("=" * 70)

    if output_dir is None:
        output_dir = Path(model_path).parent / 'evaluation'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoad model...")
    from .model_arch import VQTransAE

    checkpoint = torch.load(model_path, map_location=device)
    hyper = checkpoint.get('hyperparams', {
        'win_size': Config.WIN_SIZE, 'in_dim': Config.IN_DIM,
        'hidden': 64, 'latent': 32, 'codebook': 1024,
        'd_model': 64, 'heads': 4, 'layers': 3
    })

    model = VQTransAE(
        hyper['win_size'], hyper['in_dim'],
        hidden=hyper['hidden'], latent=hyper['latent'],
        codebook=hyper['codebook'], d_model=hyper['d_model'],
        heads=hyper['heads'], layers=hyper['layers']
    ).to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    active_tokens = checkpoint.get('active_tokens', 'N/A')
    print(f"Active tokens: {active_tokens}")

    print("\nBuild DataLoaders...")
    _, val_loader, test_loader = create_dataloaders(data_dir)
    print(f"Val windows: {len(val_loader.dataset)}")
    print(f"Test windows: {len(test_loader.dataset)}")

    print("\nCompute scores...")
    val_scores = compute_scores(model, val_loader, device)
    test_scores = compute_scores(model, test_loader, device)

    val_composite, _, _, _ = compute_composite_score(val_scores, alpha, beta, gamma)
    test_composite, E_norm, D_norm, A_norm = compute_composite_score(test_scores, alpha, beta, gamma)
    test_labels = test_scores['labels']

    print("\nComponent separability (anomaly mean - normal mean):")
    separations = {}
    for name, data in [('E', E_norm), ('D', D_norm), ('A', A_norm), ('S', test_composite)]:
        normal = data[test_labels == 0]
        anomaly = data[test_labels == 1]
        sep = anomaly.mean() - normal.mean()
        separations[name] = sep
        print(f"  {name}: {sep:+.4f}")

    valid_val = val_composite[np.isfinite(val_composite)]
    threshold = np.percentile(valid_val, percentile) if len(valid_val) > 0 else 0.0
    print(f"\nThreshold: {threshold:.4f} ({percentile}th percentile of val)")

    metrics = evaluate_with_threshold(test_composite, test_labels, threshold)

    print("\n" + "=" * 70)
    print("Eval metrics")
    print("=" * 70)
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Confusion: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")



    # Calculate PR-AUC
    valid_mask = np.isfinite(test_composite) & np.isfinite(test_labels)
    valid_composite = test_composite[valid_mask]
    valid_labels = test_labels[valid_mask]
    if len(valid_labels) > 0 and len(np.unique(valid_labels)) > 1:
        pr_auc = average_precision_score(valid_labels, valid_composite)
    else:
        pr_auc = float('nan')
    print(f"PR-AUC:  {pr_auc:.4f}")

    print("\nPerformance at multiple percentiles:")
    percentile_results = []
    for pct in [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]:
        valid_val_finite = val_composite[np.isfinite(val_composite)]
        th = np.percentile(valid_val_finite, pct) if len(valid_val_finite) > 0 else 0.0
        m = evaluate_with_threshold(test_composite, test_labels, th)
        percentile_results.append({'percentile': pct, 'threshold': th, **m})
        print(f"  {pct:2d}th: F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")

    best_pct_result = max(percentile_results, key=lambda x: x['f1'])
    print(f"Best percentile: {best_pct_result['percentile']}th (F1={best_pct_result['f1']:.4f})")

    print("\nGenerate plots...")
    plot_evaluation_results(
        val_composite, test_composite, test_labels,
        threshold, metrics, pr_auc, output_path
    )

    results = {
        'metrics': metrics,
        'pr_auc': pr_auc,
        'separations': separations,
        'threshold': threshold,
        'percentile': percentile,
        'best_percentile_result': best_pct_result
    }

    save_json(results, output_path / 'evaluation_results.json')
    return results
