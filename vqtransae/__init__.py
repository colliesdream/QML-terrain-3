"""
VQTransAE - Vector Quantized Transformer Autoencoder for bike-road anomaly detection.

Modules:
- config.py     : configuration
- utils.py      : utilities
- data.py       : preprocessing and dataloaders
- train.py      : training helpers
- evaluate.py   : evaluation helpers
- visualize.py  : plotting
- pipeline.py   : end-to-end pipeline

Usage examples:
---------------
# Full pipeline
from vqtransae import run_complete_pipeline
results = run_complete_pipeline(
    original_data_dir='./dataset/BIKE',
    output_dir='./vqtransae_results',
    epochs=100
)

# Step-by-step
from vqtransae import preprocess_data, train_model, evaluate_model
preprocess_data('./dataset/BIKE', './dataset/BIKE_processed')
model, history, best_path = train_model('./dataset/BIKE_processed', './results')
results = evaluate_model(best_path, './dataset/BIKE_processed')
"""

from .config import Config
from .utils import (
    to_serializable,
    save_json,
    speed_bins,
    robust_scale_by_group,
    token_stats
)
from .data import (
    preprocess_data,
    WindowDataset,
    create_dataloaders
)
from .train import (
    entropy_regularizer,
    diversity_regularizer,
    refresh_inactive_codes,
    train_epoch,
    validate,
    train_model
)
from .evaluate import (
    compute_scores,
    compute_composite_score,
    evaluate_with_threshold,
    evaluate_model
)
from .visualize import (
    plot_training_curves,
    plot_evaluation_results
)
from .pipeline import run_complete_pipeline
from .model_arch import VQTransAE
from .prediction_baseline import (
    segment_route,
    build_dataset,
    LSTMPredictor,
    train_predictor,
    generate_predicted_route,
    compute_disagreement_score,
    format_baseline_summary,
    print_baseline_summary,
    threshold_from_validation,
    run_prediction_baseline
)

__all__ = [
    'Config',
    'VQTransAE',
    'to_serializable', 'save_json', 'speed_bins', 'robust_scale_by_group', 'token_stats',
    'preprocess_data', 'WindowDataset', 'create_dataloaders',
    'entropy_regularizer', 'diversity_regularizer', 'refresh_inactive_codes',
    'train_epoch', 'validate', 'train_model',
    'compute_scores', 'compute_composite_score', 'evaluate_with_threshold', 'evaluate_model',
    'plot_training_curves', 'plot_evaluation_results',
    'run_complete_pipeline',
    'segment_route', 'build_dataset', 'LSTMPredictor', 'train_predictor',
    'generate_predicted_route', 'compute_disagreement_score',
    'format_baseline_summary', 'print_baseline_summary',
    'threshold_from_validation', 'run_prediction_baseline'
]
