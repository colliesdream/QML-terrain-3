# VQTransAE: Vector Quantized Transformer Autoencoder

A deep learning framework for **bike road surface anomaly detection** using accelerometer data.

## Overview

VQTransAE combines:
- **Bidirectional LSTM Encoder** for temporal feature extraction
- **Vector Quantization with EMA** for discrete latent representation
- **Temporal Convolutional Network (TCN)** for local pattern modeling
- **Transformer layers with relative position bias** for global context

The model learns normal road patterns from accelerometer data and detects anomalies (potholes, cracks, bumps) based on reconstruction error and quantization distance.

## Project Structure

```
vqtransae/
├── config.py       # Hyperparameters and paths
├── utils.py        # Utility functions (serialization, scaling)
├── data.py         # Data preprocessing and DataLoader
├── model_arch.py   # VQTransAE model architecture
├── train.py        # Training loop and regularizers
├── evaluate.py     # Scoring and evaluation metrics
├── visualize.py    # Plotting functions
├── pipeline.py     # End-to-end pipeline
├── __init__.py     # Package exports
└── requirements.txt
run_vqtransae.py    # Main entry script
```

## Installation

```bash
# Install dependencies
pip install -r vqtransae/requirements.txt

# Or install individually
pip install torch numpy pandas scikit-learn matplotlib
```

## Data Format

The model expects CSV files with the following columns:
- `X`, `Y`, `Z`, `G`: Accelerometer features (required)
- `Speed`: Vehicle speed (optional, used for normalization)
- `anomaly`: Binary label, 0=normal, 1=anomaly (required for test set)

Directory structure:
```
processed_data/
├── train.csv   # Training data (normal samples)
├── val.csv     # Validation data (normal samples)
└── test.csv    # Test data (with anomaly labels)
```

## Quick Start

### Option 1: Run Full Pipeline

```bash
python run_vqtransae.py
```

This will:
1. Skip preprocessing (uses existing processed data)
2. Train the model for 100 epochs
3. Evaluate on test set and generate plots

### Option 2: Step-by-Step in Python

```python
from vqtransae import preprocess_data, train_model, evaluate_model

# Step 1: Preprocess (if needed)
preprocess_data('./dataset/BIKE', './dataset/BIKE_processed')

# Step 2: Train
model, history, best_path = train_model(
    data_dir='./dataset/BIKE_processed',
    output_dir='./results',
    epochs=100
)

# Step 3: Evaluate
results = evaluate_model(
    model_path=str(best_path),
    data_dir='./dataset/BIKE_processed'
)
```

### Option 3: Custom Configuration

```python
from vqtransae import Config, run_complete_pipeline

# Modify config
Config.WIN_SIZE = 64
Config.EPOCHS = 50
Config.ALPHA = 1.0
Config.BETA = 1.0
Config.GAMMA = -0.5

# Run pipeline
results = run_complete_pipeline(
    original_data_dir='./my_data',
    output_dir='./my_results',
    skip_preprocessing=True
)
```

## Evaluation Metrics

The model reports:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall

Threshold is selected using the Nth percentile of validation scores (default: 20th).

## Tips

1. **Codebook Collapse**: If active tokens < 100, increase `ENTROPY_WEIGHT` or `DIVERSITY_WEIGHT`
2. **Poor Recall**: Lower `THRESHOLD_PERCENTILE` (e.g., 10th or 15th)
3. **High False Positives**: Increase `THRESHOLD_PERCENTILE` (e.g., 30th or 40th)
4. **Speed Sensitivity**: The model normalizes scores per speed bucket to handle varying baselines

## Citation

If you use this code, please cite:

```
@misc{vqtransae2025,
  title={VQTransAE: Vector Quantized Transformer Autoencoder for Road Surface Anomaly Detection},
  year={2025}
}
```

## License

MIT License
