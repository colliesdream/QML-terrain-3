"""
VQTransAE configuration
All tunable hyperparameters live here.
"""

class Config:
    """Container for all configurable parameters."""

    # Data paths
    ORIGINAL_DATA_DIR = './dataset/BIKE'
    PROCESSED_DATA_DIR = '/Volumes/hardware/UNL_smart_bike_code/anomaly_lable/vqtransae/processed_data'
    OUTPUT_DIR = './vqtransae_results'

    # Optional pretrained model path
    PRETRAINED_MODEL_PATH = './win_56/best_vqtransae.pth'

    # Feature columns
    FEATURE_COLUMNS = ['X', 'Y', 'Z', 'G']

    # Model parameters
    WIN_SIZE = 56        # Window size
    IN_DIM = 4           # Input dimension
    HIDDEN_DIM = 64      # Hidden dimension
    LATENT_DIM = 32      # Latent dimension
    CODEBOOK_SIZE = 1024 # Codebook size
    D_MODEL = 64         # Transformer model dim
    N_HEADS = 4          # Attention heads
    N_LAYERS = 3         # Transformer layers

    # Data parameters
    STEP = 5             # Sliding window stride
    BATCH_SIZE = 64      # Batch size
    VAL_RATIO = 0.15     # Validation split ratio

    # Training parameters
    EPOCHS = 100         # Training epochs
    LEARNING_RATE = 1e-4 # Base learning rate
    CODEBOOK_LR = 5e-4   # Higher LR for codebook
    WEIGHT_DECAY = 1e-5  # Weight decay

    # Regularization to avoid codebook collapse
    ENTROPY_WEIGHT = 0.3       # Entropy regularization weight
    ENTROPY_TARGET_RATIO = 0.8 # Target entropy ratio
    DIVERSITY_WEIGHT = 0.2     # Diversity regularization weight
    VQ_WEIGHT_BASE = 2.0       # Base VQ loss weight

    # Codebook refresh
    REFRESH_EVERY = 5          # Refresh every N epochs
    MIN_USAGE_THRESHOLD = 10   # Minimum usage count
    ACTIVE_TOKEN_TARGET = 100  # Target active tokens

    # Composite score weights
    ALPHA = 1.0   # Reconstruction weight
    BETA = 1.0    # Quantization distance weight
    GAMMA = -0.5  # Attention weight (negative because anomalies attend more)

    # Threshold percentile on validation scores
    THRESHOLD_PERCENTILE = 20
