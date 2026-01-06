"""
End-to-end pipeline: preprocess -> train -> evaluate.
"""

from pathlib import Path
from typing import Dict

from .config import Config
from .data import preprocess_data
from .train import train_model
from .evaluate import evaluate_model


def run_complete_pipeline(
    original_data_dir: str = Config.ORIGINAL_DATA_DIR,
    output_dir: str = Config.OUTPUT_DIR,
    pretrained_path: str = None,
    epochs: int = Config.EPOCHS,
    skip_preprocessing: bool = False,
    skip_training: bool = False
) -> Dict:
    """Run preprocessing, training, and evaluation in one call."""

    print("=" * 70)
    print("VQTransAE full pipeline")
    print("=" * 70)
    print(f"Original data: {original_data_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Epochs:        {epochs}")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_dir = output_path / 'processed_data'

    if not skip_preprocessing:
        preprocess_data(original_data_dir, str(processed_dir))
    else:
        print("Skip preprocessing (use existing preprocessed data)")
        processed_dir = Path(Config.PROCESSED_DATA_DIR)

    if not skip_training:
        model, history, best_model_path = train_model(
            data_dir=str(processed_dir),
            output_dir=str(output_path / 'models'),
            pretrained_path=pretrained_path,
            epochs=epochs
        )
    else:
        print("Skip training (use existing best_model.pth)")
        best_model_path = output_path / 'models' / 'best_model.pth'

    results = evaluate_model(
        model_path=str(best_model_path),
        data_dir=str(processed_dir),
        output_dir=str(output_path / 'evaluation'),
        alpha=Config.ALPHA,
        beta=Config.BETA,
        gamma=Config.GAMMA,
        percentile=Config.THRESHOLD_PERCENTILE
    )

    print("Pipeline finished. Results saved to:", output_path)
    return results
