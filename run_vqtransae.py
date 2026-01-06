#!/usr/bin/env python3
"""
Main entrypoint to run full VQTransAE pipeline.

Usage:
  python run_vqtransae.py

Or from code:
  from vqtransae import run_complete_pipeline
  results = run_complete_pipeline()
"""

from vqtransae import run_complete_pipeline


if __name__ == '__main__':
    run_complete_pipeline(
      original_data_dir='./dataset/BIKE',
      output_dir='./vqtransae_results',
      pretrained_path='./win_56/best_vqtransae.pth',  # optional
      epochs=100,
      skip_preprocessing=True,   # data already preprocessed at Config.PROCESSED_DATA_DIR
      skip_training=False
    )
