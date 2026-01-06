"""
Training utilities for VQTransAE: losses, regularizers, training loop.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .data import create_dataloaders
from .utils import save_json, token_stats
from .visualize import plot_training_curves


# ---------------------------------------------------------------------------
# Regularizers
# ---------------------------------------------------------------------------

def entropy_regularizer(batch_counts: torch.Tensor, target_ratio: float, device: torch.device) -> torch.Tensor:
    """Encourage uniform token usage; penalize low entropy distributions."""
    if batch_counts.sum() == 0:
        return torch.zeros(1, device=device)

    probs = batch_counts.float().to(device)
    probs = probs / probs.sum()
    probs = probs[probs > 0]

    entropy = -(probs * probs.log()).sum()
    max_entropy = math.log(batch_counts.numel())
    target_entropy = target_ratio * max_entropy

    return torch.relu(target_entropy - entropy)


def diversity_regularizer(indices: torch.Tensor) -> torch.Tensor:
    """Penalize overly concentrated token usage within a batch."""
    if indices.numel() == 0:
        return torch.zeros(1, device=indices.device)

    unique, counts = torch.unique(indices, return_counts=True)
    if unique.numel() <= 1:
        return torch.ones(1, device=indices.device)

    probs = counts.float() / counts.sum()
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    target_entropy = torch.log(torch.tensor(float(unique.numel()), device=indices.device))
    return torch.relu((target_entropy - entropy) / (target_entropy + 1e-8))


def refresh_inactive_codes(model, usage_counts: np.ndarray, min_usage: int, reuse_active: bool = True) -> int:
    """Refresh rarely used codebook vectors with active ones plus noise."""
    if usage_counts.sum() == 0:
        return 0

    dormant_mask = usage_counts < min_usage
    dormant = np.nonzero(dormant_mask)[0]
    if dormant.size == 0:
        return 0

    with torch.no_grad():
        embed = model.quant.embed.weight
        scale = 1.0 / math.sqrt(embed.shape[1])
        noise = torch.randn((dormant.size, embed.shape[1]), device=embed.device) * scale
        dormant_tensor = torch.from_numpy(dormant).to(embed.device)

        if reuse_active:
            active_idx = np.nonzero(usage_counts >= min_usage)[0]
            if active_idx.size > 0:
                chosen = np.random.choice(active_idx, size=dormant.size, replace=True)
                chosen_tensor = torch.from_numpy(chosen).to(embed.device)
                sampled = embed[chosen_tensor]
                embed[dormant_tensor] = sampled + 0.1 * noise
            else:
                embed[dormant_tensor] = noise
        else:
            embed[dormant_tensor] = noise

    return int(dormant.size)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, epoch: int,
                entropy_weight: float, entropy_target_ratio: float,
                diversity_weight: float, vq_weight: float) -> Dict:
    """Run one training epoch and return metrics and token usage histogram."""

    model.train()
    total_loss = 0
    total_recon = 0
    total_vq = 0
    codebook_size = model.quant.embed.num_embeddings
    epoch_hist = torch.zeros(codebook_size, dtype=torch.long)

    for x, _, _ in loader:
        x = x.to(device)

        recon, _, loss_vq, indices = model(x)
        rec_loss = nn.functional.mse_loss(recon, x)
        loss = rec_loss + vq_weight * loss_vq

        batch_counts = torch.bincount(indices.detach().cpu().flatten(), minlength=codebook_size)
        epoch_hist += batch_counts

        if entropy_weight > 0:
            entropy_reg = entropy_regularizer(batch_counts, entropy_target_ratio, device)
            loss = loss + entropy_weight * entropy_reg

        if diversity_weight > 0:
            div_loss = diversity_regularizer(indices)
            loss = loss + diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += rec_loss.item()
        total_vq += loss_vq.item()

    n_batches = max(1, len(loader))
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'vq': total_vq / n_batches,
        'token_counts': epoch_hist.numpy()
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, loader, device) -> Dict:
    """Compute validation reconstruction+VQ loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            recon, _, loss_vq, _ = model(x)
            rec_loss = nn.functional.mse_loss(recon, x)
            loss = rec_loss + loss_vq
            total_loss += loss.item()

    return {'loss': total_loss / max(1, len(loader))}


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    data_dir: str = Config.PROCESSED_DATA_DIR,
    output_dir: str = Config.OUTPUT_DIR,
    pretrained_path: str = None,
    epochs: int = Config.EPOCHS,
    config: Config = None
) -> Tuple[nn.Module, List[Dict], Path]:
    """Train VQTransAE and return model, history, and best checkpoint path."""

    if config is None:
        config = Config()

    print("\n" + "=" * 70)
    print("Step 2: Train model")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nBuild DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, config.WIN_SIZE, config.STEP, config.BATCH_SIZE
    )
    print(f"Train windows: {len(train_loader.dataset)}  batches: {len(train_loader)}")
    print(f"Val   windows: {len(val_loader.dataset)}  batches: {len(val_loader)}")

    print("\nInit model...")
    from .model_arch import VQTransAE

    model = VQTransAE(
        win_size=config.WIN_SIZE,
        in_dim=config.IN_DIM,
        hidden=config.HIDDEN_DIM,
        latent=config.LATENT_DIM,
        codebook=config.CODEBOOK_SIZE,
        d_model=config.D_MODEL,
        heads=config.N_HEADS,
        layers=config.N_LAYERS
    ).to(device)

    print("Reinit codebook weights (important)")
    nn.init.xavier_uniform_(model.quant.embed.weight)

    if pretrained_path and Path(pretrained_path).exists():
        print(f"Load pretrained encoder/decoder (skip codebook): {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        filtered_state = {k: v for k, v in state_dict.items() if 'quant.embed' not in k and 'ema' not in k}
        model.load_state_dict(filtered_state, strict=False)
        print("Loaded encoder/decoder weights (codebook skipped)")

    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.LEARNING_RATE},
        {'params': model.decoder.parameters(), 'lr': config.LEARNING_RATE},
        {'params': model.quant.parameters(), 'lr': config.CODEBOOK_LR},
        {'params': model.tf_layers.parameters(), 'lr': config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_loss = float('inf')
    best_active_tokens = 0
    best_model_state = None

    print("\n" + "=" * 70)
    print(f"Start training ({epochs} epochs)")
    print(f"Entropy weight: {config.ENTROPY_WEIGHT}")
    print(f"Diversity weight: {config.DIVERSITY_WEIGHT}")
    print(f"VQ weight base: {config.VQ_WEIGHT_BASE}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        if epoch < 30:
            vq_weight = config.VQ_WEIGHT_BASE * 1.5
        elif epoch < 60:
            vq_weight = config.VQ_WEIGHT_BASE
        else:
            vq_weight = config.VQ_WEIGHT_BASE * 0.8

        train_result = train_epoch(
            model, train_loader, optimizer, device, epoch,
            config.ENTROPY_WEIGHT, config.ENTROPY_TARGET_RATIO,
            config.DIVERSITY_WEIGHT, vq_weight
        )

        active_tokens, perplexity, _ = token_stats(train_result['token_counts'])

        refresh_count = 0
        if epoch % config.REFRESH_EVERY == 0:
            threshold = config.MIN_USAGE_THRESHOLD
            if active_tokens < config.ACTIVE_TOKEN_TARGET // 2:
                threshold = max(3, threshold // 2)
            refresh_count = refresh_inactive_codes(model, train_result['token_counts'], threshold)

        val_result = validate(model, val_loader, device)
        scheduler.step()

        record = {
            'epoch': epoch,
            'train_loss': train_result['loss'],
            'train_recon': train_result['recon'],
            'train_vq': train_result['vq'],
            'val_loss': val_result['loss'],
            'active_tokens': active_tokens,
            'perplexity': perplexity,
            'refresh_count': refresh_count
        }
        history.append(record)

        if active_tokens > best_active_tokens or \
           (active_tokens == best_active_tokens and val_result['loss'] < best_val_loss):
            best_val_loss = val_result['loss']
            best_active_tokens = active_tokens
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        status = f"Epoch {epoch:3d}: Loss={train_result['loss']:.4f}, Val={val_result['loss']:.4f}, "
        status += f"Active={active_tokens:4d}/1024, Perplexity={perplexity:.1f}"
        if refresh_count > 0:
            status += f", Refreshed={refresh_count}"
        print(status)

        if epoch % 20 == 0 or epoch == epochs:
            save_path = output_path / f'model_epoch_{epoch:03d}.pth'
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'active_tokens': active_tokens,
                'perplexity': perplexity,
                'hyperparams': {
                    'win_size': config.WIN_SIZE,
                    'in_dim': config.IN_DIM,
                    'hidden': config.HIDDEN_DIM,
                    'latent': config.LATENT_DIM,
                    'codebook': config.CODEBOOK_SIZE,
                    'd_model': config.D_MODEL,
                    'heads': config.N_HEADS,
                    'layers': config.N_LAYERS
                }
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

    print("\n" + "=" * 70)
    print("Training finished")
    print(f"Best active tokens: {best_active_tokens}/1024")
    print("=" * 70)

    best_path = output_path / 'best_model.pth'
    torch.save({
        'state_dict': best_model_state,
        'active_tokens': best_active_tokens,
        'hyperparams': {
            'win_size': config.WIN_SIZE,
            'in_dim': config.IN_DIM,
            'hidden': config.HIDDEN_DIM,
            'latent': config.LATENT_DIM,
            'codebook': config.CODEBOOK_SIZE,
            'd_model': config.D_MODEL,
            'heads': config.N_HEADS,
            'layers': config.N_LAYERS
        }
    }, best_path)
    print(f"Best model saved: {best_path}")

    save_json(history, output_path / 'training_history.json')
    plot_training_curves(history, output_path / 'training_curves.png')

    return model, history, best_path
