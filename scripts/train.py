#!/usr/bin/env python3
"""
Unified training script for airlift project.
Supports both simulation and real data training.
"""

import os
import sys
import time
import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_npz_pair, to_tensor_dataset, split_dataset
from src.training.trainer import train_one_epoch, evaluate, create_model, create_learning_curves
from src.utils.device import get_valid_device
from src.utils.memory import clear_gpu_memory
try:
    from src.evaluation.visualizations import create_prediction_plots
except ImportError:
    # Fallback to old location
    from src.evaluate_predictions import create_prediction_plots
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="config_real_updated.yaml", version_base=None)
def main(cfg):
    """Main training function with Hydra configuration."""
    
    # Prevent Hydra from creating output directories in sandbox/ml_airlift
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sandbox_dir = script_dir
    
    # Check and remove Hydra-created output directories
    outputs_dir = os.path.join(sandbox_dir, "outputs")
    if os.path.exists(outputs_dir):
        try:
            shutil.rmtree(outputs_dir)
            print(f"[INFO] Removed Hydra output directory: {outputs_dir}")
        except Exception as e:
            print(f"[WARN] Could not remove Hydra output directory {outputs_dir}: {e}")
    
    # Create time-based output directory
    outputs_root = cfg.output.model_save_dir
    now = datetime.datetime.now()
    run_dir = os.path.join(outputs_root, now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
    base_dir = run_dir
    logs_dir = os.path.join(run_dir, "logs")
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"[INFO] Run directory: {os.path.abspath(run_dir)}")
    print(f"[INFO] Logs will be saved under: {os.path.abspath(logs_dir)}")
    
    # Print configuration
    print("🔧 Configuration Summary")
    print("=" * 50)
    print(f"Dataset X: {cfg.dataset.x_train}")
    print(f"Dataset T: {cfg.dataset.t_train}")
    print(f"Model: {cfg.model.type}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Device: {cfg.training.device}")
    print("=" * 50)
    
    # Set reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Clear GPU memory before starting
    device = get_valid_device(cfg.training.device)
    if device.type == 'cuda':
        print("[INFO] Clearing GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        print(f"[OK] GPU memory cleared")

    # cuDNN autotune for convs
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    
    # Load data
    print("[STEP] Loading dataset files...")
    x, t = load_npz_pair(cfg.dataset.x_train, cfg.dataset.t_train, cfg.dataset.x_key, cfg.dataset.t_key)
    print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape} (elapsed {time.time()-t0:.2f}s)")
    
    # Exclude Channel 1 and Channel 3 (keep only channels 0, 2)
    if x.ndim == 4 and x.shape[1] == 4:
        print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
        x = x[:, [0, 2], :, :]  # Keep only channels 0, 2
        print(f"[OK] After excluding Channel 1 and 3: x.shape={x.shape}")
        # Update model config to reflect 2 channels
        cfg.model.in_channels = 2
    elif x.ndim == 3 and x.shape[1] == 4:
        print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
        x = x[:, [0, 2], :]  # Keep only channels 0, 2
        print(f"[OK] After excluding Channel 1 and 3: x.shape={x.shape}")
        # Update model config to reflect 2 channels
        cfg.model.in_channels = 2
    
    # Check for NaNs
    if np.isnan(x).any():
        print("[WARNING] NaN values detected in x!")
    else:
        print("[INFO] No NaN values in x.")
    if np.isnan(t).any():
        print("[WARNING] NaN values detected in t!")
    else:
        print("[INFO] No NaN values in t.")
    
    # Print data info for 4D case
    if x.ndim == 4:
        print(f"[INFO] 4D Image Data: N={x.shape[0]}, C={x.shape[1]}, H={x.shape[2]}, W={x.shape[3]}")
        print(f"[INFO] Target shape: {t.shape} (6 targets for multi-output regression)")
        print(f"[INFO] Memory usage: {x.nbytes / 1024**2:.1f} MB")
    
    # Limit samples if specified
    if cfg.dataset.limit_samples > 0:
        n = min(cfg.dataset.limit_samples, x.shape[0])
        x = x[:n]
        t = t[:n]
        print(f"[INFO] Limited to first {n} samples")
    
    # Optional downsampling
    if x.ndim == 4 and cfg.dataset.downsample_factor > 1:
        h0 = x.shape[2]
        x = x[:, :, ::cfg.dataset.downsample_factor, :]
        print(f"[INFO] Downsampled H: {h0} -> {x.shape[2]} (factor={cfg.dataset.downsample_factor})")
    elif x.ndim == 4:
        print(f"[INFO] Using full resolution: H={x.shape[2]}, W={x.shape[3]}")
    
    # Create dataset
    print("[STEP] Build dataset tensors...")
    dataset = to_tensor_dataset(x, t, cfg.training.device)
    print("[OK] Dataset ready.")
    
    # Split dataset
    print("[STEP] Split dataset...")
    train_set, val_set, test_set = split_dataset(
        dataset, 
        cfg.data_split.train_ratio, 
        cfg.data_split.val_ratio, 
        cfg.data_split.test_ratio, 
        cfg.training.seed
    )
    print(f"[OK] Sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    # Create dataloaders
    print("[STEP] Build dataloaders...")
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.workers, 
        pin_memory=cfg.training.pin_memory
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.workers, 
        pin_memory=cfg.training.pin_memory
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.workers, 
        pin_memory=cfg.training.pin_memory
    )
    print("[OK] Dataloaders ready.")
    
    # Create model
    print("[STEP] Build model...")
    x_sample = dataset.tensors[0]
    out_dim = dataset.tensors[1].shape[1] if dataset.tensors[1].ndim == 2 else 1
    model = create_model(cfg, x_sample, out_dim, device)
    
    # Create optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training
    print("[STEP] Training...")
    print(f"[INFO] Training {x_sample.shape[1]}-channel image data with {out_dim} output targets")
    print(f"[INFO] Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, cfg.training.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.logging.print_every_n_batches)
        
        # Evaluate validation set if it exists
        if len(val_loader.dataset) > 0:
            val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        else:
            val_mse, val_mae = 0.0, 0.0
        
        train_losses.append(tr)
        val_losses.append(val_mse)
        
        if epoch % cfg.logging.print_every_n_epochs == 0:
            if len(val_loader.dataset) > 0:
                print(f"Epoch {epoch:03d} | train MSE={tr:.6f} | val MSE={val_mse:.6f} | val MAE={val_mae:.6f} | {time.time()-t_ep:.2f}s")
            else:
                print(f"Epoch {epoch:03d} | train MSE={tr:.6f} | val MSE=N/A | val MAE=N/A | {time.time()-t_ep:.2f}s")
    
    # Testing
    print("[STEP] Testing...")
    test_mse, test_mae, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"Test  | MSE={test_mse:.6f} | MAE={test_mae:.6f}")
    
    # Print per-target results for multi-output regression
    if y_pred.shape[1] > 1:
        print(f"[INFO] Per-target results:")
        for i in range(y_pred.shape[1]):
            target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
            target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            print(f"  Target {i+1}: MSE={target_mse:.6f}, MAE={target_mae:.6f}")
    
    # Save model and results
    print("[STEP] Saving model and results...")
    torch.save(model.state_dict(), os.path.join(weights_dir, cfg.output.model_filename))
    np.save(os.path.join(base_dir, cfg.output.predictions_filename), y_pred)
    np.save(os.path.join(base_dir, cfg.output.ground_truth_filename), y_true)
    print(f"[OK] Saved to {base_dir}")
    
    # Create learning curves
    print("[STEP] Creating learning curves...")
    create_learning_curves(train_losses, val_losses, base_dir)
    
    # Create evaluation plots if requested
    if cfg.evaluation.create_plots and y_pred.shape[1] > 1:
        print("[STEP] Creating evaluation plots...")
        plots_dir = os.path.join(base_dir, cfg.output.evaluation_plots_dir)
        create_prediction_plots(y_pred, y_true, plots_dir, cfg.evaluation.target_names)
        print(f"[OK] Evaluation plots saved to {plots_dir}")
    
    # Save configuration
    with open(os.path.join(base_dir, "config.yaml"), 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"[OK] Configuration saved to {base_dir}/config.yaml")
    
    print(f"[DONE] Training completed. Total elapsed {time.time()-t0:.2f}s")
    print(f"[DONE] Results saved to: {base_dir}")
    
    # Clean up Hydra-created output directories
    outputs_dir = os.path.join(sandbox_dir, "outputs")
    if os.path.exists(outputs_dir):
        try:
            shutil.rmtree(outputs_dir)
            print(f"[INFO] Cleaned up Hydra output directory: {outputs_dir}")
        except Exception as e:
            print(f"[WARN] Could not remove Hydra output directory {outputs_dir}: {e}")


if __name__ == "__main__":
    main()

