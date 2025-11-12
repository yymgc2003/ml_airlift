#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for CNN model training.
Based on train_real.py structure with automatic parameter tuning.
"""

import os
import sys
import time
import datetime
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

import optuna
from optuna.trial import Trial
from omegaconf import OmegaConf, DictConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config using OmegaConf to enable attribute access
config = OmegaConf.load('config/config_real_updated.yaml')

# Import functions from new module structure
from src.data.loaders import (
    load_npz_pair,
    to_tensor_dataset,
    split_dataset,
)
from src.training.trainer import (
    train_one_epoch,
    evaluate,
    create_model,
    create_learning_curves,
)
try:
    from src.evaluation.visualizations import create_prediction_plots
except ImportError:
    # Fallback to old location
    from src.evaluate_predictions import create_prediction_plots

# Paths
OPTUNA_DIR = config.optuna.dir
OUTPUTS_ROOT = config.optuna.outputs_dir
BASE_CONFIG_PATH = config.optuna.base_config_path


def suggest_hyperparameters(trial: Trial, base_cfg: DictConfig) -> DictConfig:
    """
    Suggest hyperparameters using Optuna and update config.
    
    Args:
        trial: Optuna trial object
        base_cfg: Base configuration from YAML
    
    Returns:
        Updated configuration with suggested hyperparameters
    """
    # Create a copy of the config
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    
    # Model hyperparameters
    cfg.model.hidden = trial.suggest_int('model.hidden', 32, 128, step=16)
    cfg.model.dropout_rate = trial.suggest_float('model.dropout_rate', 0.0, 0.5, step=0.05)
    cfg.model.use_residual = trial.suggest_categorical('model.use_residual', [True, False])
    
    # Training hyperparameters
    cfg.training.learning_rate = trial.suggest_float(
        'training.learning_rate', 1e-5, 1e-2, log=True
    )
    # Reduce batch size options to avoid OOM with large images
    cfg.training.batch_size = trial.suggest_categorical('training.batch_size', [2, 4, 8])
    
    # Add weight_decay if not exists
    if 'weight_decay' not in cfg.training:
        cfg.training.weight_decay = trial.suggest_float(
            'training.weight_decay', 1e-6, 1e-3, log=True
        )
    else:
        cfg.training.weight_decay = trial.suggest_float(
            'training.weight_decay', 1e-6, 1e-3, log=True
        )
    
    # Data hyperparameters
    cfg.dataset.downsample_factor = trial.suggest_int('dataset.downsample_factor', 1, 4)
    
    # Limit epochs for faster optimization (can be adjusted)
    cfg.training.epochs = trial.suggest_int('training.epochs', 50, 200, step=50)
    
    return cfg


def create_trial_output_dir(trial_number: int) -> str:
    """
    Create output directory for a trial following train_real.py structure.
    
    Args:
        trial_number: Optuna trial number
    
    Returns:
        Output directory path
    """
    now = datetime.datetime.now()
    run_dir = os.path.join(
        OUTPUTS_ROOT,
        now.strftime('%Y-%m-%d'),
        now.strftime('%H-%M-%S')
    )
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    logs_dir = os.path.join(run_dir, "logs")
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    return run_dir


def save_trial_results(trial: Trial, cfg: DictConfig, output_dir: str, 
                       train_losses: list, val_losses: list,
                       best_val_loss: float,
                       test_mse: float, test_mae: float,
                       y_pred: np.ndarray, y_true: np.ndarray):
    """
    Save trial results following train_real.py structure.
    
    Args:
        trial: Optuna trial object
        cfg: Configuration used for this trial
        output_dir: Output directory for this trial
        train_losses: Training loss history
        val_losses: Validation loss history
        best_val_loss: Best validation loss for this trial
        test_mse: Test MSE
        test_mae: Test MAE
        y_pred: Predictions
        y_true: Ground truth
    """
    # Save configuration
    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    
    # Save trial information
    # Note: trial.state is not available during objective execution,
    # so we set it to "RUNNING" and it will be updated after completion
    trial_state = "RUNNING"
    if hasattr(trial, 'state') and trial.state is not None:
        trial_state = trial.state.name
    
    trial_info = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(best_val_loss),  # Validation loss
        'state': trial_state,
        'user_attrs': trial.user_attrs,
        'datetime': datetime.datetime.now().isoformat()
    }
    trial_info_path = os.path.join(output_dir, 'trial_info.yaml')
    OmegaConf.save(trial_info, trial_info_path)
    
    # Save metrics
    metrics = {
        'validation_loss': float(best_val_loss),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses],
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
    }
    
    # Per-target metrics
    if y_pred.shape[1] > 1:
        per_target_metrics = {}
        for i in range(y_pred.shape[1]):
            target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
            target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            per_target_metrics[f'target_{i+1}_mse'] = float(target_mse)
            per_target_metrics[f'target_{i+1}_mae'] = float(target_mae)
        metrics['per_target'] = per_target_metrics
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and ground truth
    weights_dir = os.path.join(output_dir, "weights")
    np.save(os.path.join(output_dir, cfg.output.predictions_filename), y_pred)
    np.save(os.path.join(output_dir, cfg.output.ground_truth_filename), y_true)
    
    # Create learning curves
    create_learning_curves(train_losses, val_losses, output_dir)
    
    # Create evaluation plots
    if cfg.evaluation.create_plots and y_pred.shape[1] > 1:
        plots_dir = os.path.join(output_dir, cfg.output.evaluation_plots_dir)
        create_prediction_plots(y_pred, y_true, plots_dir, cfg.evaluation.target_names)
    
    print(f"[OK] Trial {trial.number} results saved to {output_dir}")


def objective(trial: Trial, base_config_path: str) -> float:
    """
    Optuna objective function.
    
    Flow:
    1. Load base configuration
    2. Suggest hyperparameters
    3. Create trial output directory
    4. Load and preprocess data
    5. Create model
    6. Train model (with pruning support)
    7. Evaluate on test set
    8. Save results
    
    Args:
        trial: Optuna trial object
        base_config_path: Path to base configuration file
    
    Returns:
        Validation loss (optimization target)
    """
    # Clear GPU memory before starting trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset memory stats
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    
    t0 = time.time()
    
    # 1. Load base configuration
    base_cfg = OmegaConf.load(base_config_path)
    
    # 2. Suggest hyperparameters
    cfg = suggest_hyperparameters(trial, base_cfg)
    
    # 3. Create trial output directory
    output_dir = create_trial_output_dir(trial.number)
    cfg.run_dir = output_dir
    cfg.trial_number = trial.number
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} started")
    print(f"Output directory: {output_dir}")
    print(f"Hyperparameters: {trial.params}")
    print(f"{'='*60}\n")
    
    # Set reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    try:
        # 4. Load and preprocess data
        print("[STEP] Loading dataset files...")
        x, t = load_npz_pair(
            cfg.dataset.x_train, 
            cfg.dataset.t_train, 
            cfg.dataset.x_key, 
            cfg.dataset.t_key
        )
        print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape}")
        
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
        
        # Create dataset
        print("[STEP] Build dataset tensors...")
        dataset = to_tensor_dataset(x, t, cfg.training.device)
        
        # Split dataset (use fixed seed for consistency)
        print("[STEP] Split dataset...")
        train_set, val_set, test_set = split_dataset(
            dataset,
            cfg.data_split.train_ratio,
            cfg.data_split.val_ratio,
            cfg.data_split.test_ratio,
            cfg.training.seed  # Fixed seed for all trials
        )
        print(f"[OK] Sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        
        # Create dataloaders
        print("[STEP] Build dataloaders...")
        # Set num_workers=0 to avoid multiprocessing issues with large images
        # pin_memory=False for large images to reduce memory pressure
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid deadlocks
            pin_memory=False  # Disable pin_memory for large images
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # 5. Create model
        print("[STEP] Build model...")
        x_sample = dataset.tensors[0]
        device = torch.device(cfg.training.device)
        out_dim = dataset.tensors[1].shape[1] if dataset.tensors[1].ndim == 2 else 1
        model = create_model(cfg, x_sample, out_dim, device)

        # Disable DataParallel for large images to avoid deadlock/hanging issues
        # DataParallel can cause deadlocks with very large input images (1400x2500)
        # Use single GPU instead for stability
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            print(f"[WARN] Multiple GPUs available ({num_gpus}), but DataParallel disabled for large images")
            print(f"[WARN] Using single GPU: {device} to avoid potential deadlocks")
            print(f"[INFO] If you need multi-GPU, consider using DistributedDataParallel instead")
        else:
            print(f"[INFO] Using single GPU: {device}")
        
        # Disable cudnn benchmark for large images to avoid memory issues
        torch.backends.cudnn.benchmark = False
        
        # Create optimizer and loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        
        # 6. Training with pruning support
        print("[STEP] Training...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(1, cfg.training.epochs + 1):
            t_ep = time.time()
            
            # Train one epoch
            tr = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                cfg.logging.print_every_n_batches
            )
            
            # Evaluate validation set
            if len(val_loader.dataset) > 0:
                val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
            else:
                val_mse, val_mae = 0.0, 0.0
            
            train_losses.append(tr)
            val_losses.append(val_mse)
            
            # Update best validation loss
            if val_mse < best_val_loss:
                best_val_loss = val_mse
            
            # Report to Optuna for pruning
            trial.report(val_mse, step=epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                print(f"[PRUNED] Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
            
            # Print progress
            if epoch % cfg.logging.print_every_n_epochs == 0:
                print(f"Epoch {epoch:03d}/{cfg.training.epochs} | "
                      f"train MSE={tr:.6f} | val MSE={val_mse:.6f} | "
                      f"val MAE={val_mae:.6f} | {time.time()-t_ep:.2f}s")
        
        # 7. Evaluate on test set
        print("[STEP] Testing...")
        test_mse, test_mae, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        print(f"Test  | MSE={test_mse:.6f} | MAE={test_mae:.6f}")
        
        # Print per-target results
        if y_pred.shape[1] > 1:
            print(f"[INFO] Per-target results:")
            for i in range(y_pred.shape[1]):
                target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
                target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
                print(f"  Target {i+1}: MSE={target_mse:.6f}, MAE={target_mae:.6f}")
        
        # 8. Save model
        weights_dir = os.path.join(output_dir, "weights")
        # Save underlying module state_dict if wrapped by DataParallel
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, os.path.join(weights_dir, cfg.output.model_filename))
        
        # 9. Save trial results
        save_trial_results(
            trial, cfg, output_dir,
            train_losses, val_losses,
            best_val_loss,
            test_mse, test_mae,
            y_pred, y_true
        )
        
        # Set user attributes for Optuna
        trial.set_user_attr('test_mse', float(test_mse))
        trial.set_user_attr('test_mae', float(test_mae))
        trial.set_user_attr('best_val_loss', float(best_val_loss))
        trial.set_user_attr('output_dir', output_dir)
        
        elapsed = time.time() - t0
        print(f"\n[OK] Trial {trial.number} completed in {elapsed:.2f}s")
        print(f"Validation Loss: {best_val_loss:.6f}")
        print(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}\n")
        
        # Clear GPU memory after trial
        if torch.cuda.is_available():
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        return best_val_loss
        
    except optuna.TrialPruned:
        # Clean up if pruned
        print(f"[INFO] Cleaning up pruned trial {trial.number}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        raise
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        # Clear GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        raise


def find_trial_output_dir(trial_number: int) -> str:
    """
    Find output directory for a given trial number.
    
    Args:
        trial_number: Optuna trial number
    
    Returns:
        Output directory path or None if not found
    """
    for root, dirs, files in os.walk(OUTPUTS_ROOT):
        if 'trial_info.yaml' in files:
            trial_info_path = os.path.join(root, 'trial_info.yaml')
            try:
                trial_info = OmegaConf.load(trial_info_path)
                if trial_info.get('trial_number') == trial_number:
                    return root
            except Exception:
                continue
    return None


def generate_study_summary(study: optuna.Study, output_dir: str):
    """
    Generate study summary JSON file.
    
    Args:
        study: Optuna study object
        output_dir: Output directory for Optuna files
    """
    summary = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'best_trial': {
            'number': study.best_trial.number,
            'value': study.best_trial.value,
            'params': study.best_trial.params,
            'user_attrs': study.best_trial.user_attrs
        },
        'trials_summary': [
            {
                'number': t.number,
                'value': t.value,
                'state': t.state.name,
                'params': t.params,
                'user_attrs': t.user_attrs
            }
            for t in study.trials
        ]
    }
    
    summary_path = os.path.join(output_dir, 'study_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save to outputs_real
    outputs_summary_path = os.path.join(OUTPUTS_ROOT, 'optuna_study_summary.json')
    with open(outputs_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[OK] Study summary saved to {summary_path}")


def main():
    """
    Main function to run Optuna optimization.
    """
    # Create Optuna directory
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    
    # Create study database
    study_db_path = os.path.join(OPTUNA_DIR, 'study.db')
    storage = optuna.storages.RDBStorage(
        url=f'sqlite:///{study_db_path}',
        engine_kwargs={'pool_size': 20}
    )
    
    # Create or load study
    # Use new study name to avoid compatibility issues with changed hyperparameter spaces
    study_name = 'cnn_hyperparameter_optimization_v2'  # Changed to v2 for new batch_size range
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # Minimize validation loss
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,      # Don't prune first 5 trials
                n_warmup_steps=10,        # Wait 10 epochs before pruning
                interval_steps=1          # Check every epoch
            )
        )
        print(f"[INFO] Loaded existing study: {study_name}")
    except Exception as e:
        print(f"[INFO] Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
    
    print(f"[INFO] Study database: {study_db_path}")
    print(f"[INFO] Number of existing trials: {len(study.trials)}\n")
    
    # Clear GPU memory before starting optimization
    if torch.cuda.is_available():
        print("[INFO] Clearing GPU memory before optimization...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        print("[OK] GPU memory cleared")
    
    # Optimize
    print(f"[INFO] Starting optimization...")
    print(f"[INFO] Base config: {BASE_CONFIG_PATH}\n")
    
    study.optimize(
        lambda trial: objective(trial, BASE_CONFIG_PATH),
        n_trials=20,  # Adjust as needed
        n_jobs=1,      # Set to 1 for GPU usage, increase for parallel CPU trials
        show_progress_bar=True
    )
    
    # Print best trial results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Value (Validation Loss): {study.best_trial.value:.6f}")
    print(f"Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nTest Metrics:")
    print(f"  Test MSE: {study.best_trial.user_attrs.get('test_mse', 'N/A')}")
    print(f"  Test MAE: {study.best_trial.user_attrs.get('test_mae', 'N/A')}")
    print(f"{'='*60}\n")
    
    # Find and create symlink to best trial
    best_trial_dir = find_trial_output_dir(study.best_trial.number)
    if best_trial_dir:
        best_link = os.path.join(OUTPUTS_ROOT, 'optuna_best')
        if os.path.exists(best_link) or os.path.islink(best_link):
            if os.path.islink(best_link):
                os.remove(best_link)
            else:
                shutil.rmtree(best_link)
        os.symlink(best_trial_dir, best_link)
        print(f"[OK] Best trial directory linked to: {best_link}")
        print(f"      Original: {best_trial_dir}\n")
    
    # Generate summary
    generate_study_summary(study, OPTUNA_DIR)
    
    # Save best trial info
    best_trial_info = {
        'trial_number': study.best_trial.number,
        'validation_loss': study.best_trial.value,
        'params': study.best_trial.params,
        'user_attrs': study.best_trial.user_attrs,
        'output_dir': best_trial_dir if best_trial_dir else None
    }
    best_trial_info_path = os.path.join(OPTUNA_DIR, 'best_trial_info.yaml')
    OmegaConf.save(best_trial_info, best_trial_info_path)
    print(f"[OK] Best trial info saved to {best_trial_info_path}")
    
    print(f"\n[INFO] All results saved to:")
    print(f"  Optuna files: {OPTUNA_DIR}")
    print(f"  Trial outputs: {OUTPUTS_ROOT}")
    print(f"  Best trial link: {os.path.join(OUTPUTS_ROOT, 'optuna_best')}")


if __name__ == "__main__":
    main()

